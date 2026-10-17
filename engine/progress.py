"""Progress + time-remaining estimation for one enhance run.

The work is planned up front (ESRGAN tiles, VAE encode, diffusion tile passes,
VAE decode). Each phase is timed as it runs; rates measured in this run win,
otherwise rates remembered from earlier runs (same memory profile) are used.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

Emit = Callable[[str, "float | None", "float | None"], None]  # message, fraction, eta_seconds

# Share of the bar each phase gets until real timings are known.
WEIGHTS = {"esrgan": 0.15, "encode": 0.05, "passes": 0.75, "decode": 0.05}


@dataclass
class Plan:
    esrgan_tiles: int
    passes: int        # diffusion tile passes = tiles x steps
    megapixels: float  # final output size


@dataclass
class Tracker:
    plan: Plan
    emit: Emit
    memory: dict[str, float] = field(default_factory=dict)  # persisted rates for this profile
    clock: Callable[[], float] = time.monotonic

    def __post_init__(self) -> None:
        self.t0 = self.clock()
        self.phase = "esrgan"
        self.phase_t0 = self.t0
        self.esrgan_done = 0
        self.passes_done = 0
        self.encoded = False
        self.decoded = False
        self.rate: dict[str, float] = {}  # measured this run
        self._frac = 0.0                  # never let the bar move backwards

    # ── rates ──────────────────────────────────────────────────────────────
    def _r(self, key: str) -> float | None:
        if key in self.rate:
            return self.rate[key]
        if key in self.memory:
            return self.memory[key]
        if key == "decode_per_mp":  # decode costs ~2x encode on SD1.5's VAE
            enc = self._r("encode_per_mp")
            return None if enc is None else 2.0 * enc
        return None

    # ── estimates ──────────────────────────────────────────────────────────
    def eta(self) -> float | None:
        p, mp = self.plan, max(self.plan.megapixels, 1e-6)
        parts: list[tuple[str, float, str]] = []  # (phase, units remaining, rate key)
        parts.append(("esrgan", p.esrgan_tiles - self.esrgan_done, "esrgan_tile"))
        parts.append(("encode", 0 if self.encoded else mp, "encode_per_mp"))
        parts.append(("passes", p.passes - self.passes_done, "pass"))
        parts.append(("decode", 0 if self.decoded else mp, "decode_per_mp"))
        total = 0.0
        now = self.clock()
        for phase, units, key in parts:
            if units <= 0:
                continue
            r = self._r(key)
            if r is None:
                return None
            est = units * r
            # Don't double-count time already spent in a one-shot phase (encode/decode).
            if phase == self.phase and phase in ("encode", "decode"):
                est = max(0.0, est - (now - self.phase_t0))
            total += est
        return total

    def fraction(self) -> float:
        self._frac = max(self._frac, self._raw_fraction())
        return self._frac

    def _raw_fraction(self) -> float:
        eta = self.eta()
        elapsed = self.clock() - self.t0
        if eta is not None and elapsed + eta > 0:
            return min(0.999, elapsed / (elapsed + eta))
        p = self.plan
        f = WEIGHTS["esrgan"] * (self.esrgan_done / max(p.esrgan_tiles, 1))
        f += WEIGHTS["encode"] * (1.0 if self.encoded else 0.0)
        f += WEIGHTS["passes"] * (self.passes_done / max(p.passes, 1))
        f += WEIGHTS["decode"] * (1.0 if self.decoded else 0.0)
        return min(0.999, f)

    def _send(self, message: str) -> None:
        self.emit(message, self.fraction(), self.eta())

    # ── events ─────────────────────────────────────────────────────────────
    def esrgan_tile_done(self) -> None:
        self.esrgan_done += 1
        self.rate["esrgan_tile"] = (self.clock() - self.phase_t0) / self.esrgan_done
        self._send(f"Pre-upscaling (ESRGAN 4×) — tile {self.esrgan_done}/{self.plan.esrgan_tiles}")

    def encode_start(self) -> None:
        self.phase, self.phase_t0 = "encode", self.clock()
        self._send("Encoding image")

    def encode_done(self) -> None:
        self.rate["encode_per_mp"] = (self.clock() - self.phase_t0) / max(self.plan.megapixels, 1e-6)
        self.encoded = True
        self.phase, self.phase_t0 = "passes", self.clock()
        self._send("Generating detail")

    def pass_done(self) -> None:
        self.passes_done += 1
        self.rate["pass"] = (self.clock() - self.phase_t0) / self.passes_done
        self._send(f"Generating detail — pass {self.passes_done}/{self.plan.passes}")

    def decode_start(self) -> None:
        self.phase, self.phase_t0 = "decode", self.clock()
        self._send("Decoding image")

    def decode_done(self) -> None:
        self.rate["decode_per_mp"] = (self.clock() - self.phase_t0) / max(self.plan.megapixels, 1e-6)
        self.decoded = True

    def finish(self) -> float:
        """Remember this run's rates for the next estimate; returns total seconds."""
        for k, v in self.rate.items():
            old = self.memory.get(k)
            self.memory[k] = v if old is None else 0.5 * old + 0.5 * v
        return self.clock() - self.t0
