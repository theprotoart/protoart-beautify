"""Memory profiles for the enhancer: how big the work is cut so each Mac stays out of swap.

The app's Memory switch sends "8gb" or "16gb"; "16gb" is refined here by the Mac's real memory into Protoart Studio's
measured 16 GB and 24 GB+ tiers. No PyTorch here, so the tests can load it on any machine.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

ESRGAN_OVERLAP = 16
# Attention is computed in chunks of this many query tokens. At 512 px tiles (4096 tokens) this
# cuts the attention working memory ~4x, which keeps 8 GB Macs out of swap.
ATTENTION_SLICE = 1024


@dataclass(frozen=True)
class MemoryProfile:
    name: str
    max_input_short_side: int   # input is downscaled to this before enhancing
    max_output_megapixels: float
    latent_tile: tuple[int, int]  # legacy fixed tile (H, W) in latent units; unused by the planner
    tile_tokens: int              # max latent tokens per diffusion tile (planner picks the fewest tiles)
    vae_tile: int               # pixels
    esrgan_tile: int            # pixels (input side)
    esrgan_overlap: int = ESRGAN_OVERLAP
    attention_slice: int | None = ATTENTION_SLICE  # query tokens per attention chunk (None = all at once)
    low_memory: bool = True     # hand cached GPU memory back to macOS after every pass


PROFILES = {
    # 64x64 latent tiles = 512 px, SD1.5's native size: attention stays ~0.3 GB.
    "8gb": MemoryProfile("8 GB", 512, 8.0, (64, 64), 9216, 512, 192),  # 9216 = 96x96: a 768 px square in one tile
    # The app's "16 GB+" switch: Protoart Studio's measured tiers, picked by the Mac's real memory (16gb_tier).
    # 16-23 GB: a smaller tile budget and medium attention chunks keep it out of swap.
    "16gb": MemoryProfile("16 GB", 768, 16.0, (144, 112), 12288, 1024, 256, attention_slice=2048),
    # 24 GB and more: the Space's settings, whole-tile attention, big upscaler tiles, no cache flushing (faster).
    "24gb": MemoryProfile("24 GB+", 768, 16.0, (144, 112), 16384, 1024, 512, esrgan_overlap=32,
                          attention_slice=None, low_memory=False),
}


def ram_gb() -> float:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except (ValueError, OSError, AttributeError):
        return 0.0


def resolve_profile(name: str, ram: float | None = None) -> tuple[str, MemoryProfile]:
    """The app sends "8gb" or "16gb" (its Memory switch); "16gb" becomes the 16 GB or 24 GB+ tier by real memory."""
    if name != "16gb":
        return "8gb", PROFILES["8gb"]
    ram = ram_gb() if ram is None else ram
    key = "24gb" if ram >= 23.5 else "16gb"
    return key, PROFILES[key]
