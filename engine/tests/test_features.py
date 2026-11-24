"""Tests for the features ported from Protoart Studio: the memory tiers, the upscaler choice and Remove spots.

Standard library + numpy + Pillow only (no models, no PyTorch, no network):
    python3 -m unittest discover -s engine/tests -v
"""

from __future__ import annotations

import ast
import importlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
ENGINE = ROOT / "engine"
sys.path.insert(0, str(ENGINE))


def upscalers_table() -> dict:
    """esrgan.UPSCALERS, read without importing PyTorch."""
    tree = ast.parse((ENGINE / "esrgan.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == "UPSCALERS":
            return ast.literal_eval(node.value)
    raise AssertionError("UPSCALERS not found")


class MemoryTiers(unittest.TestCase):
    def test_tiers_follow_the_macs_memory(self):
        import memory
        self.assertEqual(memory.resolve_profile("8gb", 64)[0], "8gb")      # the switch on 8 GB stays 8 GB
        self.assertEqual(memory.resolve_profile("16gb", 16)[0], "16gb")
        self.assertEqual(memory.resolve_profile("16gb", 18)[0], "16gb")
        self.assertEqual(memory.resolve_profile("16gb", 24)[0], "24gb")
        self.assertEqual(memory.resolve_profile("16gb", 96)[0], "24gb")
        self.assertEqual(memory.resolve_profile("something", 96)[0], "8gb")  # unknown: the safe one

    def test_bigger_tiers_never_use_less(self):
        from memory import PROFILES
        a, b, c = PROFILES["8gb"], PROFILES["16gb"], PROFILES["24gb"]
        self.assertTrue(a.tile_tokens <= b.tile_tokens <= c.tile_tokens)
        self.assertTrue(a.esrgan_tile <= b.esrgan_tile <= c.esrgan_tile)
        self.assertTrue(a.low_memory and b.low_memory and not c.low_memory)
        self.assertIsNone(c.attention_slice)
        self.assertEqual((a.max_input_short_side, a.max_output_megapixels), (512, 8.0))  # 8 GB unchanged
        # the app's size estimate (EnhanceSettings.predictedOutput) assumes 768 px / 16 MP for "16gb"
        for p in (b, c):
            self.assertEqual((p.max_input_short_side, p.max_output_megapixels), (768, 16.0))


class Upscalers(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["PAB_DATA_DIR"] = self.tmp.name
        import models
        self.models = importlib.reload(models)

    def tearDown(self):
        self.tmp.cleanup()

    def test_every_upscaler_is_a_pinned_download(self):
        table = upscalers_table()
        remotes = {r.key: r for r in self.models.REMOTES}
        self.assertIn("esrgan", table)
        for key in table:
            self.assertIn(key, remotes, key)
            self.assertIn(key, self.models.PINNED_SHA256, key)
        extra = {k for k in remotes if k.startswith("up_")} - set(table)
        self.assertEqual(extra, set(), "downloads that can't be picked")
        for key, (label, fp32, max_tile) in table.items():
            self.assertTrue(label and isinstance(fp32, bool) and max_tile >= 0, key)

    def test_full_precision_for_the_new_networks(self):
        table = upscalers_table()
        for key in ("up_ultrasharp_v2", "up_ultrasharp_v2_lite", "up_nomos_webphoto"):  # DAT and RealPLKSR
            self.assertTrue(table[key][1], key)

    def test_app_offers_the_same_upscalers(self):
        swift = (ROOT / "app" / "ProtoArtBeautify.swift").read_text()
        block = swift[swift.index("static let all: [Upscaler]"):]
        block = block[:block.index("]\n")]
        import re
        ids = re.findall(r'Upscaler\(id: "([^"]+)"', block)
        self.assertEqual(sorted(ids), sorted(upscalers_table()))

    def test_downloaded_only_finds_the_apps_file(self):
        m = self.models
        self.assertIsNone(m.downloaded("up_nomos8ksc"))
        m.DOWNLOADS.mkdir(parents=True)
        (m.DOWNLOADS / "up_nomos8ksc.pth").write_bytes(b"x")
        self.assertEqual(m.downloaded("up_nomos8ksc"), m.DOWNLOADS / "up_nomos8ksc.pth")
        self.assertIsNone(m.downloaded("not_a_model"))

    def test_optional_files_are_skipped_until_the_app_downloads_them(self):
        # ensure_models must not try to fetch the new add-ons itself (it would need huggingface_hub and the network)
        for key in list(upscalers_table()) + ["lama"]:
            if key == "esrgan":
                continue
            r = next(x for x in self.models.REMOTES if x.key == key)
            self.assertTrue(r.optional, key)


class FakeLaMa:
    """Stands in for spandrel's LaMa: fills the masked pixels with a flat colour, keeps the rest exactly."""

    def __init__(self):
        self.calls = []

    def run(self, desc, device, crop, crop_mask):
        self.calls.append(crop.size)
        out = crop.copy()
        out.paste((0, 200, 0), mask=crop_mask.point(lambda v: 255 if v > 127 else 0))
        return out


class RemoveSpots(unittest.TestCase):
    def setUp(self):
        import heal
        self.heal = heal
        self.fake = FakeLaMa()
        self._orig = heal._run
        heal._run = self.fake.run

    def tearDown(self):
        self.heal._run = self._orig

    def picture(self, w=600, h=400):
        rng = np.random.default_rng(1)
        return Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8))

    def test_only_the_painted_area_changes(self):
        img = self.picture()
        mask = Image.new("L", img.size, 0)
        ImageDraw.Draw(mask).ellipse((280, 180, 320, 220), fill=255)
        out = self.heal.heal(img, mask, (None, None))
        a, b = np.asarray(img).astype(int), np.asarray(out).astype(int)
        changed = np.abs(a - b).sum(axis=2) > 0
        self.assertTrue(changed[200, 300])                  # the middle of the spot is filled in
        far = np.ones(changed.shape, bool)
        far[160:240, 260:340] = False                        # beyond the grown, softened edge nothing moves
        self.assertFalse(changed[far].any())

    def test_spots_far_apart_are_healed_close_up(self):
        img = self.picture(1200, 800)
        mask = Image.new("L", img.size, 0)
        d = ImageDraw.Draw(mask)
        d.ellipse((50, 50, 70, 70), fill=255)
        d.ellipse((1100, 700, 1120, 720), fill=255)
        self.heal.heal(img, mask, (None, None))
        self.assertEqual(len(self.fake.calls), 2)           # one close-up each, not the whole picture
        for w, h in self.fake.calls:
            self.assertLess(max(w, h), 400)

    def test_regions_and_context_box(self):
        m = np.zeros((300, 300), bool)
        m[10:20, 10:20] = True
        m[250:260, 250:260] = True
        parts = self.heal.regions(m)
        self.assertEqual(len(parts), 2)
        self.assertEqual(int(sum(p.sum() for p in parts)), int(m.sum()))
        x0, y0, x1, y1 = self.heal.box(parts[0])
        self.assertTrue(0 <= x0 < 10 and 0 <= y0 < 10 and x1 >= 20 and y1 >= 20 and x1 <= 300 and y1 <= 300)
        self.assertGreaterEqual(x1 - x0, min(300, self.heal.MIN_CONTEXT))   # always some surroundings

    def test_nothing_painted_is_refused(self):
        img = self.picture(64, 64)
        with self.assertRaises(ValueError):
            self.heal.heal(img, Image.new("L", img.size, 0), (None, None))

    def test_engine_heal_command(self):
        import server
        img = self.picture(200, 120)
        with tempfile.TemporaryDirectory() as d:
            ip, mp, op = Path(d, "in.png"), Path(d, "mask.png"), Path(d, "out.png")
            img.save(ip)
            mask = Image.new("L", img.size, 0)
            ImageDraw.Draw(mask).rectangle((90, 50, 110, 70), fill=255)
            mask.save(mp)
            buf = io.StringIO()
            orig_out, orig_get = server._out, server.get_heal_model
            server._out, server.get_heal_model = buf, (lambda status: (None, None))
            try:
                server.handle({"id": "h1", "cmd": "heal", "input": str(ip), "mask": str(mp), "output": str(op)})
            finally:
                server._out, server.get_heal_model = orig_out, orig_get
            msgs = [json.loads(x) for x in buf.getvalue().splitlines()]
            done = [m for m in msgs if m.get("event") == "done"]
            self.assertEqual(len(done), 1)
            self.assertEqual((done[0]["id"], done[0]["width"], done[0]["height"]), ("h1", 200, 120))
            out = Image.open(op)
            self.assertEqual(out.format, "PNG")
            self.assertEqual(np.asarray(out.convert("RGB"))[60, 100].tolist(), [0, 200, 0])

    def test_heal_refuses_a_non_png_mask(self):
        import server
        img = self.picture(64, 64)
        with tempfile.TemporaryDirectory() as d:
            ip, mp = Path(d, "in.png"), Path(d, "mask.tif")
            img.save(ip)
            Image.new("L", img.size, 255).save(mp, format="TIFF")
            orig_out, orig_get = server._out, server.get_heal_model
            server._out, server.get_heal_model = io.StringIO(), (lambda status: (None, None))
            try:
                with self.assertRaises(Exception):
                    server.handle({"id": "h2", "cmd": "heal", "input": str(ip), "mask": str(mp),
                                   "output": str(Path(d, "o.png"))})
            finally:
                server._out, server.get_heal_model = orig_out, orig_get


if __name__ == "__main__":
    unittest.main()
