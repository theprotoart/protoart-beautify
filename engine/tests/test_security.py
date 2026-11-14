"""Security tests for ProtoArt Beautify (run: python3 -m unittest discover -s engine/tests -v).

They need only the Python standard library (Pillow too, for the image-decoder test, which is skipped
without it); no models, no PyTorch and no network.
"""

from __future__ import annotations

import importlib
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENGINE = ROOT / "engine"
SWIFT = ROOT / "app" / "ProtoArtBeautify.swift"
BUILD = ROOT / "build.sh"
sys.path.insert(0, str(ENGINE))


def load_models(data_dir: Path):
    """engine/models.py with its data folder pointed at a temporary directory."""
    os.environ["PAB_DATA_DIR"] = str(data_dir)
    import models
    return importlib.reload(models)


def swift_models() -> tuple[dict[str, tuple[str, str, str]], dict[str, str]]:
    """The app's download list and pinned hashes, read from the Swift source."""
    src = SWIFT.read_text()
    files = {m.group(1): (m.group(2), m.group(3), m.group(4)) for m in re.finditer(
        r'ModelFile\(key: "([^"]+)", repo: "([^"]+)", file: "([^"]+)",\s*rev: "([0-9a-f]{40})"', src)}
    block = src[src.index("static let pinnedSHA256"):]
    block = block[:block.index("]\n")]
    pins = dict(re.findall(r'"([a-z0-9_]+)": "([0-9a-f]{64})"', block))
    return files, pins


class ModelIntegrity(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.models = load_models(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def test_every_model_file_has_a_pinned_sha256(self):
        missing = [r.key for r in self.models.REMOTES if r.key not in self.models.PINNED_SHA256]
        self.assertEqual(missing, [], "every downloaded model file must be checked against a pinned SHA-256")
        for key, h in self.models.PINNED_SHA256.items():
            self.assertRegex(h, r"^[0-9a-f]{64}$", key)

    def test_app_and_engine_agree_on_files_revisions_and_hashes(self):
        files, pins = swift_models()
        remotes = {r.key: (r.repo_id, r.filename, r.revision) for r in self.models.REMOTES}
        self.assertEqual(files, remotes, "app/ProtoArtBeautify.swift and engine/models.py list different models")
        for key, h in pins.items():
            self.assertEqual(h, self.models.PINNED_SHA256.get(key), f"hash for {key} differs between app and engine")

    def test_model_repositories_use_their_current_names(self):
        # These four were renamed on Hugging Face; the old names only work through redirects.
        old = ["juggernaut.reborn.sd1_5.unet", "juggernaut.reborn.sd1_5.text_encoder",
               "juggernaut.reborn.sd1_5.autoencoder", "controlnet.sd1_5.tile"]
        for name in old:
            self.assertNotIn(name, SWIFT.read_text())
            self.assertNotIn(name, (ENGINE / "models.py").read_text())

    def test_tampered_large_model_is_deleted_before_it_is_converted(self):
        m = self.models
        m.DOWNLOADS.mkdir(parents=True)
        bad = m.DOWNLOADS / "unet.safetensors"
        bad.write_bytes(b"not the real model")
        with self.assertRaisesRegex(RuntimeError, "failed its security check"):
            m.ensure_models(lambda *a: None)
        self.assertFalse(bad.exists(), "a file that fails its check must be removed")
        self.assertFalse((m.FP16_DIR / "unet.fp16.safetensors").exists(), "it must never be converted or used")

    def test_tampered_pickle_file_is_deleted_before_loading(self):
        m = self.models
        p = Path(self.tmp.name) / "esrgan.pth"
        p.write_bytes(b"\x80\x04 pretend pickle")
        with self.assertRaises(RuntimeError):
            m._verify("esrgan", p)
        self.assertFalse(p.exists())

    def test_matching_file_passes(self):
        m = self.models
        p = Path(self.tmp.name) / "x.bin"
        p.write_bytes(b"hello")
        m.PINNED_SHA256["_test"] = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        try:
            m._verify("_test", p)  # no exception
        finally:
            del m.PINNED_SHA256["_test"]
        self.assertTrue(p.exists())


class ImageDecoding(unittest.TestCase):
    def test_engine_only_decodes_png_and_jpeg(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow not installed")
        import server
        with tempfile.TemporaryDirectory() as d:
            for fmt, ok in (("PNG", True), ("JPEG", True), ("TIFF", False), ("GIF", False), ("BMP", False)):
                p = Path(d) / f"x.{fmt.lower()}"
                Image.new("RGB", (8, 8), (10, 20, 30)).save(p, format=fmt)
                if ok:
                    self.assertEqual(server.open_image(str(p)).size, (8, 8))
                else:
                    with self.assertRaises(Exception, msg=f"{fmt} must be refused"):
                        server.open_image(str(p))


class BuildPinning(unittest.TestCase):
    """The build installs exactly the package versions that were reviewed, not whatever is newest that day."""

    PAIRS = (("requirements.txt", "constraints.txt"), ("requirements-intel.txt", "constraints-intel.txt"))

    @staticmethod
    def entries(path: Path) -> list[str]:
        out = []
        for line in path.read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                out.append(line)
        return out

    @staticmethod
    def name(req: str) -> str:
        return re.split(r"[<>=!~ @\[;]", req, maxsplit=1)[0].strip().lower().replace("_", "-")

    def test_every_package_is_pinned_to_an_exact_version(self):
        for req_file, con_file in self.PAIRS:
            con = ENGINE / con_file
            self.assertTrue(con.exists(), f"engine/{con_file} is missing")
            pinned = {}
            for line in self.entries(con):
                self.assertRegex(line, r"^[A-Za-z0-9_.\-]+==[0-9][0-9A-Za-z.+\-]*$", f"{con_file}: {line!r} is not an exact pin")
                pinned[self.name(line)] = line
            for req in self.entries(ENGINE / req_file):
                n = self.name(req)
                if "git+" in req:
                    self.assertRegex(req, r"@[0-9a-f]{40}$", f"{req_file}: {n} must be pinned to a commit")
                    continue
                self.assertIn(n, pinned, f"{req_file}: {n} has no exact version in {con_file}")
            for must in ("torch", "pillow", "numpy", "safetensors", "spandrel", "huggingface-hub", "filelock"):
                self.assertIn(must, pinned, f"{con_file} must pin {must}")

    def test_build_uses_the_pins(self):
        src = BUILD.read_text()
        self.assertIn('-c "$CON"', src, "build.sh must install with the constraints file")
        self.assertRegex(src, r'REQ_SHA="\$\(cat "\$REQ" "\$CON" \| shasum', "a change of pins must trigger a reinstall")
        self.assertRegex(src, r'PY_PATCH="3\.12\.\d+"', "the bundled Python must be an exact release")


if __name__ == "__main__":
    unittest.main()
