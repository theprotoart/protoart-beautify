# Third-party notices

ProtoArt Beautify is licensed under the Apache License 2.0 (see `LICENSE`).
Copyright 2026 ProtoArt Beautify contributors.

## Design origin

The enhancement pipeline and its default settings follow the
[Finegrain Image Enhancer](https://huggingface.co/spaces/finegrain/finegrain-image-enhancer) Space (MIT), built on
Finegrain's [refiners](https://github.com/finegrain-ai/refiners) library (MIT). The code in this repository was
written for ProtoArt Beautify; refiners is used as a library, not copied.

## Code included from other projects

| File | Origin | Licence |
| --- | --- | --- |
| `engine/restoreformer_arch.py` | [GFPGAN](https://github.com/TencentARC/GFPGAN) `gfpgan/archs/restoreformer_arch.py`, © 2021 THL A29 Limited (Tencent); RestoreFormer by Wang et al., CVPR 2022 | Apache-2.0, see `engine/LICENSE-gfpgan-restoreformer.txt` |
| `engine/codeformer_arch.py` | [CodeFormer](https://github.com/sczhou/CodeFormer) `basicsr/archs/vqgan_arch.py` and `codeformer_arch.py` @ b33cc7d, © 2022 S-Lab; CodeFormer by Zhou et al., NeurIPS 2022 | **S-Lab License 1.0 (non-commercial use only)**, see `engine/LICENSE-codeformer.txt` |

The CodeFormer code is only used when you choose the CODE face model. Its licence permits redistribution and use for
non-commercial purposes; for commercial use, contact its authors.

## Software the app is built with (bundled in the macOS app)

These are not stored in this repository. The packaged app includes them, each with its own licence file (inside
`ProtoArt Beautify.app/Contents/Resources/site-packages/*.dist-info/`).

| Component | Licence |
| --- | --- |
| Python (python-build-standalone) | PSF License |
| PyTorch, torchvision | BSD-3-Clause |
| refiners | MIT |
| spandrel | MIT |
| safetensors, huggingface_hub | Apache-2.0 |
| NumPy | BSD-3-Clause |
| Pillow | MIT-CMU (HPND) |
| jaxtyping, einops | MIT |

## Models (not included: downloaded on first launch)

ProtoArt Beautify ships no model weights. They are downloaded from Hugging Face the first time the app runs, and each
has its own licence, which you accept by using it. **Some are non-commercial.**

| Model | Source | Licence |
| --- | --- | --- |
| Juggernaut Reborn (SD 1.5) UNet, text encoder, autoencoder | `refiners/sd15.juggernaut.reborn.*` | CreativeML OpenRAIL-M |
| Tile ControlNet (SD 1.5) | `refiners/sd15.controlnet.tile` | CreativeML OpenRAIL-M |
| 4x-UltraSharp (ESRGAN) | `philz1337x/upscaler` | CC BY-NC-SA 4.0 (**non-commercial**) |
| JuggernautNegative embedding | `philz1337x/embeddings` | see its source |
| more_details, SDXLrender LoRAs | `philz1337x/loras` | see their sources |
| LCM-LoRA SD 1.5 (Turbo) | `latent-consistency/lcm-lora-sdv1-5` | OpenRAIL++ |
| RestoreFormer (Face Fix, downloaded on first use) | GFPGAN v1.3.4 release file, via `fofr/comfyui` (byte-identical, SHA-256 pinned) | see the GFPGAN repository |
| CodeFormer (Face Fix, downloaded on first use) | CodeFormer v0.1.0 release file, via `fofr/comfyui` (byte-identical, SHA-256 pinned) | S-Lab License 1.0 (**non-commercial**) |
| IP-Adapter Plus SD 1.5 (Likeness, downloaded on first use) | `refiners/sd15.ip_adapter.plus` (converted from `h94/IP-Adapter` `ip-adapter-plus_sd15`) | Apache-2.0 |
| OpenCLIP ViT-H/14 image encoder (Likeness, downloaded on first use) | `refiners/sd21.unclip.image_encoder` | OpenRAIL++ |
| 4x-UltraSharp V2 and V2 Lite (Upscaler, on first use) | `Kim2091/UltraSharpV2` | CC BY-NC-SA 4.0 (**non-commercial**) |
| 4x-AnimeSharp (Upscaler, on first use) | `Kim2091/AnimeSharp` | CC BY-NC-SA 4.0 (**non-commercial**) |
| 4xNomosWebPhoto RealPLKSR, 4xNomos8kSC (Upscaler, on first use) | Philip Hofmann's releases, `github.com/Phhofm/models` | CC BY 4.0 |
| Real-ESRGAN x4plus, general x4v3, x4plus Anime 6B (Upscaler, on first use) | `github.com/xinntao/Real-ESRGAN` releases | BSD-3-Clause |
| LaMa big-lama (Remove, on first use) | `michaelgold/big-lama` (safetensors of advimman/lama's big-lama) | Apache-2.0 |

Check these licences before any commercial use of pictures made with the AI enhancement.
