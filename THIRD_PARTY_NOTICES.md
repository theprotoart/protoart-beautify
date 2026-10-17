# Third-party notices

ProtoArt Beautify is licensed under the Apache License 2.0 (see `LICENSE`).
Copyright 2026 ProtoArt Beautify contributors.

## Design origin

The enhancement pipeline and its default settings follow the
[Finegrain Image Enhancer](https://huggingface.co/spaces/finegrain/finegrain-image-enhancer) Space (MIT), built on
Finegrain's [refiners](https://github.com/finegrain-ai/refiners) library (MIT). The code in this repository was
written for ProtoArt Beautify; refiners is used as a library, not copied.

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
| Juggernaut Reborn (SD 1.5) UNet, text encoder, autoencoder | `refiners/juggernaut.reborn.sd1_5.*` | CreativeML OpenRAIL-M |
| Tile ControlNet (SD 1.5) | `refiners/controlnet.sd1_5.tile` | CreativeML OpenRAIL-M |
| 4x-UltraSharp (ESRGAN) | `philz1337x/upscaler` | CC BY-NC-SA 4.0 (**non-commercial**) |
| JuggernautNegative embedding | `philz1337x/embeddings` | see its source |
| more_details, SDXLrender LoRAs | `philz1337x/loras` | see their sources |
| LCM-LoRA SD 1.5 (Turbo) | `latent-consistency/lcm-lora-sdv1-5` | OpenRAIL++ |

Check these licences before any commercial use of pictures made with the AI enhancement.
