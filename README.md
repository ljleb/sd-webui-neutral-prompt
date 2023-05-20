# Neutral Prompt

Neutral prompt is an a1111 webui extension that replaces the webui's CFG denoising algorithm with a higher-quality implementation based on more recent research.

## Features

- [Perp-Neg](https://perp-neg.github.io/) cfg algorithm, using the `AND_PERP` keyword
- standard deviation based CFG rescaling (https://arxiv.org/abs/2305.08891)

## Known issues

- The webui does not support composable diffusion via [`AND`](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features#composable-diffusion) for samplers DDIM, PLMS and UniPC. Since Perp-Neg relies on composable diffusion, it falls back on the appropriate unmodified sampler implementation.

## Special Mentions

Special thanks to these people for helping make this extension possible:

- [Ai-Casanova](https://github.com/AI-Casanova) : shared mathematical knowledge, time and proof-testing of implementation to make the extension more robust
