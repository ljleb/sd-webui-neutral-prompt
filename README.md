# Neutral Prompt

Neutral prompt is an a1111 webui extension that replaces the webui's CFG denoising algorithm with a higher-quality implementation based on more recent research.

## Features

- [Perp-Neg](https://perp-neg.github.io/) cfg algorithm, using the `AND_PERP` keyword
- standard deviation based CFG rescaling (https://arxiv.org/abs/2305.08891)

## Usage

Perp-Neg has been implemented using the `AND_PERP` prompt keyword, which stands for "perpendicular `AND`". In one sentence, `AND_PERP` takes advantage of composable diffusion to make it possible prompt for concepts that would otherwise highly overlap with the regular prompts, by removing contradicting noise.

Another way too look at it is that if `AND` prompts are "greedy" (meaning they will try to take as much space as possible in the output), then `AND_PERP` prompts are as lazy as possible, giving up easilly as soon as there is a disagreement in the generated output.

## Examples

Here is an example to illustrate this idea. Prompt:

`beautiful castle landscape AND monster house castle :-1`

This is an XY grid with prompt S/R `AND, AND_PERP`:

![image](https://github.com/ljleb/sd-webui-neutral-prompt/assets/32277961/29f3cf34-2ed4-45d2-b73a-b6fadec21d61)

Takeaways:

- You can see that the dynamic range of the picture is much greater in the `AND_PERP` images than in the `AND` images.
- The `AND` images also somewhat struggle to create a castle sometimes, where that isn't the case for `AND_PERP` images.
- The `AND` images are skewed towards a color similar to purple in this instance, because this was the path of least resistance between the two contradicting prompts during generation. On the left, there is no struggle to generate one thing or a different one, so the image is much clearer.

## Known issues

- The webui does not support composable diffusion via [`AND`](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features#composable-diffusion) for samplers DDIM, PLMS and UniPC. Since Perp-Neg relies on composable diffusion, the extension will fallback on the appropriate unmodified sampler implementation whenever they are used.

## Special Mentions

Special thanks to these people for helping make this extension possible:

- [Ai-Casanova](https://github.com/AI-Casanova) : shared mathematical knowledge, time and proof-testing of implementation to make the extension more robust
