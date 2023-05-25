# Neutral Prompt

Neutral prompt is an a1111 webui extension that replaces the webui's CFG denoising algorithm with a higher-quality implementation based on more recent research.

## Features

- [Perp-Neg](https://perp-neg.github.io/) cfg algorithm, using the `AND_PERP` keyword
- standard deviation based CFG rescaling (https://arxiv.org/abs/2305.08891)

## Usage

Perp-Neg has been implemented using the `AND_PERP` prompt keyword, which stands for "perpendicular `AND`". In one sentence, `AND_PERP` takes advantage of composable diffusion to make it possible to prompt for concepts that would otherwise highly overlap with the regular prompts, by removing contradicting noise.

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

## Advanced features

### Nesting AND_PERP prompts

The extension provides a way to describe prompts that are to be orthogonalized with respect to other perpendicular prompts:

```
a red hot desert canion AND_PERP [
    cold blue everest montain :1
    AND a beautiful woman climbing a massive rocks wall :1.1
    AND_PERP far away, ugly, ants, water :-0.6
] :0.9
AND a rocky sahara climbing party :0.7
```

In this example, to obtain the final noise from the diffusion model, the extension will:

1. take the noise generated from the prompt `far away, ugly, ants, water :-0.6`
2. orthogonalize it with respect to `cold blue everest montain :1` and `a beautiful woman climbing a massive rocks wall :1.1` combined
3. add this orthogonal noise with the prompts it was orthogonalized against
4. orthogonalize the resulting noise with respect to `a red hot desert canion :1` and `a rocky sahara climbing party :0.7` combined
5. add this orthogonal noise multiplied by 0.9 with the prompts it was orthogonalized against

The resulting single noise map is composed of all the normal `AND` prompts + all the orthogonalized `AND_PERP` prompts.

In other words, each use of the `AND_PERP` keyword provides an isolated denoising space within its square brackets `[...]`, where the prompts inside of it are combined into a single noise map before being further processed down the prompt tree.

Experimentally, it does not seem useful to go beyond a depth of 2. I have yet to figure whether this allows to control more precisely the generations. If you find interesting ways of controling the generations using nested `AND_PERP` prompts, please let me know in the discussions!

![image](https://github.com/ljleb/sd-webui-neutral-prompt/assets/32277961/f6d0c95b-8efd-4ce2-b5e4-928597facd34)

## Known issues

- The webui does not support composable diffusion via [`AND`](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features#composable-diffusion) for samplers DDIM, PLMS and UniPC. Since Perp-Neg relies on composable diffusion, the extension will fallback on the appropriate unmodified sampler implementation whenever they are used.

## Special Mentions

Special thanks to these people for helping make this extension possible:

- [Ai-Casanova](https://github.com/AI-Casanova) : shared mathematical knowledge, time and proof-testing of implementation to make the extension more robust
