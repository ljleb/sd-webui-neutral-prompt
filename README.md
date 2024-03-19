# Neutral Prompt

Neutral prompt is an a1111 webui extension that adds alternative composable diffusion keywords to the prompt language. It enhances the original implementation using more recent research.

## Features

- Now compatible wih [stable-diffusion-webui-forge](https://github.com/lllyasviel/stable-diffusion-webui-forge)!
- [Perp-Neg](https://perp-neg.github.io/) orthogonal prompts, invoked using the `AND_PERP` keyword
- saliency-aware noise blending, invoked using the `AND_SALT` keyword (credits to [Magic Fusion](https://magicfusion.github.io/) for the algorithm used to determine SNB maps from epsilons)
- semantic guidance top-k filtering, invoked using the `AND_TOPK` keyword (reference: https://arxiv.org/abs/2301.12247)
- standard deviation based CFG rescaling (Reference: https://arxiv.org/abs/2305.08891, section 3.4)

## Usage

### Keyword `AND_PERP`

The `AND_PERP` keyword, standing for "PERPendicular `AND`", integrates the orthogonalization process described in the Perp-Neg paper. Essentially, `AND_PERP` allows for prompting concepts that highly overlap with regular prompts, by negating contradicting concepts.

You could visualize it as such: if `AND` prompts are "greedy" (taking as much space as possible in the output), `AND_PERP` prompts are opposite, relinquishing control as soon as there is a disagreement in the generated output.

### Keyword `AND_SALT`

Saliency-aware blending is made possible using the `AND_SALT` keyword, shorthand for "SALienT `AND`". In essence, `AND_SALT` keeps the highest activation pixels at each denoising step.

Think of it as a territorial dispute: the image generated by the `AND` prompts is one country, and the images generated by `AND_SALT` prompts represent neighbouring nations. They're all vying for the same land - whoever strikes the strongest at a given time (denoising step) and location (latent pixel) claims it.

### Keyword `AND_TOPK`

The `AND_TOPK` keyword refers to "TOP-K filtering". It keeps only the "k" highest activation latent pixels in the noise map and discards the rest. It works similarly to `AND_SALT`, except that the high-activation regions are simply added instead of replacing previous content.

Currently, k is constantly 5% of all latent pixels, meaning 95% of the weakest latent pixel values at each step are discarded.

Top-k filtering is useful when you want to have a more targeted effect on the generated image. It should work best with smaller objects and details.

## Examples

### Using the `AND_PERP` Keyword

Here is an example to illustrate one use case of the `AND_PREP` keyword. Prompt:

`beautiful castle landscape AND monster house castle :-1`

This is an XY grid with prompt S/R `AND, AND_PERP`:

![image](https://github.com/ljleb/sd-webui-neutral-prompt/assets/32277961/29f3cf34-2ed4-45d2-b73a-b6fadec21d61)

Key observations:

- The `AND_PERP` images exhibit a higher dynamic range compared to the `AND` images.
- Since the prompts have a lot of overlap, the `AND` images sometimes struggle to depict a castle. This isn't a problem for the `AND_PERP` images.
- The `AND` images tend to lean towards a purple color, because this was the path of least resistance between the two opposing prompts during generation. In contrast, the `AND_PERP` images, free from this tug-of-war, present a clearer representation.

### Using the `AND_SALT` Keyword

The `AND_SALT` keyword can be used to invoke saliency-aware blending. It spotlights and accentuates areas of high-activation in the output.

Consider this example prompt utilizing `AND_SALT`:

```
a vibrant rainforest with lush green foliage
AND_SALT the glimmering rays of a golden sunset piercing through the trees
```

In this case, the extension identifies and isolates the most salient regions in the sunset prompt. Then, the extension applies this marsked image to the rainforest prompt. Only the portions of the rainforest prompt that coincide with the salient areas of the sunset prompt are affected. These areas are replaced by pixels from the sunset prompt.

This is an XY grid with prompt S/R `AND_SALT, AND, AND_PERP`:

![xyz_grid-0008-1564977627-a vibrant rainforest with lush green foliage_AND_SALT the glimmering rays of a golden sunset piercing through the trees](https://github.com/ljleb/sd-webui-neutral-prompt/assets/32277961/2404f20b-47f6-457f-b4c5-76b9fd919345)

Key observations:

- `AND_SALT` behaves more diplomatically, enhancing areas where its impact makes the most sense and aligning with high activity regions in the output
- `AND` gives equal weight to both prompts, creating a blended result
- `AND_PERP` will find its way through anything not blocked by the regular prompt

## Advanced Features

### Nesting prompts

The extension supports nesting of all prompt keywords including `AND`, allowing greater flexibility and control over the final output. Here's an example of how these keywords can be combined:

```
magical tree forests, eternal city
AND_PERP [
    electrical pole voyage
    AND_SALT small nocturne companion
]
AND_SALT [
    electrical tornado
    AND_SALT electric arcs, bzzz, sparks
]
```

To generate the final image from the diffusion model:

1. The extension first processes the root `AND` prompts. In this case, it's just `magical tree forests, eternal city`
2. It then processes the `AND_SALT` prompt `small nocturne companion` in the context of `electrical pole voyage`. This enhances salient features in the `electrical pole voyage` image
3. This new image is orthogonalized with the image from `magical tree forests, eternal city`, blending the details of the 'electrical pole voyage' into the main scene without creating conflicts
4. The extension then turns to the second `AND_SALT` group. It processes `electric arcs, bzzz, sparks` in the context of `electrical tornado`, amplifying salient features in the electrical tornado image
5. The image from this `AND_SALT` group is then combined with the `magical tree forests, eternal city` image. The final output retains the strongest features from both the `electrical tornado` (enhanced by 'electric arcs, bzzz, sparks') and the earlier 'magical tree forests, eternal city' scene influenced by the 'electrical pole voyage'

Each keyword can define a distinct denoising space within its square brackets `[...]`. Prompts inside it merge into a single image before further processing down the prompt tree.

While there's no strict limit on the depth of nesting, experimental evidence suggests that going beyond a depth of 2 is generally unnecessary. We're still exploring the added precision from deeper nesting. If you discover innovative ways of controlling the generations using nested prompts, please share in the discussions!

![image](https://github.com/ljleb/sd-webui-neutral-prompt/assets/32277961/f16587fe-2244-4832-a253-98f819a9e2e0)

## Special Mentions

Special thanks to these people for helping make this extension possible:

- [Ai-Casanova](https://github.com/AI-Casanova) : for sharing mathematical knowledge, time, and conducting proof-testing to enhance the robustness of this extension
