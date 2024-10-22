# Reference Point-Based Interpolation of CLIP Embeddings for Controlling Text-To-Image Generation

### Abstract
This thesis investigates interpolation methods within the context of text-
to-image generation, focusing on the latent space of the CLIP (Contrastive
Language-Image Pretraining) model. Our work explores the effectiveness of
linear interpolation (lerp) and spherical linear interpolation (slerp) in generat-
ing coherent and smooth transitions between text prompts. Results indicate
that slerp outperforms lerp, particularly with complex prompts, by producing
more visually coherent images. Additionally, a novel reference-based interpo-
lation method is introduced, leveraging cosine similarity to guide the inter-
polation path through the latent space. While manual selection of reference
points demonstrated improved interpolation quality, automatic selection meth-
ods showed varying levels of success. Despite these advancements, limitations
related to dataset quality and the initial embeddings were identified, highlighting
areas for future research. The findings contribute to the broader understanding
of interpolation methods in multimodal AI, offering insights into sampling from
text-to-image generation models


### Structure

- 00 interpolates CLIP embeddings using lerp and slerp and generates images from the interpolated embeddings
- 01 evaluates the generated images from 00\\
- 02 uses manual selected prompts to improve the interpolation quality\\
- 03 evaluates the generated images from 02
- 04 for experiments with the EOT Token of CLIP
- 05 selects the reference points for the interpolation automatically
- 06 evaluates the generated images from 05

### Some Results

#### using slerp and reference points
![out5_semiautomatic_0](https://github.com/user-attachments/assets/b8bdca3b-23c7-46b2-8c93-a86c47369595)
![out5_semiautomatic_1](https://github.com/user-attachments/assets/b17e106f-3090-4a3d-9788-b9c72c5a02f0)
![out5_semiautomatic_2](https://github.com/user-attachments/assets/e647eab5-023e-4176-9801-e01764c768d7)

#### using slerp and reference points
![experiment5_manual_0](https://github.com/user-attachments/assets/a07e657f-c45d-4880-94dc-944871075b7e)
![experiment5_manual_1](https://github.com/user-attachments/assets/b24ed94e-39b6-41f5-96a2-bf6e576926a0)
![experiment5_manual_2](https://github.com/user-attachments/assets/3c34f203-7804-4295-aba8-7387e3645dd7)


