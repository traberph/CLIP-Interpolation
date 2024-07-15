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