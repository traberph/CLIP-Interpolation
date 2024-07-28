from diffusers.image_processor import VaeImageProcessor
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

import torch
import numpy as np
from tqdm.auto import tqdm
from ba_utils.vector import getNoise
import umap


repo = 'CompVis/stable-diffusion-v1-4'  # works


# just include umap to this file to reduce clutter in notebooks
UMAP = umap.UMAP



class CLIP():
    """
    CLIP (Contrastive Language-Image Pretraining) class for creating text embeddings.

    Args:
        gpu (bool): Whether to use GPU for computation. Default is True.
        gpu_id (int): GPU device ID to use. Default is 0.

    Attributes:
        device (str): Device to use for computation (either 'cuda:<gpu_id>' or 'cpu').
        tokenizer (CLIPTokenizer): Tokenizer for CLIP model.
        text_encoder (CLIPTextModel): Text encoder model for CLIP.

    Methods:
        get_config(): Get the configuration of the text encoder model.
        embed(text, pooled=False): Create embeddings from text.
        tokenize(text): Create tokens from text.
        encode(tokens, pooled=False): Create embeddings from tokens.
        getEmpty(batch_size): Get an empty embedding.

    """

    def __init__(self, gpu=True, gpu_id=0):
        """Load and initialize CLIP"""
        self.device = f'cuda:{gpu_id}' if gpu else 'cpu'
        self.tokenizer = CLIPTokenizer.from_pretrained(repo, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(repo, subfolder="text_encoder").to(self.device)

    def get_config(self):
        """
        Get the configuration of the text encoder model.

        Returns:
            dict: Configuration of the text encoder model.
        """
        return self.text_encoder.config
    
    def embed(self, text, pooled=False):
        """
        Create embeddings from text.

        Args:
            text (str or List[str]): Input text or list of texts.
            pooled (bool): Whether to return pooled embeddings. Default is False.

        Returns:
            torch.Tensor: Embeddings of the input text.
        """
        tokens = self.tokenize(list(text))
        return self.encode(tokens, pooled)

    def tokenize(self, text):
        """
        Create tokens from text.

        Args:
            text (str or List[str]): Input text or list of texts.

        Returns:
            dict: Tokenized input text.
        """
        return self.tokenizer(text, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')

    def encode(self, tokens, pooled=False):
        """
        Create embeddings from tokens.

        Args:
            tokens (dict): Tokenized input text.
            pooled (bool): Whether to return pooled embeddings. Default is False.

        Returns:
            torch.Tensor: Embeddings of the input tokens.
        """
        with torch.no_grad():
            output = self.text_encoder(tokens.input_ids.to(self.device))
        return output.pooler_output if pooled else output

    def getEmpty(self, batch_size):
        """
        Get an empty embedding.

        Args:
            batch_size (int): Number of empty embeddings to generate.

        Returns:
            torch.Tensor: Empty embeddings.
        """
        uncond_input = self.tokenizer([""] * batch_size, padding="max_length", max_length=self.tokenizer.model_max_length, return_tensors="pt")
        return self.text_encoder(uncond_input.input_ids.to(self.device))[0]
        

class UNET():
    """
    A class representing the U-Net model and providing functions to apply it.
    Args:
        gpu_id (int): The ID of the GPU to use.
        clip (CLIP, optional): The CLIP instance to use. If not provided, a new instance will be created.
    
    Attributes:
        unet (UNet2DConditionModel): The U-Net model.
        scheduler (PNDMScheduler): The scheduler for the U-Net model.
        device (str): The device to run the model on.
        clip (CLIP): The CLIP instance used by the U-Net model.
        num_inference_steps (int): The number of inference steps.
        guidance_scale (float): The scale factor for the guidance.
        max_length (int): The maximum length.
    """
    
    def __init__(self, gpu_id=0, clip=None):
        """Load and initialize the U-Net model.
        
        Args:
            gpu_id (int, optional): The ID of the GPU to use. Defaults to 0.
            clip (CLIP, optional): The CLIP instance to use. If not provided, a new instance will be created.
        """
        
        self.unet = UNet2DConditionModel.from_pretrained(repo, subfolder="unet", use_safetensors=True)
        self.scheduler = PNDMScheduler.from_pretrained(repo, subfolder="scheduler")

        self.device = f'cuda:{gpu_id}'
        self.unet.to(self.device)

        if clip is not None:
            self.clip = clip
        else:
            print('UNET created own clip instance')
            self.clip = CLIP(gpu_id=self.device)

        self.num_inference_steps = 25
        self.guidance_scale = 7.5
        self.max_length = 77
        
    def iterate(self, text_embeddings, seed=0):
        """Denoise latents with U-Net conditioned on text_embeddings.
        
        Args:
            text_embeddings (torch.Tensor): The text embeddings.
            seed (int, optional): The seed for random number generation. Defaults to 0.
        
        Returns:
            torch.Tensor: The denoised latents.
        """

        batch_size = len(text_embeddings)
        uncond_embeddings = self.clip.getEmpty(batch_size)
        text_embeddings_cfg = torch.cat([uncond_embeddings, text_embeddings])
        
        latents = getNoise(batch_size, seed=seed)
        latents = latents.to(self.device)
        latents = latents * self.scheduler.init_noise_sigma
        self.scheduler.set_timesteps(self.num_inference_steps)
        for t in tqdm(self.scheduler.timesteps):
            latent_model_input = torch.cat([latents] *2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep=t)
        
            with torch.no_grad():
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings_cfg).sample
        
            noise_pred_uncond, noise_pred_text =noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
        
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        return latents

class VAE():
    """
    Variational Autoencoder (VAE) class.
    
    This class represents a Variational Autoencoder model. It provides methods for loading and initializing the VAE,
    as well as decoding a batch of latents.
    """
    
    def __init__(self, gpu_id=0):
        """Load and initialize VAE"""
        
        self.vae = AutoencoderKL.from_pretrained(repo, subfolder="vae", use_safetensors=True)
        self.vae.to(f'cuda:{gpu_id}')
        
    def decode(self, latent):
        """Decode a batch of latents"""
        latents_scaled = 1 / 0.18215 * latent
        with torch.no_grad():
            self.vae.enable_slicing()
            image = self.vae.decode(latents_scaled).sample
        processor = VaeImageProcessor()
        return processor.postprocess(image)

class Generator():
    def __init__(self, gpu_id=0, clip=None):
        """
        Initializes a Wrapper for the Stable Diffusion pipeline.

        Args:
            gpu_id (int): The ID of the GPU to use. Defaults to 0.
            clip (object): An instance of the CLIP class. If not provided, a new instance will be created.

        """
        self.clip = clip if clip is not None else CLIP(gpu_id=gpu_id)
        self.unet = UNET(gpu_id=gpu_id, clip=self.clip)
        self.vae = VAE(gpu_id=gpu_id)
    
    def pipe(self, text, seed=0):
        """
        Performs the stable diffusion pipeline.

        Args:
            text (str): The input text.
            seed (int): The seed value for random number generation. Defaults to 0.

        Returns:
            The output of the pipeline.

        """
        te = self.clip.embed(text)
        ie = self.unet.iterate(te[0], seed)
        o = self.vae.decode(ie)
        return o

    def pipe2(self, clipe, seed=0):
        """
        Performs the pipeline without text encoder.

        Args:
            clipe (object): The input object.
            seed (int): The seed value for random number generation. Defaults to 0.

        Returns:
            The output of the pipeline.

        """
        ie = self.unet.iterate(clipe, seed)
        o = self.vae.decode(ie)
        return o
        
        


def pyarrow_to_torch(batch, gpu=True):
    """
    Convert a PyArrow batch to a Torch tensor.

    Args:
        batch (pyarrow.Table): The PyArrow batch to convert.

    Returns:
        torch.Tensor: The converted Torch tensor.
    """
    batch_np = batch.to_numpy(zero_copy_only=False)
    batch_float = np.array([[i.astype(np.float32) for i in j] for j in batch_np])
    batch_tensor = torch.tensor(batch_float, dtype=torch.float32)
    return batch_tensor.to('cuda' if gpu else 'cpu')
        









        
