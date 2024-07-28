import seaborn as sns
import matplotlib.pyplot as plt

from math import log2
import torch
import numpy as np
import os
from pandas import DataFrame as df

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from pytorch_fid import fid_score


v = 4

model = None
config = None
transform = None


def entropy(probabilities):
    """
    Calculate the entropy of a probability distribution.

    Parameters:
    probabilities (list): A list of probabilities.

    Returns:
    float: The entropy value.

    """
    return -sum([p*log2(p) for p in probabilities])

def load_model():
    """
    Loads the model if it is not already loaded.
    
    This function checks if the model, config, and transform variables are None. If any of them is None, it loads the model, resolves the data configuration, and creates the transformation.
    """
    global model, config, transform
    if model is None:
        print("Loading model - this takes some time...")
        model = timm.create_model(f'inception_v{v}', pretrained=True)
        model.eval()
    if config is None:
        config = resolve_data_config({}, model=model)
    if transform is None:
        transform = create_transform(**config)

def ice(images):
    """
    Calculate Inception Class Entropy of provided images.

    Parameters:
    - images: A list of PIL images.

    Returns:
    - entropies: A list of float values representing the calculated entropies.
    """
    
    global model, config, transform
    load_model()

    # preprocess
    prepped_images = [transform(image.convert('RGB')).unsqueeze(0) for image in images]
    batch = torch.cat(prepped_images, dim=0)

    # get weights
    with torch.no_grad():
        out = model(batch)
    probabilities = torch.nn.functional.softmax(out, dim=1)

    # calculate entropy
    entropies = [entropy(prob) for prob in probabilities]
    return entropies








def fid(images):
    """
    Calculate FID between COCO subset and provided images

    Parameters:
    - images: A list of images to compare with the COCO subset.

    Returns:
    - fid_value: The calculated FID (Fréchet Inception Distance) score.
    """

    # create folder if not exists
    if not os.path.exists('gen_out'):
        os.makedirs('gen_out')
    if not os.path.exists('gen_out/fid'):
        os.makedirs('gen_out/fid')

    # clear all images
    def clear():
        for f in os.listdir('gen_out/fid'):
            if f.endswith('.png') or f.endswith('.jpg'):
                os.remove(f'gen_out/fid/{f}')

    clear()

    # save all images
    for i, img in enumerate(images):
        img.resize((299,299)).save(f"gen_out/test/img_{i}.png")
    
    fid_value = fid_score.calculate_fid_given_paths(['data/dump/val.npz', 'gen_out/test'], 50, 'cuda', 2048)
    clear()

    return fid_value


