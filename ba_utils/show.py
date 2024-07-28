import math
import matplotlib.pyplot as plt
from IPython import display
from time import sleep

def flipbook(out, s=0):
    """
    Displays a sequence of outputs as a flipbook animation.
    
    Parameters:
    - out (list): A list of outputs to display as frames in the flipbook.
    - s (float): The time delay (in seconds) between each frame. Default is 0.
    """
    for o in out:
        display.clear_output(wait=True)
        display.display(o)
        sleep(s)

def grid(out, show=5, range=None, save=None):
    """
    Display a grid of images.

    Parameters:
    - out: list of images to display
    - show: number of images to show in the grid (default: 5)
    - range: range of indices to display (default: start to end equally distributed)
    - save: file path to save the grid as an image (default: None, does not save)

    """

    png = save[-4:] == '.png' if save else False
    # if saved as png optimize for dark background
    
    if(range is None):
        range = (0, len(out)-1)
    fig, ax = plt.subplots(1, show)
    fig.set_figwidth(20)
    for i, x in enumerate(ax):
        oidx = int((range[1]-range[0])*i/(show-1))+range[0]
        x.set_title(oidx, color='white' if png else 'black')
        x.imshow(out[oidx])
        x.axis('off')
    if save:
        if png:
            plt.savefig(save, bbox_inches='tight', pad_inches=0, dpi=300, transparent=True)
        else:
            plt.savefig(save, bbox_inches='tight', pad_inches=0)

def grid_all(out, n_cols=3, max_width=20, save=None):
    """
    Display a grid of images.

    Args:
        out (list): List of images to display.
        n_cols (int, optional): Number of columns in the grid. Defaults to 3.
        max_width (int, optional): Maximum width of the grid in inches. Defaults to 20.
        save (str, optional): Filepath to save the grid image. Defaults to None.

    Returns:
        None
    """
    n_rows = math.ceil(len(out) / n_cols)

    aspect_ratio = max_width / n_cols
    height = n_rows * aspect_ratio
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(max_width, height))
    
    for i, ax in enumerate(axs.flat):
        ax.axis('off')
        if i < len(out):
            ax.imshow(out[i])   
            
    if save:
         plt.savefig(save, bbox_inches='tight', pad_inches=0)
    plt.show()

def single(out):
    """wraper for display function"""
    display.display(out)

