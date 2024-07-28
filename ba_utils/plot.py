import seaborn as sns
import numpy as np
from pandas import DataFrame as df
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from ba_utils.model import CLIP, UMAP
import pandas as pd

theme_color = '#dddddd'

def enable_darkmode():
    ax = plt.gca()
    
    # Set the spines to the theme color
    ax.spines['top'].set_color(theme_color)
    ax.spines['right'].set_color(theme_color)
    ax.spines['bottom'].set_color(theme_color)
    ax.spines['left'].set_color(theme_color)
    
    # Set the tick parameters to the theme color
    ax.tick_params(colors=theme_color)
    
    # Set the axis labels and title to the theme color
    ax.yaxis.label.set_color(theme_color)
    ax.xaxis.label.set_color(theme_color)
    ax.title.set_color(theme_color)
    
    # Set the figure background to be transparent
    fig = plt.gcf()
    fig.patch.set_alpha(0)
    
    # Set the axes background to be transparent
    ax.patch.set_alpha(0)
    
    # Customize the legend
    legend = ax.legend(facecolor='none', edgecolor=theme_color)
    for text in legend.get_texts():
        text.set_color(theme_color)

def save_plt(save):
    plt.savefig(save, bbox_inches='tight', pad_inches=0.01, transparent=True, dpi=300)

def plot(data, meta=None):
    """helper function to create labled scatterplot"""
    
    plt.rcParams['figure.figsize'] = [10, 10]
    if meta is None:
        p = sns.scatterplot(x=data[:,0], y=data[:,1])
    else:
        p = sns.scatterplot(x=data[:,0], y=data[:,1],hue=meta[:,1], style=meta[:,2], s=80)
        sns.move_legend(p, "upper left", bbox_to_anchor=(1, 1))


clip = None
def plot_prompts(prompts, dim=1, method='PCA', save=None, nominal=False, show_legend=False, rotation=90):
    """
    Plot prompts in a scatter plot.

    Args:
        prompts (list): List of prompts.
        dim (int, optional): Dimensionality of the plot. Defaults to 1.
        method (str, optional): Dimensionality reduction method. Can be 'PCA' or 'UMAP'. Defaults to 'PCA'.
        save (str, optional): File path to save the plot. Defaults to None.
        nominal (bool, optional): Whether the prompts are nominal values. Defaults to False.
        show_legend (bool, optional): Whether to show the legend. Defaults to False.
        rotation (int, optional): Rotation angle for the text labels. Defaults to 90.

    Returns:
        None
    """
    clip = CLIP()
    embeddings = clip.embed(prompts).pooler_output

    if method == 'PCA':
        scaled = StandardScaler().fit_transform(embeddings.cpu().numpy())
        pca = PCA(n_components=dim)
        reduced = pca.fit_transform(scaled)
    elif method == 'UMAP':
        umap = UMAP(n_components=dim)
        reduced = umap.fit_transform(embeddings.cpu().numpy())

    if dim == 1:        
        # plot
        data = np.array([[value[0],label] for value,label in sorted(zip(reduced, prompts))], dtype='object')
        plt.figure(figsize=(20,0.5))
        ax = sns.stripplot(x=data[:,0], jitter=False, hue=data[:,1], alpha=0.5, s=10, palette="tab10" if nominal else 'viridis')
        if show_legend:
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 5))
        else:
            plt.legend([],[], frameon=False)
            
        # add text labels to the plot
        for i, txt in enumerate(data[:,1]):
            plt.annotate(txt, (data[i,0], 0), ha='left', va='bottom', rotation=rotation, xytext=(-5,20), textcoords='offset points', fontsize=20)
            
    elif dim == 2:
        #sns.scatterplot(x=reduced[:,0], y=reduced[:,1], hue=prompts, s=100, palette="tab10")
        sns.scatterplot(x=reduced[:,0], y=reduced[:,1])
        plt.legend([],[], frameon=False)

        # add text labels to the plot
        x_mean, y_mean = np.mean(reduced[:,0]), np.mean(reduced[:,1])
        for i, txt in enumerate(prompts):
            ha = 'left' if reduced[i,0] < x_mean else 'right'
            px = 5 if reduced[i,0] < x_mean else -5
            va = 'bottom' if reduced[i,1] < y_mean else 'top'
            py = 5 if reduced[i,1] < y_mean else -5
            #plt.text(reduced[i,0], reduced[i,1], f'{txt}', rotation=0, fontsize=12, ha=ha, va=va, xytext=(px, py), textcoords='offset points')
            plt.annotate(txt, (reduced[i,0], reduced[i,1]), textcoords="offset points", xytext=(px, py), ha=ha, va=va, fontsize=12)
        
    else:
        print("Invalid dimensionality. Must be 1 or 2.")
        return

    
    if save:
        save_plt(save)
    plt.show()


### METRICS RELATED

def plot_ice(entropy, save=None):
    """
    Plots ICE values for a single interpolation row.
    Values have to be precomputed

    Parameters:
    - entropy: The ICE values to plot.
    - save: The file path to save the plot. If not provided, the plot will be just displayed.
    """
    entropy = np.array(entropy)
    sns.lineplot(x=range(len(entropy)), y=entropy)
    sns.scatterplot(x=range(len(entropy)), y=entropy, marker='x')

    # calculate the average entropy
    avg = np.mean(entropy)
    sns.lineplot(x=[0,len(entropy)-1], y=[avg, avg], linestyle='--')

    if save:
        save_plt(save)
    plt.show()


def plot_ice_batch(experiment, save=None, method_names=['lerp', 'slerp'], color_offset=0):
    """
    Plots the ICE (Individual Conditional Expectation) curves for a batch of experiments.

    Args:
        experiment (ndarray): The batch of experiments. Each experiment should be a 2D array.
        save (str, optional): The file path to save the plot. Defaults to None.
        method_names (list, optional): The names of the methods. Defaults to ['lerp', 'slerp'].
        color_offset (int, optional): The offset for the color palette. Defaults to 0.
    """
    
    experiment = np.array(experiment)
    means = []
    png = save[-4:] == '.png' if save else False
    
    if len(experiment.shape) == 2:
        experiment = np.expand_dims(experiment, axis=0)
    
    for idx, method in enumerate(experiment):
        method_array = np.array(method)
        data = []
        for i, row in enumerate(method_array):
            for j, value in enumerate(row):
                data.append([i, j, value])
        data = pd.DataFrame(data, columns=['seed', 'step', 'ice'])
        mean = method_array.mean()
        means.append(mean)
        color = sns.color_palette()[idx + color_offset]
        sns.lineplot(data=data, x='step', y='ice', label=method_names[idx] if idx < len(method_names) else f'Method {idx+1}', color=color)
        sns.lineplot(x=[0, len(method_array[0])-1], y=[mean, mean], linestyle='--', label=f'{method_names[idx] if idx < len(method_names) else f"Method {idx+1}"} mean', color=color)

    mean_of_means = np.mean(means)
    # annotate means, if they are higher than the mean of means plot above line else plot below line
    for mean in means:
        offset = 5 if mean > mean_of_means else -5
        va = 'bottom' if mean > mean_of_means else 'top'
        plt.annotate(f'{mean:.2f}', (experiment.shape[2]-1, mean), textcoords="offset points", xytext=(0, offset), ha='right', va=va, color=theme_color if png else 'black')

    if png:
        enable_darkmode()

    if save:
        save_plt(save)
    plt.show()

def plot_fid_batch(fid_values, save=None, method_names=['lerp', 'slerp'], ylim=(220,320)):
    """
    Plots FID (Fréchet Inception Distance) values for different methods.

    Parameters:
    - fid_values (list of lists): FID values for each method and batch.
    - save (str, optional): Filepath to save the plot. Default is None.
    - method_names (list, optional): Names of the methods. Default is ['lerp', 'slerp'].
    - ylim (tuple, optional): Y-axis limits for the plot. Default is (220, 320).

    Returns:
    None
    """

    png = save[-4:] == '.png' if save else False
    
    mean_fid = np.array(fid_values).mean(axis=1)
    min_fid = np.array(fid_values).min(axis=1)
    max_fid = np.array(fid_values).max(axis=1)

    data = []
    for i, row in enumerate(fid_values):
        for j, value in enumerate(row):
            data.append([i, value])
    data = np.array(data)

    sns.scatterplot(x=data[:,0], y=data[:,1], label='fid')
    
    plt.xticks(np.arange(0, len(fid_values), 1), method_names)
    plt.xlim(-0.5, len(fid_values)-0.5)
    if ylim:
        plt.ylim(*ylim)

    plt.ylabel('FID')
    plt.xlabel('Method')

    for i in range(len(fid_values)):
        sns.lineplot(x=[i, i], y=[min_fid[i], max_fid[i]], color=sns.color_palette()[0])

    sns.scatterplot(x=np.arange(len(fid_values)), y=mean_fid, color='orange', label='mean fid')

    for i, value in enumerate(mean_fid):
        plt.annotate(f'{value:.2f}', (i, value), textcoords="offset points", xytext=(10,0), ha='left', va='center', color=theme_color if png else 'black')

    # annotate min and max
    for i, value in enumerate(min_fid):
        plt.annotate(f'{value:.2f}', (i, value), textcoords="offset points", xytext=(-10,0), ha='right', va='center', color=theme_color if png else 'black')
    for i, value in enumerate(max_fid):
        plt.annotate(f'{value:.2f}', (i, value), textcoords="offset points", xytext=(-10,0), ha='right', va='center', color=theme_color if png else 'black')

    if png:
        enable_darkmode()
    
    if save:
        save_plt(save)
    plt.show()

    
    