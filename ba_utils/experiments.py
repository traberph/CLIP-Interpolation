from ba_utils.model import CLIP, UMAP, UNET, VAE, Generator
from ba_utils.vis import fig_post
from ba_utils import show
from ba_utils.show import grid_all

import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch

from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA


class ClusterResult:
    """
    Represents the result of a clustering algorithm.

    Attributes:
        prompts_sorted (numpy.ndarray): Sorted array of prompts.
        embeddings_pooled_sorted (numpy.ndarray): Sorted array of pooled embeddings.
        embeddings_last_sorted (numpy.ndarray): Sorted array of last embeddings.
        cids (numpy.ndarray): Array of cluster IDs.
        unique_clusters (list): List of unique cluster IDs.
        cluster_avg_idx_pooled (list): List of indices of cluster average embeddings in the pooled embeddings array.
        cluster_avg_idx_last (list): List of indices of cluster average embeddings in the last embeddings array.
    """

    def __init__( self, prompts_sorted, embeddings_pooled_sorted, embeddings_last_sorted, cids, unique_clusters, cluster_avg_idx_pooled, cluster_avg_idx_last):
        """
        Initializes a ClusterResult object.

        Args:
            prompts_sorted (numpy.ndarray): Sorted array of prompts.
            embeddings_pooled_sorted (numpy.ndarray): Sorted array of pooled embeddings.
            embeddings_last_sorted (numpy.ndarray): Sorted array of last embeddings.
            cids (numpy.ndarray): Array of cluster IDs.
            unique_clusters (list): List of unique cluster IDs.
            cluster_avg_idx_pooled (list): List of indices of cluster average embeddings in the pooled embeddings array.
            cluster_avg_idx_last (list): List of indices of cluster average embeddings in the last embeddings array.
        """
        self.prompts_sorted = prompts_sorted
        self.embeddings_pooled_sorted = embeddings_pooled_sorted
        self.embeddings_last_sorted = embeddings_last_sorted
        self.cids = cids
        self.unique_clusters = unique_clusters
        self.cluster_avg_idx_pooled = cluster_avg_idx_pooled
        self.cluster_avg_idx_last = cluster_avg_idx_last

    def debug(self):
        """
        Returns a DataFrame with debug information about the ClusterResult object.

        Returns:
            pandas.DataFrame: DataFrame with debug information.
        """
        def get_i(x):
            if type(x) == np.ndarray:
                return x.shape
            else:
                return len(x)

        return pd.DataFrame({
            'name' : ['prompts_sorted', 'embeddings_pooled_sorted', 'embeddings_last_sorted', 'cids', 'unique_clusters', 'cluster_avg_idx_pooled', 'cluster_avg_idx_last'],
            'type': [get_i(self.prompts_sorted), get_i(self.embeddings_pooled_sorted), get_i(self.embeddings_last_sorted), get_i(self.cids), get_i(self.unique_clusters), get_i(self.cluster_avg_idx_pooled), get_i(self.cluster_avg_idx_last)],
            'shape': [self.prompts_sorted.shape, self.embeddings_pooled_sorted.shape, self.embeddings_last_sorted.shape, self.cids.shape, len(self.unique_clusters), len(self.cluster_avg_idx_pooled), len(self.cluster_avg_idx_last)],
        })

    def get_samples(self, n=2):
        """
        Returns a DataFrame with randomly sampled prompts from each cluster.

        Args:
            n (int, optional): Number of samples to retrieve from each cluster. Defaults to 2.

        Returns:
            pandas.DataFrame: DataFrame with sampled prompts.
        """
        prompts_only = np.delete(self.prompts_sorted, self.cluster_avg_idx_pooled)
        cids_only = np.delete(self.cids, self.cluster_avg_idx_pooled)

        d = pd.DataFrame({'prompt': prompts_only, 'cluster': cids_only})
        d = d.groupby('cluster').apply(lambda x: x.sample(n, replace=True))
        d = d.drop(columns='cluster')
        return d

    def get_centers(self, gpu=True):
        """
        Returns the cluster centers as tensors.

        Args:
            gpu (bool, optional): Whether to return the tensors on GPU. Defaults to True.

        Returns:
            torch.Tensor or numpy.ndarray: Cluster centers.
        """
        if gpu:
            return torch.tensor(self.embeddings_last_sorted[self.cluster_avg_idx_last], dtype=torch.float ).to('cuda')
        else:
            return self.embeddings_last_sorted[self.cluster_avg_idx_last]

    def plot_3d(self):
        """
        Plots a 3D scatter plot of the embeddings.

        Returns:
            ClusterResult: The ClusterResult object itself.
        """
        # UMAP
        reducer_3d = UMAP(n_components=3, metric='cosine')
        reducer_3d_all = reducer_3d.fit(self.embeddings_pooled_sorted)

        e_3d = reducer_3d_all.transform(self.embeddings_pooled_sorted)

        # prepare color map
        color_map = {category: px.colors.qualitative.D3[category] if category !=-1 else '#555' for category in set(self.cids)}
        categories = pd.Categorical(self.cids, categories=color_map.keys()) 

        # create 3d scatter plot
        fig = px.scatter_3d(x=e_3d[:,0], y=e_3d[:,1], z=e_3d[:,2], text=self.prompts_sorted, color=categories, color_discrete_map=color_map)
        fig.update_scenes(xaxis_visible=False, yaxis_visible=False,zaxis_visible=False )

        # add line from base to avg
        for x, idx in enumerate(self.cluster_avg_idx_pooled):
            if self.unique_clusters[x] == -1:
                continue
            v = e_3d[idx]
            b = e_3d[0]
            fig.add_trace(go.Scatter3d(x=[v[0], b[0]], y=[v[1], b[1]], z=[v[2], b[2]], mode='lines', line=dict(color=color_map[self.cids[idx]], width=3), name=f'c{self.unique_clusters[x]}'))

        # hide cluster id -1 (noise)
        fig.update_traces(visible='legendonly', selector=dict(name='-1'))

        fig_post(fig, 5)
        return self

    def plot_2d(self, method='umap', save=None):
        """
        Plots a 2D scatter plot of the embeddings.

        Args:
            method (str, optional): Dimensionality reduction method to use ('umap' or 'pca'). Defaults to 'umap'.
            save (str, optional): Filepath to save the plot. Defaults to None.

        Returns:
            ClusterResult: The ClusterResult object itself.
        """
        if method == 'umap':
            reducer_2d = UMAP(n_components=2, metric='cosine')
            reducer_2d_all = reducer_2d.fit(self.embeddings_pooled_sorted)
            e_2d = reducer_2d_all.transform(self.embeddings_pooled_sorted)
        elif method == 'pca':
            reducer_2d = PCA(n_components=2)
            reducer_2d_all = reducer_2d.fit(self.embeddings_pooled_sorted)
            e_2d = reducer_2d_all.transform(self.embeddings_pooled_sorted)
        else:
            raise ValueError('method must be one of [umap, pca]')

        plt.figure(figsize=(10,10))     

        # draw line from base to avg
        b = e_2d[0]
        plt.plot([b[0], b[0]], [b[1], b[1]], marker='o', markersize=5, color='red')

        for x, idx in enumerate(self.cluster_avg_idx_pooled):
            if self.unique_clusters[x] == -1:
                continue
            v = e_2d[idx]
            plt.plot([v[0], b[0]], [v[1], b[1]], '--', linewidth=1, color='black')

        # filter noise 
        e_2d = e_2d[self.cids != -1]
        cids = self.cids[self.cids != -1]
        prompts = self.prompts_sorted[self.cids != -1]

        plt.scatter(e_2d[:,0], e_2d[:,1], c=cids, cmap='tab20')
        x_mean, y_mean = np.mean(e_2d[:,0]), np.mean(e_2d[:,1])
        # annotate
        for i, txt in enumerate(prompts):
            ha = 'left' if e_2d[i,0] < x_mean else 'right'
            px = 5 if e_2d[i,0] < x_mean else -5
            va = 'bottom' if e_2d[i,1] < y_mean else 'top'
            py = 5 if e_2d[i,1] < y_mean else -5
            plt.annotate(txt, (e_2d[i,0], e_2d[i,1]), textcoords="offset points", xytext=(px,py), ha=ha, va=va)

        if save:
            plt.savefig(save, bbox_inches='tight', pad_inches=0)

        plt.show()
        return self

    def __repr__(self):
        """
        Returns a string representation of the ClusterResult object.

        Returns:
            str: String representation of the ClusterResult object.
        """
        return f'ClusterResult(prompts={len(self.prompts_sorted)}, clusters={len(self.unique_clusters)})'

    def generate(self, idx=None):
        """
        Generates and displays images for each cluster.

        Args:
            idx (int, optional): Index of the cluster to generate images for. Defaults to None.

        Returns:
            ClusterResult: The ClusterResult object itself.
        """
        unet = UNET()
        latents = unet.iterate(self.get_centers())

        vae = VAE()
        images = vae.decode(latents)

        samples = self.get_samples()

        for idx, img in enumerate(images):
            print(f'cluster {samples.index.get_level_values(0).unique()[idx]}')
            print(f'samples : \"{samples.iloc[idx*2].prompt}\", \"{samples.iloc[idx*2+1].prompt}\"')
            show.single(img)

        return self



def cluster(prompts, eps=0.15, min_samples=3, plot=False):
    """
    Cluster the given prompts using DBSCAN algorithm based on their embeddings.

    Args:
        prompts (list): List of prompts to be clustered.
        eps (float, optional): The maximum distance between two samples for them to be considered as in the same neighborhood. Defaults to 0.15.
        min_samples (int, optional): The number of samples in a neighborhood for a point to be considered as a core point. Defaults to 3.
        plot (bool, optional): Whether to plot the clusters. Defaults to False.

    Returns:
        ClusterResult: An object containing the sorted prompts, cluster IDs, and cluster averages.

    """
    # Function implementation goes here
def cluster(prompts, eps=0.15, min_samples=3, plot=False):



    # CLIP    
    clip = CLIP()
    output = clip.embed(prompts)
    embeddings_pooled = output.pooler_output.cpu()
    embeddings_last = output.last_hidden_state.cpu()

    

    # DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
    clusters = dbscan.fit_predict(embeddings_pooled)

    # empty arrays to store the sorted embeddings, prompts, and cluster ids
    embeddings_pooled_sorted = np.empty((0,768)) 
    embeddings_last_sorted = np.empty((0,77,768))
    prompts_sorted = np.empty((0))
    cids = np.empty((0), dtype=int)
    
    cluster_avg_idx_pooled = []
    cluster_avg_idx_last = []
    
    # unique clusters
    _, idx = np.unique(clusters, return_index=True)
    unique_clusters = clusters[np.sort(idx)]

    for cid in unique_clusters:

        # determine the cluster prompts and embeddings
        cluster_prompts = np.array(prompts)[clusters == cid]
        cluster_embeddings_pooled = embeddings_pooled[clusters == cid]
        cluster_embeddings_last = embeddings_last[clusters == cid]

        # calculate the average of the cluster
        cluster_embeddings_pooled_avg = cluster_embeddings_pooled.mean(0)

        # add the average to the cluster embeddings and prompts
        cluster_embeddings_pooled_iavg = np.concatenate([cluster_embeddings_pooled, [cluster_embeddings_pooled_avg]])
        cluster_prompts_iavg = np.concatenate([cluster_prompts, [f'center of cluster {cid}',]])

        # add the prompts, embeddings, and cluster ids to the sorted arrays
        embeddings_pooled_sorted = np.concatenate([embeddings_pooled_sorted, cluster_embeddings_pooled_iavg], axis=0)
        embeddings_last_sorted = np.concatenate([embeddings_last_sorted, cluster_embeddings_last], axis=0)
        prompts_sorted = np.concatenate([prompts_sorted, cluster_prompts_iavg])

        cids = np.concatenate([cids, [cid]*len(cluster_prompts_iavg)]) 

        # index where the cluster average is located in sorted arrays
        cluster_avg_idx_pooled = cluster_avg_idx_pooled + [len(embeddings_pooled_sorted)-1]
        cluster_avg_idx_last = cluster_avg_idx_last + [len(embeddings_last_sorted)-1]
        
    # return pompts (ordered by cluster), cluster ids, and the cluster averages
    return ClusterResult(
        prompts_sorted = prompts_sorted,
        cids = cids,
        embeddings_pooled_sorted = embeddings_pooled_sorted,
        embeddings_last_sorted = embeddings_last_sorted,
        unique_clusters = unique_clusters,
        cluster_avg_idx_pooled = cluster_avg_idx_pooled,
        cluster_avg_idx_last = cluster_avg_idx_last
    )


clip_global = None
gen_global = None

def interpolation_pipe(prompts, interp ,n_steps=20, n_cols=None, seed=0, clip=None, gen=None, gpu_id=0):
    """
    Interpolates between prompts using the given interpolation function.

    Args:
        prompts (list): List of prompts to interpolate between.
        interp (function): Interpolation function that takes two embeddings and the number of steps as input.
        n_steps (int, optional): Number of interpolation steps. Defaults to 20.
        n_cols (int, optional): Number of columns in the output grid. Defaults to None.
        seed (int, optional): Seed for random number generation. Defaults to 0.
        clip (object, optional): CLIP object for embedding calculation. Defaults to None.
        gen (object, optional): Generator object for generating images. Defaults to None.

    Returns:
        out: Output of the interpolation and generation process.
    """
    global clip_global, gen_global
    
    if clip is None:
        if clip_global is None:
            print("initializing CLIP")
            clip_global = CLIP(gpu_id=gpu_id)
        clip = clip_global
    if gen is None:
        if gen_global is None:
            print("initializing Gen")
            gen_global = Generator(gpu_id=gpu_id)
        gen = gen_global

    embeddings = clip.embed(prompts).last_hidden_state
    path = []
    for i in range(len(embeddings) - 1):
        sub_path = interp(embeddings[i], embeddings[i + 1], n_steps)
        path.extend(sub_path)
    path = torch.stack(path)
    out = gen.pipe2(path, seed=seed)
    grid_all(out, n_cols=n_cols if n_cols is not None else n_steps)
    return out
    