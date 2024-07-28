import joblib

def get_reducer(folder):
    """
    Load UMAP reducers from the specified folder.

    Args:
        folder (str): The folder path where the UMAP reducers are stored.

    Returns:
        tuple: A tuple containing the UMAP reducers for n_neighbors=2, n_neighbors=10, and n_neighbors=50.
    """
    umap_n2 = joblib.load(f'data/dump/{folder}/umap_n2.pkl')
    umap_n10 = joblib.load(f'data/dump/{folder}/umap_n10.pkl')
    umap_n50 = joblib.load(f'data/dump/{folder}/umap_n50.pkl')
    return umap_n2, umap_n10, umap_n50

def get_reduced(folder):
    """
    Load reduced data from the specified folder.

    Args:
        folder (str): The folder path where the reduced data is stored.

    Returns:
        tuple: A tuple containing the reduced data for n_neighbors=2, n_neighbors=10, and n_neighbors=50.
    """
    pooled_n2 = joblib.load(f'data/dump/{folder}/pooled_n2.pkl')
    pooled_n10 = joblib.load(f'data/dump/{folder}/pooled_n10.pkl')
    pooled_n50 = joblib.load(f'data/dump/{folder}/pooled_n50.pkl')
    return pooled_n2, pooled_n10, pooled_n50