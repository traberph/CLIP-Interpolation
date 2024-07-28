import numpy as np
import pyarrow.parquet as pq
import joblib

locations7 = ['', 'on the moon', 'in the jungle', 'at the beach', 'in the city', 'on a mountain', 'in the snow', 'in the woods']
locations3 = ['', 'on the moon', 'in the jungle', 'at the beach']
colors3 = ['blue', 'orange', 'red']
colors8 = ['blue', 'red', 'green', 'yellow', 'orange', 'black', 'white', 'purple']
objects3 = ['monkey', 'child', 'car']
sentences = [[f'a {color} {object}', color, object] for color in colors3 for object in objects3]


descriptions = np.array(sentences)



adjectives_all = ['',  'kind',  'ugly',  'adventurous',  'thirsty',  'sad',  'real',  'witty',  'lucky',  'yummy',  'patient',  'sophisticated',  'strange',  'nervous',  'young',  'talented',  'uninterested',  'charming',  'curious',  'obedient',  'smart',  'sore',  'upset',  'uptight',  'strong',  'scary',  'stupid',  'repulsive',  'optimistic',  'worried',  'fast',  'puzzled',  'happy',  'worrisome',  'scared',  'tender',  'prickly',  'silly',  'old-fashioned',  'unsightly',  'blissful',  'vast',  'open',  'troubled',  'poor',  'diligent',  'brave',  'vibrant',  'ambitious',  'wild',  'graceful',  'stormy',  'vivacious',  'funny',  'plain',  'tough',  'thankful',  'wandering',  'outstanding',  'tense',  'terrible',  'handsome',  'sleepy',  'smiling',  'wicked',  'zealous',  'spotless',  'successful',  'thoughtless',  'super',  'obnoxious',  'humble',  'outrageous',  'jovial',  'wide-eyed',  'nice',  'shy',  'polite',  'panicky',  'ugliest',  'victorious',  'thoughtful',  'smoggy',  'shiny',  'radiant',  'lazy',  'perfect',  'angry',  'sparkling',  'poised',  'energetic',  'precious',  'relieved',  'zany',  'rich',  'proud',  'weak',  'crazy',  'selfish',  'imaginative',  'tired',  'mindful',  'powerful',  'youthful',  'testy',  'tame',  'splendid',  'old',  'tall',  'noisy',  'wet',  'creative',  'pleasant',  'unusual',  'kind-hearted',  'quaint',  'odd',  'weary',  'lively']

prompts_man_all = [f'a {a} man' for a in adjectives_all]
prompts_woman_all = [f'a {a} woman' for a in adjectives_all]


human_keywords = [
    "Human",
    "Person",
    "People",
    "Man",
    "Woman",
    "Child",
    "Adult",
    "Portrait",
    "Face",
    "Body",
    "Humanoid",
    "Male",
    "Female",
    "Model",
    "Elderly",
    "Infant",
    "Toddler",
    "Teenager",
    "Gesture",
]

def get_dataset(dataset='portrait', return_np=True, unique=True):
    """
    Load the dataset from a parquet file.
    Laion400M portrait subset with pooled embeddings.

    Parameters:
    - np (bool): If True, return the dataset as numpy arrays. If False, return as pandas DataFrame.

    Returns:
    - If np is True, returns a tuple containing two numpy arrays: (text, pooled).
    - If np is False, returns the dataset as a pandas DataFrame.
    """
    df = pq.read_table(f'./data/laion400m_{dataset}_pooled.parquet').to_pandas()
    if unique:
        df.drop_duplicates(subset='TEXT', inplace=True)
    if return_np:
        text = df['TEXT'].values
        pooled = np.stack(df['POOLER_OUTPUT'].values, dtype=np.float32)
        return (text, pooled)
    else:
        return df


def get_pooled_reduced():
    pooled_n2 = joblib.load('data/dump/pooled_n2.pkl')
    pooled_n10 = joblib.load('data/dump/pooled_n10.pkl')
    pooled_n50 = joblib.load('data/dump/pooled_n50.pkl')
    return pooled_n2, pooled_n10, pooled_n50

def get_torus( num_points = 1000):
    np.random.seed(1)

    # Parameters for the torus
    R = 3  # Major radius
    r = 1  # Minor radius

    # Generate angles theta and phi
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    phi = np.random.uniform(0, 2 * np.pi, num_points)

    # Compute the (x, y, z) coordinates
    x = (R + r * np.cos(theta)) * np.cos(phi)
    y = (R + r * np.cos(theta)) * np.sin(phi)
    z = r * np.sin(theta)

    return np.array([*zip(x, y, z)])


