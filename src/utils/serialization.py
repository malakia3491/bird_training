import pickle
import os

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_pickle(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)