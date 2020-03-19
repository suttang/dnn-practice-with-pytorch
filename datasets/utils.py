import os

DATASET_DIR = os.path.dirname(__file__)
CACHE_DIR = os.path.join(DATASET_DIR, ".cache")


def get_cache_dir():
    return CACHE_DIR
