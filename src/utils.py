import os
import sys

import numpy as np 
import pandas as pd
from src import logger
from src.logger import logging
import dill

from src.Exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def load_obj(file_path):
    try:
        with open(file_path,'rb') as fileObj:
            return dill.load(fileObj)

    except Exception as e:
        raise CustomException(e,sys)


def load_object(file_path):
    """Alias for load_obj to maintain consistency"""
    return load_obj(file_path)


# In src/utils.py, add this at the end:

def text_to_vector(text: str, word_dict: dict, vector_dim: int = 300) -> np.ndarray:
    """
    Converts cleaned text into a 300-dimension vector using a dictionary.
    """
    try:
        # Standardize text to match dictionary keys
        words = str(text).lower().split()
        word_vectors = []
        
        for word in words:
            if word in word_dict:
                word_vectors.append(word_dict[word])
        
        if len(word_vectors) == 0:
            return np.zeros(vector_dim)
        
        return np.mean(word_vectors, axis=0)
    except Exception as e:
        logger.error(f"Error converting text to vector: {str(e)}")
        return np.zeros(vector_dim)