import numpy as np
import logging
import logging.config
import sys
import torch

from typing import List


# ================== RotatE Loader ================================
def loadRotatEModel(file, entityForUse: List = None):
    """
    Args

        File: file path
        entityForUse: entity list that the user is going to use
    """

    print('Load RotatE Vectors')
    file = open(file, 'r', encoding='utf-8')
    rotateModel = dict()
    for line in file:
        splitLines = line.split()
        entity = splitLines[0]

        if not entityForUse:
            embeddings = np.array([float(v) for v in splitLines[3:]])
            rotateModel[entity] = embeddings
        else:
            if entity in entityForUse:
                embeddings = np.array([float(v) for v in splitLines[3:]])
                rotateModel[entity] = embeddings

    print(f"{len(rotateModel)} entity vectors are loaded")
    return rotateModel
