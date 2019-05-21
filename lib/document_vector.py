# coding: utf-8
import numpy as np
from typing import List


class DocumentVector(object):
    def __init__(self, distributed: np.ndarray, vocabulary: List[str]):
        # vocabulary
        self.id_to_word = vocabulary
        self.word_to_id = {word: i for i, word in enumerate(self.id_to_word)}
        # vector
        self.distributed = distributed
        self.vector = []
    
    def build(self, documents: List[List[str]], zero=False):
        self.vector = [self.get_vector(text) for text in documents if zero or self.get_vector(text).sum()!=0]
    
    def get_vector(self, text: List[str]):
        wv = lambda word: self.distributed[self.word_to_id[word]].reshape(-1)
        vector = [wv(word) for word in text if self.word_to_id.get(word) and wv(word).sum() != 0]
        if vector:
            return np.array(vector).sum(axis=0) / len(vector)
        return np.zeros_like(self.distributed[0].reshape(-1))

class SparceDocumentVector(DocumentVector):

    def build(self, documents: List[List[str]], sparsity_percentage: float, zero=False):
        assert 0 <= sparsity_percentage <= 1
        dv = np.array([self.get_vector(text) for text in documents if zero or self.get_vector(text).sum()!=0])
        #sparse
        abs_ave_max = lambda array : np.abs(np.average(np.max(array, axis=1)))
        threshold = sparsity_percentage * (abs_ave_max(dv) + abs_ave_max(-dv))/2
        if zero:
            dv[np.abs(dv) < threshold] = 0
            self.vector = dv
        else:
            self.vector = dv[np.abs(dv) >= threshold]