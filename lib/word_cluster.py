# coding: utf-8
import numpy as np
from sklearn.mixture import GaussianMixture
from lib.cosine_similarity import cosine_similarity_2d_1d

class WordCluster(object):
    def __init__(self, cluster_size, max_iter):
        self.cluster_size = cluster_size
        self.max_iter = max_iter
        # vocabulary
        self.id_to_word = []
        self.word_to_id = {}
        # np.ndarray
        self.embedding = []
        self.idf = []
        self.probability = []
        self.composite = []
    
    def build(self, embedding, idf):
        # vocabulary
        self.id_to_word = list(embedding.vocab)
        self.word_to_id = {word: i for i, word in enumerate(self.id_to_word)}
        # embedding
        self.embedding = np.array([embedding[word] for word in self.id_to_word])
        # GMM
        gm = GaussianMixture(n_components=self.cluster_size, 
                             max_iter=self.max_iter)
        gm.fit(self.embedding)
        # probability
        self.probability = gm.predict_proba(self.embedding)
        # idf
        self.idf = np.array([idf[word] if idf.get(word) else 0.0 for word in self.id_to_word])
        # vector
        vocab_size, embedd_size = self.embedding.shape
        e = self.embedding.reshape(vocab_size, 1, embedd_size)
        p = self.probability.reshape(vocab_size, self.cluster_size, 1)
        i = self.idf.reshape(vocab_size, 1, 1)
        self.composite = (e * p) * i