#coding: utf-8
import numpy as np
from lib.cosine_similarity import cosine_similarity_2d_2d


class SimilarWord(object):
    def __init__(self, id_to_word):
        self.id_to_word = id_to_word
        self.word_to_id = {word: i for i, word in enumerate(self.id_to_word)}
        self.vectors = {}
    
    def set_model(self, **vectors):
        self.vectors = vectors
    
    def _get(self, words, top_n=10):
        if type(words)==str: words = [words]
        word_ids = [self.word_to_id[w] for w in words]
        similar_words = {i: {} for i in word_ids}
        for model_name, vectors in self.vectors.items():
            vectors = vectors.reshape(vectors.shape[0], -1)
            target_vectors = vectors[word_ids]
            cosine_similarity = cosine_similarity_2d_2d(target_vectors, vectors)
            cosine_similarity[np.arange(len(word_ids)), word_ids] = -1
            similar_indexs = (-cosine_similarity).argsort(axis=1)
            for wi in word_ids:
                similar_words[wi][model_name] = (similar_indexs[:, :top_n], cosine_similarity[:, wi])
        return similar_words

class Similar(object):
    def __init__(self, corpus):
        self.indexs = corpus.indexs
        self.documents = corpus.documents
    
    def _get(self, vectors, document_ids=None, top_n=10):
        vectors = vectors.reshape(vectors.shape[0], -1)
        if document_ids is None: document_ids = 20
        if type(document_ids) == int:
            document_ids = np.random.choice(len(vectors), document_ids, replace=False)
        target_vectors = vectors[document_ids]
        cosine_similarity = cosine_similarity_2d_2d(target_vectors, vectors)
        cosine_similarity[np.arange(len(document_ids)), document_ids] = -1
        similar_indexs = (-cosine_similarity).argsort(axis=1)
        return document_ids, similar_indexs[:, :top_n], cosine_similarity

