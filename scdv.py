# coding: utf-8
import os
import numpy as np
from gensim.models import word2vec
from sklearn.feature_extraction.text import TfidfVectorizer
# luigi
import pickle
import luigi
import luigi.format
from luigi.util import inherits, requires
# lib
import lib.config as con
from lib.corpus import Corpus
from lib.word_cluster import WordCluster
from lib.document_vector import DocumentVector, SparceDocumentVector

def target(file_path: str):
    return luigi.LocalTarget(file_path, format=luigi.format.Nop)

def dump(target: luigi.LocalTarget, obj):
    with target.open('w') as f:
        f.write(pickle.dumps(obj, protocol=4))

def load(target: luigi.LocalTarget):
    with target.open('r') as f:
        return pickle.load(f)


class BuildCorpus(luigi.Task):
    
    model = luigi.Parameter()
    input_file = luigi.Parameter()
    min_length = luigi.IntParameter(default=con.MIN_LENGTH)
    
    def requires(self):
        pass
    
    def output(self):
        corpus_file = con.MODEL + self.model + '/corpus.pkl'
        return target(corpus_file)
    
    def run(self):
        corpus = Corpus(self.min_length)
        corpus.build(self.input_file, tokenizer=con.TOKENIZER)
        dump(self.output(), corpus)

        
@requires(BuildCorpus)
class BuildEmbedding(luigi.Task):
    
    # embedding
    vector_size = luigi.IntParameter(default=con.VECTOR_SIZE)
    min_count = luigi.IntParameter(default=con.MIN_COUNT)
    window = luigi.IntParameter(default=con.WINDOW)
    iter_num = luigi.IntParameter(default=con.ITER_NUM)
    skip_gram = luigi.BoolParameter(default=con.SKIP_GRAM)
    
    def output(self):
        word2vec_model = con.MODEL + self.model + '/embedding.pkl'
        return target(word2vec_model)
    
    def run(self):
        corpus = load(self.input())
        # model
        embedding = word2vec.Word2Vec(corpus.documents,
                                      size=self.vector_size,
                                      min_count=self.min_count, 
                                      window=self.window,
                                      sg=self.skip_gram, 
                                      iter=self.iter_num)
        dump(self.output(), embedding.wv)

        
@requires(BuildCorpus)
class BuildIdf(luigi.Task):

    # idf
    #max_df = luigi.FloatParameter(default=con.MAX_DF)
    max_df = 0.9
    #min_df = luigi.FloatParameter(default=con.MIN_DF)
    min_df = 3
    stop_words = luigi.ListParameter(default=con.STOP_WORDS)
    smooth_idf = luigi.BoolParameter(default=con.SMOOTH_IDF)
    
    def output(self):
        idf_file = con.MODEL + self.model + '/idf.pkl'
        return target(idf_file)
    
    def run(self):
        corpus = load(self.input())
        tokenizer = ' '
        tfv = TfidfVectorizer(dtype=np.float32,
                              max_df=self.max_df,
                              min_df=self.min_df,
                              stop_words=self.stop_words,
                              smooth_idf=self.smooth_idf)
        documents = [tokenizer.join(text) for text in corpus.documents]
        tfv.fit_transform(documents)
        
        idf = {}
        featurenames = tfv.get_feature_names()
        for name, feature in zip(featurenames, tfv._tfidf.idf_):
            idf[name] = feature
        dump(self.output(), idf)


@inherits(BuildEmbedding)
@inherits(BuildIdf)
class BuildWordClusterVector(luigi.Task):
    
    cluster_size = luigi.IntParameter(default=con.CLUSTER_SIZE) # TODO : unkwnown
    max_iter = luigi.IntParameter(default=con.MAX_ITER)
    
    def requires(self):
        return dict(embedding = self.clone(BuildEmbedding), 
                    idf = self.clone(BuildIdf))
    
    def output(self):
        pcw_file = con.MODEL + self.model + '/word_cluster.pkl'
        return target(pcw_file)
    
    def run(self):
        embedding = load(self.input()['embedding'])
        idf = load(self.input()['idf'])
        pcw = WordCluster(self.cluster_size, self.max_iter)
        pcw.build(embedding, idf)
        dump(self.output(), pcw)


@inherits(BuildCorpus)
@inherits(BuildWordClusterVector)
class BuildSparceCompositeDocumentVector(luigi.Task):
    
    sparsity_percentage = luigi.FloatParameter(default=con.SPARSITY_PERCENTAGE)
    
    def requires(self):
        return dict(corpus = self.clone(BuildCorpus),
                    wcv = self.clone(BuildWordClusterVector))
        
    def output(self):
        composite_document_file = con.MODEL + self.model + '/composite_document.pkl'
        return target(composite_document_file)
    
    def run(self):
        corpus = load(self.input()['corpus'])
        wcv = load(self.input()['wcv'])
        composite_document = SparceDocumentVector(wcv.composite, wcv.id_to_word)
        composite_document.build(corpus.documents, self.sparsity_percentage, zero=True)
        dump(self.output(), composite_document.vector)
        
if __name__ == '__main__':
    luigi.run()
