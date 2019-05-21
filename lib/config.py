# coding: utf-8

# path
PROJECT = '~/project/scdv/'
MODEL = PROJECT + 'model/'
DATA = PROJECT + 'data/'

# corpus parameter
MIN_LENGTH = 5
FILE_DELIMITER = '\t'
TOKENIZER = lambda line: line.split()

# embedding parameter
MIN_COUNT = 3
VECTOR_SIZE = 50
WINDOW = 12
SKIP_GRAM = False
ITER_NUM = 5

# idf
MIN_DF = 3
MAX_DF = 0.90
STOP_WORDS = []
SMOOTH_IDF = True

MAX_ITER = 200
CLUSTER_SIZE = 100
SPARSITY_PERCENTAGE = 0.75
