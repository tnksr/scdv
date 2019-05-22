# coding: utf-8
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import TSNE


def t_sne(data, file_name):
    file_name = '/Users/tanaka-so/project/scdv/demo/' + file_name +'.pkl'
    if os.path.exists(file_name):
        with open(file_name, 'rb') as f:
            return pickle.load(f)
    else:
        data = np.array(data)
        if data.shape[1] != 2:
            tsne = TSNE(n_components=2, random_state=0, verbose=2)
            tsne.fit(data.reshape(data.shape[0], -1))
        with open(file_name, 'wb') as f:
            f.write(pickle.dumps(tsne.embedding_))
    return tsne.embedding_

def plot(tsne_data, hue=None, annotate=None):
    df = pd.DataFrame(tsne_data, columns=['x', 'y'])
    df['hue'] = hue
    df['annotate'] = annotate
    
    sns.set(font='IPAexGothic')
    graph = sns.scatterplot(data=df, x='x', y='y', hue='hue', palette='Set2', marker='o', size=3, legend=False)
    for k, (x, y, hue, annotate) in df.iterrows():
        if annotate:
            graph.annotate(annotate, (x, y), horizontalalignment='left', size=7, color='black', weight='semibold')
    plt.show()
