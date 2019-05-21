# coding: utf-8
import numpy as np


def cosine_similarity(x, y, eps=1e-8):
    nx = x / (np.sqrt(np.sum(np.square(x))) + eps)
    ny = y / (np.sqrt(np.sum(np.square(y))) + eps)
    return np.dot(nx, ny)


def cosine_similarity_2d_1d(x, y, eps=1e-8):
    nx = x / (np.sqrt(np.sum(np.square(x), axis=1)) + eps)[:, np.newaxis]
    ny = y / (np.sqrt(np.sum(np.square(y))) + eps)
    return np.dot(nx, ny)


def cosine_similarity_2d_2d(x, y, eps=1e-8):
    nx = x / (np.sqrt(np.sum(np.square(x), axis=1)) + eps)[:, np.newaxis]
    ny = y / (np.sqrt(np.sum(np.square(y), axis=1)) + eps)[:, np.newaxis]
    return np.dot(nx, ny.swapaxes(0, 1))


if __name__ == '__main__':
    print ('cosine_similar.py')
