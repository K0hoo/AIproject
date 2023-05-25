import numpy as np

import os
import csv
import pickle

from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize

from data import menu


def get_ingredient_similarity(menus:np.array, s_idx:int):

    sparse_matrix = create_sparse_matrix()

    similarity = np.empty(len(menus), dtype=float)
    return similarity


def get_menu_similarity(menus:np.array, s_idx:int):
    similarity = np.empty(len(menus), dtype=float)
    return similarity


def get_recipe_similarity(menus:np.array, s_idx:int):
    similarity = np.empty(len(menus), dtype=float)
    return similarity


def create_sparse_matrix(write_csv:bool=False):

    if os.path.exists('data_binary/matrix.pkl'):
        return pickle.load(open('data_binary/matrix.pkl', 'rb'))

    ingds = pickle.load(open('data_binary/ingd.pkl', 'rb'))
    menus = pickle.load(open('data_binary/menu.pkl', 'rb'))
    ingd_name = ingds[:, 0]

    ingd_cnt, menu_cnt = len(ingds), len(menus)
    sparse_matrix = np.zeros((menu_cnt, ingd_cnt), dtype=float)
    
    for mi, menu in enumerate(menus):
        weights = menu.weight
        for ingd, weight in weights.items():
            i = np.where(ingd_name == ingd)[0]
            sparse_matrix[mi, i] += weight

    sparse_matrix_pickle = open('data_binary/matrix.pkl', 'wb')
    pickle.dump(sparse_matrix, sparse_matrix_pickle, pickle.HIGHEST_PROTOCOL)
    sparse_matrix_pickle.close()

    if write_csv:
        sparse_matrix_file = open('data_csv/matrix.csv', 'w', encoding='utf8', newline='')
        writer = csv.writer(sparse_matrix_file)
        writer.writerows(sparse_matrix)
        sparse_matrix_file.close()

    return sparse_matrix


if __name__=="__main__":
    create_sparse_matrix(write_csv=True)    
