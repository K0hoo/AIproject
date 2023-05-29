import numpy as np

import os
import sys
import csv
import pickle

from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize

from data import menu


def write_csv_file(path:str, data:np.array):
    
    if len(data.shape) != 2:
        print(f'Dimension of data should be 2 to write in CSV file.', file=sys.stderr)

    if not os.path.exists(path):
        csv_file = open(path, 'w', encoding='utf8', newline='')
        writer = csv.writer(csv_file)
        writer.writerows(data)
        csv_file.close()
        return True
    return False


def get_ingredient_similarity(s_idx:int, n_components:int=40, write_csv:bool=False):

    if os.path.exists(f'data_binary/similarity/ingredient/{s_idx:04d}_{n_components:2d}.pkl'):
        similarity_pkl = open(f'data_binary/similarity/ingredient/{s_idx:04d}_{n_components:2d}.pkl', 'rb')
        similarity = pickle.load(similarity_pkl)
        if write_csv: 
            write_csv_file(f'data_csv/similarity/ingredient/{s_idx:04d}_{n_components:2d}.csv', np.expand_dims(similarity, axis=1))
        return similarity

    sparse_matrix = create_sparse_matrix(write_csv=write_csv)

    if os.path.exists(f'data_binary/matrix/nmf_{n_components}.pkl'):
        matrix_pkl = open(f'data_binary/matrix/nmf_{n_components}.pkl', 'rb')
        embedding_matrix = pickle.load(matrix_pkl)
        matrix_pkl.close()
    else:
        nmf = NMF(n_components, init='random', random_state=0, max_iter=1000, tol=1e-8, verbose=True)
        embedding_matrix = nmf.fit_transform(sparse_matrix)
        embedding_matrix = normalize(embedding_matrix)
        matrix_pkl = open(f'data_binary/matrix/nmf_{n_components}.pkl', 'wb')
        pickle.dump(embedding_matrix, matrix_pkl, pickle.HIGHEST_PROTOCOL)
        matrix_pkl.close()

    if write_csv:
        write_csv_file(f'data_csv/matrix/nmf_{n_components}.csv', embedding_matrix)

    s_weight = embedding_matrix[s_idx]
    similarity = np.empty(len(sparse_matrix), dtype=float)

    for i, weight in enumerate(embedding_matrix):
        similarity[i] = np.sum(s_weight * weight)

    similarity_pkl = open(f'data_binary/similarity/ingredient/{s_idx:04d}_{n_components:2d}.pkl', 'wb')
    pickle.dump(similarity, similarity_pkl, pickle.HIGHEST_PROTOCOL)
    similarity_pkl.close()

    if write_csv: 
        write_csv_file(f'data_csv/similarity/ingredient/{s_idx:04d}_{n_components:2d}.csv', np.expand_dims(similarity, axis=1))
    
    return similarity


def get_menu_similarity(s_idx:int):
    menus = pickle.load(open('data_binary/menu.pkl', 'rb'))
    similarity = np.empty(len(menus), dtype=float)

    # TODO !!!
    with open(f"data_binary/menu_similarity_matrix.pickle","rb") as fi:
        similarity_matrix = pickle.load(fi)
    similarity = similarity_matrix[s_idx]

    return similarity

def get_recipe_similarity(s_idx:int):
    menus = pickle.load(open('data_binary/menu.pkl', 'rb'))
    similarity = np.empty(len(menus), dtype=float)

    # TODO !!!
    with open(f"data_binary/recipe_similarity_matrix.pickle","rb") as fi:
        similarity_matrix = pickle.load(fi)
    similarity = similarity_matrix[s_idx]

    return similarity

def get_view(s_idx:int=0):
    with open(f"data_binary/menu_view.pickle","rb") as fi:
        data = pickle.load(fi)
        data = normalize(data.reshape(1, -1))
        return np.array(data[0], dtype=float)

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

    if write_csv: write_csv_file('data_csv/matrix.csv', sparse_matrix)

    return sparse_matrix


if __name__=="__main__":
    
    print(get_ingredient_similarity(0, write_csv=True))
    print(get_recipe_similarity(0))
    print(get_menu_similarity(0))