import numpy as np
import pandas
import pickle

def make_csv_name_view():
    # read csv data
    data = pandas.read_csv('data_csv/recipes.csv')
    data = data.dropna(subset=['name'])
    data_to_list = data.values.tolist()
    data_to_list = [['name', '청국장찌개']] + data_to_list
    data_to_list= data_to_list

    # make name - view list
    name_view_list = []
    for data in data_to_list:
        if data[0]=='view':
            name_view_list.append(data[1])

    # save in picke
    with open("data_binary/menu_view.pickle", "wb") as file:
        pickle.dump(np.array(name_view_list), file)

if __name__ == '__main__':
    make_csv_name_view()

