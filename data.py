import numpy as np

import os
import csv
import pickle
import time


cnt_menu = 0
user_status = {
    '달걀': 2,
    '연근': 3,
    '파프리카': 2,
    '돼지고기': 3,
    '브로콜리': 2,
    '당근': 3,
    '고추장': 3,
    '깻잎': 2,
    '단무지': 5
}

class menu:

    def __init__(self, doc:dict):
        self.name = doc['name']
        self.view = int(doc['view'])
        self.cooktime = doc['cooktime']
        self.url = doc['url']
        self.ingredient = doc['ingredient']
        self.weight = doc['weight']
        self.recipe = doc['recipe']

    def getWeight(self):
        return self.cur_weight
    
    def getName(self):
        return self.name
    
    def getWeight(self, state):
        result = 0
        for k, v in self.weight.items():
            if k in state: result += v
        return result 
                


def create_menus(
        input_file:str='data_csv/recipes.csv', 
        menu_output_file:str='data_binary/menu.pkl',
        ingd_output_file:str='data_binary/ingd.pkl',
        write_file:bool=True
        ):

    global cnt_menu
    with open(input_file, 'r', encoding='utf8') as menu_file:
        menu_reader = csv.reader(menu_file)
        for line in menu_reader:
            if line[0] == 'name': cnt_menu += 1
    
    menus = np.empty(cnt_menu, dtype=menu)
    normal_info = ['name', 'view', 'people', 'cooktime', 'condition', 'url', 'recipe']
    menu_idx, b_ingd, b_weight, tmp_dict = 0, 0, 0, {}
    ingd_dict, ingd_list = {}, []
    with open(input_file, 'r', encoding='utf8') as menu_file:
        menu_reader = csv.reader(menu_file)
        for line in menu_reader:
            cmd, value = (line[0], 0) if len(line) == 1 else (line[0], line[1])
            if cmd in normal_info:
                tmp_dict.update({cmd: value})
            elif cmd == 'ingredient':
                tmp_dict.update({cmd: {}})
                b_ingd = True
            elif cmd == 'weight':
                tmp_dict.update({cmd: {}})
                b_ingd, b_weight = False, True
            elif cmd == '':
                menus[menu_idx] = menu(tmp_dict)
                menu_idx, tmp_dict, b_weight = menu_idx + 1, {}, False
            elif b_ingd:
                tmp_dict['ingredient'].update({cmd: value})
            elif b_weight:
                tmp_dict['weight'].update({cmd: float(value)})
                ingd_dict.update({cmd: float(value)})
            else:
                print(f'There is an exceptional case: {menu_idx} {line}')
        
        for k, v in ingd_dict.items():
            ingd_list.append([k, v])
        ingds = np.array(ingd_list)

        if write_file:
            ingd_pickle = open(ingd_output_file, 'wb')
            pickle.dump(ingds, ingd_pickle, pickle.HIGHEST_PROTOCOL)
            ingd_pickle.close()

            menu_pickle = open(menu_output_file, 'wb')
            pickle.dump(menus, menu_pickle, pickle.HIGHEST_PROTOCOL)
            menu_pickle.close()

    return (menus, ingds)


def get_current_weights(menus:np.array, state:dict=user_status):
    
    assert(menus.dtype == menu)
    current_weights = np.empty(len(menus), dtype=float)
    for i, m in enumerate(menus):
        current_weights[i] = m.getWeight(state)
    if not os.path.exists('data_binary/weights'): os.mkdir('data_binary/weights')
    weights_pickle = open(f'data_binary/weights/w_{int(time.time())}.pkl', 'wb')
    pickle.dump(current_weights, weights_pickle, pickle.HIGHEST_PROTOCOL)
    weights_pickle.close()
    return current_weights
    

if __name__ == "__main__":

    menus, ingds = create_menus()
    weights = get_current_weights(menus)
