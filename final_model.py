import similarity as sim
import numpy as np
import data 
import os
import csv

FOOD_POOL = 100
COEFF = [1,1,1,1,1]

############ get information from user ############
user_status = {
    '청국장' : 1,
    '돼지고기' : 1
}
user_preference_menus = [[1, 1]]
####################################################

# get food list about ingredient
def get_FOOD_POOL_sort_index(food_pool: int=100) -> np.array:
    menus, _ = data.create_menus()
    menus_weights = data.get_current_weights(menus, state=user_status)
    return menus_weights.argsort()[::-1][:food_pool], menus_weights

# get food similarity list about user preference
def get_similarity_by_user_preference(similarity_function=None) -> np.array:
    try:
        if similarity_function == None:
            raise Exception("similarity_function is \"None\"")
    except Exception as e:
        print(f"Error : {e}")
        exit()
    
    flag = 0
    for menu in user_preference_menus:
        if flag == 0:
            m = menu[1] * similarity_function(menu[0])
        else:
            m += menu[1] * similarity_function(menu[0])
    return m

# get name index
def get_name_array() -> np.array:
    if os.path.exists('data_csv\\name.csv'):
        with open('data_csv\\name.csv','r', encoding='utf-8') as f:
            rdr = csv.reader(f)
            name_list = []
            for name in rdr:
                name_list.append(name)
            f.close()
            return np.array(name_list)
    return None

if __name__=="__main__":
    menus_FOOD_POOL_sort_index, menus_FOOD_POOL_weight = get_FOOD_POOL_sort_index(FOOD_POOL)

    # ensemble models
    # m0 : have ingredient weight
    m0 = menus_FOOD_POOL_weight

    # m1 : ingredient_similarity
    m1 = get_similarity_by_user_preference(sim.get_ingredient_similarity)

    # m2 : menu_similarity
    m2 = get_similarity_by_user_preference(sim.get_menu_similarity)

    # m3 : recipe_similarity
    m3 = get_similarity_by_user_preference(sim.get_recipe_similarity)

    # m4 : view
    m4 = get_similarity_by_user_preference(sim.get_view)

    # ensemble m0 m1 m2 m3 m4
    ensemble_model = m0*COEFF[0]+m1*COEFF[1]+m2*COEFF[2]+m3*COEFF[3]+m4*COEFF[4]
    FOOD_POOL_ensemble_model = ensemble_model[menus_FOOD_POOL_sort_index]

    # final ensemble model
    final_ensemble_model_index = FOOD_POOL_ensemble_model.argsort()[::-1]
    final_ensemble_model = FOOD_POOL_ensemble_model[final_ensemble_model_index]

    # result
    name_array = get_name_array()
    #print(menus_FOOD_POOL_sort_index[final_ensemble_model_index])
    print(name_array[menus_FOOD_POOL_sort_index[final_ensemble_model_index]])

