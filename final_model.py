import similarity as sim
import numpy as np
import data 
import os
import csv

FOOD_POOL = 100
COEFF = [1,1,1,1]

############ get information from user ############
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
user_preference_menus = [[0, 1], [1, 1], [2, 1]]
####################################################

# get food list about ingredient
def get_FOOD_POOL_sort_index(food_pool: int=100) -> np.array:
    menus, _ = data.create_menus()
    menus_weights = data.get_current_weights(menus, state=user_status)
    return menus_weights.argsort()[::-1][:food_pool]

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
    menus_FOOD_POOL_sort_index = get_FOOD_POOL_sort_index(FOOD_POOL)

    # ensemble models
    # m0 : ingredient_similarity
    m0 = get_similarity_by_user_preference(sim.get_ingredient_similarity)

    #
    m1 = np.zeros(1545)

    #
    m2 = np.zeros(1545)

    # m3 : recipe_similarity
    m3 = get_similarity_by_user_preference(sim.get_recipe_similarity)

    # ensemble m0 m1 m2 m3
    ensemble_model = m0*COEFF[0]+m1*COEFF[1]+m2*COEFF[2]+m3*COEFF[3]
    FOOD_POOL_ensemble_model = ensemble_model[menus_FOOD_POOL_sort_index]

    # final ensemble model
    final_ensemble_model_index = FOOD_POOL_ensemble_model.argsort()[::-1]
    final_ensemble_model = FOOD_POOL_ensemble_model[final_ensemble_model_index]

    # result
    name_array = get_name_array()
    print(name_array[menus_FOOD_POOL_sort_index[final_ensemble_model_index]])

