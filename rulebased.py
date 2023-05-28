import numpy as np
import pandas
import pickle

# nan 없는 버전
data = pandas.read_csv('data_csv/recipes.csv')
data = data.dropna(subset=['name'])
data_tolist2 = data.values.tolist()
data_tolist2 = [['name', '청국장찌개']] + data_tolist2
data_tolist2 = np.array(data_tolist2)

count = 0
name = ""
weight_dict = {}
is_weight = False
for twolist in data_tolist2:
    if(twolist[0]=='recipe'):
        is_weight = False
        name_weight_recipe['recipe'] = twolist[1]
    elif(twolist[0]=="name"): # 값이 name인 경우, 새로운 음식이 시작되기 때문에 각종 값들을 초기화해준다.
        name_weight_recipe = {}
        name_weight_recipe['name'] = twolist[1]
        weight_stack = {}   # weight_stack[ingredient] = weight 의 형식으로 담겨진다.
        
        name_weight_recipe['weight'] = weight_stack
        weight_dict[count] = name_weight_recipe
        count += 1
    elif(twolist[0]=="weight"):
        is_weight = True

    elif(is_weight):
        if(twolist[0] not in weight_stack):
            weight_stack[twolist[0]] = twolist[1]

MENU_COUNT = len(weight_dict)
# 전체 메뉴개수는 0~1525 =>1526개이다.

ingredient_list = []
for index in weight_dict:
    for ingredient in weight_dict[index]['weight']:
        if(ingredient not in ingredient_list):
            ingredient_list.append(ingredient)
ingredient_list.remove('기본')
ingredient_list.sort()
INGREDIENT_COUNT = len(ingredient_list)

# for i, letter in enumerate(['A', 'B', 'C']):

ingredient_toindex = {}
for i, ingredient in enumerate(ingredient_list):
    ingredient_toindex[ingredient] = i
# print(ingredient_toindex)

# 재료 종류 빈도 행렬 생성
ingredient_frequency = np.zeros((INGREDIENT_COUNT,INGREDIENT_COUNT))
# 재료 종류 weight 기반 유사도 행렬 생성
ingredient_similarity = np.zeros((INGREDIENT_COUNT,INGREDIENT_COUNT))

for index in weight_dict:
    for ii in range(INGREDIENT_COUNT):
        for jj in range(INGREDIENT_COUNT):
            if(ingredient_list[ii]==ingredient_list[jj]):
                if(ingredient_list[ii] in weight_dict[index]['weight']):    # 1가지 재료만 포함된 음식
                    ingredient_frequency[ii][ii] += 1
                    # print(ingredient_list[ii])
            else:
                if(ingredient_list[ii] in weight_dict[index]['weight'] and ingredient_list[jj] in weight_dict[index]['weight']):    # 2가지 재료가 포함된 음식
                    ingredient_frequency[ii][jj] += 1
                    # print(ingredient_list[ii], ingredient_list[jj], weight_dict[index]['weight'])

for ii in range(INGREDIENT_COUNT):
    for jj in range(INGREDIENT_COUNT):
        if(ii==jj): # 같은 재료일 때에는 유사도 1
            # print(jj)
            ingredient_similarity[ii][jj] = 1
        else:       # 다른 재료일 때에는 유사도 = A&B/A|B = A and B / (A + B - A and B)
            if(ingredient_frequency[ii][ii] * ingredient_frequency[jj][jj] * ingredient_frequency[ii][jj] != 0):
                # print(ingredient_frequency[ii][ii] , ingredient_frequency[jj][jj] , ingredient_frequency[ii][jj])
                ingredient_similarity[ii][jj] = ingredient_frequency[ii][jj]/(ingredient_frequency[ii][ii] + ingredient_frequency[jj][jj] - ingredient_frequency[ii][jj])

# 재료 유사도기반 음식 유사도 행렬 생성
ingredient_menu_similarity = np.zeros((MENU_COUNT,MENU_COUNT))


for index_a in range(MENU_COUNT):
    for index_b in range(MENU_COUNT):
        # 2가지 메뉴의 각 재료 m*n개에 대해서 
        sum = 0
        for ingredient_a in weight_dict[index_a]['weight']:
            for ingredient_b in weight_dict[index_b]['weight']:
                if(ingredient_a == "기본" or ingredient_b == "기본"):
                    pass
                else:
                    sum += ingredient_similarity[ingredient_toindex[ingredient_a]][ingredient_toindex[ingredient_b]]
        # m*n으로 나누기
        if(len(weight_dict[index_a]['weight']) * len(weight_dict[index_b]['weight'])!=0):
            sum = sum / len(weight_dict[index_a]['weight']) / len(weight_dict[index_b]['weight'])
            ingredient_menu_similarity[index_a][index_b] = sum
            
# 같은 메뉴끼리의 유사도를 1로 설정
for i, row in enumerate(ingredient_menu_similarity):
    if(row[i]==0):
        print(i, row[i])
    else:
        ingredient_menu_similarity[i] = row/row[i]




# 피클 파일로 저장
with open("data_binary/menu_similarity_matrix.pickle", "wb") as file:
    pickle.dump(ingredient_menu_similarity, file)