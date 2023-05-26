import similarity as sim
import numpy as np

COEFF = [-1,1,-1,1]




selected_food_index = np.random.randint(0, 123, 10)
#

m0 = np.random.randn(123,2)
# m1 = sim.get_ingredient_similarity(0,50)

m1 = np.random.randn(123,2)
#

m2 = np.random.randn(123,2)
#

m3 = np.random.randn(123,2)
#

ensemble_model = m0*COEFF[0]+m1*COEFF[1]+m2*COEFF[2]+m3*COEFF[3]

#print(ensemble_model.shape)
print(ensemble_model.where(selected_food_index))

