from scipy.stats import poisson
import numpy as np
import pickle
import pandas as pd
import sys

# >>> k=np.arange(40)
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
# NameError: name 'np' is not defined
# >>> import numpy as np
# >>> k=np.arange(40)
# >>> k
# array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
#        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
#        34, 35, 36, 37, 38, 39])
# >>> k=np.arange(40)
# KeyboardInterrupt
# >>> poisson.logpmf(k, 20)

# ========================================================
# Create the possion mat from mean variance data
# Example command : python utility/create_possion_table.py
#                   data/gtea_meanvar_actions.pkl 11 data/gtea_possion_class_dict.pkl


mean_var_file = sys.argv[1]
num_class = sys.argv[2]
poisson_file = sys.argv[3]

loaded_mean_var_actions = pickle.load(open(mean_var_file, "rb"))

k = 20000
k_arr = np.arange(k)
mat_poission = {}  # np.zeros(num_class, k)

for class_n in loaded_mean_var_actions:
    mean_class, std_class = loaded_mean_var_actions[class_n]

    mat_poission[class_n] = poisson.logpmf(k_arr, mean_class)

pickle.dump(mat_poission, open(poisson_file, "wb"))

