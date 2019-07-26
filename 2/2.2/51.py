# python3
# Description:  
# Author:       xiaoshi
# Time:         2019/7/23 11:33
import numpy as np

M = np.array([[1,2], [2,4]])
print(np.linalg.matrix_rank(M, tol=None))