# python3
# Name:         2.1.1
# Description:  
# Author:       xiaoshi
# Time:         2019/7/15 18:22
import matplotlib.pyplot  as plt
import numpy as np

x = np.arange(-10, 10, 0.5)
y = 1.0 / (1 + np.exp(-x))

plt.plot(x, y)
plt.xlabel('z')
plt.ylabel('g')
plt.show()