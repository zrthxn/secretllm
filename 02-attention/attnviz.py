#%%
import numpy as np
import matplotlib.pyplot as plt

#%%
key = np.random.rand(10,1)
value = np.random.rand(1,1)

#%%
query = np.ones((2,1)) / 2

#%%
f, axs = plt.subplots(nrows=2, ncols=2, height_ratios=(0.9, 0.1), width_ratios=(0.1, 0.9))
axs[0][0].imshow(query)
axs[0][1].imshow(np.matmul(query, key.T))
axs[1][1].imshow(key.T)
# axs[1][0]
plt.show()
# %%
