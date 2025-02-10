# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
df = pd.read_csv(
    "/Users/mohit/Developer/Machine Learning/Machine_Learning_Algorithms/Random Forest/train.csv"
)

# %%
df.head()

# %%
df.shape

# %%
df.sample(5)

# %%
plt.imshow(df.iloc[11111, 1:].values.reshape(28, 28), cmap="gray")

# %%
X = df.iloc[:, 1:]

# %%
y = df.iloc[:, 0]
