# %% [markdown]
# Generating Random data

# %%
import numpy as np
import pandas as pd

np.random.seed(23)
mu_vec1 = np.array([0, 0, 0])
cov_mat1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
class1_sample = np.random.multivariate_normal(mu_vec1, cov_mat1, 20)


# %%
df = pd.DataFrame(class1_sample, columns=["feature1", "feature2", "feature3"])
df["target"] = 1

mu_vec2 = np.array([1, 1, 1])
cov_mat2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
class2_sample = np.random.multivariate_normal(mu_vec2, cov_mat2, 20)

df1 = pd.DataFrame(class2_sample, columns=["feature1", "feature2", "feature3"])

df1["target"] = 0

# df = df.append(df1,ignore_index=True)
df = pd.concat([df, df1], ignore_index=True)

df = df.sample(40)

# %%
df.head()


# %%
import plotly.express as px

fig = px.scatter_3d(df, x="feature1", y="feature2", z="feature3", color="target")
fig.show()

# %% [markdown]
# 1.Applying Standard Scalling

# %%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

df.iloc[:, 0:3] = scaler.fit_transform(df.iloc[:, 0:3])


# %% [markdown]
# 2. Find Covariance Matrix
#

# %%
covariance_matrix = np.cov([df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2]])
print(covariance_matrix)

# %% [markdown]
# 3. Finding EV and EVs

# %%
eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)

# %%
eigen_values, eigen_vectors

# %% [markdown]
# Plotting the graph

# %%
%matplotlib inline

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch


class Arrow3D(FancyArrowPatch):
    def init(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.init(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)
        return np.min(zs)


fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection="3d")

ax.plot(
    df["feature1"],
    df["feature2"],
    df["feature3"],
    "o",
    markersize=8,
    color="blue",
    alpha=0.2,
)
ax.plot(
    [df["feature1"].mean()],
    [df["feature2"].mean()],
    [df["feature3"].mean()],
    "o",
    markersize=10,
    color="red",
    alpha=0.5,
)
for v in eigen_vectors.T:
    a = Arrow3D(
        [df["feature1"].mean(), v[0]],
        [df["feature2"].mean(), v[1]],
        [df["feature3"].mean(), v[2]],
        mutation_scale=20,
        lw=3,
        arrowstyle="-|>",
        color="r",
    )
    ax.add_artist(a)
ax.set_xlabel("x_values")
ax.set_ylabel("y_values")
ax.set_zlabel("z_values")

plt.title("Eigenvectors")

plt.show()