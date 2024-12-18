# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
data = pd.read_csv("Social_Netwoks_Ads.csv")
data.head()

# %%
X = data.iloc[:, 2:4].values

# %%
X.shape

# %%
y = data.iloc[:, -1].values

# %%
y.shape

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# %%
X_train.shape

# %%
X_test.shape

# %% [markdown]
# As we are using KNN 2 input colums have very large difference we need to scale it dowm to age's value
#
# we will use class standardscaler inside sklearn.preprocessor

# %%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()


# %%
X_train = scaler.fit_transform(X_train)

# %%
X_train

# %%
X_test = scaler.transform(X_test)

# %%
X_test

# %% [markdown]
# Applying KNN algo :
#
#  We need to find K by applying square root

# %%
np.sqrt(X_train.shape[0])

# %%
k = 17

# %%
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=k)

# %%
# for training our model

knn.fit(X_train, y_train)

# %%
y_pred = knn.predict(X_test)

# %%
y_pred.shape

# %%
# need to check y_test shape to ensure both are same
y_test.shape

# %%
# to verify Accuracy
from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)


# %%
# to verify confusion matrix
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_pred)


# %% [markdown]
# Let's try 2nd method which is trail and error method
#

# %%
error_train = []
error_test = []

for i in range(1, 29):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    x = confusion_matrix(y_train, knn.predict(X_train))
    y = confusion_matrix(y_test, knn.predict(X_test))
    error_train.append((x[0][1] + x[1][0]) / x.sum())
    error_test.append((y[0][1] + y[1][0]) / y.sum())

# %%
error_train

# %%
len(error_train)

# %%
len(error_test)

# %%
plt.plot(range(1, 29), error_train, label="Training error rate")
plt.plot(range(1, 29), error_test, label="Test error rate")
plt.xlabel("K value")
plt.ylabel("Error")
plt.legend()

# %%
knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(X_train, y_train)

# %%
y_pred = knn.predict(X_test)

# %%
accuracy_score(y_test, y_pred)

# %% [markdown]
# Creating a function for predecting the output
#


# %%
def predict_output():
    age = int(input("Enter your age : "))
    salary = int(input("Enter your salary : "))

    X_new = np.array([[age], [salary]]).reshape(1, 2)
    X_new = scaler.transform(X_new)

    if knn.predict(X_new)[0] == 0:
        return "Item can not be purchased"
    else:
        return "Item can be purchased"


# %%
predict_output()

# %% [markdown]
# Creating a Meshgrid
#

# %%
a = np.arange(start=X_train[:, 0].min() - 1, stop=X_train[:, 0].max() + 1, step=0.01)
a.shape

# %%
b = np.arange(start=X_train[:, 1].min() - 1, stop=X_train[:, 1].max() + 1, step=0.01)
b.shape

# %%
XX, YY = np.meshgrid(a, b)

# %%
XX.shape

# %%
YY

# %% [markdown]
# classify every point on the meshgrid
#

# %% [markdown]
# Fetching 1st value
#

# %%
print(XX[0][0])
print(YY[0][0])
knn.predict(np.array([-2.862945294656013, -2.5686411260352955]).reshape(1, 2))

# %% [markdown]
# Need to perform for entire data

# %%
# Example of ravel
# m=np.array([[1,2,3],[4,5,6]])
# n=np.array([[7,8,9],[0,0,0]])
# m
# n
# np.array([m.ravel(),n.ravel()]).shape

np.array([XX.ravel(), YY.ravel()]).shape
input_array = np.array([XX.ravel(), YY.ravel()]).T
labels = knn.predict(input_array)


# %%
labels.shape

# %% [markdown]
# Plotting the array as image
#
#

# %%
plt.contourf(XX, YY, labels.reshape(XX.shape))

# %% [markdown]
# Plotting all the training data on plot
#

# %%
plt.contourf(XX, YY, labels.reshape(XX.shape), alpha=0.75)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)

# %%
