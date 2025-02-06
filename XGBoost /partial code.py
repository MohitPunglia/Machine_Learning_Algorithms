# %%
import numpy as np
import pandas as pd

# %%
df = pd.read_csv("titanic.csv")

# %%
df.head()

# %%
df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], inplace=True)

# %%
df

# %% [markdown]
# 1. Missing values in Age and Embarked to fill
# 2. one Hot Endocing in Sex and Embarked

# %%
X = df.iloc[:, 1:]
y = df.iloc[:, 0]

# %%
X

# %%
y

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# %%
X_train.head()

X_train.shape

# %%
X_test.shape

# %% [markdown]
# Adding column transformer from SKlearn
#

# %%
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# %%
trf1 = ColumnTransformer(
    [
        ("imputer", SimpleImputer(), [2]),
        ("imputer2", SimpleImputer(strategy="most_frequent"), [6]),
    ],
    remainder="passthrough",
)

# %%
trf1.fit_transform(X_train).shape

# %%
trf1.named_transformers_

# %%
trf1.named_transformers_["imputer"].statistics_

# %%
trf1.named_transformers_["imputer2"].statistics_

# %%
trf2 = ColumnTransformer(
    [("ohe", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), [1, 6])]
)

# %%
from xgboost import XGBClassifier


# %%
clf = XGBClassifier()

# %%
from sklearn.pipeline import Pipeline

# %%
pipe = Pipeline([("step1", trf1), ("step22", trf2), ("step3", clf)])

# %%
pipe.fit(X_train, y_train)

# %%
y_pred = pipe.predict(X_test)

# %%
from sklearn.metrics import accuracy_score

# %%
accuracy_score(y_test, y_pred)

# %%
