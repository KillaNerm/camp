import pandas as pd
pd.set_option('display.max_columns', None)

from sklearn.datasets import load_diabetes

X, y = load_diabetes(return_X_y=True, as_frame=True)

all_variables = pd.concat([X, y], axis=1)


print(X.head())
print(y.head())
print(all_variables.head())