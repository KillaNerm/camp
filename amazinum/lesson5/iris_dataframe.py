import pandas as pd
import numpy as np

num_seed = 2021
np.random.seed = num_seed

from sklearn.datasets import load_iris

# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)



iris = load_iris()
# print(iris)
# print ('data contains:',iris.keys())

# Х - зберігає у собі значення чашолистків, і пелюсток
X = iris.data
# print('Print: X\n', X)

# Значення/таргет квітки.
y = iris.target
# print('Print: y\n', y)

# назви квіток - вони пов'язані з "у" - таргет.
labels = iris.target_names
# print('Print: target_names\n', labels)

# назви кожного значення в Х,
feature_names = iris['feature_names']
# print('Print: feature_names', feature_names)

# Створюємо таблицю
df_iris = pd.DataFrame(X, columns= feature_names)
# print('Print: df_iris\n', df_iris)

# додаємо стовпець з номером таргету квітки.
df_iris['label'] =  y

# Створюємо пари ключ значення з квітками, додаємо назви за таргетом.    ???
features_dict = {k:v for k,v in  enumerate(labels)}
df_iris['label_names'] = df_iris.label.apply(lambda x: features_dict[x])

df_iris = df_iris.sample(frac=1, random_state=num_seed).reset_index(drop=True)


































