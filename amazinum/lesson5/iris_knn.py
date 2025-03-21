from iris_dataframe import df_iris
from algo_knn import train_test_split_df, Scaler, KNN_classifier





# Розділення на тренувальні та тестові дані (75% на тренування, 25% на тестування)
train_df, test_df = train_test_split_df(df_iris, percentage=0.75)

# Масштабування даних
scaler = Scaler()
X_train_scaled = scaler.fit_transform(train_df.drop(['label', 'label_names'], axis=1))
X_test_scaled = scaler.transform(test_df.drop(['label', 'label_names'], axis=1))

# Мітки для тренувальних та тестових даних
y_train = train_df['label']
y_test = test_df['label']

# Створення та тренування моделі KNN з k=3

for i in range(1, 52, 2):
    knn = KNN_classifier(k_number=i)
    knn.fit(X_train_scaled, y_train)

# Оцінка точності на тестових даних
    accuracy = knn.score(X_test_scaled, y_test)
    print(f"Точність моделі KNN, при k={i}: {accuracy:.2f}")




# print(df_iris)