import pandas as pd
import ML_mst as mst
import math
import numpy as np



# there is alternative from sklearn.model_selection import train_test_split
def train_test_split_df(df, percentage = 0.75):
    """
    Функція приймає 2 параметри.
    Перший - це ДатаФрейм/Таблиця.
    Другий - це якй відсоток рядків піде на тренування. За замовчуванням 75
    """
    train_amount = int(len(df) * percentage)
    # Повертає 2 ДатаФрейми, перший для тренування, другий для тесту
    return (df.iloc[:train_amount], df.iloc[train_amount:])


class Scaler():
    """
    Цей клас для нормалізації даних
    fit_transform - Пошук найменшого, і найбільшого, нормалізуємо дані, менше, і більше зберігається
    transform - нормалізуємо дані. на основі попередніх.
    """

    def fit_transform(self, X):
        """
        Функція використовується для нормалізації даних,
        можна використати як для одного стовпця, так і для цілого ДФ.
        """
        self._min = X.min()
        self._max = X.max()
        self.diff =  self._max - self._min
        # if self.diff == 0:
        #     self.diff = 1
        return (X - self._min) / self.diff

    def transform(self, X):
        """
        Функція для нормалізації даних на основі попередніх даних
        """
        return (X - self._min) / self.diff




class KNN_classifier():
    def __init__(self, k_number = 1):
        '''
        Зберігаємо параметр скільки сусідів будемо брати на розгляд.
        За замовчуванням 1.
        Рекомендується брати не парні числа.
        '''
        self._k = k_number

    def fit(self, X, y):
        '''
        Зберігає набір даних.
        :param X: dataframe of features
        :param y: pd.Series of labels
        '''
        self._X = X
        self._y = y

    def calculate_distance(self, x, t):
        """
        x, t - pd.Series - Вони є рядками DataFrame***
        Обчислюємо відстань* від x до t
        Тобто обчислюємо відстань між квітками/атрибутами квіток.
        return: float
        """
        return math.sqrt(np.sum((x - t)**2))


    def predict(self, target_observation):
        """
        :param target_observation: - Отримуємо рядок для обчислення відстані
        :return: Найбільш ймовірний клас квітки
        """

        # Копіюємо Х що б не змінювати його
        df = self._X.copy()

        # створюємо новий стовпець з дистанцією до кожної квітки.
        df["distance"] = df.apply(self.calculate_distance, axis=1, t=target_observation)

        # Створюємо копію міток в "_у"
        y_sr = self._y.copy()
        # Додаємо стовпець
        y_sr.name = "label_reserved_name"
        # Проводимо склеювання/конкатенацію
        df = pd.concat([df, y_sr], axis=1)


        # Сортуємо по стовпцю дистанція
        df.sort_values("distance", inplace=True)

        # Скидаємо індекси, тобто оновлюємо їх
        df.reset_index(inplace=True, drop=True)

        # Робимо зріз до К. зберігаємо його в змінну
        df_k = df.iloc[0:self._k]

        #
        df_grouped = df_k.groupby("label_reserved_name")["label_reserved_name"].agg(["count"]).reset_index().sort_values("count",ascending=False)
        return df_grouped.iloc[0]["label_reserved_name"]


    def score(self, X_test, y_test):
        '''
        Обчислюємо кількість вірних передбачень.
        :param X_test: features
        :param y_test: labels
        :return: float in [0:1]
        '''
        return np.mean(y_test == X_test.apply (self.predict,axis =1))

