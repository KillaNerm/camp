import numpy as np
import pandas as pd

# Візуалізація
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


from sklearn.datasets import load_diabetes # Завантажуємо дата сет
from sklearn import preprocessing # масштабування/нормалізація
from sklearn.model_selection import train_test_split, GridSearchCV # 1 - розділення на тренування/тест, 2 - пошук гіперпараметрів.
from sklearn.metrics import r2_score, mean_squared_error #Аналіз моделі
from sklearn.linear_model import LinearRegression, Ridge, Lasso # Моделі регресії


diabetes_df = load_diabetes() # Завантажуємо дата сет
df = pd.DataFrame(diabetes_df.data, columns=diabetes_df.feature_names) # Створюємо датафрейм
df['TARGET'] = diabetes_df.target # Додаємо стовпець з цільовими значеннями
print(df.head(5))


diabetes_df.data = preprocessing.scale(diabetes_df.data) # Нормалізація
X_train, X_test, y_train, y_test = train_test_split(
    diabetes_df.data, diabetes_df.target, test_size=0.3, random_state=10) # Розділяємо дані на тренувальні та тестові


# Ordinary Least Squares - Навчання по методу най менших квадратів
ols_reg = LinearRegression() # Лінійна регресія. Метод найменших квадратів
ols_reg.fit(X_train, y_train) # Навчаємо модель на тренувальних даних
ols_predict = ols_reg.predict(X_test) # Тестуємо модель
res_ols = pd.DataFrame({'variable': diabetes_df.feature_names, 'estimate': ols_reg.coef_}) # Зберігаємо результат


# Створюємо модель рідж-регресії
ridge_reg = Ridge(alpha=0) # створюємо модель лінійної регресії рідж, з регуляризацією, alpha=0 означає що регуляризація не застосовується - Це проста регресія
ridge_reg.fit(X_train, y_train) # Навчання моделі на тестових даних.
ridge_df = pd.DataFrame({'variable': diabetes_df.feature_names, 'estimate0': ridge_reg.coef_}) # Зберігаємо результати

# Списки для зберігання прогнозів
ridge_train_predict = [] # Тренувальні
ridge_test_predict = [] # Тестові

num_alpha = np.arange(0, 200, 1) # Створюємо масив з чисел, для перебору.

# Перебір різних варіантів альфа для рідж, але є проблема у швидкодії, через те, що це можна зробити комплексно, що б скоротити час.
for alpha in num_alpha:
    ridge_reg = Ridge(alpha=alpha) # Змінюємо регуляризацію
    ridge_reg.fit(X_train, y_train) # Навчаємо модель на тренувальних даних
    var_name = 'estimate' + str(alpha)# Створюємо ім'я стовпця для кожного значення альфа
    ridge_df[var_name] = ridge_reg.coef_ # Присвоюємо стовпцю зі значенням альфа кефи які дала модель
    # Передбачення для тренувальних та тестових даних, додаємо їх в списки для зберігання прогнозів.
    ridge_train_predict.append(ridge_reg.predict(X_train))
    ridge_test_predict.append(ridge_reg.predict(X_test))


# Змінюємо рядки, і стовпці місцями. Має бути так що б  ***???
ridge_df = ridge_df.set_index('variable').T.reset_index().rename(columns={'index': 'variable'})
print(ridge_df)


# Створюємо графік за розміром 10х5
fig, ax = plt.subplots(figsize=(10, 5))

# Задаємо колі для ліній, для різних ознак, та базова пунктирна.
ax.plot(ridge_df.bmi, 'r', ridge_df.s1, 'g', ridge_df.age, 'b', ridge_df.sex, 'c', ridge_df.bp, 'y')
ax.axhline(y=0, color='black', linestyle='--') # Додаємо базову пунктирну лінію

# Назви осів координат, та заголовку
ax.set_xlabel("Lambda")
ax.set_ylabel("Beta Estimate")
ax.set_title("Ridge Regression Trace", fontsize=16)

ax.legend(labels=['bmi','s1','age','sex','bp']) # Легенда визначає/поєднує назви та кольори ліній.
ax.grid(True) # Встановлюємо сітку
plt.show() # Запускаємо візуалізацію

# =======================================

# Створюємо модель Ласо-регресії
lasso_reg = Lasso(alpha=1) # створюємо модель ласо регресії з регуляризацією
lasso_reg.fit(X_train, y_train) # Навчаємо модель на тренувальних даних
lasso_df = pd.DataFrame({'variable': diabetes_df.feature_names, 'estimate0': lasso_reg.coef_}) # Зберігаємо результати


# Зберігаємо тренувальні, і тестувальні дані сюди
lasso_train_predict = [] # Тренувальні
lasso_test_predict = [] # Тестувальні

lasso_alphas = np.arange(0.01, 8.01, 0.04) # Створюємо масив для перебору, ласо регресія дуже чутлива, тому значення менші.

# Тренуємо модель з різними значеннями альфа.
for alpha in lasso_alphas:
    lasso_reg = Lasso(alpha=alpha) # Змінюємо регуляризацію
    lasso_reg.fit(X_train, y_train) # Навчання моделі
    var_name = 'estimate' + str(alpha) # Назва для стовпця
    lasso_df[var_name] = lasso_reg.coef_ # Додаємо кефи до стовпця в ДФ

    # Збереження прогнозів тренування і тесту в списки
    lasso_train_predict.append(lasso_reg.predict(X_train))
    lasso_test_predict.append(lasso_reg.predict(X_test))

# Перевертаємо таблицю, estimate - мають бути рядками. А категорії стовпцями.
lasso_df = lasso_df.set_index('variable').T.rename_axis('estimate').reset_index() # Робимо назви категорій* індексами.


fig, ax = plt.subplots(figsize=(10, 5)) # Задаємо розміри таблиці
ax.plot(lasso_df.bmi, 'r', lasso_df.s1, 'g', lasso_df.age, 'b', lasso_df.sex, 'c', lasso_df.bp, 'y') # Задаємо кольори параметрам

# Задаємо назви осей, тайтл, і тд.
ax.set_xlabel("Lambda")
ax.set_ylabel("Beta Estimate")
ax.set_title("Lasso Regression Trace", fontsize=16)
ax.legend(labels=['bmi','s1','age','sex','bp']) # Додаємо легенду з назвами
ax.grid(True) # Включаємо сітку
plt.show() # Запуск


# =============================
#
# print("R2 Score (OLS):", r2_score(y_test, ols_predict))
# print("MSE (OLS):", mean_squared_error(y_test, ols_predict))
#
# ridge_best_alpha = num_alpha[np.argmax([r2_score(y_test, pred) for pred in ridge_test_predict])]
# print(f"Найкраще значення альфа для Ridge: {ridge_best_alpha}")
#
# lasso_best_alpha = lasso_alphas[np.argmax([r2_score(y_test, pred) for pred in lasso_test_predict])]
# print(f"Найкраще значення альфа для Lasso: {lasso_best_alpha}")
#
# print('===================================')
#
# ridge_best = Ridge(alpha=34)
# ridge_best.fit(X_train, y_train)
# ridge_predict = ridge_best.predict(X_test)
#
# print("R2 Score (Ridge):", r2_score(y_test, ridge_predict))
# print("MSE (Ridge):", mean_squared_error(y_test, ridge_predict))
#
# lasso_best = Lasso(alpha=2.81)
# lasso_best.fit(X_train, y_train)
# lasso_predict = lasso_best.predict(X_test)
#
# print("R2 Score (Lasso):", r2_score(y_test, lasso_predict))
# print("MSE (Lasso):", mean_squared_error(y_test, lasso_predict))


# # Обчислюємо кефи детермінації, на тренувальних даних(Якщо число ближче до 1 - це означає що модель добре працює)
# ridge_r_squared_train = [r2_score(y_train, p) for p in ridge_train_predict]
# lasso_r_squared_train = [r2_score(y_train, p) for p in lasso_train_predict]
#
# # Обчислюємо кефи детермінації, на тестових даних(Якщо число ближче до 1 - це означає що модель добре працює)
# ridge_r_squared_test = [r2_score(y_test, p) for p in ridge_test_predict]
# lasso_r_squared_test = [r2_score(y_test, p) for p in lasso_test_predict]









#
#
#
#
#
#

# print(df)



