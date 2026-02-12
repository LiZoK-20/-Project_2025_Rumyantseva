# Шаг 1: Импорт необходимых библиотек и классов

import numpy as np # Импортируем NumPy для работы с массивами и математическими операциями
from sklearn.datasets import load_iris # Импортируем функцию для загрузки датасета Iris из библиотеки Scikit-learn
from sklearn.model_selection import train_test_split # Импортируем функцию для разделения данных на обучающую и тестовую выборки
from sklearn.metrics import mean_squared_error # Импортируем функцию для подсчёта ошибки
from tensorflow.keras.models import Sequential # Импортируем модель для создания нейронной сети (последовательные слои)
from tensorflow.keras.layers import Dense # Импортируем Dense слой (полносвязный слой) для построения нейронной сети
from tensorflow.keras.optimizers import Adam # Импортируем оптимизатор Adam для обновления весов нейронной сети

import os # Для создания папок

# ДОБАВИМ МАТПЛОТЛИБ ДЛЯ ВИЗУАЛИЗАЦИИ
import matplotlib.pyplot as plt 

# Шаг 2: Импортировать датасет

iris = load_iris() # Загружаем датасет ирисов в переменную iris
X = iris.data # Присваиваем переменной х данные о характеристиках цветов (признаки)
y = iris.target # Присваиваем переменной y целевые значения (вид ириса: 0, 1, 2)

# ПРОСТАЯ ВИЗУАЛИЗАЦИЯ: РАЗМЕРЫ ЧАШЕЛИСТИКОВ И ЛЕПЕСТКОВ

# Создаем фигуру с 4 графиками
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Визуализация датасета Iris', fontsize=16)

# Цвета для разных классов ирисов
colors = ['red', 'green', 'blue']
class_names = ['Setosa', 'Versicolor', 'Virginica']

# 1. Длина чашелистика vs Ширина чашелистика
for i in range(3):
    axes[0, 0].scatter(X[y == i, 0], X[y == i, 1], 
                      color=colors[i], label=class_names[i], alpha=0.7)
axes[0, 0].set_xlabel('Длина чашелистика (см)')
axes[0, 0].set_ylabel('Ширина чашелистика (см)')
axes[0, 0].set_title('Чашелистики')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Длина лепестка vs Ширина лепестка
for i in range(3):
    axes[0, 1].scatter(X[y == i, 2], X[y == i, 3], 
                      color=colors[i], label=class_names[i], alpha=0.7)
axes[0, 1].set_xlabel('Длина лепестка (см)')
axes[0, 1].set_ylabel('Ширина лепестка (см)')
axes[0, 1].set_title('Лепестки')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Гистограмма распределения по классам
# Простой способ: использовать bar вместо hist
counts = [np.sum(y == i) for i in range(3)]
axes[1, 0].bar([0, 1, 2], counts, color=colors, width=0.6)
axes[1, 0].set_xlabel('Класс ириса')
axes[1, 0].set_ylabel('Количество образцов')
axes[1, 0].set_title('Распределение по классам')
axes[1, 0].set_xticks([0, 1, 2])
axes[1, 0].set_xticklabels(class_names)
# Добавляем числа над столбцами
for i, count in enumerate(counts):
    axes[1, 0].text(i, count + 1, str(count), ha='center', va='bottom')

# 4. Размеры всех признаков (боксплот)
box_data = [X[:, i] for i in range(4)]
axes[1, 1].boxplot(box_data)
axes[1, 1].set_xlabel('Признаки')
axes[1, 1].set_ylabel('Значения (см)')
axes[1, 1].set_title('Распределение признаков')
axes[1, 1].set_xticklabels(['Дл. чаш.', 'Шир. чаш.', 'Дл. леп.', 'Шир. леп.'])

plt.tight_layout()
plt.show()

# ПРОСТАЯ ВИЗУАЛИЗАЦИЯ 2: МАТРИЦА КОРРЕЛЯЦИИ

# Создаем корреляционную матрицу
correlation_matrix = np.corrcoef(X.T)

# Визуализируем матрицу корреляции
plt.figure(figsize=(8, 6))
plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')
plt.colorbar(label='Корреляция')
plt.xticks([0, 1, 2, 3], ['Дл. чаш.', 'Шир. чаш.', 'Дл. леп.', 'Шир. леп.'])
plt.yticks([0, 1, 2, 3], ['Дл. чаш.', 'Шир. чаш.', 'Дл. леп.', 'Шир. леп.'])
plt.title('Матрица корреляции признаков')
plt.show()


# Шаг 3: Разделить датасет на входные и целевые данные

# Разделяем данные на обучающую и тестовую выборки (20% для проверки, 80% на обучение)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Шаг 4: В полученных данных выделить обучающую и тестовую выборку
# (уже сделано на шаге 3)

# Шаг 5: Определить функцию ошибки (MSE)
# (будет использоваться в качестве метрики при обучении модели)

# Шаг 6: Создать Нейросеть
model = Sequential() # Создаем модель Sequential (слои идут друг за другом)

# activation='relu' - функция активации (если положительно то используем)
model.add(Dense(16, input_shape=(4,), activation='relu'))  # Входной слой с 16 нейронами 
model.add(Dense(16, activation='relu'))    # Скрытый слой с 16 нейронами

# activation='softmax' преобразует выходы в вероятности распределения по классам
model.add(Dense(3, activation='softmax'))    # Выходной слой с 3 нейронами (3 класса)

# Компилируем модель (говорим как обучать)

# metrics=['accuracy'] - точность в процессе обучения
model.compile(optimizer=Adam(learning_rate=0.001), # optimizer=Adam(learning_rate=0.001) - используем Adam с шагом 0.001
    loss='sparse_categorical_crossentropy', metrics=['accuracy']) # loss='sparse_categorical_crossentropy' - функция потерь для многоклассовой классификации

# Шаг 7: Обучить модель

# вход. данные для обучения, значения для обучения, кол-во образцов на 1 итерцию, вывод о процессе обучения
model.fit(X_train, y_train, epochs=100, batch_size=8, verbose=1)

# Шаг 8: Протестировать модель

#X_test, y_test - вход. тест. данные, целевые тестовые значения
# verbose=0 - отключаем вывод процесса оценки
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}") # Выводим значение функции потерь
print(f"Test Accuracy: {accuracy:.4f}") # Выводим точность модели на тестовой выборке

# Шаг 9: Провести оценку прогнозирования
y_pred = model.predict(X_test) # Получаем предсказания модели для тестовых данных

# Преобразуем вероятности в классы
# np.argmax находит индекс максимального значения по второй оси
y_pred_classes = np.argmax(y_pred, axis=1) # axis=1 означает операцию по строкам

# Вычисляем MSE (среднеквадратичную ошибку)
# Сравниваем истинные значения (y_test) с предсказанными классами (y_pred_classes)
mse = mean_squared_error(y_test, y_pred_classes) 
print(f"Mean Squared Error: {mse:.4f}") # Выводим значение среднеквадратичной ошибки

# ПРОСТАЯ ВИЗУАЛИЗАЦИЯ 3: РЕЗУЛЬТАТЫ КЛАССИФИКАЦИИ

# Создаем график для сравнения предсказаний с истинными значениями
plt.figure(figsize=(10, 4))

# График 1: Истинные классы
plt.subplot(1, 2, 1)
for i in range(3):
    # ИСПРАВЛЕНО: используем np.where для получения индексов
    indices = np.where(y_test == i)[0]
    plt.scatter(indices, 
                y_test[y_test == i], 
                color=colors[i], label=class_names[i], alpha=0.7, s=50)
plt.xlabel('Номер образца')
plt.ylabel('Класс')
plt.title('Истинные классы')
plt.yticks([0, 1, 2], class_names)
plt.legend()
plt.grid(True, alpha=0.3)

# График 2: Предсказанные классы
plt.subplot(1, 2, 2)
for i in range(3):
    # ИСПРАВЛЕНО: используем np.where для получения индексов
    indices = np.where(y_pred_classes == i)[0]
    plt.scatter(indices, 
                y_pred_classes[y_pred_classes == i], 
                color=colors[i], label=class_names[i], alpha=0.7, s=50)
plt.xlabel('Номер образца')
plt.ylabel('Класс')
plt.title('Предсказанные классы')
plt.yticks([0, 1, 2], class_names)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ПРОСТАЯ ВИЗУАЛИЗАЦИЯ 4: МАТРИЦА ОШИБОК (CONFUSION MATRIX)

# Создаем простую матрицу ошибок
confusion = np.zeros((3, 3), dtype=int)
for true, pred in zip(y_test, y_pred_classes):
    confusion[true, pred] += 1

# Визуализируем матрицу ошибок
plt.figure(figsize=(6, 5))
plt.imshow(confusion, cmap='Blues', interpolation='nearest')
plt.colorbar(label='Количество образцов')

# Добавляем текст в каждую ячейку
for i in range(3):
    for j in range(3):
        plt.text(j, i, str(confusion[i, j]), 
                ha='center', va='center', 
                color='white' if confusion[i, j] > confusion.max()/2 else 'black')

plt.xticks([0, 1, 2], class_names)
plt.yticks([0, 1, 2], class_names)
plt.xlabel('Предсказанный класс')
plt.ylabel('Истинный класс')
plt.title('Матрица ошибок классификации')
plt.show()

# Статистика по точности
print("\n" + "="*50)
print("СТАТИСТИКА КЛАССИФИКАЦИИ:")
print("="*50)

for i, class_name in enumerate(class_names):
    correct = confusion[i, i]
    total = np.sum(confusion[i, :])
    accuracy_class = correct / total * 100
    print(f"{class_name}: {correct}/{total} верно ({accuracy_class:.1f}%)")
    
# СОХРАНЕНИЕ МОДЕЛИ
import os

if not os.path.exists('models'):
    os.makedirs('models')
    
model.save('models/iris_model.keras')