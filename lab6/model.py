from tensorflow.keras.models import load_model #импортируем функцию load_model из Keras для загрузки сохраненной нейросети
import numpy as np #импортируем NumPy для работы с массивами чисел
import os #импортируем os для проверки существования файла модели

MODEL_PATH = 'models/iris_model.keras' 
CLASS_NAMES = ['Setosa', 'Versicolor', 'Virginica'] #список названий классов в порядке обучения модели

#создаем глобальную переменную для модели
_model = None

#функция для загрузки модели из файла
def load_iris_model():
    global _model 
    if os.path.exists(MODEL_PATH):  #проверяем, существует ли файл с моделью по указанному пути
        _model = load_model(MODEL_PATH)  #если файл существует - загружаем модель и сохраняем в переменную _model
        print('Модель успешно загружена!') 
    return _model #возвращаем загруженную модель

#функция для предсказания вида ириса по четырем параметрам
def predict_iris(sepal_length, sepal_width, petal_length, petal_width):
    global _model
    if _model is None: #проверяем, загружена ли модель (если переменная все еще пустая)
        load_iris_model()  #если модель не загружена - вызываем функцию загрузки

    #создаем двумерный массив NumPy из четырех входных параметров
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    #передаем массив в модель для предсказания
    pred = _model.predict(features, verbose=0)
    
    # verbose=0 отключает вывод информации о процессе предсказания
    # pred[0] - первый (и единственный) элемент массива предсказаний
    # np.argmax() находит индекс элемента с максимальным значением (вероятностью)

    return CLASS_NAMES[np.argmax(pred[0])] # CLASS_NAMES[...] - получаем название класса по найденному индексу
