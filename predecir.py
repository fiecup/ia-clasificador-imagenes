import numpy as np
from tensorflow import keras
import keras
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform

with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
    longitud, altura = 150, 150
    modelo = './modelo/modelo.h5'
    pesos_modelo = './modelo/pesos.h5'
    cnn = load_model(modelo)
    cnn.load_weights(pesos_modelo)

    def predict(file):
        x = load_img(file, target_size = (longitud, altura))
        x = img_to_array(x)
        # print("La x:: ", x)
        x = np.expand_dims(x, axis=0)
        array = cnn.predict(x)
        # print("array:: ", array)
        result = array[0]
        # print("result:: ", result)
        answer = np.argmax(result)
        msg = "--->Predicción: "
        if answer == 0:
            print(msg," No imagen")
        elif answer == 1:
            print(msg," Si imagen")

        return answer

    predict('no-test2.png') ## buscar una imagen de ejemplo