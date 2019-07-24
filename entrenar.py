import sys
import os
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Sequential # Nos permite hacer redes neuronales secuenciales
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.python.keras import backend as K # Nos permite matar subprocesos que ya no son necesarios

K.clear_session()

datos_entrenamiento = './datos_entrenamiento'
datos_prueba = './datos_prueba'

## Parametros

# 'epocas = 20' Numero de veces que vamos a
# iterar sobre nuestros datos de entrenamientos
epocas = 20 

# Ajustamos el tamaño de las imagenes
altura, longitud = 100, 100

# Cantidad de Imagenes
cantidad_datos = 57

# Numero de veces que se va a procesar la 
# información en cada una de las epocas
pasos = 1000

pasos_validacion = 200

filtros_conv1 = 32
filtros_conv2 = 64

tamano_filtro1 = (3,3)
tamano_filtro2 = (2,2)

tamano_pool = (2,2)
clases=1

# Learning Rate 
lr=0.0005


## PreProcesamiento de Imagenes

entrenamiento_datagen = ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.3,
    zoom_range = 0.3,
    horizontal_flip = True
)

validacion_datagen = ImageDataGenerator(
    rescale = 1./255
)

imagen_entrenamiento = entrenamiento_datagen.flow_from_directory(
    datos_entrenamiento,
    target_size = (altura, longitud),
    batch_size = cantidad_datos,
    class_mode = 'categorical'
)

imagen_validacion = validacion_datagen.flow_from_directory(
    datos_prueba,
    target_size = (altura, longitud),
    batch_size = cantidad_datos,
    class_mode = 'categorical'
)

## Creando la Red CNN

cnn = Sequential()

cnn.add(
    Convolution2D(
        filtros_conv1,
        tamano_filtro1,
        padding = 'same',
        input_shape = (altura, longitud, 3),
        activation = 'relu'
    )
)

cnn.add(
    MaxPooling2D(pool_size = tamano_pool)
)

cnn.add(
    Convolution2D(
        filtros_conv2,
        tamano_filtro2,
        padding = 'same',
        activation = 'relu'
    )
)

cnn.add(
    MaxPooling2D(pool_size = tamano_pool)
)

cnn.add(Flatten())

cnn.add(
    Dense(256, activation = 'relu')
)

cnn.add(
    Dropout(0.5)
)

cnn.add(
    Dense(clases, activation = 'softmax')
)

cnn.compile(
    loss='categorical_crossentropy',
    optimizer = optimizers.Adam(lr=lr),
    metrics = ['accuracy']
)

cnn.fit(
    imagen_entrenamiento,
    steps_per_epoch = pasos,
    epochs = epocas,
    validation_data = imagen_validacion,
    validation_steps = pasos_validacion
)

div = './modelo/'

if not os.path.exists(dir):
    os.mkdir(dir)

cnn.save('./modelo/modelo.h5')
cnn.save_weights('./modelo/pesos.h5')

print ('¡Has ejecutado el entrenamiento!')
