# ia-clasificador-imagenes

Un código simple :coffee: para un :bar_chart: Clasificador de imágenes en Python utilizando Inteligencia Artificial.

1. Crea una entorno virtual de Python versión 3+.
```console
virtualenv -p python3 env     # Creas entorno virtual con Python3+.
source env/bin/activate       # Activar entorno virtual.
deactivate                    # Para salir del entorno virtual (EJECUTAR SOLO SI YA NO LO VAS A USAR).
```

2. Ejecuta el archivo _install.sh_ (si estas en MAC OSX o GNU/Linux, funcionará), esto instalara las librerias necesarias del proyecto.
```console
source install-lib.sh
```

3. Ejecuta primeramente el _entrenar.py_, para que puedar entrenar la red neuronal y crear el modelo.
```console
python3 entrenar.py
```

4. Ahora, ejecuta el _predecir.py_, para que puedar decirte si la imagen se encuentra o no. En el código de predecir.py la función _predict('**nombre-imagen**')_, en nombre-imagen reemplazar por una imagen cualquiera para probar el modelo.
```console
python3 predecir.py
```

Al final deberia lanzar el resultado.





_Si te gustaria colaborar, sería fantástico :v:_
