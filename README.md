# TRABAJO-FIN-DE-GRADO
En este repositorio se encuentran todos los códigos de la red neuronal profunda creada para el TFG del grado en tecnologías de telecomunicación. Los códigos incluyen cierto texto explicativo de cada parte del código, aunque el funcionamiento general de la DNN se explica de forma detallada en la memoria.

La función "genera_datos_ECG.m" no funcionará ya que no se proporcionan en este repositorio las funciones "FUN_ECG.m" ni "FUN_NOISE.m" . Para poder simular el modelo de la red neuronal profunda se han de cargar los datos que se proporcionan en los ficheros "datos_train.mat", "datos_val.mat", "datos_test.mat" con la siguiente instrucción

load('datos_train.mat')

Una vez cargados los datos, se puede comprobar el funcionamiento del modelo proporcionado de una red neuronal profunda ejecutando el fichero "main.m"
