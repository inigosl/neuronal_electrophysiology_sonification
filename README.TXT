Las funciones diferenciaciónNeuronas.py, diferenciacionNeuronasPDintra.py y sonificaciónInvariantes.py son las relativas a la sonificación presentadas en el TFG. Las dos primeras corresponden
al primer objetivo, la tercera lleva a cabo el segundo objetivo.

Estas pueden reproducir y/o grabar los audios generados.

Para crear los vídeos se usan las funciones crear.py, invariante_video_functions_1.py y combine.py.

La función crear.py está pensada para ser ejecutada, esta llamará a invariante_video_functions_1.py para crear el video. Esta última función es, en esencia, la función de Manuel Reyes
con ligeros cambios. Tras esto, si se ha puesto en las variables, se habrá guardado el video generado.

Una vez tenemos video y audio, se puede ejecutar el script combine.py, donde se pueden sincronizar ambos archivos de audio y crear un video sonificado. Es una función muy torpe y sin duda 
hay otras formas de hacer esto, pero funciona.