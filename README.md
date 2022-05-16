En este desafío se implementaron 2 perceptrones multicapa uno usando la librería de Python keras y otro usando matrices para representar la red neuronal. Para el caso de las matrices, se utilizó la función de activación Sigmoide y su derivada, en general la precisión obtenida para predecir la categoría de ropa de una prenda dada por su imagen representada en pixeles fue eficaz, ya que en general la precisión ronda alrededor del 0.9. Sin embargo, para el caso de las camisas cuyo label es el 6, los resultados fueron bajos en comparación con el resto, resultando un 0.64. Los casos más eficientes fueron los pantalones con 0.98 los bolsos, carteras con 0.97 y sandalias con 0.96.
En la implementación con keras se usaron 4 capas de neuronas con 320, 160,80,30 respectivamente, en todas se usó como función de activación relu y solo en la capa de salida se usó softmax con un optimizador Adam y para las pérdidas se usa regresión del error cuadrático medio. Para definir las operaciones de testeo y aprendizaje se usó un epoch de 20, batch size de 400 y un validation-split de 0.2. En este caso se usó matriz de confusión para cada una de las prendas, en términos generales se puede apreciar que los verdaderos negativos tienden a ser el valor significativamente más alto, oscilando alrededor de los 9000 a los 9200 casos, mientras que los verdaderos positivos, oscila entre de 700 a 1000, cuando se observan los falsos positivos y negativos, se obtienen resultados bastante bajos con respecto a los valores verdaderos, dándonos a entender que las neuronas aprenden y aciertan respecto a la clasificación esperada de cada prenda.
La prenda de la camisa en ambos casos fue la que obtuvo valores menos eficientes, para el caso de la matriz de confusión obtuvo el falso positivo más alto 413 y dándonos unos de los verdaderos positivos más bajos con 728. Al observar la precisión implementada con matrices, también entrega el valor más bajo, dándonos 0.64.
En ambos casos las gráficas de testeo y aprendizaje cumplen con arrojar una disminución de los costos, lo que comprueba que las neuronas están aprendiendo. Además, esto se evidencia en los resultados arrojados, pues las probabilidades giraron en torno al 0.9 en cada clasificación según el tipo de prenda.

–Asistencia: Nicolás Espinoza: +1 Carlos Méndez: +1 Luis González: -1

–Integración: Nicolás Espinoza: +1 Carlos Méndez:-1 Luis González:+1

–Responsabilidad: Nicolás Espinoza: -1 Carlos Méndez: +1 Luis González: 0

–Contribución: Nicolás Espinoza:-1 Carlos Méndez:+1 Luis González:-1

–Resolución de Conflictos: Nicolás Espinoza:0 Carlos Méndez:0 Luis González:+1

–Aspectos Positivos: Nicolás Espinoza: Buena disposición a trabajar en equipo. Carlos Méndez: Responsabilidad y manejo de la materia. Luis González: Buena disposición para trabajar en equipo.

–Aspectos a mejorar: Nicolás Espinoza: Manejo del lenguaje de programación. Carlos Méndez: Trabajo en equipo. Luis González: Manejo del tiempo.