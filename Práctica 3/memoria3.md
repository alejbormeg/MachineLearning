---
title: Práctica 3
subtitle: Aprendizaje Automático
author: Alejandro Borrego Megías
date: Curso 2020-2021
documentclass: report
toc: true
toc-depth: 2
toc-title: Índice
header-includes:
- \usepackage[spanish,es-tabla]{babel}
---

# Ajuste de Modelos Lineales


## 1.Problema

**Comprender el problema a resolver. Identificar los elementos X, Y, $f$ del problema y describirlos en detalle**

Vamos a tratar de resolver con técnicas de Machine Learning usando modelos lineales dos supuestos prácticos. Uno de ellos se resolverá usando Regresión y el otro usando Clasificación.

### Regresión

Usaremos la base de datos "Superconductividad" del siguiente enlace: https://archive.ics.uci.edu/ml/datasets/Superconductivty+Data

En dicho enlace vemos que no hay mucha información del problema, pero te redirecciona al siguiente enlace, donde sí que se explica el problema que vamos a tratar: https://www.sciencedirect.com/science/article/abs/pii/S0927025618304877?via%3Dihub 

Aquí podemos ver que se pretende estimar por medio de regresión la temperatura crítica superconductora en función de las características extraidas de la fórmula química del superconductor. Por lo tanto, esto no permite diferenciar los elementos del problema: 

- Por un lado $\mathcal{X}$ será la matriz de características, dónde en cada columna medimos una de las características de la fórmula química del superconductor. En total tenemos 81 características. Se nos proporciona una matriz de 21263x81.
  
- Por otra parte $\mathcal{Y}$ (nuestro vector de etiquetas) contiene las temperaturas para cada uno de los superconductores en los que hemos medido las 81 características. En nuestro caso, el vector $\mathcal{Y}$  viene pegado a la matriz $\mathcal{X}$, por lo que para manipularlo tendremos que separarlo.

- Finalmente la función que pretendemos estimar, y que es desconocida sería $f:\mathcal{X}\to\mathcal{Y}$ que nos asocia para cada vector de características una etiqueta.


### Clasificación

Usaremos la base de datos "Sensorless Drive Diagnosis" del siguiente enlace: https://archive.ics.uci.edu/ml/datasets/Dataset+for+Sensorless+Drive+Diagnosis

El problema a tratar consiste en el Diagnóstico de unidad sin sensor, si leemos la descripción del problema nos dicen que las características se extraen de la corriente del motor, el motor además tiene componentes intactos y defectuosos y esto se traduce en 11 clases diferentes, con diferentes condiciones. Por lo que nos encontramos ante un problema de *Clasificación multiclase*.

Vamos a tratar de identificar los elementos: 

- Por un lado $\mathcal{X}$ nuevamente es la matriz de características observadas en los distintos elementos. En este caso tenemos un total de 58509 instancias en las que se han medido 49 atributos reales, por lo que disponemos de una matriz $\mathcal{X}$ de 58509x49
- Como se ha mencionado, disponemos de 11 posibles clases dónde agrupar los elementos, y si miramos el fichero txt dónde vienen las características de cada elemento, podemos ver que al final de cada tupla viene un número del 1 al 11 haciendo referencia a la clase a la que pertenece cada elemento. Luego nuestro vector $\mathcal{Y}$ de etiquetas será un vector columna de 58509x1 dónde cada posición en el vector toma un valor entre 1 y 11.
- Finalmente la función que deseamos aproximar es $f:\mathcal{X}\to\mathcal{Y}$ que es desconocida y nos asocia cada elemento a una clase de entre las 11.

## 2.Obtención y visualización de los datos
Una vez nos descargamos los datos de las bases de datos mencionadas en el apartado anterior, las metemos en nuestro directorio de trabajo en una carpeta llamada "datos", de manera que en nuestro código definiremos las siguientes variables para acceder a dichos ficheros: 

~~~py
#En el Fichero Regresion.py
NOMBRE_FICHERO_REGRESION = './datos/train.csv'


#En el Fichero Clasificacion.py
NOMBRE_FICHERO_CLASIFICACION = './datos/Sensorless_drive_diagnosis.txt'
~~~

Después leemos los datos y los separamos en los conjuntos de Entrenamiento y Test (función LeerDatos)
dejando un 80% de los datos en Entrenamiento y el 20% restante en Test (esta proporción de división es la más empleada y la que mejor resultados suele dar). Desde este momento en adelante no miraremos en ningún momento los datos de Test y todas las transformaciones que hagamos será en el conjunto de Entrenamiento, así evitaremos cometer "Data Snooping".


### Caso Regresión

En el caso de Regresión, al tener una gran dimensionalidad los datos ($\mathbb{R}^{81}$), vamos a optar por no visualizar el conjunto de entrenamiento y simplemente haremos un estudio de las correlaciones existentes entre los datos en la posterior etapa de preprocesado.

### Caso Clasificación
En el caso de Clasificación si que vamos a optar por visualizar los datos de entrenamiento. Para ello dado que los datos están en una dimensión que no podemos visualizar ($\mathbb{R}^{49}$) vamos a aplicar el algoritmo TSNE, una algoritmo muy empleado para reducir la dimensionalidad y tratar de visualizar datos en 2-3 dimensiones conservando parcialmente su estructura. El funcionamiento de dicho algoritmo, en rasgos generales es el siguiente:

- Calcular una distribución de probabilidad conjunta que represente las similitudes entre los puntos de datos.

- Crear un conjunto de datos de puntos en la dimensión objetivo y luego calcular la distribución de probabilidad conjunta para ellos también.

- Usar el descenso de gradiente para cambiar el conjunto de datos en el espacio de baja dimensión de modo que la distribución de probabilidad conjunta que lo representa sea lo más similar posible a la de la dimensión alta.

La pega que tiene el algoritmo es que es algo lento en tiempo de ejecución, por eso, únicamente pondré los resultados obtenidos (para repetir el experimento descomentar las líneas comentadas en el archivos Clasificacion.py). La sentencia en cuestión es: 

~~~py
X_visualizar=TSNE(n_components=2).fit_transform(x_entrenamiento)
~~~

Usamos el objeto TSNE de la biblioteca sklearn.manifold, y especificamos en la variable n_components las dimensiones a las que reducimos los datos, y luego con fit_transform decimos que se apliquen estas transformaciones a los datos de entrenamiento y se almacene el resultado en X_visualizar. El resultado obtenido es el siguiente:

---------------------------------------------------------------------
![TSNE](TSNEClass.png)
---------------------------------------------------------------------


Como podemos observar, las distintas clases se encuentran muy solapadas, lo que nos hace pensar que quizá en el preprocesamiento de datos tendremos que hacer algo para tratar de diferenciarlas un poco y que nuestro modelo funcione mejor.



## 3.Clase de funciones a usar 

A continuación, tal y como se ha visto en Teoría, una vez comprendido el problema vamos a establecer la clase $\mathcal{H}$ de funciones que usaremos en nuestro modelo para estimar $f$.

### Caso Regresión

Para el caso de regresión, dado que siempre vamos a tratar de elegir la clase $\mathcal{H}$ más simple posible (de acuerdo al principio de la Navaja de Okham) vamos a establecer la clase de combinaciones lineales con las características, de manera que la hipótesis que obtengamos tenga la forma: 
$$g(x)=w^{T}x$$


Es decir, nuestra clase $\mathcal{H}$ será el conjunto de todos los hiperplanos de $\mathbb{R}^{81}$, de manera que nuestro problema se reducirá a encontrar el vector de pesos $w$ empleando SGD (Gradiente Descendente estocástico). 

### Caso Clasificación

Para el caso de Regresión Logística
... Completar


## 4. Identificar las hipótesis finales que usará.

## 5. Partición en Training y Test

En los dos datasets nos venía un único fichero con los datos para entrenamiento y test unidos, es por ello que no hemos tenido que juntar datos para luego separarlos. 

Para separar los datos vamos a diferenciar lo que hemos hecho en Regresión y en Clasificación: 


### Caso Regresión
Para dividir el dataset en Entrenamiento y Test usamos la función de sklearn.model_selection llamada  train_test_split de la siguiente forma:

~~~py
x_entrenamiento, x_test, y_entrenamiento, y_test = train_test_split(x,
 y, test_size = 0.2, random_state = 1)
~~~

Como podemos observar nos devuelve el conjunto de entrenamiento y test con sus respectivas etiquetas y los parámetros a utilizar son, el conjunto x de características, el vector y de datos, la fracción de datos que usaremos como test (usamos un 20% pues es lo más habitual) y usamos una semilla para que al repetir el experimeto en una máquina diferente se produzcan los mismos resultados.


### Caso Clasificación

Este caso es prácticamente idéntico al de Regresión, pero con una salvedad importante. Al tener 11 clases diferentes, debemos ver si están balanceadas y que la estratificación de los datos se produzca de acuerdo al número de elementos que haya de cada clase, pues cabría el riesgo de que al dividir los datos, nuestro conjunto de Entrenamiento no fuera representativo. 

Es por eso que usamos la misma función pero especificando que se estratifique según las etiquetas de y (stratify=y): 

~~~py
x_entrenamiento, x_test, y_entrenamiento, y_test = train_test_split(x, 
y, test_size = 0.2, random_state = 1, stratify=y)
~~~

A cotinuación mostramos la cantidad de elementos por clase en el conjunto de datos antes de separarlo en entrenamiento y test y tras la división: 

---------------------------------------------------------------------
![Balanceo Dataset](balanceodataset.png)
---------------------------------------------------------------------

---------------------------------------------------------------------
![Balanceo Entrenamiento](balanceotraining.png)
---------------------------------------------------------------------

Como vemos, en el datset original las clases estaban perfectamente balanceadas y tras la división se sigue manteniendo ese balanceo.

## 6. Preprocesado de datos

En esta etapa detallaremos todo el proceso de preprocesado de datos que se realiza en nuestro modelo de Regresión y Clasificación: 

### Caso Regresión

En el caso de Regresión, dado que vamos a usar el gradiente descendente estocástico para la obtención del vector de pesos $w$, sabemos que este método es sensible al escalado de características (las características con valores más altos tienen más peso para el algoritmo)  y como podemos observar en la matriz de características $X$ del conjunto de entrenamiento, las distintas columnas toman valores en unos rangos muy diferentes: 

~~~
[[  5.          92.94072     50.18273333 ...   0.48989795   0.24944383
   78.        ]
 [  3.          64.17263953  78.70759272 ...   0.81649658   0.80467385
   16.5       ]
 [  7.         112.27482857  89.88988857 ...   1.12485827   0.79702892
   80.        ]
 ...
 [  1.          47.867       47.867      ...   0.           0.
    0.39      ]
 [  5.          74.290174    56.81209207 ...   0.4          0.44095212
   22.3       ]
 [  4.          76.5177175   56.84544955 ...   0.4330127    0.44133321
   32.2       ]]
~~~

Es por esto que en esta etapa de preprocesado vamos a hacer una estandarización utilizando la función StandardScaler() de sklearn.preprocessing y logrando así que nuestros datos se distribuyan según una Gausiana de media 0 y varianza 1.

~~~py
x_estandarizado=StandardScaler().fit_transform(x_entrenamiento)
~~~

Aplicamos la función StandardScaler() a los datos de entrenamiento almacenando el resultado en x_estandarizado. Si ahora visualizamos x_estandarizado vemos lo siguiente: 

~~~
[[ 0.61417205  0.18370187 -0.67942496 ... -0.71934604 -0.93186082
   1.27666332]
 [-0.77802766 -0.78863442  0.17582239 ... -0.04587185  0.28973162
  -0.52493387]
 [ 2.00637176  0.83717805  0.51109586 ...  0.5899962   0.2729116
   1.33525184]
 ...
 [-2.17022736 -1.33975096 -0.74885649 ... -1.72955732 -1.48067605
  -0.99686445]
 [ 0.61417205 -0.44667049 -0.48066004 ... -0.90472327 -0.51051277
  -0.35502715]
 [-0.0819278  -0.37138144 -0.47965989 ... -0.83664826 -0.50967432
  -0.06501394]]
~~~

Ahora todas las características tienen un rango de valores similar.

### Caso Clasificación


## 7. Métrica de error usada.


### Caso Regresión
En el caso de regresión, en el modelo que hemos elegido usaremos como métrica de error el error cuadrático medio. Que nos da la media de los cuadrados de las distancias de los distintos puntos al hiperplano de regresión elegido. Su expresión es la siguiente: 

![Error Cuadrático Medio](ErrorCuadraticoMedio.png)


### Caso Clasificación

