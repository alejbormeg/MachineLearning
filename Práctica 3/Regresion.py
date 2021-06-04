#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Creado Viernes Mayo 14 20:04:08 2021

@author: Alejandro Borrego Megías
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import LinearSVR
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
import time


np.random.seed(1)
######### CONSTANTES #########  
NOMBRE_FICHERO_REGRESION = './datos/train.csv'

################ funciones auxiliares


def LeerDatos (nombre_fichero, cabecera = None):
    '''
    Input:
    - nombre_fichero: bombre del fichero path relativo a dónde se ejecute o absoluto
    La estructura de los datos debe ser:
        - Cada fila un vector de características con su etiqueta en la última columna.
    
    Outputs: x,y
    x: matriz de características
    y: vector fila de las etiquetas

    '''
    
    #función de la biblioteca de pandas para leer datos. El parámetro sep es el 
    #delimitador que utilizamos y header son las filas que se utilizan para los nombres
    # de las variables, en este caso ninguna
    data = pd.read_csv(nombre_fichero,
                       sep = ',',
                       header = cabecera)
    values = data.values
    
    # Nos quedamos con todas las columnas salvo la última (la de las etiquetas)
    x = values [:,:-1]
    y = values [:, -1] # guardamos las etiquetas

    return x,y


### Validación cruzada
def Evaluacion( modelos, x, y, x_test, y_test, k_folds, nombre_modelo):
    '''
    Función para automatizar el proceso de experimento: 
    1. Ajustar modelo.
    2. Aplicar validación cruzada.
    3. Medir tiempo empleado en ajuste y validación cruzada.
    4. Medir Error cuadrático medio.   
    INPUT:
    - modelo: Modelo con el que buscar el clasificador
    - X datos entrenamiento. 
    - Y etiquetas de los datos de entrenamiento
    - x_test, y_test
    - k-folds: número de particiones para la validación cruzada
    OUTPUT:
    '''

    ###### constantes a ajustar
    numero_trabajos_paralelos_en_validacion_cruzada = 2 
    ##########################
    
    print('\n','-'*60)
    print (f' Evaluando {nombre_modelo}')
    print('-'*60)


    print('\n------ Comienza Validación Cruzada------\n')        

    #validación cruzada
    np.random.seed(0)
    tiempo_inicio_validacion_cruzada = time.time()

    best_score = 10000
    for model in modelos:
        print(model)
        score = -np.mean(cross_val_score(model, x, y, cv = k_folds, scoring="neg_mean_squared_error",n_jobs=-1))
        print('Error cuadrático medio del modelo con cv: ',score)
        #plot_confusion_matrix(model, x_train_reduced, y_train_unidime)
        if best_score > score:
            best_score = score
            best_model = model

    tiempo_fin_validacion_cruzada = time.time()
    tiempo_validacion_cruzada = (tiempo_fin_validacion_cruzada
                                 - tiempo_inicio_validacion_cruzada)

    print(f'\nTiempo empleado para validación cruzada: {tiempo_validacion_cruzada}s\n')
    
    print('\n\nEl mejor modelo es: ', best_model)
    print('E_val calculado en cross-validation: ', best_score)

    # Error cuadrático medio
    # predecimos test acorde al modelo
    best_model.fit(x, y)
    prediccion = best_model.predict(x)
    prediccion_test = best_model.predict(x_test)

    Etest=mean_squared_error(y_test, prediccion_test)
    Ein=mean_squared_error(y, prediccion)
    print("Error cuadratico medio en entrenamiento: ",Ein)
    print("Error cuadratico medio en test: ",Etest)

    return best_model
                

def VisualizaDatos(x):
    '''
    Input:
    - x: matriz de características a visualizar (Debe estar en 2 dimensiones)
    '''
    X=TSNE(n_components=2).fit_transform(x)
    plt.scatter(X[:, 0], X[:, 1],  c = 'blue', marker='o') 
    plt.title('Visualización de datos de entrenamiento por medio de TSNE')
    plt.legend()
    plt.show()
    
    
    
def VisualizarEstandarizacion(x):
    '''
    Input:
    - x: matriz de características a estandarizar
    '''
    print('\nMatriz de características sin estandarizar: \n')
    print(x)
    print('\nMedia: ', x.mean())
    print('\nVarianza: ', x.var())
    scaler=StandardScaler()
    x=scaler.fit_transform(x)
    
    print ('\nMatriz de características estandarizada: \n')
    print(x)
    print('\nMedia: ', x.mean())
    print('\nVarianza: ', x.var())
def VisualizarMatrizCorrelacion(matriz_correlacion):
    '''
    Input:
    - matriz_correlacion: matriz de correlaciones
    '''
    sn.heatmap(matriz_correlacion)
    plt.title('Mapa de calor de la matriz de correlaccion')
    plt.show()
   
def VisualizarElementosCorrelados(matriz_correlacion): 
    '''
    Input:
    - matriz_correlacion: matriz de correlaciones
    '''
    matriz_correlacion = matriz_correlacion[matriz_correlacion > 0.9]
    plt.figure(figsize=(12,8))
    sn.heatmap(matriz_correlacion)
    plt.title('Mapa de calor de elementos con mayor coeficiente de Pearson')
    plt.show()
    
# Las siguientes funciones están tomadas del siguiente enlace: https://stackoverflow.com/questions/17778394/list-highest-correlation-pairs-from-a-large-correlation-matrix-in-pandas
def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(matriz_corr, df):
    au_corr = matriz_corr.unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    strong_pairs = au_corr[abs(au_corr) > 0.9]
    return strong_pairs    
    
    
################################################################
######################   Partición  ############################

print('\n------------Leemos los datos ---------------\n')
x,y=LeerDatos(NOMBRE_FICHERO_REGRESION, 0)
x_entrenamiento, x_test, y_entrenamiento, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)

input("\n--- Pulsar tecla para continuar ---\n")

################################################################
###################### Visualización ###########################

#Descomentar si se quiere visualizar
#VisualizaDatos(x_entrenamiento)
print('Visualización del proceso de Estandarizar los datos')
VisualizarEstandarizacion(x_entrenamiento)
input("\n--- Pulsar tecla para continuar ---\n")

#################################################################
###################### Modelos a usar ###########################
k_folds=10 #Número de particiones para cross-Validation


print('\nPrimer Modelo: Regresión Lineal con SGD para obtener vector de pesos\n')
#Primer Modelo: Regresión Lineal con SGD para obtener vector de pesos
#Hago un vector con modelos del mismo tipo pero variando los parámetros
modelos1=[Pipeline([('scaler', StandardScaler()),('sgdregressor',SGDRegressor(loss='squared_loss', penalty=pen, alpha=a, learning_rate = lr, eta0 = 0.01, max_iter=5000) )]) for a in [0.0001,0.001] for pen in ['l1', 'l2'] for lr in ['optimal', 'adaptive'] ]

#Usando cross-Validation tomo el modelo con los parámetros que mejor comportamiento tiene
modelo_elegido1=Evaluacion( modelos1, x_entrenamiento, y_entrenamiento, x_test, y_test, k_folds, 'Regresion Lineal usando SGD')
#Estimacion del Eout con validación cruzada y k grande
modelos1=[Pipeline([('scaler', StandardScaler()),('sgdregressor',SGDRegressor(loss='squared_loss', penalty=pen, alpha=a, learning_rate = 'adaptive', eta0 = 0.01, max_iter=5000) )]) for a in [0.0001,0.001] for pen in ['l1', 'l2']]
Evaluacion( modelos1, x, y, x_test, y_test, 450, 'Regresion Lineal usando SGD')


input("\n--- Pulsar tecla para continuar ---\n")
print('\nSegundo Modelo: Regresión lineal con SVM\n')
#Segundo Modelo: Regresión Lineal con SVM
#Hago un vector con modelos del mismo tipo pero variando los parámetros
modelos2=[Pipeline([('scaler', StandardScaler()),('SVR',LinearSVR(epsilon=e, random_state=0, max_iter=10000))]) for e in [1, 1.5, 2, 2.5, 3, 3.5]]

#Usando cross-Validation tomo el modelo con los parámetros que mejor comportamiento tiene
modelo_elegido2=Evaluacion( modelos2, x_entrenamiento, y_entrenamiento, x_test, y_test, k_folds, 'SVM aplicado a Regresión')

#Finalmente de entre los dos modelos elegidos previamente tomo aquel con un mejor comportamiento
modelos=[modelo_elegido1,modelo_elegido2]
modelo_final= Evaluacion(modelos, x_entrenamiento, y_entrenamiento, x_test, y_test, k_folds, 'Elección entre SVM o Regresion Lineal')

input("\n--- Pulsar tecla para continuar ---\n")
print('\nTercer Modelo: Regresión lineal con SGD para obtener vector de pesos y reducción de dimensionalidad con matriz de correlación\n')
#Vamos a probar con un último modelo
#Tercer Modelo: Regresión lineal con SGD pero con reducción de atributos usando la matriz de correlación
#Convertimos los datos de entrenamiento en un Data Frame de  Pandas por comodidad
df_entrenamiento=pd.DataFrame(x_entrenamiento)
df_test=pd.DataFrame(x_test)
#Con esta función generamos la matriz de correlaciones
matriz=df_entrenamiento.corr()

input("\n--- Pulsar tecla para continuar ---\n")
#Visualizamos la matriz completa en un mapa de calor
VisualizarMatrizCorrelacion(matriz)
input("\n--- Pulsar tecla para continuar ---\n")

print("\n------Matriz de Correlación------\n")
#Imprimimos la matriz de correlación
print(matriz)
input("\n--- Pulsar tecla para continuar ---\n")

#Visualizamos en un mapa de calor aquellas parejas con coeficiente de pearson mayor de 0.95 y que son susceptibles de eliminarse
VisualizarElementosCorrelados(matriz)
input("\n--- Pulsar tecla para continuar ---\n")

print("\n------Parejas con coeficiente de correlación de Pearson mayor que 0.9------\n")
correlaciones = get_top_abs_correlations(matriz, df_entrenamiento)
print(correlaciones)
input("\n--- Pulsar tecla para continuar ---\n")

#Eliminamos las características señaladas del data frame y generamos los nuevos conjuntos de entrenamiento y test
df_entrenamiento.drop([0,2,5,6,7,11,12,15,17,22,26,25,27,33,37,47,52,57,67,69,71,72,77],axis=1)
df_test.drop([0,2,5,6,7,11,12,15,17,22,26,25,27,33,37,47,52,57,67,69,71,72,77],axis=1)
x_entrenamiento_reducido = np.delete(x_entrenamiento, [0,2,5,6,7,11,12,15,17,22,26,25,27,33,37,47,52,57,67,69,71,72,77],axis=1)
x_test_reducido= np.delete(x_test,[0,2,5,6,7,11,12,15,17,22,26,25,27,33,37,47,52,57,67,69,71,72,77],axis=1)

input("\n--- Pulsar tecla para continuar ---\n")
#Hago un vector con modelos del mismo tipo pero variando los parámetros
modelos3=[Pipeline([('scaler', StandardScaler()),('sgdregressor',SGDRegressor(loss=algoritmo, penalty=pen, alpha=a, learning_rate = lr, eta0 = 0.01, max_iter=5000) )]) for a in [0.001,0.01] for algoritmo in ['squared_loss', 'epsilon_insensitive'] for pen in ['l1', 'l2'] for lr in ['optimal', 'adaptive'] ]
k_folds=10

#Usando cross-Validation tomo el modelo con los parámetros que mejor comportamiento tiene
modelo=Evaluacion( modelos3, x_entrenamiento_reducido, y_entrenamiento, x_test_reducido, y_test, k_folds, 'Regresion Lineal usando SGD')