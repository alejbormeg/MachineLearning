#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 18:13:51 2021

@author: alejandro
"""


import numpy as np
import matplotlib.pyplot as plt
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
from time import time
SEED=42

np.random.seed(1)
######### CONSTANTES #########  
NOMBRE_FICHERO_REGRESION = './datos/train.csv'
SEPARADOR_REGRESION= ','

################ funciones auxiliares

def LeerDatos (nombre_fichero, separador):
    '''
    Input:
    - nombre_fichero: bombre del fichero path relativo a dónde se ejecute o absoluto
    La estructura de los datos debe ser:
        - Cada fila un vector de características con su etiqueta en la última columna.
    
    Outputs: x,y
    x: matriz de características
    y: vector fila de las etiquetas

    '''
    #Ponemos header=0 porque en la primera línea están los nombres de las columnas. Así los capturamos
    datos=pd.read_csv(nombre_fichero,sep=separador,header=0) 
    
    valores=datos.values
    x=valores [:: -1]
    y=valores [:, -1]
    
    return x,y




print("Leemos los datos")
x, y = LeerDatos(NOMBRE_FICHERO_REGRESION,SEPARADOR_REGRESION)   

input("\n--- Pulsar tecla para continuar ---\n")

print("Separamos en test y training y vemos que los conjuntos están balanceados")


#dividimos en test y training
x_train, x_test, y_train_unidime, y_test_unidime = train_test_split(x, y, test_size = 0.2, random_state = SEED)

print("Normalizamos los datos")
#Normalizamos los datos para que tengan media 0 y varianza 1
scaler = StandardScaler()
x_train = scaler.fit_transform( x_train )
x_test = scaler.transform( x_test)


x_train_reduced = x_train #np.delete(x_train, [0,2,5,6,7,11,12,15,17,20,22,26,25,27,33,37,47,52,57,67,69,70,71,72,77],axis=1)
x_test_reduced= x_test #np.delete(x_test,[0,2,5,6,7,11,12,15,17,20,22,26,25,27,33,37,47,52,57,67,69,70,71,72,77],axis=1)


modelos = [SGDRegressor(loss=algoritmo, penalty=pen, alpha=a, learning_rate = lr, eta0 = 0.01, max_iter=5000) for a in [0.0001,0.001] for algoritmo in ['squared_loss', 'epsilon_insensitive'] for pen in ['l1', 'l2'] for lr in ['optimal', 'adaptive'] ]


start_time = time()



best_score = 0
for model in modelos:
    print(model)
    score = np.mean(cross_val_score(model, x_train_reduced, y_train_unidime, cv = 5, scoring="r2",n_jobs=-1))
    print(score)
    #plot_confusion_matrix(model, x_train_reduced, y_train_unidime)
    if best_score < score:
           best_score = score
           best_model = model
    

print("Hacemos el entrenamiento")
print(best_model)
best_model.fit(x_train_reduced, y_train_unidime)

print("Hacemos prediccion")
y_pred_logistic = best_model.predict(x_test_reduced)
y_pred_logistic_train = best_model.predict(x_train_reduced)
print("Calculamos coeficientes de determinación")

coef_regres_test = best_model.score(x_test_reduced, y_test_unidime)
coef_regres_train = best_model.score(x_train_reduced, y_train_unidime)
print("\tCoeficiente de determinación en test: ", coef_regres_test)
print("\tCoeficiente de determinación en entrenamiento: ", coef_regres_train)

#y_aleatorio = np.random.randint(0,11,len(y_test))
#numero_aciertos_aleatorio = accuracy_score(y_test,y_aleatorio)
#print("\tPorcentaje de aciertos de forma aleatoria: ", numero_aciertos_aleatorio)
#print(100*best_model.score(x_train_trans, y_train_unidime))
#print(100* best_model.score(x_test_trans, y_test_unidime))


input("\n--- Pulsar tecla para continuar ---\n")
Etest=mean_squared_error(y_test_unidime, y_pred_logistic)
print("Error cuadratico medio en test: ",Etest)
Etrain=mean_squared_error(y_train_unidime, y_pred_logistic_train)
print("Error cuadratico medio en test: ",Etest)





'''
x,y=LeerDatos(NOMBRE_FICHERO_REGRESION, SEPARADOR_CLASIFICACION)
x_entrenamiento, x_test, y_entrenamiento, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

print("Normalizamos los datos")
#Normalizamos los datos para que tengan media 0 y varianza 1
scaler = StandardScaler()
x_entrenamiento = scaler.fit_transform( x_entrenamiento )
x_test = scaler.transform( x_test)

modelos = [SGDRegressor(loss=algoritmo, penalty=pen, alpha=a, learning_rate = lr, eta0 = 0.01, max_iter=5000) for a in [0.0001,0.001] for algoritmo in ['squared_loss', 'epsilon_insensitive'] for pen in ['l1', 'l2'] for lr in ['optimal', 'adaptive'] ]


#start_time = time()



best_score = 0
for model in modelos:
    print(model)
    score = np.mean(cross_val_score(model, x_entrenamiento, y_entrenamiento, cv = 5, scoring="r2",n_jobs=-1))
    print(score)
    #plot_confusion_matrix(model, x_train_reduced, y_train_unidime)
    if best_score < score:
           best_score = score
           best_model = model
    

print("Hacemos el entrenamiento")
print(best_model)
best_model.fit(x_entrenamiento, y_entrenamiento)

print("Hacemos prediccion")
y_pred_logistic = best_model.predict(x_test)
y_pred_logistic_train = best_model.predict(x_entrenamiento)
print("Calculamos coeficientes de determinación")

coef_regres_test = best_model.score(x_test, y_test)
coef_regres_train = best_model.score(x_entrenamiento, y_entrenamiento)
print("\tCoeficiente de determinación en test: ", coef_regres_test)
print("\tCoeficiente de determinación en entrenamiento: ", coef_regres_train)

#y_aleatorio = np.random.randint(0,11,len(y_test))
#numero_aciertos_aleatorio = accuracy_score(y_test,y_aleatorio)
#print("\tPorcentaje de aciertos de forma aleatoria: ", numero_aciertos_aleatorio)
#print(100*best_model.score(x_train_trans, y_train_unidime))
#print(100* best_model.score(x_test_trans, y_test_unidime))


input("\n--- Pulsar tecla para continuar ---\n")
Etest=mean_squared_error(y_test, y_pred_logistic)
print("Error cuadratico medio en test: ",Etest)
Etrain=mean_squared_error(y_entrenamiento, y_pred_logistic_train)
print("Error cuadratico medio en test: ",Etest)
#print(100* best_model.score(x_test_reduced, y_test_unidime))
input("\n--- Pulsar tecla para continuar ---\n")
print("Matriz de confusión")
'''

























