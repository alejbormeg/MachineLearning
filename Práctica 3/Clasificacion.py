#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Creado Viernes Mayo 14 20:04:08 2021

@author: Alejandro Borrego Megías
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
import time

np.random.seed(1)

######### CONSTANTES #########  
NOMBRE_FICHERO_CLASIFICACION = './datos/Sensorless_drive_diagnosis.txt'

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
                       sep = ' ',
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

    best_score = 0
    for model in modelos:
        print(model)
        score = np.mean(cross_val_score(model, x, y, cv = k_folds, scoring="accuracy",n_jobs=-1))
        print('\nPrecisión usando validación cruzada: ',score)
        print('\n')
        #plot_confusion_matrix(model, x_train_reduced, y_train_unidime)
        if best_score < score:
            best_score = score
            best_model = model

    tiempo_fin_validacion_cruzada = time.time()
    tiempo_validacion_cruzada = (tiempo_fin_validacion_cruzada
                                 - tiempo_inicio_validacion_cruzada)

    print(f'Tiempo empleado para validación cruzada: {tiempo_validacion_cruzada}s')
    
    print('\n\nEl mejor modelo es: ', best_model)
    print('E_in calculado en cross-validation: ', best_score)

    # Precisión
    # predecimos test acorde al modelo
    best_model.fit(x, y)
    Etest = best_model.score(x_test,y_test)
    print("Error cuadratico medio en test: ",Etest)

    return best_model

def VisualizaDatos(x):
    '''
    Input:
    - x: matriz de características a visualizar (Debe estar en 2 dimensiones)
    '''
    X_visualizar=TSNE(n_components=2).fit_transform(x)
    colores=["blue","red","darkgreen","purple","yellow","orange","black","brown","pink","grey","lightgreen"]

    for i in range(11):
        y0=np.where(y_entrenamiento==i+1)
        x_auxiliar=np.array(X_visualizar[y0[0]])
        plt.scatter(x_auxiliar[:, 0], x_auxiliar[:, 1],  c = colores[i],marker='+',label = i+1) #Dibujamos los puntos con etiqueta 1
    
    plt.title('Visualización de datos de entrenamiento por medio de TSNE')
    plt.legend()
    plt.show()
    
################################################################
######################   Partición  ############################
print('\n------------Leemos los datos ---------------\n')
x,y=LeerDatos(NOMBRE_FICHERO_CLASIFICACION)

input("\n--- Pulsar tecla para continuar ---\n")

print('Vemos si están balanceados\n')
etiquetas=np.arange(1,12) 
elementos_por_etiqueta=[]

for i in range(11):
    n=np.where(y==i+1)
    elementos_por_etiqueta.append(len(n[0]))

plt.bar(etiquetas,elementos_por_etiqueta, color='lightgreen', align='center')
plt.title ('Elementos por clase en el dataset')
plt.show()

x_entrenamiento, x_test, y_entrenamiento, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1, stratify=y)

etiquetas=np.arange(1,12) 
elementos_por_etiqueta=[]

for i in range(11):
    n=np.where(y_entrenamiento==i+1)
    elementos_por_etiqueta.append(len(n[0]))

plt.bar(etiquetas,elementos_por_etiqueta, color='lightblue', align='center')
plt.title ('Elementos por clase en entrenamiento')
plt.show()


input("\n--- Pulsar tecla para continuar ---\n")


################################################################
###################### Visualización ###########################
#DESCOMENTAR PARA VER RESULTADO DE TSNE

#VisualizaDatos(x_entrenamiento)    

################################################################
################ PREPROCESAMIENTO DE DATOS y MODELOS #####################
k_folds=5

#Primer modelo
print('\nPrimer Modelo: Regresión Logística con SGD para obtener vector de pesos aplicado a clasificación multiclase\n')
input("\n--- Pulsar tecla para continuar ---\n")

#El mejor modelo es:  Pipeline(steps=[('scaler', StandardScaler()),
#                ('Regresion Logística',
#                 SGDClassifier(alpha=0.001, average=True, eta0=0.0001,
#                               max_iter=10000, random_state=1))])
modelos1=[Pipeline([('scaler', StandardScaler()),('Regresion Logística',SGDClassifier(loss='hinge', penalty=pen, alpha=a,max_iter=10000,random_state=1,learning_rate=lr,eta0=0.001, early_stopping=early,average=av))]) for pen in ['l1','l2'] for a in [0.001,0.01] for lr in ['optimal', 'adaptive'] for early in [True, False] for av in [True, False]]          
modelo_elegido1=Evaluacion( modelos1, x_entrenamiento, y_entrenamiento, x_test, y_test, k_folds, 'Regresion Logística')

#Segundo modelo
print('\Segundo Modelo: SVM aplicado a clasificación multiclase\n')
input("\n--- Pulsar tecla para continuar ---\n")
modelos2=[Pipeline([('scaler', StandardScaler()),('SVM',SVC(C=1.5, kernel='linear', shrinking=True ,max_iter=-1,decision_function_shape='ovr',break_ties=True,random_state=1))])]            
modelo_elegido1=Evaluacion( modelos2, x_entrenamiento, y_entrenamiento, x_test, y_test, k_folds, 'Regresion Logística')

#Tercer Modelo
print('\Tercer Modelo: SVM aplicado a clasificación multiclase con reducción de dimensionalidad y características cuadráticas\n')
input("\n--- Pulsar tecla para continuar ---\n")
modelos1=[Pipeline([('scaler', StandardScaler()),('pca',PCA(n_components=2)),('poly',PolynomialFeatures(2,include_bias='False')),('Regresion Logística',SGDClassifier(loss='hinge', penalty=pen, alpha=a,max_iter=10000,epsilon=e,random_state=1,learning_rate=lr,eta0=et0, early_stopping=early,average=av))]) for pen in ['l1','l2'] for a in [0.001,0.01,0.1] for e in [0.1,0.3,0.5] for lr in ['optimal', 'adaptive'] for early in [True, False] for av in [True, False] for et0 in [0.0001,0.001] ]          
modelo_elegido1=Evaluacion( modelos1, x_entrenamiento, y_entrenamiento, x_test, y_test, k_folds, 'Regresion Logística')


modelos2=[Pipeline([('scaler', StandardScaler()),('pca',PCA(n_components=2)),('poly',PolynomialFeatures(2,include_bias='False')),('SVM',SVC(C=c, kernel='linear', shrinking=True ,max_iter=-1,decision_function_shape='ovr',break_ties=b,random_state=1))]) for c in [1, 1.5] for b in [True,False]]            
modelo_elegido2=Evaluacion( modelos2, x_entrenamiento, y_entrenamiento, x_test, y_test, k_folds, 'Regresion Logística')
















