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

    
    print('\n------ Ajustando modelos------\n')        
    tiempo_inicio_ajuste = time.time()
    
    #ajustamos modelo 
    for modelo in modelos:
        modelo.fit(x,y) 
    tiempo_fin_ajuste = time.time()

    tiempo_ajuste = tiempo_fin_ajuste - tiempo_inicio_ajuste
    print(f'Tiempo empleado para el ajuste: {tiempo_ajuste}s')

    #validación cruzada
    np.random.seed(0)
    tiempo_inicio_validacion_cruzada = time.time()
    '''
    resultado_validacion_cruzada = cross_val_score(
        modelo,
        x, y,
        scoring = 'neg_mean_squared_error',
        cv = k_folds,
        n_jobs = numero_trabajos_paralelos_en_validacion_cruzada
    )
    '''
    best_score = 0
    for model in modelos:
        print(model)
        score = np.mean(cross_val_score(model, x, y, cv = 5, scoring="accuracy",n_jobs=-1))
        print(score)
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

################################################################
######################   Partición  ############################

x,y=LeerDatos(NOMBRE_FICHERO_CLASIFICACION)

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

################################################################
###################### Visualización ###########################


#DESCOMENTAR PARA VER RESULTADO DE TSNE
'''
X_visualizar=TSNE(n_components=2).fit_transform(x_entrenamiento)
colores=["blue","red","darkgreen","purple","yellow","orange","black","brown","pink","grey","lightgreen"]

for i in range(11):
    y0=np.where(y_entrenamiento==i+1)
    x_auxiliar=np.array(X_visualizar[y0[0]])
    plt.scatter(x_auxiliar[:, 0], x_auxiliar[:, 1],  c = colores[i],marker='+',label = i+1) #Dibujamos los puntos con etiqueta 1

plt.title('Visualización de datos de entrenamiento por medio de TSNE')
plt.legend()
plt.show()
'''
    

################################################################
################ PREPROCESAMIENTO DE DATOS #####################
modelos1=[Pipeline([('scaler', StandardScaler()),('pca',PCA(n_components=2)),('poly',PolynomialFeatures(2,include_bias='False')),('Regresion Logística',SGDClassifier(loss='hinge', penalty='l2', alpha=0.01, max_iter=10000))])]             #LogisticRegression(penalty='l2',dual=False,C=c ,class_weight=we,max_iter=1000, multi_class='ovr') )]) for we in ['balanced', 'none'] for c in [0.2,0.4,0.6,0.8,1] ]

k_folds=10

modelo_elegido1=Evaluacion( modelos1, x_entrenamiento, y_entrenamiento, x_test, y_test, k_folds, 'Regresion Logística')






'''
print (x_entrenamiento)
scaler = StandardScaler().fit(x_entrenamiento)
x_entrenamiento=scaler.transform(x_entrenamiento)

print ('\n\n', x_entrenamiento)
print (x_entrenamiento.shape)

#PCA
pca=PCA(n_components=2)
pca.fit(x_entrenamiento)
x_entrenamiento=pca.transform(x_entrenamiento)

poly=PolynomialFeatures(2,include_bias='False')
poly.fit(x_entrenamiento)
x_entrenamiento=poly.transform(x_entrenamiento)
#print(x)
colores=["blue","red","green","purple","yellow","orange","black","brown","pink","grey","magenta"]

#plt.scatter(x[:,0], x[:,1])

for i in range(11):
    print(i)
    y0=np.where(y==i+1)
    x_auxiliar=np.array(x[y0[0]])
    plt.scatter(x_auxiliar[:, 0], x_auxiliar[:, 1],  c = colores[i], label = i+1) #Dibujamos los puntos con etiqueta 1

plt.ylim(-0.00075,0.00075)
plt.xlim(-0.0001, 0.0001)

regresionLogistica=LogisticRegression(penalty='l2',class_weight='balanced',max_iter=500, multi_class='ovr')
regresionLogistica.fit(x_entrenamiento, y_entrenamiento)

x_test=scaler.transform(x_test)
x_test=pca.transform(x_test)
x_test=poly.transform(x_test)

print(regresionLogistica.score(x_test,y_test))
'''











