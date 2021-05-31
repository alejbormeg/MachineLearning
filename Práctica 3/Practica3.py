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


######### CONSTANTES #########  
NOMBRE_FICHERO_CLASIFICACION = './datos/Sensorless_drive_diagnosis.txt'
SEPARADOR_CLASIFICACION= ' '

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
    
    datos=pd.read_csv(nombre_fichero,sep=separador,header=None)
    valores=datos.values
    
    #Los datos son todas las filas de todas las columnas salvo la última
    x=valores [:: -1]
    y=valores [:, -1] #el vector de etiquetas es la última columna
    
    return x,y

x,y=LeerDatos(NOMBRE_FICHERO_CLASIFICACION, SEPARADOR_CLASIFICACION)
#print(x)
#print(y)

etiquetas=np.arange(1,12) 
elementos_por_etiqueta=[]

for i in range(11):
    n=np.where(y==i+1)
    elementos_por_etiqueta.append(len(n[0]))

plt.bar(etiquetas,elementos_por_etiqueta, color='green', align='center')
plt.title ('Elementos por clase')
plt.show()
    
################################################################
################ PREPROCESAMIENTO DE DATOS #####################
scaler = StandardScaler().fit(x)
x=scaler.transform(x)
#PCA




pca=PCA(n_components=2)
pca.fit(x)
x=pca.transform(x)
#print(x)
colores=["blue","red","green","purple","yellow","orange","black","brown","pink","grey","magenta"]

#plt.scatter(x[:,0], x[:,1])

for i in range(11):
    print(i)
    y0=np.where(y==i+1)
    x_auxiliar=np.array(x[y0[0]])
    plt.scatter(x_auxiliar[:, 0], x_auxiliar[:, 1],  c = colores[i], label = i+1) #Dibujamos los puntos con etiqueta 1

plt.ylim(-10,10)
plt.xlim(-10, 20)


'''
X=TSNE(n_components=2,perplexity=50, learning_rate=10).fit_transform(x)
colores=["blue","red","green","purple","yellow","orange","black","brown","pink","grey","magenta"]

for i in range(11):
    y0=np.where(y==i+1)
    x_auxiliar=np.array(X[y0[0]])
    plt.scatter(x_auxiliar[:, 0], x_auxiliar[:, 1],  c = colores[i], label = i+1) #Dibujamos los puntos con etiqueta 1

plt.ylim(-2,2)
plt.xlim(-2,2)
'''











