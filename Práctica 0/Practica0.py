# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from sklearn import datasets 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Leer la base de datos Iris y mostrar algunos de los datos leidos
iris=datasets.load_iris()
X=np.array(iris.data)
Y=np.array(iris.target)+1 #sumamos 1 porque la etiqueta 0 nos da problemas para usar weights en el ejercicio 2, así cambiamos las etiquetas a 1,2,3
print (Y) #Mostramos todas las etiquetas
print (X[0:10]) #Mostramos las 10 primeras filas a modo de ejemplo
input("Pulsa una tecla para continuar")

#Nos quedamos con la columna 1 y 3 de X

X_1=X[:,::2]

print (X_1[0:3]);

input("Pulsa una tecla para continuar")
#Iremos representando los datos en la gráfica por etiquetas
idx=np.where (Y==1); #tomo los índices con etiqueta 1 en este caso
Y0=Y[idx];  #me quedo con los valores en esos índices (todos los 1 de Y)
X0=X_1[idx]; #me quedo con los datos correspondientes a los elementos con etiqueta 1

#Repito el proceso con el resto de etiquetas

idx=np.where (Y==2);
Y1=Y[idx];
X1=X_1[idx];

idx=np.where (Y==3);
Y2=Y[idx];
X2=X_1[idx];

#Hago los gráficos de puntos con cada pareja de datos especificando los colores del ejercicio
plt.scatter(X0[:,0],X0[:,1], c='orange', label='Iris Setosa')
plt.scatter(X1[:,0],X1[:,1], c='black', label='Iris Versicolor')
plt.scatter(X2[:,0],X2[:,1], c='green', label='Iris Virginica')


plt.xlabel("Longitud de Sépalo");
plt.ylabel("Longitud de Pétalo");
plt.legend();

#Los muestro todos juntos
plt.show();



################## Ejercicio 2 ##############################

print("\n\n\n---------------------- Ejercicio 2 ---------------------- \n\n\n ")
A=X.copy();
B=Y.copy(); 
B=B.reshape(-1,1);

# Concatenamos X e Y por columnas (axis=1) para tener datos y etiquetas juntos
C=np.concatenate((A,B), axis=1);
print("Mostramos como se queda la matriz una vez pegamos las etiquetas al final: "); 
print(C);

#Usaré la librería Pandas para separar los datos, consultada en este enlace https://www.analyticslane.com/2018/12/14/seleccion-de-una-submuestra-en-python-con-pandas/
D=pd.DataFrame(data=C, columns=('L. Sepalo', 'An. Sepalo', 'L. Petalo' , 'An. Petalo ', 'Especie')); #Convierto la matriz C en un Dataframe de Pandas, que es más cómodo de usar y más visual.
Training_set=D.sample(frac=0.75,random_state=1, weights='Especie'); #Hacemos que tome las muestras dependiendo de las etiquetas y cúantas haya de cada tipo por eso especificamos que weights='Especie'
print("\n Así quedaría el Training Set: \n");  
print(Training_set);
Test_set=D.drop(Training_set.index); #El test set será el resultado de eliminar al conjunto total D los elementos obtenidos con sample

print("\n Así quedaría el Test Set: \n");
print(Test_set);
input("Pulsa una tecla para continuar")


################## Ejercicio 3 ##############################

print("\n\n\n---------------------- Ejercicio 3 ---------------------- \n\n\n ")
print("\n Lista de 100 valores equiespaciados entre 0 y 4*Pi \n")
lista=np.linspace(0,4.0*np.pi,100) #Vector de 100 elementos equiespaciados entre 0 y 4*PI
print (lista)

input("Pulsa una tecla para continuar")
print("\n Seno de los valores anteriores \n")
seno=np.sin(lista)
print (seno);

print("\n Cossenoeno de los valores anteriores \n")
coseno=np.cos(lista)
print (coseno);

print("\n Tangente hiperbólica de sen + cos \n")
tanh=np.tanh(seno+coseno);
print (tanh);
input("Pulsa una tecla para continuar")

print("\n\nFinalmente dibujamos en un mismo gráfico las tres funciones \n")
fig,ax=plt.subplots(); #Utilizo subplots para especificar cómo quiero que se visualice cada función y con qué colores
ax.plot(lista, seno, 'k--',color='green' ,label='y=sen(x)')
ax.plot(lista, coseno, 'k--',color='black', label='y=cos(x)')
ax.plot(lista, tanh, 'k--',color='red', label='y=tanh(sen(x)+cos(x))')
legend=ax.legend(loc='upper right', shadow=True) #Hago la leyenda para más claridad
ax.set_ylim((-1,1.75))

legend.get_frame()
plt.grid()
plt.show() #Muestro todos los gráficos conjuntamente
























