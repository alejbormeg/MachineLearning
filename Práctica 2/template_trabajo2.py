# -*- coding: utf-8 -*-
"""
TRABAJO 2
Nombre Estudiante: Alejandro Borrego Megías
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Fijamos la semilla
np.random.seed(1)


def simula_unif(N, dim, rango):
	return np.random.uniform(rango[0],rango[1],(N,dim))

def simula_gaus(N, dim, sigma):
    media = 0    
    out = np.zeros((N,dim),np.float64)        
    for i in range(N):
        # Para cada columna dim se emplea un sigma determinado. Es decir, para 
        # la primera columna (eje X) se usará una N(0,sqrt(sigma[0])) 
        # y para la segunda (eje Y) N(0,sqrt(sigma[1]))
        out[i,:] = np.random.normal(loc=media, scale=np.sqrt(sigma), size=dim)
    
    return out


def simula_recta(intervalo):
    points = np.random.uniform(intervalo[0], intervalo[1], size=(2, 2))
    x1 = points[0,0]
    x2 = points[1,0]
    y1 = points[0,1]
    y2 = points[1,1]
    # y = a*x + b
    a = (y2-y1)/(x2-x1) # Calculo de la pendiente.
    b = y1 - a*x1       # Calculo del termino independiente.
    
    return a, b


# EJERCICIO 1.1: Dibujar una gráfica con la nube de puntos de salida correspondiente

x = simula_unif(50, 2, [-50,50])
#CODIGO DEL ESTUDIANTE
# Dibujo el Scatter Plot de los puntos generados
plt.scatter(x[:,0],x[:,1], c='blue')
plt.title('Ejercicio1.1 Puntos generados con simula uniformemente')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

x = simula_gaus(50, 2, np.array([5,7]))
#CODIGO DEL ESTUDIANTE
# Dibujo el Scatter Plot de los puntos generados
plt.scatter(x[:,0],x[:,1], c='red')
plt.title('Ejercicio1.1 Puntos generados con Distribución Gaussiana')
plt.xlabel('x1')
plt.ylabel('x2')

plt.show()

input("\n--- Pulsar tecla para continuar ---\n")


###############################################################################
###############################################################################
###############################################################################


# EJERCICIO 1.2: Dibujar una gráfica con la nube de puntos de salida correspondiente

# La funcion np.sign(0) da 0, lo que nos puede dar problemas
def signo(x):
	if x >= 0:
		return 1
	return -1

def f(x, y, a, b):
	return signo(y - a*x - b)

#CODIGO DEL ESTUDIANTE
#1.2 a) 
x = simula_unif(100, 2, [-50,50])
# Dibujo el Scatter Plot de los puntos generados
plt.scatter(x[:,0],x[:,1], c='blue')
plt.title('Ejercicio1.1 Puntos generados con simula uniformemente')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

#Calculamos los coeficientes de la recta
a,b=simula_recta([-50,50])

#Generamos las etiquetas
y=[]

for i in x :
    y.append(f(i[0],i[1],a,b))

y=np.array(y)
y0 = np.where(y == -1) #capturo los índices de los elementos con -1
y1 = np.where(y == 1) #capturo los índices de los elementos con 1
#x_2 contiene dos arrays, uno en cada componente, el primero tiene los valores de x con etiqueta -1 y la segunda los de etiqueta 1
x_2 = np.array([x[y0[0]],x[y1[0]]])
plt.scatter(x_2[0][:, 0], x_2[0][:, 1],  c = 'blue', label = '-1') #Dibujamos los puntos con etiqueta 1
plt.scatter(x_2[1][:, 0], x_2[1][:, 1],  c = 'orange', label = '1')#Dibujamos los de etiqueta -1

#Calculamos las imagenes de los puntos (sin aplicar la función signo) y así dibujar la recta de regresión
imagenes=[]

for i in x :
    imagenes.append(a*i[0]+b)
    
plt.plot( x[:,0], imagenes, c = 'red', label='Recta') #Para representarlo, despejo x2 de la ecuación y represento la función resultante en 2D
plt.legend();
plt.title("Ejercicio 1.2 a)")
plt.xlabel('x1')
plt.ylabel('x2')
plt.figure()
plt.show()
input("\n--- Pulsar tecla para continuar ---\n")


# 1.2.b. Dibujar una gráfica donde los puntos muestren el resultado de su etiqueta, junto con la recta usada para ello
### Como en el apartado siguiente vamos a calcular la accuracy del clasificador (TP+TN)/(P+N) capturo los TP y TN de este método ya que (P+N)=100
TN=len(y0[0]) #Numero de etiquetas con -1, pues antes de meter ruido la recta clasifica perfectamente
TP=len(y1[0]) #Numero de etiquetas con +1, pues antes de meter ruido la recta clasifica perfectamente

# Array con 10% de indices aleatorios para introducir ruido
y0=pd.DataFrame(data=y0[0]); #Convierto la matriz X en un Dataframe de Pandas, que es más cómodo de usar 
y0=y0.sample(frac=0.10,random_state=1); #Hacemos que tome un 10% de los datos de forma aleatoria
y0=y0.to_numpy()
for i in y0:
    y[i]=1
    
TN=TN-len(y0); #Como hemos etiquetado "mal" el 10% de los elementos pues actualizamos los TN

y1=pd.DataFrame(data=y1[0]); #Convierto la matriz X en un Dataframe de Pandas, que es más cómodo de usar 
y1=y1.sample(frac=0.10,random_state=1); #Hacemos que tome un 10% de los datos de forma aleatoria
y1=y1.to_numpy()
for i in y1:
    y[i]=-1

TP=TP-len(y1); #Como hemos etiquetado "mal" el 10% de los elementos pues actualizamos los TN

y0 = np.where(y == -1) #capturo los índices de los elementos con -1
y1 = np.where(y == 1) #capturo los índices de los elementos con 1
#x_2 contiene dos arrays, uno en cada componente, el primero tiene los valores de x con etiqueta -1 y la segunda los de etiqueta 1
x_2 = np.array([x[y0[0]],x[y1[0]]])
plt.scatter(x_2[0][:, 0], x_2[0][:, 1],  c = 'blue', label = '-1') #Dibujamos los puntos con etiqueta 1
plt.scatter(x_2[1][:, 0], x_2[1][:, 1],  c = 'orange', label = '1')#Dibujamos los de etiqueta -1

#Calculamos las imagenes de los puntos (sin aplicar la función signo) y así dibujar la recta de regresión
imagenes=[]

for i in x :
    imagenes.append(a*i[0]+b)
    
plt.plot( x[:,0], imagenes, c = 'red', label='Recta') #Para representarlo, despejo x2 de la ecuación y represento la función resultante en 2D
plt.legend();
plt.title("Ejercicio 1.2 b)")
plt.xlabel('x1')
plt.ylabel('x2')
plt.figure()
plt.show()


input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 1.3: Supongamos ahora que las siguientes funciones definen la frontera de clasificación de los puntos de la muestra en lugar de una recta

def plot_datos_cuad(X, y, fz, title='Point cloud plot', xaxis='x axis', yaxis='y axis'):
    #Preparar datos
    min_xy = X.min(axis=0)
    max_xy = X.max(axis=0)
    border_xy = (max_xy-min_xy)*0.01
    
    #Generar grid de predicciones
    xx, yy = np.mgrid[min_xy[0]-border_xy[0]:max_xy[0]+border_xy[0]+0.001:border_xy[0], 
                      min_xy[1]-border_xy[1]:max_xy[1]+border_xy[1]+0.001:border_xy[1]]
    grid = np.c_[xx.ravel(), yy.ravel(), np.ones_like(xx).ravel()]
    pred_y = fz(grid)
    # pred_y[(pred_y>-1) & (pred_y<1)]
    pred_y = np.clip(pred_y, -1, 1).reshape(xx.shape)
    
    #Plot
    f, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(xx, yy, pred_y, 50, cmap='RdBu',vmin=-1, vmax=1)
    ax_c = f.colorbar(contour)
    ax_c.set_label('$f(x, y)$')
    ax_c.set_ticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
    ax.scatter(X[:, 0], X[:, 1], c=y, s=50, linewidth=2, 
                cmap="RdYlBu", edgecolor='white')
    
    XX, YY = np.meshgrid(np.linspace(round(min(min_xy)), round(max(max_xy)),X.shape[0]),np.linspace(round(min(min_xy)), round(max(max_xy)),X.shape[0]))
    positions = np.vstack([XX.ravel(), YY.ravel()])
    ax.contour(XX,YY,fz(positions.T).reshape(X.shape[0],X.shape[0]),[0], colors='black')
    
    ax.set(
       xlim=(min_xy[0]-border_xy[0], max_xy[0]+border_xy[0]), 
       ylim=(min_xy[1]-border_xy[1], max_xy[1]+border_xy[1]),
       xlabel=xaxis, ylabel=yaxis)
    plt.title(title)
    plt.show()
    
    
#CODIGO DEL ESTUDIANTE

print("Mostramos la precisión del método del apartado anterior")
print ("ACCURACY= ", (TN+TP)/100.0)

###### Definimos las nuevas funciones
def f1(x):
    y=[]
    for i in x:
        y.append((i[0]-10)**2 + (i[1]-20)**2-400)
    
    return np.asarray(y)

plot_datos_cuad(x,y,f1,'Frontera de decision con función 1', 'x1', 'x2')

#Comparamos a continuación la precisión de este nuevo clasificador con la accuracy de la recta
imagenes=f1(x) #Capturo las imágenes de cada punto
TN=0
TP=0
cont=0

for i in imagenes:
    
    if i>0 and y[cont]>0: #Si tienen la misma etiqueta (+1 en este caso)
        TP+=1
    if i<0 and y[cont]<0: # (-1 en este caso)
        TN+=1
    cont+=1
    
print("Mostramos la accuracy del método f1")
print ("ACCURACY= ", (TN+TP)/100.0)

input("\n--- Pulsar tecla para continuar ---\n")

def f2(x):
    y=[]
    for i in x:
        y.append(0.5*(i[0]-10)**2 + (i[1]-20)**2-400)
    
    return np.asarray(y)

plot_datos_cuad(x,y,f2,'Frontera de decision con función 2', 'x1', 'x2')

#Comparamos a continuación la precisión de este nuevo clasificador con la accuracy de la recta
imagenes=f2(x) #Capturo las imágenes de cada punto
TN=0
TP=0
cont=0

for i in imagenes:
    
    if i>0 and y[cont]>0: #Si tienen la misma etiqueta (+1 en este caso)
        TP+=1
    if i<0 and y[cont]<0: # (-1 en este caso)
        TN+=1
    cont+=1
    
print("Mostramos la accuracy del método f2")
print ("ACCURACY= ", (TN+TP)/100.0)

input("\n--- Pulsar tecla para continuar ---\n")

def f3(x):
    y=[]
    for i in x:
        y.append(0.5*(i[0]-10)**2 - (i[1]+20)**2 - 400)
    
    return np.asarray(y)

plot_datos_cuad(x,y,f3,'Frontera de decision con función 3', 'x1', 'x2')

#Comparamos a continuación la precisión de este nuevo clasificador con la accuracy de la recta
imagenes=f3(x) #Capturo las imágenes de cada punto
TN=0
TP=0
cont=0

for i in imagenes:
    
    if i>0 and y[cont]>0: #Si tienen la misma etiqueta (+1 en este caso)
        TP+=1
    if i<0 and y[cont]<0: # (-1 en este caso)
        TN+=1
    cont+=1
    
print("Mostramos la accuracy del método f3")
print ("ACCURACY= ", (TN+TP)/100.0)
input("\n--- Pulsar tecla para continuar ---\n")

def f4(x):
    y=[]
    for i in x:
        y.append(i[1]-20*i[0]**2-5*i[0]+3)
    
    return np.asarray(y)

plot_datos_cuad(x,y,f4,'Frontera de decision con función 4', 'x1', 'x2')
#Comparamos a continuación la precisión de este nuevo clasificador con la accuracy de la recta
imagenes=f4(x) #Capturo las imágenes de cada punto
TN=0
TP=0
cont=0

for i in imagenes:
    
    if i>0 and y[cont]>0: #Si tienen la misma etiqueta (+1 en este caso)
        TP+=1
    if i<0 and y[cont]<0: # (-1 en este caso)
        TN+=1
    cont+=1
    
print("Mostramos la accuracy del método f4")
print ("ACCURACY= ", (TN+TP)/100.0)
input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################
###############################################################################
###############################################################################
'''
# EJERCICIO 2.1: ALGORITMO PERCEPTRON

def ajusta_PLA(datos, label, max_iter, vini):
    #CODIGO DEL ESTUDIANTE
    
    return ?  

#CODIGO DEL ESTUDIANTE

# Random initializations
iterations = []
for i in range(0,10):
    #CODIGO DEL ESTUDIANTE
    
print('Valor medio de iteraciones necesario para converger: {}'.format(np.mean(np.asarray(iterations))))

input("\n--- Pulsar tecla para continuar ---\n")

# Ahora con los datos del ejercicio 1.2.b

#CODIGO DEL ESTUDIANTE


input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 3: REGRESIÓN LOGÍSTICA CON STOCHASTIC GRADIENT DESCENT

def sgdRL(?):
    #CODIGO DEL ESTUDIANTE

    return w



#CODIGO DEL ESTUDIANTE

input("\n--- Pulsar tecla para continuar ---\n")
    


# Usar la muestra de datos etiquetada para encontrar nuestra solución g y estimar Eout
# usando para ello un número suficientemente grande de nuevas muestras (>999).


#CODIGO DEL ESTUDIANTE


input("\n--- Pulsar tecla para continuar ---\n")


###############################################################################
###############################################################################
###############################################################################
#BONUS: Clasificación de Dígitos


# Funcion para leer los datos
def readData(file_x, file_y, digits, labels):
	# Leemos los ficheros	
	datax = np.load(file_x)
	datay = np.load(file_y)
	y = []
	x = []	
	# Solo guardamos los datos cuya clase sea la digits[0] o la digits[1]
	for i in range(0,datay.size):
		if datay[i] == digits[0] or datay[i] == digits[1]:
			if datay[i] == digits[0]:
				y.append(labels[0])
			else:
				y.append(labels[1])
			x.append(np.array([1, datax[i][0], datax[i][1]]))
			
	x = np.array(x, np.float64)
	y = np.array(y, np.float64)
	
	return x, y

# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy', [4,8], [-1,1])
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy', [4,8], [-1,1])


#mostramos los datos
fig, ax = plt.subplots()
ax.plot(np.squeeze(x[np.where(y == -1),1]), np.squeeze(x[np.where(y == -1),2]), 'o', color='red', label='4')
ax.plot(np.squeeze(x[np.where(y == 1),1]), np.squeeze(x[np.where(y == 1),2]), 'o', color='blue', label='8')
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (TRAINING)')
ax.set_xlim((0, 1))
plt.legend()
plt.show()

fig, ax = plt.subplots()
ax.plot(np.squeeze(x_test[np.where(y_test == -1),1]), np.squeeze(x_test[np.where(y_test == -1),2]), 'o', color='red', label='4')
ax.plot(np.squeeze(x_test[np.where(y_test == 1),1]), np.squeeze(x_test[np.where(y_test == 1),2]), 'o', color='blue', label='8')
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (TEST)')
ax.set_xlim((0, 1))
plt.legend()
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

#LINEAR REGRESSION FOR CLASSIFICATION 

#CODIGO DEL ESTUDIANTE


input("\n--- Pulsar tecla para continuar ---\n")



#POCKET ALGORITHM
  
#CODIGO DEL ESTUDIANTE




input("\n--- Pulsar tecla para continuar ---\n")


#COTA SOBRE EL ERROR

#CODIGO DEL ESTUDIANTE
'''