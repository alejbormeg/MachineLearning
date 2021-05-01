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

##################################################################################################################################
####################################Funciones proporcionadas######################################################################

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


##################################################################################################################################
##################################################################################################################################
######################## EJERCICIO 1.1: Dibujar una gráfica con la nube de puntos de salida correspondiente#######################

#Generamos los puntos con Distribución uniforme
x = simula_unif(50, 2, [-50,50])
# Dibujo el Scatter Plot de los puntos generados
plt.scatter(x[:,0],x[:,1], c='blue')
plt.title('Ejercicio1.1 Puntos generados con simula uniformemente')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

#Generamos los puntos con una distribución normal
x = simula_gaus(50, 2, np.array([5,7]))
# Dibujo el Scatter Plot de los puntos generados
plt.scatter(x[:,0],x[:,1], c='red')
plt.title('Ejercicio1.1 Puntos generados con Distribución Gaussiana')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")


##################################################################################################################################
##################################################################################################################################
############################# EJERCICIO 1.2: Dibujar una gráfica con la nube de puntos de salida correspondiente##################


####################### FUNCIONES QUE VAMOS A USAR################################
# La funcion np.sign(0) da 0, lo que nos puede dar problemas
def signo(x):
	if x >= 0:
		return 1
	return -1

def f(x, y, a, b):
	return signo(y - a*x - b)

##################################################################################################################################
################################################ Ejercicio 1.2 a) ################################################################
#Generamos los puntos
x = simula_unif(100, 2, [-50,50])
# Dibujo el Scatter Plot de los puntos generados
plt.scatter(x[:,0],x[:,1], c='blue')
plt.title('Ejercicio1.1 Puntos generados con simula uniformemente')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

#Calculamos los coeficientes de la recta
a,b=simula_recta([-50,50])

print("Los coeficientes a y b: ", a, b)
print('\n\n')
#Generamos las etiquetas
y=[]

for i in x :
    y.append(f(i[0],i[1],a,b))

y=np.array(y)
labels=y.copy() ####### PARA EJERCICIO 2 #######
y0 = np.where(y == -1) #capturo los índices de los elementos con -1
y1 = np.where(y == 1) #capturo los índices de los elementos con 1
#x_2 contiene dos arrays, uno en cada componente, el primero tiene los valores de x con etiqueta -1 y la segunda los de etiqueta 1
x_2 = np.array([x[y0[0]],x[y1[0]]])
plt.scatter(x_2[0][:, 0], x_2[0][:, 1],  c = 'blue', label = '-1') #Dibujamos los puntos con etiqueta 1
plt.scatter(x_2[1][:, 0], x_2[1][:, 1],  c = 'orange', label = '1')#Dibujamos los de etiqueta -1

#Calculamos las imagenes de los puntos (sin aplicar la función signo) y así dibujar la recta 
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

##################################################################################################################################
##################################################################################################################################
####Ejercicio 1.2.b. Dibujar una gráfica donde los puntos muestren el resultado de su etiqueta, junto con la recta usada para ello


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


####DIBUJAMOS LOS DATOS
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

##################################################################################################################################
##################################################################################################################################
# EJERCICIO 1.3: Supongamos ahora que las siguientes funciones definen la frontera de clasificación de los puntos de la muestra en lugar de una recta

################### FUNCION DE AYUDA ###################
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
    
    
##################################################################################################################################


print("Mostramos la precisión del método del apartado anterior")
print ("ACCURACY= ", (TN+TP)/100.0)


##################################################################################################################################
############################################# Definimos las nuevas funciones #####################################################
def f1(x):
    y=[]
    for i in x:
        y.append((i[0]-10)**2 + (i[1]-20)**2-400)
    
    return np.asarray(y)

##USAMOS FUNCIÓN DE AYUDA PARA DIBUJAR LAS FRONTERAS DE DECISIÓN
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
        y.append(0.5*(i[0]+10)**2 + (i[1]-20)**2-400)
    
    return np.asarray(y)


##USAMOS FUNCIÓN DE AYUDA PARA DIBUJAR LAS FRONTERAS DE DECISIÓN
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


##USAMOS FUNCIÓN DE AYUDA PARA DIBUJAR LAS FRONTERAS DE DECISIÓN
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

##USAMOS FUNCIÓN DE AYUDA PARA DIBUJAR LAS FRONTERAS DE DECISIÓN
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



##################################################################################################################################
##################################################################################################################################
############################################## EJERCICIO 2.1: ALGORITMO PERCEPTRON ###############################################

#ALGORITMO PLA
def ajusta_PLA(datos, label, max_iter, vini):
    '''
    Algoritmo Perceptrón
    Parameters
    ----------
    datos : Matriz de datos
    
    label : Vector de etiquetas
    
    max_iter : número máximo de iteraciones
    
    vini : vector inicial de pesos

    Returns
    -------
    w : vector de pesos.
    
    it : número de iteraciones empleadas

    '''
    mejora=True
    it=0
    w=np.array(vini)
    w=w.reshape(-1,1) #Lo transformo en un vector columna
    
    while (mejora and it<max_iter): # 
        mejora=False
        it+=1
        for i in range(len(datos)):
            valor=datos[i,:].dot(w)
            sign=signo(valor.item())

            if sign!=label[i]:
                actualiza=label[i]*datos[i,:]
                actualiza=np.array(actualiza)
                actualiza=actualiza.reshape(-1,1)
                w=w+actualiza
                mejora=True
    
            
        
                
    return w, it  

#########################################
vini=[0.0,0.0,0.0]
unos=np.ones((x.shape[0],1))
x=np.concatenate((unos,x),axis=1)
w, iteraciones=ajusta_PLA(x,labels,1000,vini)

print('Vector obtenido: ', w)
print('\n\n')

#REPRESENTAMOS EL HIPERPLANO OBTENIDO
y0 = np.where(labels == -1) #capturo los índices de los elementos con -1
y1 = np.where(labels == 1) #capturo los índices de los elementos con 1
#x_2 contiene dos arrays, uno en cada componente, el primero tiene los valores de x con etiqueta -1 y la segunda los de etiqueta 1
x_2 = np.array([x[y0[0]],x[y1[0]]])
plt.scatter(x_2[0][:, 1], x_2[0][:, 2],  c = 'blue', label = '-1') #Dibujamos los puntos con etiqueta 1
plt.scatter(x_2[1][:, 1], x_2[1][:, 2],  c = 'orange', label = '1')#Dibujamos los de etiqueta -1

#Calculamos las imagenes de los puntos (sin aplicar la función signo) y así dibujar la recta de regresión
imagenes=[]

for i in x :
    imagenes.append((-w[1]*i[1]-w[0])/w[2]) #y=(-ax-c)/b
     
plt.plot( x[:,1], imagenes, c = 'black', label='g(x,y)') #Para representarlo, despejo x2 de la ecuación y represento la función resultante en 2D
plt.ylim(-50,50)
plt.legend();

#Dibujamos recta clasificadora
imagenes=[]

for i in x :
    imagenes.append(a*i[1]+b)
    
plt.plot( x[:,1], imagenes, c = 'red', label='f(x,y)') #Para representarlo, despejo x2 de la ecuación y represento la función resultante en 2D
plt.legend();
plt.title("Ejercicio PERCEPTRON")
plt.xlabel('x1')
plt.ylabel('x2')
plt.figure()
plt.show()

print('\n\nValor medio de iteraciones necesario para converger un vector de 0: ', iteraciones)

input("\n--- Pulsar tecla para continuar ---\n")

# Random initializations
iterations = []
#np.random.seed(0)
media=0
for i in range(0,10):
    #CODIGO DEL ESTUDIANTE
    vini=np.random.rand(3)
    w, iteraciones=ajusta_PLA(x,labels,1000,vini)
    media+=iteraciones
    print('Iteracion: ',i)
    print('vector inicial= ',vini)
    print('vector obtenido= ', w)
    print('Iteraciones: ', iteraciones)
    print('\n\n')

print('Valor medio de iteraciones necesario para converger con 10 vectores random: ', media/10.0)

input("\n--- Pulsar tecla para continuar ---\n")

print('Mismo experimento pero con la muestra con ruido')
vini=[0.0,0.0,0.0]
w, iteraciones=ajusta_PLA(x,y,1000,vini) #esta vez usamos el vector y de etiquetas alteradas
print('Vector obtenido: ', w)
print('\n\n')

#REPRESENTAMOS EL HIPERPLANO OBTENIDO
y0 = np.where(y == -1) #capturo los índices de los elementos con -1
y1 = np.where(y == 1) #capturo los índices de los elementos con 1
#x_2 contiene dos arrays, uno en cada componente, el primero tiene los valores de x con etiqueta -1 y la segunda los de etiqueta 1
x_2 = np.array([x[y0[0]],x[y1[0]]])
plt.scatter(x_2[0][:, 1], x_2[0][:, 2],  c = 'blue', label = '-1') #Dibujamos los puntos con etiqueta 1
plt.scatter(x_2[1][:, 1], x_2[1][:, 2],  c = 'orange', label = '1')#Dibujamos los de etiqueta -1

#Calculamos las imagenes de los puntos (sin aplicar la función signo) y así dibujar la recta de regresión
imagenes=[]

for i in x :
    imagenes.append((-w[1]*i[1]-w[0])/w[2]) #y=(-ax-c)/b
     
plt.plot( x[:,1], imagenes, c = 'black', label='g(x,y)') #Para representarlo, despejo x2 de la ecuación y represento la función resultante en 2D
plt.ylim(-50,50)
plt.legend();


#Representamos el clasificador
imagenes=[]

for i in x :
    imagenes.append(a*i[1]+b)
    
plt.plot( x[:,1], imagenes, c = 'red', label='f(x,y)') #Para representarlo, despejo x2 de la ecuación y represento la función resultante en 2D
plt.legend();
plt.title("Ejercicio PERCEPTRON apartado b) 1000 iteraciones")
plt.xlabel('x1')
plt.ylabel('x2')
plt.figure()
plt.show()

print('\n\n')
print('Valor medio de iteraciones necesario para converger un vector de 0: ', iteraciones)

input("\n--- Pulsar tecla para continuar ---\n")

##################################################################################################################################
##################################################################################################################################
############## EXPERIMENTO EXTRA ################
vini=[0.0,0.0,0.0]
w, iteraciones=ajusta_PLA(x,y,3000,vini) #Experimento con 3000 iteraciones
print('Vector obtenido: ', w)
print('\n\n')

y0 = np.where(y == -1) #capturo los índices de los elementos con -1
y1 = np.where(y == 1) #capturo los índices de los elementos con 1
#x_2 contiene dos arrays, uno en cada componente, el primero tiene los valores de x con etiqueta -1 y la segunda los de etiqueta 1
x_2 = np.array([x[y0[0]],x[y1[0]]])
plt.scatter(x_2[0][:, 1], x_2[0][:, 2],  c = 'blue', label = '-1') #Dibujamos los puntos con etiqueta 1
plt.scatter(x_2[1][:, 1], x_2[1][:, 2],  c = 'orange', label = '1')#Dibujamos los de etiqueta -1

#Calculamos las imagenes de los puntos (sin aplicar la función signo) y así dibujar la recta
imagenes=[]

for i in x :
    imagenes.append((-w[1]*i[1]-w[0])/w[2]) #y=(-ax-c)/b
     
plt.plot( x[:,1], imagenes, c = 'black', label='g(x,y)') #Para representarlo, despejo x2 de la ecuación y represento la función resultante en 2D
plt.ylim(-50,50)
plt.legend();


#Representamos el clasificador
imagenes=[]

for i in x :
    imagenes.append(a*i[1]+b)
    
plt.plot( x[:,1], imagenes, c = 'red', label='f(x,y)') #Para representarlo, despejo x2 de la ecuación y represento la función resultante en 2D
plt.legend();
plt.title("Ejercicio PERCEPTRON apartado b) 3000 iteraciones")
plt.xlabel('x1')
plt.ylabel('x2')
plt.figure()
plt.show()
########################## FIN EXPERIMENTO EXTRA #####################################
##################################################################################################################################
##################################################################################################################################

input("\n--- Pulsar tecla para continuar ---\n")

########## SEGUIMOS CON EL APARTADO b) DEL EJERCICIO 2.1
# Random initializations
iterations = []
#np.random.seed(0)
media=0
for i in range(0,10):
    #CODIGO DEL ESTUDIANTE
    vini=np.random.rand(3)
    w, iteraciones=ajusta_PLA(x,y,1000,vini)
    media+=iteraciones
    print('Iteracion: ',i)
    print('vector inicial= ',vini)
    print('vector obtenido= ', w)
    print('Iteraciones: ', iteraciones)
    print('\n\n')

print('Valor medio de iteraciones necesario para converger con 10 vectores random: ', media/10.0)


input("\n--- Pulsar tecla para continuar ---\n")
##################################################################################################################################
##################################################################################################################################
##################################################################################################################################
# EJERCICIO 3: REGRESIÓN LOGÍSTICA CON STOCHASTIC GRADIENT DESCENT

# FUNCIONES QUE VAMOS A NECESITAR
def err(x, y, w):
    '''
    Calcula el error en Regresión Logística
    Parameters
    ----------
    X : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    w : TYPE
        DESCRIPTION.

    '''
    y=y.reshape(-1,1)
    return np.mean(np.log(1 + np.exp(-y * x.dot(w))))


def gradiente(x,y,w): #w e y deben ser vectores columna esta función calcula el gradiente para regresión logística
    '''
    Calcula el gradiente en el caso N=1 
    Parameters
    ----------
    x : Matriz de datos
    
    y : vector de etiqyetas
    
    w : vector de pesos

    '''
    return -y * np.transpose(x) / (1 + np.exp(y * x.dot(w)))

    
def sgdRL(x,y,eta, tolerancia,tam_Minibatch=1):
    '''
    Algoritmo Gradiente Descendiente Estocástico aplicado a Regresión Logística
    
    Parameters
    ----------
    x : Matriz de datos
    
    y : vector de etiqyetas
    
    eta : learning rate
    
    tolerancia : tolerancia tal que una vez superada la función acaba
    
    tam_Minibatch : Tamaño del minibatch, en este caso usamos 1 por defecto

    Returns
    -------
    w : vector de pesos
    
    it: número de épocas
        

    '''
    y=y.reshape(-1,1) #transformamos y en un vector columna
    N=len(y) #Numero de filas de X e y
    iterations=0
    dif=1000.0
    w=np.zeros((x.shape[1],1)) #Inicializo w a un vector columna de ceros
    w=w.reshape(-1,1)
    xy=np.c_[x.copy(),y.copy()] #Esta función de numpy concatena dos matrices por columnas cuando el segundo parámetro es un vector columna

    while dif>tolerancia:
        w_anterior=w.copy()   #Guardo el vector de pesos de la iteración anterior
        np.random.shuffle(xy) #Mezclo los datos 

        for i in range(0,N,tam_Minibatch): #Recorro lo minibatches
        #Para cada minibatch actualizao el vector de pesos w con los datos del minibatch
            parada= i + tam_Minibatch
            x_mini,y_mini=xy[i:parada, :-1], xy[i:parada,-1:]
            grad=gradiente(x_mini,y_mini,w)
            w=w - eta*grad
            
        #Al acabar la actualización de los w incremento el número de iteraciones del bucle while
        iterations=iterations + 1
        dif= np.linalg.norm(w_anterior - w)
        
    return w, iterations



###############################################################################################################################################################
#EXPLICACIÓN DEL EXPERIMENTO
a,b=simula_recta([0,2]) # Generamos la recta que usaremos
print("Obtenemos los coeficientes a y b: ", a, b)

print('\n\nVAMOS A REALIZAR UN EJEMPLO DE UNA ITERACIÓN DEL EXPERIMENTO\n\n')
# Generamos los datos de la muestra
x = simula_unif(100, 2, [0,2])
#Generamos las etiquetas
y=[]

for i in x :
    y.append(f(i[0],i[1],a,b))


y=np.array(y)
#Añadimos la primera columna de unos
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
    
plt.plot( x[:,0], imagenes, c = 'red', label='f(x,y)') #Para representarlo, despejo x2 de la ecuación y represento la función resultante en 2D
plt.legend();


unos=np.ones((x.shape[0],1))
x=np.concatenate((unos,x),axis=1)
#LLamamos al algoritmo RL con una tolerancia de 0.01 y un lerning rate de 0.01
w,it=sgdRL(x,y,0.01,0.01)
print('\n\nCoeficientes obtenidos: ',w)
print('\n\n')

#Calculamos las imagenes de los puntos (sin aplicar la función signo) y así dibujar la recta de regresión
imagenes=[]

for i in x :
    imagenes.append((-w[1]*i[1]-w[0])/w[2]) #y=(-ax-c)/b
     
plt.plot( x[:,1], imagenes, c = 'black', label='g(x,y)') #Para representarlo, despejo x2 de la ecuación y represento la función resultante en 2D
plt.ylim(0,2)
plt.legend();
plt.title("Ejercicio 2.2 Ejemplo RL")
plt.xlabel('x1')
plt.ylabel('x2')
plt.figure()
plt.show()

Ein=err(x,y,w)

print('\n\nNúmero de épocas en converger: ', it)
print ('Calculamos el Error en la muestra (Ein): ', Ein)
print('\n\n')

input("\n--- Pulsar tecla para continuar ---\n")


#Probamos el modelo en otra muestra de 1000 datos
x = simula_unif(1000, 2, [0,2])
#Generamos las etiquetas
y=[]

for i in x :
    y.append(f(i[0],i[1],a,b))


y=np.array(y)
#Añadimos la primera columna de unos
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
    
plt.plot( x[:,0], imagenes, c = 'red', label='f(x,y)') #Para representarlo, despejo x2 de la ecuación y represento la función resultante en 2D
plt.legend();

unos=np.ones((x.shape[0],1))
x=np.concatenate((unos,x),axis=1)
imagenes=[]

for i in x :
    imagenes.append((-w[1]*i[1]-w[0])/w[2]) #y=(-ax-c)/b
     
plt.plot( x[:,1], imagenes, c = 'black', label='g(x,y)') #Para representarlo, despejo x2 de la ecuación y represento la función resultante en 2D
plt.ylim(0,2)
plt.legend();
plt.title("Comportamiento en muestra de 1000 datos")
plt.xlabel('x1')
plt.ylabel('x2')
plt.figure()
plt.show()
Eout=err(x,y,w)
print ('\n\nCalculamos el Error fuera de la muestra (Eout): ', Eout)



###############################################################################################################################################################
input("\n--- Pulsar tecla para continuar ---\n")
   
print ('\n\n\n COMIENZA EL EXPERIMENTO \n\n\n')
epocas=0
for i in range(100):
    print('Iteracion: ',i)
    x = simula_unif(100, 2, [0,2]) #obtenemos training set
    #Generamos las etiquetas
    y=[]

    for i in x :
        y.append(f(i[0],i[1],a,b))


    y=np.array(y)

    #Preparamos Regresión Logística
    unos=np.ones((x.shape[0],1))
    x=np.concatenate((unos,x),axis=1) #Añadimos columna de unos al principio de x

    w,it=sgdRL(x,y,0.01,0.01) #Ejecutamos el algoritmo con un learning rate de 0.01 y una tolerancia de 0.01
    Ein+=err(x,y,w) #Calculamos el Error interno
    #Probamos el modelo en otra muestra de 1000 datos (Test Set)
    x = simula_unif(1000, 2, [0,2])
    #Generamos las etiquetas
    y=[]

    for i in x :
        y.append(f(i[0],i[1],a,b))


    y=np.array(y)    
    unos=np.ones((x.shape[0],1))
    x=np.concatenate((unos,x),axis=1)
    Eout+=err(x,y,w)
    epocas+=it


print('\n\n ---------------- TRAS 100 ITERACIONES ----------------')
print('Ein medio: ', Ein/100.0)
print('Eout medio: ', Eout/100.0)
print('Número medio de épocas en converger: ', epocas/100.0)


input("\n--- Pulsar tecla para continuar ---\n")

print('EXPERIMENTO EXTRA')
print('Vamos a estudiar el comportamiento del algoritmo en una muestra con ruido')

a,b=simula_recta([0,2])
x = simula_unif(100, 2, [0,2])
y=[]

for i in x :
    y.append(f(i[0],i[1],a,b))


y=np.array(y)
#Añadimos la primera columna de unos
y0 = np.where(y == -1) #capturo los índices de los elementos con -1
y1 = np.where(y == 1) #capturo los índices de los elementos con 1

# Array con 10% de indices aleatorios para introducir ruido
y0=pd.DataFrame(data=y0[0]); #Convierto la matriz X en un Dataframe de Pandas, que es más cómodo de usar 
y0=y0.sample(frac=0.10,random_state=1); #Hacemos que tome un 10% de los datos de forma aleatoria
y0=y0.to_numpy()
for i in y0:
    y[i]=1
    

y1=pd.DataFrame(data=y1[0]); #Convierto la matriz X en un Dataframe de Pandas, que es más cómodo de usar 
y1=y1.sample(frac=0.10,random_state=1); #Hacemos que tome un 10% de los datos de forma aleatoria
y1=y1.to_numpy()
for i in y1:
    y[i]=-1


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
    
plt.plot( x[:,0], imagenes, c = 'red', label='f(x,y)') #Para representarlo, despejo x2 de la ecuación y represento la función resultante en 2D


unos=np.ones((x.shape[0],1))
x=np.concatenate((unos,x),axis=1)


#LLamamos al algoritmo RL con una tolerancia de 0.01 y un lerning rate de 0.01
w,it_RL=sgdRL(x,y,0.01,0.01)
print('\n\nCoeficientes obtenidos: ',w)
print('\n\n')

#Calculamos las imagenes de los puntos (sin aplicar la función signo) y así dibujar la recta de regresión
imagenes=[]

for i in x :
    imagenes.append((-w[1]*i[1]-w[0])/w[2]) #y=(-ax-c)/b
     
plt.plot( x[:,1], imagenes, c = 'black', label='RL') #Para representarlo, despejo x2 de la ecuación y represento la función resultante en 2D


Ein_RL=err(x,y,w)

# AHORA PLA
vini=[0.0,0.0,0.0]
w,it_PLA=ajusta_PLA(x,y,1000,vini)
imagenes=[]

for i in x :
    imagenes.append((-w[1]*i[1]-w[0])/w[2]) #y=(-ax-c)/b
     
plt.plot( x[:,1], imagenes, c = 'green', label='PLA') #Para representarlo, despejo x2 de la ecuación y represento la función resultante en 2D
plt.ylim(0,2)
plt.legend();
plt.title("Experimento Extra")
plt.xlabel('x1')
plt.ylabel('x2')
plt.figure()
plt.show()
Ein_PLA=err(x,y,w)

print('\n\nNúmero de épocas en convergerRL: ', it_RL)
print('Número de épocas en convergerPLA: ', it_PLA)
print ('Calculamos el Error en la muestra de RL: ', Ein_RL)
print ('Calculamos el Error en la muestra de PLA: ', Ein_PLA)
print('\n\n')
input("\n--- Pulsar tecla para continuar ---\n")




##################################################################################################################################
##################################################################################################################################
##################################################################################################################################
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
def error_clasificacion(x, y, w):
    '''
    Calcula el error de clasificación del hiperplano w para un
    conjunto de datos  x con  etiquetas y.
    '''
    incorrect = [signo(i.dot(w)) for i in x] != y
    return np.mean(incorrect)
 
# Pseudoinversa	
def pseudoinverse(x,y):
    '''
    Algoritmo de la Pseudoinversa

    Parameters
    ----------
    x : Matriz de datos 
    
    y : Vector de etiquetas
        
    Returns
    -------
    w : Vector de pesos ajustados

    '''
    pseudoinverse=np.linalg.pinv(np.transpose(x).dot(x))
    X=pseudoinverse.dot(np.transpose(x))
    w=X.dot(y)
    return w


print('\n----------------------REGRESIÓN LINEAL----------------------')
print('Usamos el algoritmo de la pseudoinversa para estimar la recta de regresión')

w=pseudoinverse(x,y)

print('\nVector de pesos obtenido con Pseudoinversa: ', w)
print('Error de clasificación cometido por la Pseudoinversa: ', error_clasificacion(x,y,w))

fig, ax = plt.subplots()
ax.plot(np.squeeze(x[np.where(y == -1),1]), np.squeeze(x[np.where(y == -1),2]), 'o', color='red', label='4')
ax.plot(np.squeeze(x[np.where(y == 1),1]), np.squeeze(x[np.where(y == 1),2]), 'o', color='blue', label='8')
imagenes=[]

for i in x :
    imagenes.append((-w[1]*i[1]-w[0])/w[2]) #y=(-ax-c)/b
     
plt.plot( x[:,1], imagenes, c = 'green', label='Pseudoinversa') #Para representarlo, despejo x2 de la ecuación y represento la función resultante en 2D


input("\n--- Pulsar tecla para continuar ---\n")

print('\n----------------------POCKET ALGORITHM----------------------')
print('Vamos a usar el algoritmo POCKET, en primer lugar sobre el vector de pesos devuelto por la pseudoinversa')


#POCKET ALGORITHM

def pla_pocket(x, y, max_iter, w_ini):
    '''
    Algoritmo PLA POCKET, ajusta el vector de pesos w_ini a un hiperplano de clasificación

    Parameters
    ----------
    x : matriz de datos
    
    y : Vector de etiquetas
    
    max_it : máximo de iteraciones
    
    w_ini : vector de pesos inicial

    Returns
    -------
    w_best : TYPE
        DESCRIPTION.
    evol : TYPE
        DESCRIPTION.

    '''
    mejora=True
    it=0
    w=np.array(w_ini)
    w=w.reshape(-1,1) #Lo transformo en un vector columna
    w_best = w_ini.copy()
    error_minimo = error_clasificacion(x, y, w_best)
    evolucion = [w_ini]

    while (mejora and it<max_iter): # 
        mejora=False
        it+=1
        for i in range(len(x)):
            valor=x[i,:].dot(w)
            sign=signo(valor.item())

            if sign!=y[i]:
                actualiza=y[i]*x[i,:]
                actualiza=np.array(actualiza)
                actualiza=actualiza.reshape(-1,1)
                w=w+actualiza
                mejora=True
                
        error_actual=error_clasificacion(x,y,w)
        if error_actual<error_minimo:
            error_minimo=error_actual
            w_best=w.copy()
        
        evolucion.append(w_best.copy())

            
    return w_best, evolucion
     
#Hacemos 500 iteraciones
w,evolucion=pla_pocket(x,y,500,w)
EinPseudo=error_clasificacion(x,y,w)
print('\nVector de pesos obtenido con mejora PLA POCKET: ', w)
print('Ein obtenido con Pseudoinversa+PLA POCKET: ', EinPseudo)
imagenes=[]

for i in x :
    imagenes.append((-w[1]*i[1]-w[0])/w[2]) #y=(-ax-c)/b
     
plt.plot( x[:,1], imagenes, c = 'black', label='Pseudoinversa+PLA POCKET') #Para representarlo, despejo x2 de la ecuación y represento la función resultante en 2D
plt.legend();
plt.title("Ajuste Training Set con Pseudoinversa+PLA POCKET")
plt.xlabel('Intensidad Promedio')
plt.ylabel('Simetría')
plt.figure()
plt.show()
input("\n--- Pulsar tecla para continuar ---\n")

print('VEAMOS COMPORTAMIENTO SOBRE EL TEST SET')

fig, ax = plt.subplots()
ax.plot(np.squeeze(x_test[np.where(y_test == -1),1]), np.squeeze(x_test[np.where(y_test == -1),2]), 'o', color='red', label='4')
ax.plot(np.squeeze(x_test[np.where(y_test == 1),1]), np.squeeze(x_test[np.where(y_test == 1),2]), 'o', color='blue', label='8')

imagenes=[]

for i in x_test :
    imagenes.append((-w[1]*i[1]-w[0])/w[2]) #y=(-ax-c)/b
     
plt.plot( x_test[:,1], imagenes, c = 'black', label='Pseudoinversa+PLA POCKET') #Para representarlo, despejo x2 de la ecuación y represento la función resultante en 2D
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (TEST)')
ax.set_xlim((0, 1))
plt.legend()
plt.show()
EtestPseudo=error_clasificacion(x_test, y_test,w)
print('Etest obtenido con Pseudoinversa+PLA POCKET: ', EtestPseudo)
input("\n--- Pulsar tecla para continuar ---\n")

#GRAFICO DE EVOLUCIÓN DEL ERROR CON PLA POCKET
iteraciones=np.arange(501)
error=[]
for i in evolucion:
    error.append(error_clasificacion(x,y,i))
    
plt.plot(iteraciones,error, c='black', label='Error')
plt.title("Evolución del error para PLA POCKET")
plt.xlabel('Iteraciones')
plt.ylabel('error')
plt.figure()
plt.show()
input("\n--- Pulsar tecla para continuar ---\n")


print('\n\nAhora a usar el algoritmo POCKET sobre un vector de pesos inicializado a 0')
w_ini=[0.0, 0.0, 0.0]
w,evolucion=pla_pocket(x,y,500,w_ini)
print('\nVector de pesos obtenido con PLA POCKET: ', w)


#REPRESENTAMOS LOS RESULTADOS OBTENIDOS
fig, ax = plt.subplots()
ax.plot(np.squeeze(x[np.where(y == -1),1]), np.squeeze(x[np.where(y == -1),2]), 'o', color='red', label='4')
ax.plot(np.squeeze(x[np.where(y == 1),1]), np.squeeze(x[np.where(y == 1),2]), 'o', color='blue', label='8')
imagenes=[]

for i in x :
    imagenes.append((-w[1]*i[1]-w[0])/w[2]) #y=(-ax-c)/b
     
plt.plot( x[:,1], imagenes, c = 'black', label='PLA POCKET') #Para representarlo, despejo x2 de la ecuación y represento la función resultante en 2D
plt.legend();
plt.title("Ajuste Training Set con PLA POCKET")
plt.xlabel('Intensidad Promedio')
plt.ylabel('Simetría')
plt.figure()
plt.show()


input("\n--- Pulsar tecla para continuar ---\n")

#GRAFICO DE EVOLUCIÓN DEL ERROR CON PLA POCKET
iteraciones=np.arange(501)
error=[]
for i in evolucion:
    error.append(error_clasificacion(x,y,i))
    
plt.plot(iteraciones,error, c='black', label='Error')
plt.title("Evolución del error para PLA POCKET")
plt.xlabel('Iteraciones')
plt.ylabel('error')
plt.figure()
plt.show()


input("\n--- Pulsar tecla para continuar ---\n")

#OBTENEMOS EL ERROR EN LA MUESTRA Y EN TEST SET
EinPLA=error_clasificacion(x,y,w)
print('Ein obtenido con PLA en 500 iteraciones: ', EinPLA)
print('VEAMOS COMPORTAMIENTO SOBRE EL TEST SET')

#REPRESENTAMOS EL RESULTADO EN EL TEST SET
fig, ax = plt.subplots()
ax.plot(np.squeeze(x_test[np.where(y_test == -1),1]), np.squeeze(x_test[np.where(y_test == -1),2]), 'o', color='red', label='4')
ax.plot(np.squeeze(x_test[np.where(y_test == 1),1]), np.squeeze(x_test[np.where(y_test == 1),2]), 'o', color='blue', label='8')

imagenes=[]

for i in x_test :
    imagenes.append((-w[1]*i[1]-w[0])/w[2]) #y=(-ax-c)/b
     
plt.plot( x_test[:,1], imagenes, c = 'black', label='Pseudoinversa+PLA POCKET') #Para representarlo, despejo x2 de la ecuación y represento la función resultante en 2D
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (TEST)')
ax.set_xlim((0, 1))
plt.legend()
plt.show()

#CALCULAMOS ERROR EN TEST SET
EtestPLA=error_clasificacion(x_test, y_test,w)
print('Etest obtenido con PLA POCKET: ', EtestPLA)

input("\n--- Pulsar tecla para continuar ---\n")

#COTAS SOBRE EL ERROR

def Hoeffding(error, N, H, delta):
    '''
    Cota de Hoeffding para el error de generalización.
    Parameters
    ----------
    error : Error del que partimos
        
    N : Tamaño de la muestra
    
    H : Cardinal de H para calcular el error del que partimos.
    
    delta : tolerancia

    Returns
    -------
    cota de hoeffding

    '''
    return error+np.sqrt((1/(2*N))*np.log((2*H)/delta))

def VC(error, N, dvc, delta):
    '''
    Cota para el error de generalización de VC.

    Parameters
    ----------
    error : error del que partimos
    
    N : Tamaño de la muestra
    
    dvc : dimensión de VC
    
    delta : tolerancia

    Returns
    -------
    cota de VC

    '''
    return error+np.sqrt((8/N)*np.log(4*((2*N)**dvc+1)/delta))

print('----------------------COTAS DE ERROR----------------------\n')
print('COTA DE HOEFFDING\n')
print('Pseudoinversa+PLA POCKET:')
print('Ein: ', Hoeffding(EinPseudo,x.shape[0], 2**(64*3),0.05))
print('Etest: ', Hoeffding(EtestPseudo,x_test.shape[0], 1,0.05))
print('\nPLA POCKET:')
print('Ein: ', Hoeffding(EinPLA,x.shape[0], 2**(64*3),0.05))
print('Etest: ', Hoeffding(EtestPLA,x_test.shape[0], 1,0.05) )

print('\nCOTA DE VC\n')
print('Pseudoinversa+PLA POCKET:')
print('Ein: ', VC(EinPseudo,x.shape[0],3,0.05))

print('\nPLA POCKET:')
print('Ein: ', VC(EinPLA,x.shape[0],3,0.05))

