# -*- coding: utf-8 -*-
"""
TRABAJO 1. 
Nombre Estudiante: 
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

np.random.seed(1)

print('EJERCICIO SOBRE LA BUSQUEDA ITERATIVA DE OPTIMOS\n')
print('Ejercicio 1\n')

def E(u,v):
    return  (u**3*np.e**(v-2)-2*v**2*np.e**(-u))**2

#Derivada parcial de E con respecto a u
def dEu(u,v):
    return 2*(np.e**(v-2)*u**3-2*v**2*np.e**(-u))*(2*v**2*np.e**(-u)+3*np.e**(v-2)*u**2)
    
#Derivada parcial de E con respecto a v
def dEv(u,v):
    return 2*(u**3*np.e**(v-2)-4*np.e**(-u)*v)*(u**3*np.e**(v-2)-2*np.e**(-u)*v**2)

#Gradiente de E
def gradE(u,v):
    return np.array([dEu(u,v), dEv(u,v)])

def gradient_descent(w,eta,num_iterations, error): #Función para el caso del ejercicio 1 b) w: puntos iniciales, eta:tasa aprendizaje, num_iterations:num max de iteraciones, error: precisión requerida
    #
    # gradiente descendente
    # 
    iterations=0 
    Err=1000.0

    while Err>error and iterations<num_iterations:
       partial_derivative=gradE(w[0],w[1])
       w=w - eta*partial_derivative
       iterations=iterations + 1 
       Err=E(w[0],w[1]) #Preguntar entre usar esto y el error cuadrático medio
        
    return w, iterations    

def gradient_descent_linear_regresion(X,y,w,eta,num_iterations):
    N=len(y)
    iterations=0 
    Err=1000.0
    
    while Err>error and iterations<num_iterations:
        h_x= X*w 
        partial_derivative = np.transpose(h_x - y)*X #multiplico el vector fila transpose(h_x-y) por X así consigo la sum de 1 a N de el xnj*(h(xn)-yn) en cada componente del vector patial_derivative
        w=w - (2/N)*(eta*np.transpose(partial_derivative))
        iterations=iterations + 1 
        Err=calc_coste(w)
        
        
    
def calc_coste(X,y,w): #X tienen que ser np.matrix, y np.array, w np.array
    N=len(y) #Calculo el número de filas de y
    Err=(1/N)*np.transpose(X*w-y)*(X*w-y)
    return Err.item()



eta = 0.1 
maxIter = 10000000000
error2get = 1e-14
initial_point = np.array([1.0,1.0])
w, it = gradient_descent(initial_point, eta,maxIter, error2get)


print('Funcion a minimizar: E(u,v)=(u^3*e^(v-2)-2*v^2*e^(-u))^2')
print('Gradiente: [2*(e^(v-2)*u^3-2*v^2*e^(-u))*(2*v^2*e^(-u)+3*e^(v-2)*u^2), 2*(u^3*e^(v-2)-4*e^(-u)*v)*(u^3*e^(v-2)-2*e^(-u)*v^2)]')
print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')

# DISPLAY FIGURE
x = np.linspace(-30, 30, 50)
y = np.linspace(-30, 30, 50)
X, Y = np.meshgrid(x, y)
Z = E(X, Y) #E_w([X, Y])
fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_surface(X, Y, Z, edgecolor='none', rstride=1,
                        cstride=1, cmap='jet')
min_point = np.array([w[0],w[1]])
min_point_ = min_point[:, np.newaxis]
ax.plot(min_point_[0], min_point_[1], E(min_point_[0], min_point_[1]), 'r*', markersize=10)
ax.set(title='Ejercicio 1.2. Función sobre la que se calcula el descenso de gradiente')
ax.set_xlabel('u')
ax.set_ylabel('v')
ax.set_zlabel('E(u,v)')
plt.show()
input("\n--- Pulsar tecla para continuar ---\n")


#Ejercicio 1.2

def f(x,y):
    return  (x+2)**2 + 2*(y-2)**2 + 2*np.sin(2*np.pi*x)*np.sin(2*np.pi*y)

#Derivada parcial de f con respecto a x
def dfx(x,y):
    return 4*np.pi*np.sin(2*np.pi*y)*np.cos(2*np.pi*x)+2*(x+2)
    
#Derivada parcial de f con respecto a y
def dfy(x,y):
    return 4*np.pi*np.sin(2*np.pi*x)*np.cos(2*np.pi*y)+4*(y-2)

#Gradiente de f
def gradf(x,y):
    return np.array([dfx(x,y), dfy(x,y)])

def gradient_descent2(w,eta,num_iterations, error): #Función para el caso del ejercicio 1 b) w: puntos iniciales, eta:tasa aprendizaje, num_iterations:num max de iteraciones, error: precisión requerida
    #
    # gradiente descendente
    # 
    iterations=0 
    Err=1000.0
    N=1.0 #En este caso N vale 1 porque el vector de etiquetas y solo contiene un elemento que es el 0
    vector_puntos=np.array([[w[0],w[1]]])
    while Err>error and iterations<num_iterations:
       h_x=f(w[0],w[1])
       partial_derivative=gradf(w[0],w[1])
       w=w -(eta*np.transpose(partial_derivative))
       iterations=iterations + 1 
       Err=f(w[0],w[1])
       vector_puntos=np.append(vector_puntos, [[w[0],w[1]]], axis=0)
          
    
    return w, iterations, vector_puntos



eta = 0.01 
maxIter = 50
error2get = 1e-14
initial_point = np.array([-1.0,1.0])
w, it, vector_puntos = gradient_descent2(initial_point, eta,maxIter, error2get)

print('Funcion a minimizar: f(x,y)=(x+2)^2 + 2*(y-2)^2 + 2*sin(2*pi*x)*sin(2*pi*y)')
print('Gradiente: [4*pi*sin(2*pi*y)*cos(2*pi*x)+2*(x+2), 4*pi*sin(2*pi*x)*cos(2*pi*y)+4*(y-2)]')
print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas con eta=0.01 : (', w[0], ', ', w[1],')')

################## DISPLAY FIGURE ##############################

x = np.linspace(-0.5, -1.5, 50)
y = np.linspace(0.5, 1.5, 50)
X, Y = np.meshgrid(x, y)
Z = f(X, Y) #E_w([X, Y])
fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_surface(X, Y, Z,edgecolor='none', rstride=1,
                        cstride=1, cmap=cm.coolwarm,linewidth=0)
min_point=vector_puntos

for i in min_point:
    ax.plot(i[0],i[1],f(i[0],i[1]), 'r*' , markersize=10)


ax.set(title='Ejercicio 1.3 f(x,y) con eta=0.01')
ax.set_xlabel('y')
ax.set_ylabel('x')
ax.set_zlabel('f(x,y)')
ax.view_init(70, 60) #Uso esta función para rotar el gráfico y que se vea mejor cómo funciona el gradiente descendente el primer parámetro es la elevación y el segundo el ángulo de rotación de la cámara
plt.draw()
plt.show()
input("\n--- Pulsar tecla para continuar ---\n")

eta = 0.1 

initial_point = np.array([-1.0,1.0])
w, it, vector_puntos = gradient_descent2(initial_point, eta,maxIter, error2get)
print('Funcion a minimizar: f(x,y)=(x+2)^2 + 2*(y-2)^2 + 2*sin(2*pi*x)*sin(2*pi*y)')
print('Gradiente: [4*pi*sin(2*pi*y)*cos(2*pi*x)+2*(x+2), 4*pi*sin(2*pi*x)*cos(2*pi*y)+4*(y-2)]')
print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas con eta=0.1: (', w[0], ', ', w[1],')')

################## DISPLAY FIGURE ##############################

x = np.linspace(-10, 10, 50)
y = np.linspace(-10, 10, 50)
X, Y = np.meshgrid(x, y)
Z = f(X, Y) #E_w([X, Y])
fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_surface(X, Y, Z,edgecolor='none', rstride=1,
                        cstride=1, cmap=cm.coolwarm,linewidth=0) #He usado cm.coolwarm porque dibuja con colores más suaves el gráfico
min_point=vector_puntos

for i in min_point:
    ax.plot(i[0],i[1],f(i[0],i[1]), 'r*' , markersize=10) #del vector de puntos devuelto por la función gradient_descent2 voy representando cada punto.


ax.set(title='Ejercicio 1.3 f(x,y) con eta=0.1')
ax.set_xlabel('y')
ax.set_ylabel('x')
ax.set_zlabel('f(x,y)')
ax.view_init(70, 60) #Uso esta función para rotar el gráfico y que se vea mejor cómo funciona el gradiente descendente el primer parámetro es la elevación y el segundo el ángulo de rotación de la cámara
plt.draw()
plt.show()
input("\n--- Pulsar tecla para continuar ---\n")

eta = 0.01
x=-0.5
y=-0.5 
maxIter = 50
error2get = 1e-14
initial_point = np.array([x,y])
w, it, vector_puntos = gradient_descent2(initial_point, eta,maxIter, error2get)
print('Puntos iniciales (x,y)= (',x, ',', y,')')
print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')
print ('valor obtenido: ', f(w[0],w[1]))

print("\n--------------------------\n")
x=1
y=1
initial_point = np.array([x,y])
w, it, vector_puntos = gradient_descent2(initial_point, eta,maxIter, error2get)
print('Puntos iniciales (x,y)= (',x, ',', y,')')
print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')
print ('valor obtenido: ', f(w[0],w[1]))

print("\n--------------------------\n")
x=2.1
y=-2.1
initial_point = np.array([x,y])
w, it, vector_puntos = gradient_descent2(initial_point, eta,maxIter, error2get)
print('Puntos iniciales (x,y)= (',x, ',', y,')')
print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')
print ('valor obtenido: ', f(w[0],w[1]))

print("\n--------------------------\n")
x=-3
y=3
initial_point = np.array([x,y])
w, it, vector_puntos = gradient_descent2(initial_point, eta,maxIter, error2get)
print('Puntos iniciales (x,y)= (',x, ',', y,')')
print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')
print ('valor obtenido: ', f(w[0],w[1]))

print("\n--------------------------\n")
x=-2
y=2
initial_point = np.array([x,y])
w, it, vector_puntos = gradient_descent2(initial_point, eta,maxIter, error2get)
print('Puntos iniciales (x,y)= (',x, ',', y,')')
print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')
print ('valor obtenido: ', f(w[0],w[1]))


input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################
###############################################################################
###############################################################################
###############################################################################
print('EJERCICIO SOBRE REGRESION LINEAL\n')
print('Ejercicio 1\n')

label5 = 1
label1 = -1

# Funcion para leer los datos
def readData(file_x, file_y):
	# Leemos los ficheros	
	datax = np.load(file_x)
	datay = np.load(file_y)
	y = []
	x = []	
	# Solo guardamos los datos cuya clase sea la 1 o la 5
	for i in range(0,datay.size):
		if datay[i] == 5 or datay[i] == 1:
			if datay[i] == 5:
				y.append(label5)
			else:
				y.append(label1)
			x.append(np.array([1, datax[i][0], datax[i][1]]))
			
	x = np.array(x, np.float64)
	y = np.array(y, np.float64)
	
	return x, y

# Funcion para calcular el error
def Err(x,y,w):
    N=len(y) #Calculo el número de filas de y
    producto=x.dot(w)
    Err=(1/N)*(np.transpose(producto-y).dot(producto-y))
    return Err.item()
 

# Gradiente Descendente Estocastico
def sgd(x,y,eta,num_iterations,error,tam_Minibatch=1):
    N=len(y)
    iterations=0 
    Error=1000.0
    w=np.ones(x.shape[1]) #A que valor inicializar w
    w=w.reshape(-1,1)
    while Error>error and iterations<num_iterations:
        #################################### Tomamos 10 filas aleatorias de la matriz x #####################################
        #Idea tomada de https://www.it-swarm-es.com/es/python/numpy-obtener-un-conjunto-aleatorio-de-filas-de-la-matriz-2d/1069900142/
        # con random choice elegimos de un conjunto de x.shape[0] (filas de x) elementos tantos como indique la variable tam_minibatch y sin reemplazamiento   
        filas=np.random.choice(x.shape[0], size=tam_Minibatch, replace=False)
        x_mini=x[filas,:] 
        y_mini=y[filas]
        h_x= np.dot(x_mini,w)
        partial_derivative = np.dot(np.transpose(h_x - y_mini),x_mini) #multiplico el vector fila transpose(h_x-y) por X así consigo la sum de 1 a N de el xnj*(h(xn)-yn) en cada componente del vector patial_derivative
        w=w - (2/N)*(eta*np.transpose(partial_derivative))
        iterations=iterations + 1 
        Error= Err(x,y,w)
    
    return w

# Pseudoinversa	
def pseudoinverse(x,y,w):
    #
    pseudoinverse=np.linalg.pinv(np.transpose(x).dot(x))
    X=pseudoinverse.dot(np.transpose(x));
    w=X.dot(y);
    return w


# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy')
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy')

y=y.reshape(-1,1) #redimensionamos el vector para convertirlo en un vector columna pongo -1 porque a priori no se cuantas filas saldrán
y_test=y_test.reshape(-1,1) #redimensionamos el vector para convertirlo en un vector columna pongo -1 porque a priori no se cuantas filas saldrán
num_iterations=10000
errorerror2get = 1e-14
eta=0.1

w = sgd(x,y,eta,num_iterations,error2get,tam_Minibatch=10)
print ('Bondad del resultado para grad. descendente estocastico:\n')
print('Uso eta=0.1, error=1e-14 , max_iteraciones=10000 y w inicializado a [1. 1. 1.]')
print('w final: ', w)
print ("Ein: ", Err(x,y,w))
print ("Eout: ", Err(x_test, y_test, w))

################################################################# Pseudoinversa #####################################################

print ('\n Bondad del resultado para algoritmo de la pseudoinversa:\n')
w=pseudoinverse(x,y,w)
print('w final: ', w)
print ("Ein: ", Err(x,y,w))
print ("Eout: ", Err(x_test, y_test, w))

input("\n--- Pulsar tecla para continuar ---\n")

#Seguir haciendo el ejercicio...

print('Ejercicio 2\n')
# Simula datos en un cuadrado [-size,size]x[-size,size]
def simula_unif(N, d, size):
	return np.random.uniform(-size,size,(N,d))


def sign(x):
	if x >= 0:
		return 1
	return -1

def f1(x1, x2):
    x=(x1-0.2)**2 + x2**2 -0.6
    return sign(x)

#Seguir haciendo el ejercicio...

#APARTADO a)
np.random.seed(2)

X=simula_unif(1000,2,1)

# Dibujo el Scatter Plot
plt.scatter(X[:,0],X[:,1], c='blue')

#Los muestro todos juntos
plt.show();


#Apartado b)
y=[]
for i in X:
    y.append(f1(i[0],i[1]))


X_=pd.DataFrame(data=X); #Convierto la matriz X en un Dataframe de Pandas, que es más cómodo de usar 
X_=X_.sample(frac=0.10,random_state=1); #Hacemos que tome un 10% de los datos de forma aleatoria

for i in X_.index:
    y[i]=y[i]*-1 #Cambio el signo de esos elementos


plt.scatter(X[:,0],X[:,1], c=y) #Uso el vector y como vector de colores

#Muestro el gráfico
plt.show();

#Apartado c)
#Concatenamos El vector de unos con la matriz X, para ello  usamos np.concatenate especificando que es por columnas (axis=1)
y=np.array(y)
y=y.reshape(-1,1) #convertimos y en un vector columna 
unos=np.ones((X.shape[0],1))
X=np.concatenate((unos,X),axis=1)

w = sgd(X,y,eta,num_iterations,error2get,tam_Minibatch=10)
print ('Bondad del resultado para grad. descendente estocastico:\n')
print('Uso eta=0.1, error=1e-14 , max_iteraciones=10000 y w inicializado a [1. 1. 1.]')
print('w final: ', w)
print ("Ein: ", Err(X,y,w))


#Apartado d)

Ein=0
E_out=0
'''
for i in range(100):
    print ("iteracion: ", i)
    X=simula_unif(1000,2,1)
    y=[]
    for i in X:
        y.append(f1(i[0],i[1]))
        
    X_=pd.DataFrame(data=X); #Convierto la matriz X en un Dataframe de Pandas, que es más cómodo de usar 
    X_=X_.sample(frac=0.10,random_state=1); #Hacemos que tome un 10% de los datos de forma aleatoria

    for i in X_.index:
        y[i]=y[i]*-1 #Cambio el signo de esos elementos
    
    y=np.array(y)
    y=y.reshape(-1,1) #convertimos y en un vector columna 
    unos=np.ones((X.shape[0],1))
    X=np.concatenate((unos,X),axis=1)
    w = sgd(X,y,eta,num_iterations,error2get,tam_Minibatch=10)
    Ein+=Err(X,y,w)
    X=simula_unif(1000,2,1) #Generamos 1000 datos nuevos 
    X=np.concatenate((unos,X),axis=1)
    
    ###########################PREPARAMOS EL TEST SET ###########################
    y=[]
    for i in X:
        y.append(f1(i[0],i[1])) #Generamos las etiquetas para los nuevos datos
    y=np.array(y)
    y=y.reshape(-1,1) #convertimos y en un vector columna 
    E_out += Err(X,y,w)
'''

print ('Tras mil iteraciones repitiendo el ejemplo anterior:\n')
print ("Ein medio: ", Ein/100.0)   
print ("Eout medio: ", E_out/100.0)   
input("\n--- Pulsar tecla para continuar ---\n")



#################################################################################################
#################################################################################################
#################################################################################################
#################### Mismo experimento con distinto vector de características####################

#APARTADO a)
np.random.seed(2)

X=simula_unif(1000,2,1)

# Dibujo el Scatter Plot
plt.scatter(X[:,0],X[:,1], c='blue')

#Los muestro todos juntos
plt.show();


#Apartado b)
y=[]
for i in X:
    y.append(f1(i[0],i[1]))


X_=pd.DataFrame(data=X); #Convierto la matriz X en un Dataframe de Pandas, que es más cómodo de usar 
X_=X_.sample(frac=0.10,random_state=1); #Hacemos que tome un 10% de los datos de forma aleatoria

for i in X_.index:
    y[i]=y[i]*-1 #Cambio el signo de esos elementos


plt.scatter(X[:,0],X[:,1], c=y) #Uso el vector y como vector de colores

#Muestro el gráfico
plt.show()

#Apartado c)
#Concatenamos El vector de unos con la matriz X, para ello  usamos np.concatenate especificando que es por columnas (axis=1)
y=np.array(y)
y=y.reshape(-1,1) #convertimos y en un vector columna 


############### Preparamos la nueva matriz de características
x1x2=X[:,0]*X[:,1] #multiplicación de las dos columnas elemento a elemento
x1x2=x1x2.reshape(-1,1)
x1_cuadrado=X[:,0]*X[:,0] 
x1_cuadrado=x1_cuadrado.reshape(-1,1)
x2_cuadrado=X[:,1]*X[:,1] 
x2_cuadrado=x2_cuadrado.reshape(-1,1)
unos=np.ones((X.shape[0],1))
X=np.concatenate((unos,X,x1x2,x1_cuadrado,x2_cuadrado),axis=1) #Unimos por columnas todo

w = sgd(X,y,eta,num_iterations,error2get,tam_Minibatch=10)
print ('Bondad del resultado para grad. descendente estocastico:\n')
print('Uso eta=0.1, error=1e-14 , max_iteraciones=10000 y w inicializado a [1. 1. 1.]')
print('w final: ', w)
print ("Ein: ", Err(X,y,w)) #Me sale Ein:  0.608731055311969 mejora


##################################Experimento con 1000 iteraciones##################################
Ein=0
E_out=0

for i in range(100):
    print ("iteracion: ", i)
    X=simula_unif(1000,2,1)
    y=[]
    for i in X:
        y.append(f1(i[0],i[1]))
        
    X_=pd.DataFrame(data=X); #Convierto la matriz X en un Dataframe de Pandas, que es más cómodo de usar 
    X_=X_.sample(frac=0.10,random_state=1); #Hacemos que tome un 10% de los datos de forma aleatoria

    for i in X_.index:
        y[i]=y[i]*-1 #Cambio el signo de esos elementos
    
    y=np.array(y)
    y=y.reshape(-1,1) #convertimos y en un vector columna 
    x1x2=X[:,0]*X[:,1] #multiplicación de las dos columnas elemento a elemento
    x1x2=x1x2.reshape(-1,1)
    x1_cuadrado=X[:,0]*X[:,0] 
    x1_cuadrado=x1_cuadrado.reshape(-1,1)
    x2_cuadrado=X[:,1]*X[:,1] 
    x2_cuadrado=x2_cuadrado.reshape(-1,1)
    unos=np.ones((X.shape[0],1))
    X=np.concatenate((unos,X,x1x2,x1_cuadrado,x2_cuadrado),axis=1) #Unimos por columnas todo
    w = sgd(X,y,eta,num_iterations,error2get,tam_Minibatch=10)
    Ein+=Err(X,y,w)
   
    
    ###########################PREPARAMOS EL TEST SET ###########################
    X=simula_unif(1000,2,1) #Generamos 1000 datos nuevos 
    x1x2=X[:,0]*X[:,1] #multiplicación de las dos columnas elemento a elemento
    x1x2=x1x2.reshape(-1,1)
    x1_cuadrado=X[:,0]*X[:,0] 
    x1_cuadrado=x1_cuadrado.reshape(-1,1)
    x2_cuadrado=X[:,1]*X[:,1] 
    x2_cuadrado=x2_cuadrado.reshape(-1,1)
    X=np.concatenate((unos,X,x1x2,x1_cuadrado,x2_cuadrado),axis=1) #Unimos por columnas todo
    y=[]
    for i in X:
        y.append(f1(i[0],i[1])) #Generamos las etiquetas para los nuevos datos
    y=np.array(y)
    y=y.reshape(-1,1) #convertimos y en un vector columna 
    E_out += Err(X,y,w)

print ('Tras mil iteraciones repitiendo el ejemplo anterior:\n')
print ("Ein medio: ", Ein/100.0)   
print ("Eout medio: ", E_out/100.0)   
input("\n--- Pulsar tecla para continuar ---\n")

#### Mis resultados---> Ein medio:  0.5883792408813793  Eout medio:  1.2217528079915623

