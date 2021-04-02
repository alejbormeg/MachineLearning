# -*- coding: utf-8 -*-
"""
TRABAJO 1. 
Nombre Estudiante: Alejandro Borrego Megías
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from sklearn.utils import shuffle

np.random.seed(1)

print('EJERCICIO SOBRE LA BUSQUEDA ITERATIVA DE OPTIMOS\n')
print('Ejercicio 1\n')


#######################################################################################################################################
###################################### Funciones a usar en el ejercicio 1.1 y 1.2 #####################################################
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
       partial_derivative=gradE(w[0],w[1]) #Calculamos el valor del gradiente en el punto actual
       w=w - eta*partial_derivative #Modificamos el valor de los pesos en la dirección opuesta al gradiente
       iterations=iterations + 1 
       Err=E(w[0],w[1]) #Calculamos el valor de la función en el nuevo punto
        
    return w, iterations    

#######################################################################################################################################
#######################################################################################################################################

#################################################### Ejercicio 1.2 ####################################################################
eta = 0.1  #Tasa de aprendizaje
maxIter = 10000000000 #Máximo de iteraciones del gradiente descendente
error2get = 1e-14 #Error a alcanzar
initial_point = np.array([1.0,1.0])
w, it = gradient_descent(initial_point, eta,maxIter, error2get) #Llamamos a la función Gradiente DEscendente especificando en el primer parámetro el punto inicial (1,1), la tasa de aprendizaje, el máximo de iteraciones y el error a alcanzar


####### Imprimimos los resultados por pantalla
print('Funcion a minimizar: E(u,v)=(u^3*e^(v-2)-2*v^2*e^(-u))^2')
print('Gradiente: [2*(e^(v-2)*u^3-2*v^2*e^(-u))*(2*v^2*e^(-u)+3*e^(v-2)*u^2), 2*(u^3*e^(v-2)-4*e^(-u)*v)*(u^3*e^(v-2)-2*e^(-u)*v^2)]')
print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')

# Dibujamos el gráfico en 3D
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
#######################################################################################################################################



###################################### Funciones a usar en el ejercicio 1.3 ###########################################################

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

def gradient_descent2(w,eta,num_iterations): #Función para el caso del ejercicio 1 b) w: puntos iniciales, eta:tasa aprendizaje, num_iterations:num max de iteraciones, error: precisión requerida
    #
    # gradiente descendente
    # 
    iterations=0 
    vector_puntos=np.array([[w[0],w[1]]])
    while iterations<num_iterations:
       h_x=f(w[0],w[1])
       partial_derivative=gradf(w[0],w[1])
       w=w -(eta*np.transpose(partial_derivative))
       iterations=iterations + 1 
       Err=f(w[0],w[1])
       vector_puntos=np.append(vector_puntos, [[w[0],w[1]]], axis=0) #Voy guardando los puntos obtenidos en un array de numpy para luego hacer el gráfico
          
    
    return w, iterations, vector_puntos
#######################################################################################################################################
#################################################### Ejercicio 1.3 apartado a)#########################################################


eta = 0.01 
maxIter = 50
initial_point = np.array([-1.0,1.0])
w, it, vector_puntos = gradient_descent2(initial_point, eta,maxIter) #En este caso como no se especifica el error que se desea obtener, hace las 50 iteraciones unicamente

print('Funcion a minimizar: f(x,y)=(x+2)^2 + 2*(y-2)^2 + 2*sin(2*pi*x)*sin(2*pi*y)')
print('Gradiente: [4*pi*sin(2*pi*y)*cos(2*pi*x)+2*(x+2), 4*pi*sin(2*pi*x)*cos(2*pi*y)+4*(y-2)]')
print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas con eta=0.01 : (', w[0], ', ', w[1],')')
print ('valor obtenido: ', f(w[0],w[1]))
################## Representamos un un gráfico los datos ##############################
imagenes=[]
for i in vector_puntos:
    imagenes.append(f(i[0],i[1])) #Calculamos las imágenes de los puntos obtenidos en el gradiente descendente

iteraciones=np.arange(it+1) #HAcemos un array con las 50 iteraciones
plt.plot(iteraciones, imagenes, label='Eta=0.01')
plt.xlabel('Numero de Iteraciones')
plt.ylabel('Valor de la función ')
plt.title('Ejercicio 1.3 apartado a)')
input("\n--- Pulsar tecla para continuar ---\n")

eta = 0.1  #Cambiamos el valor de la tasa de aprendizaje

initial_point = np.array([-1.0,1.0])
w, it, vector_puntos = gradient_descent2(initial_point, eta,maxIter)
print('Funcion a minimizar: f(x,y)=(x+2)^2 + 2*(y-2)^2 + 2*sin(2*pi*x)*sin(2*pi*y)')
print('Gradiente: [4*pi*sin(2*pi*y)*cos(2*pi*x)+2*(x+2), 4*pi*sin(2*pi*x)*cos(2*pi*y)+4*(y-2)]')
print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas con eta=0.1: (', w[0], ', ', w[1],')')
print ('valor obtenido: ', f(w[0],w[1]))

################## Representamos un un gráfico los datos ##############################
imagenes=[]
for i in vector_puntos:
    imagenes.append(f(i[0],i[1]))

iteraciones=np.arange(it+1)
plt.plot(iteraciones, imagenes, label='Eta=0.1')
plt.legend()
plt.show() 
input("\n--- Pulsar tecla para continuar ---\n")

#######################################################################################################################################
#################################################### Ejercicio 1.3 apartado b)#########################################################
eta = 0.01
x=-0.5
y=-0.5 
maxIter = 50
error2get = 1e-14
initial_point = np.array([x,y])
w, it, vector_puntos = gradient_descent2(initial_point, eta,maxIter)
print('Puntos iniciales (x,y)= (',x, ',', y,')')
print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')
print ('valor obtenido: ', f(w[0],w[1]))

#Representamos en un gráfico
imagenes=[]
for i in vector_puntos:
    imagenes.append(f(i[0],i[1]))

iteraciones=np.arange(it+1)
plt.plot(iteraciones, imagenes, label='Eta=0.01 (-0.5,-0.5)')

print("\n--------------------------\n")
x=1
y=1
initial_point = np.array([x,y])
w, it, vector_puntos = gradient_descent2(initial_point, eta,maxIter)
print('Puntos iniciales (x,y)= (',x, ',', y,')')
print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')
print ('valor obtenido: ', f(w[0],w[1]))

#Representamos en un gráfico
imagenes=[]
for i in vector_puntos:
    imagenes.append(f(i[0],i[1]))

iteraciones=np.arange(it+1)
plt.plot(iteraciones, imagenes, label='Eta=0.01 (1,1)')

print("\n--------------------------\n")
x=2.1
y=-2.1
initial_point = np.array([x,y])
w, it, vector_puntos = gradient_descent2(initial_point, eta,maxIter)
print('Puntos iniciales (x,y)= (',x, ',', y,')')
print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')
print ('valor obtenido: ', f(w[0],w[1]))

#Representamos en un gráfico
imagenes=[]
for i in vector_puntos:
    imagenes.append(f(i[0],i[1]))

iteraciones=np.arange(it+1)
plt.plot(iteraciones, imagenes, label='Eta=0.01 (2.1,-2.1)')

print("\n--------------------------\n")
x=-3
y=3
initial_point = np.array([x,y])
w, it, vector_puntos = gradient_descent2(initial_point, eta,maxIter)
print('Puntos iniciales (x,y)= (',x, ',', y,')')
print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')
print ('valor obtenido: ', f(w[0],w[1]))
#Representamos en un gráfico
imagenes=[]
for i in vector_puntos:
    imagenes.append(f(i[0],i[1]))

iteraciones=np.arange(it+1)
plt.plot(iteraciones, imagenes, label='Eta=0.01 (-3,3)')

print("\n--------------------------\n")
x=-2
y=2
initial_point = np.array([x,y])
w, it, vector_puntos = gradient_descent2(initial_point, eta,maxIter)
print('Puntos iniciales (x,y)= (',x, ',', y,')')
print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')
print ('valor obtenido: ', f(w[0],w[1]))
#Representamos en un gráfico
imagenes=[]
for i in vector_puntos:
    imagenes.append(f(i[0],i[1]))

iteraciones=np.arange(it+1)
plt.plot(iteraciones, imagenes, label='Eta=0.01 (-2,2)')
plt.xlabel('Numero de Iteraciones')
plt.ylabel('Valor de la función ')
plt.title('Ejercicio 1.3 apartado b)')
plt.legend()
plt.show()

# Dibujamos el gráfico en 3D
x = np.linspace(-3, 3, 50)
y = np.linspace(-3, 3, 50)
X, Y = np.meshgrid(x, y)
Z = f(X, Y) #E_w([X, Y])
fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_surface(X, Y, Z, edgecolor='none', rstride=1,
                        cstride=1, cmap='jet')
ax.set(title='Ejercicio 1.3. Función sobre la que se calcula el descenso de gradiente')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x,y)')
ax.view_init(0,60)
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")





#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
print('EJERCICIO SOBRE REGRESION LINEAL\n')
print('Ejercicio 2\n')

label5 = 1
label1 = -1

###################################### Funciones a usar en el ejercicio 2 ###########################################################

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
def Err(x,y,w): #Los parámetros son la matriz x de características, el vector y de etiquetas y el vector w de pesos
    N=len(y) #Calculo el número de filas de y
    producto=x.dot(w)
    Err=(1/N)*(np.transpose(producto-y).dot(producto-y))
    return Err.item()
 

# Gradiente Descendente Estocastico
def sgd(x,y,eta,num_iterations,error,tam_Minibatch=1):
    N=len(y) #Numero de filas de X e y
    iterations=0
    Error=1000.0
    w=np.ones(x.shape[1]) #Inicializo w a un vector de unos del tamaño de las columnas de x
    w=w.reshape(-1,1) #Lo transformo en un vector columna
    
    xy=np.c_[x.copy(),y.copy()] #Esta función de numpy concatena dos matrices por columnas cuando el segundo parámetro es un vector columna

    while Error>error and iterations<num_iterations:
        #La manera en que está implementado el algoritmo está inspirado en el algoritmo implementado en la siguiente dirección:
        #https://realpython.com/gradient-descent-algorithm-python
            
        np.random.shuffle(xy) #Mezclo los datos 

        for i in range(0,N,tam_Minibatch): #Recorro lo minibatches
        #Para cada minibatch actualizao el vector de pesos w con los datos del minibatch
            parada= i + tam_Minibatch
            x_mini,y_mini=xy[i:parada, :-1], xy[i:parada,-1:]
            h_x= np.dot(x_mini,w)
            partial_derivative = (2/tam_Minibatch)*np.dot(np.transpose(h_x - y_mini),x_mini) #multiplico el vector fila transpose(h_x-y) por X así consigo la sum de 1 a N de el xnj*(h(xn)-yn) en cada componente del vector patial_derivative
            w=w - eta*np.transpose(partial_derivative)
            
        #Al acabar la actualización de los w incremento el número de iteraciones del bucle while
        iterations=iterations + 1 
        Error= Err(x,y,w)
        
    return w

# Pseudoinversa	
def pseudoinverse(x,y,w):

    pseudoinverse=np.linalg.pinv(np.transpose(x).dot(x))
    X=pseudoinverse.dot(np.transpose(x));
    w=X.dot(y);
    return w

#####################################################################################################################################
############################################# Ejercicio 2.1 #########################################################################
############################################# Usando SGD ############################################################################
# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy')
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy')

y=y.reshape(-1,1) #redimensionamos el vector para convertirlo en un vector columna pongo -1 porque a priori no se cuantas filas saldrán
y_test=y_test.reshape(-1,1) #redimensionamos el vector para convertirlo en un vector columna pongo -1 porque a priori no se cuantas filas saldrán
num_iterations=200
errorerror2get = 1e-14

###################################################################################################################################################
###################################################################################################################################################
'''
eta=np.linspace(0.01,0.5, 10) #Experimento para ver con que valor de eta se obtienen mejores resultados
tam_Minibatch=np.arange(1,100) #Experimento para ver con que tamaño de minibatch se obtienen mejores resultados
Error=1000.
etamejor=0
tam_mejor=0
contador=0
for i in tam_Minibatch:
    print('Iteracion: ',contador)
    for j in eta:
        w=sgd(x,y,j,num_iterations,error2get,i)
        if Err(x,y,w)<Error:
            Error=Err(x,y,w)
            etamejor=j
            tam_mejor=i
    contador+=1
    
#Con este experimento se obtiene que el mejor eta es 0.06444 y el mejor tamaño para el minibatch es de 34
print ('El mejor eta es ', etamejor)
print ('El mejor Tamaño de minibatch es ', tam_mejor)
'''
###################################################################################################################################################
###################################################################################################################################################
eta=0.1
tam_Minibatch=32
w = sgd(x,y,eta,num_iterations,error2get,tam_Minibatch)
print ('Bondad del resultado para grad. descendente estocastico:\n')
print('Uso eta=0.1, error=1e-14 , max_iteraciones=200 y w inicializado a un vector de unos')
print('w final: ', w)
print ("Ein: ", Err(x,y,w))
print ("Eout: ", Err(x_test, y_test, w)) #Comprobamos como de bueno es el ajuste con los datos del Test set

###### DIBUJO NUBE DE PUNTOS DE X ######## Este código está basado en el código de una compañera de prácticas, Celia Arias Martínez
y0 = np.where(y == -1) #capturo los índices de los elementos con -1
y1 = np.where(y == 1) #capturo los índices de los elementos con 1
#x_2 contiene dos arrays, uno en cada componente, el primero tiene los valores de x con etiqueta -1 y la segunda los de etiqueta 1
x_2 = np.array([x[y0[0]],x[y1[0]]])

plt.scatter(x_2[0][:, 1], x_2[0][:, 2],  c = 'blue', label = 'Numero 1') #Dibujamos los puntos con etiqueta 1
plt.scatter(x_2[1][:, 1], x_2[1][:, 2],  c = 'orange', label = 'Numero 5')#Dibujamos los de etiqueta -1
t = np.linspace(0,1, 100) #Tomo 100 valores equiespaciados para evaluarlos en la función obtenida on sgd
plt.plot( t, (-w[0]-w[1]*t)/w[2], c = 'red', label='Recta de regresión') #Para representarlo, despejo x2 de la ecuación y represento la función resultante en 2D
plt.legend();
plt.title("Ejercicio1 SGD")
plt.xlabel('Intensidad')
plt.ylabel('Simetría')
plt.figure()
plt.show()
input("\n--- Pulsar tecla para continuar ---\n")
################################################################# Pseudoinversa #####################################################

print ('\n Bondad del resultado para algoritmo de la pseudoinversa:\n')
w=pseudoinverse(x,y,w)
print('w final: ', w)
print ("Ein: ", Err(x,y,w))
print ("Eout: ", Err(x_test, y_test, w))

###### DIBUJO NUBE DE PUNTOS DE X ######## Mismo procedimiento de antes
y0 = np.where(y == -1)
y1 = np.where(y == 1)
x_2 = np.array([x[y0[0]],x[y1[0]]])

plt.scatter(x_2[0][:, 1], x_2[0][:, 2],  c = 'blue', label = 'Numero 1')
plt.scatter(x_2[1][:, 1], x_2[1][:, 2],  c = 'orange', label = 'Numero 5')
t = np.linspace(0,1, 100)
plt.plot( t, (-w[0]-w[1]*t)/w[2], c = 'red', label='Recta de regresión')
plt.legend();
plt.title("Ejercicio1 Pseudoinversa")
plt.xlabel('Intensidad')
plt.ylabel('Simetría')
plt.figure()
plt.show()
input("\n--- Pulsar tecla para continuar ---\n")


#####################################################################################################################################
############################################# Ejercicio 2.2 #########################################################################
print('Ejercicio 2.2\n')
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

num_iterations=200
errorerror2get = 1e-14
eta=0.01 #Esta vez no hacemos el estudio del apartado anterior para elegir el mejor eta
tam_Minibatch=32 #Esta vez no hacemos el estudio del apartado anterior para elegir el mejor tamaño de minibatch, pongo 32 porque suele dar buenos resultados
############################################# Apartado a) #########################################################################

np.random.seed(2) #Establezco la semilla para que los procesos aleatorios sean reproducibles en cualquier máquina

X=simula_unif(1000,2,1) #Genero mil puntos en el cuadrado [-1,1]x[-1,1]

# Dibujo el Scatter Plot de los puntos generados
plt.scatter(X[:,0],X[:,1], c='purple')
#Los muestro todos juntos
plt.title('Ejercicio2.2 a) Puntos generados')
plt.show();

input("\n--- Pulsar tecla para continuar ---\n")


############################################# Apartado b) #########################################################################
y=[] #Vector de etiquetas
for i in X:
    y.append(f1(i[0],i[1])) #para cada fila de X genero el valor de su etiqueta correspondiente en y usando la función del ejercicio

X_=pd.DataFrame(data=X); #Convierto la matriz X en un Dataframe de Pandas, que es más cómodo de usar 
X_=X_.sample(frac=0.10,random_state=1); #Hacemos que tome un 10% de los datos de forma aleatoria para generar ruido en la muestra

for i in X_.index:
    y[i]=y[i]*-1 #Cambio el signo de esos elementos


plt.scatter(X[:,0],X[:,1], c=y) #Uso el vector y como vector de colores
#Muestro el gráfico
plt.title('Ejercicio2.2 b) coloreo los puntos por etiquetas')

plt.show();
input("\n--- Pulsar tecla para continuar ---\n")


############################################# Apartado c) #########################################################################

#Concatenamos El vector de unos con la matriz X, para ello  usamos np.concatenate especificando que es por columnas (axis=1)
y=np.array(y)
y=y.reshape(-1,1) #convertimos y en un vector columna 
unos=np.ones((X.shape[0],1))
X=np.concatenate((unos,X),axis=1)

w = sgd(X,y,eta,num_iterations,error2get,tam_Minibatch)
print ('Bondad del resultado para grad. descendente estocastico:\n')
print('Uso eta=0.1, error=1e-14 , max_iteraciones=200 y w inicializado a [1. ... 1. 1.]')
print('w final: ', w)
print ("Ein: ", Err(X,y,w))

####################################################################################################################################
#################Para la representación gráfica de este apartado usamos mismo método que el apartado 2.1
y0 = np.where(y == -1)
y1 = np.where(y == 1)
x_2 = np.array([X[y0[0]],X[y1[0]]])

plt.scatter(x_2[0][:, 1], x_2[0][:, 2],  c = 'purple', label = 'Valores con etiqueta 1')
plt.scatter(x_2[1][:, 1], x_2[1][:, 2],  c = 'yellow', label = 'Valores con etiqueta -1')
t = np.linspace(-0.1,0.35, 100)
plt.plot( t, (-w[0]-w[1]*t)/w[2], c = 'red', label='Recta de regresión obtenida')
plt.legend();
plt.title("Ejercicio2.2 apartado c) Recta de regresión")

plt.figure()
plt.show()

w=np.ones(x.shape[1]) #Inicializo w a un vector de unos del tamaño de las columnas de x
w=w.reshape(-1,1) #Lo transformo en un vector columna
w = pseudoinverse(X,y,w)
print ('Bondad del resultado para pseudoinversa:\n')
print('Uso eta=0.1, error=1e-14 , max_iteraciones=200 y w inicializado a [1. ... 1. 1.]')
print('w final: ', w)
print ("Ein: ", Err(X,y,w))

y0 = np.where(y == -1)
y1 = np.where(y == 1)
#Hacemos 3 arrays para separar los indices de las clases 0,1 y 2
x_2 = np.array([X[y0[0]],X[y1[0]]])

plt.scatter(x_2[0][:, 1], x_2[0][:, 2],  c = 'purple', label = 'Valores con etiqueta 1')
plt.scatter(x_2[1][:, 1], x_2[1][:, 2],  c = 'yellow', label = 'Valores con etiqueta -1')
t = np.linspace(-0.1,0.35, 100)
plt.plot( t, (-w[0]-w[1]*t)/w[2], c = 'red', label='Recta de regresión obtenida')
plt.legend();
plt.title("Ejercicio2.2 con la pseudoinversa")

plt.figure()
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")


############################################# Apartado d) #########################################################################
Ein=0
E_out=0

for i in range(1000):
    print ("iteracion: ", i)
    X=simula_unif(1000,2,1)
    y=[]
    for i in X:
        y.append(f1(i[0],i[1]))   #para cada fila de X genero el valor de su etiqueta correspondiente en y usando la función del ejercicio

        
    X_=pd.DataFrame(data=X); #Convierto la matriz X en un Dataframe de Pandas, que es más cómodo de usar 
    X_=X_.sample(frac=0.10,random_state=1); #Hacemos que tome un 10% de los datos de forma aleatoria

    for i in X_.index:
        y[i]=y[i]*-1 #Cambio el signo de esos elementos
    
    y=np.array(y)
    y=y.reshape(-1,1) #convertimos y en un vector columna 
    unos=np.ones((X.shape[0],1))
    X=np.concatenate((unos,X),axis=1)
    w = sgd(X,y,eta,num_iterations,error2get,tam_Minibatch)
    Ein+=Err(X,y,w)
   
    
    ###########################PREPARAMOS EL TEST SET ###########################
    X=simula_unif(1000,2,1) #Generamos 1000 datos nuevos 
    X=np.concatenate((unos,X),axis=1)
    y=[]
    for i in X:
        y.append(f1(i[0],i[1])) #Generamos las etiquetas para los nuevos datos
    X_=pd.DataFrame(data=X); #Convierto la matriz X en un Dataframe de Pandas, que es más cómodo de usar 
    X_=X_.sample(frac=0.10,random_state=1); #Hacemos que tome un 10% de los datos de forma aleatoria

    for i in X_.index:
        y[i]=y[i]*-1 #Cambio el signo de esos elementos
    y=np.array(y)
    y=y.reshape(-1,1) #convertimos y en un vector columna 
    E_out += Err(X,y,w)


print ('Tras mil iteraciones repitiendo el ejemplo anterior:\n')
print ("Ein medio: ", Ein/1000.0)   
print ("Eout medio: ", E_out/1000.0)   
input("\n--- Pulsar tecla para continuar ---\n")

#Ein medio:  0.9258693113953839
#Eout medio:   1.0025869854054248
###################################################################################################################################
###################################################################################################################################
###################################################################################################################################
#################### Mismo experimento con distinto vector de características######################################################

############################################# Apartado a) #########################################################################
np.random.seed(2)

X=simula_unif(1000,2,1)

# Dibujo el Scatter Plot
plt.scatter(X[:,0],X[:,1], c='purple')

#Los muestro todos juntos
plt.title('Ejercicio2.2 a) Puntos generados')
plt.show();
input("\n--- Pulsar tecla para continuar ---\n")


############################################# Apartado b) #########################################################################
y=[]
for i in X:
    y.append(f1(i[0],i[1]))


X_=pd.DataFrame(data=X); #Convierto la matriz X en un Dataframe de Pandas, que es más cómodo de usar 
X_=X_.sample(frac=0.10,random_state=1); #Hacemos que tome un 10% de los datos de forma aleatoria

for i in X_.index:
    y[i]=y[i]*-1 #Cambio el signo de esos elementos


plt.scatter(X[:,0],X[:,1], c=y) #Uso el vector y como vector de colores

#Muestro el gráfico
plt.title('Ejercicio2.2 b) coloreo los puntos por etiquetas')
plt.show()
input("\n--- Pulsar tecla para continuar ---\n")

############################################# Apartado c) #########################################################################
#Concatenamos El vector de unos con la matriz X, para ello  usamos np.concatenate especificando que es por columnas (axis=1)
y=np.array(y)
y=y.reshape(-1,1) #convertimos y en un vector columna 


############### Preparamos la nueva matriz de características#######################################################################
x1x2=X[:,0]*X[:,1] #multiplicación de las dos columnas elemento a elemento
x1x2=x1x2.reshape(-1,1)
x1_cuadrado=X[:,0]*X[:,0] 
x1_cuadrado=x1_cuadrado.reshape(-1,1)
x2_cuadrado=X[:,1]*X[:,1] 
x2_cuadrado=x2_cuadrado.reshape(-1,1)
unos=np.ones((X.shape[0],1))
X=np.concatenate((unos,X,x1x2,x1_cuadrado,x2_cuadrado),axis=1) #Unimos por columnas todo

w = sgd(X,y,eta,num_iterations,error2get,tam_Minibatch)
print ('Bondad del resultado para grad. descendente estocastico:\n')
print('Uso eta=0.1, error=1e-14 , max_iteraciones=200 y w inicializado a [1.  ... 1. 1.]')
print('w final: ', w)
print ("Ein: ", Err(X,y,w)) 
#Ein:  0.5989154650667663

############### Representación gráfica####################################################################
#Procedemos como en 2.1
y0 = np.where(y == -1)
y1 = np.where(y == 1)
x_2 = np.array([X[y0[0]],X[y1[0]]])

#Representamos la nube de puntos
plt.scatter(x_2[0][:, 1], x_2[0][:, 2],  c = 'purple', label = 'Valores con etiqueta 1')
plt.scatter(x_2[1][:, 1], x_2[1][:, 2],  c = 'yellow', label = 'Valores con etiqueta -1')


def h(x,y,w):
    return w[0] + w[1]*x + w[2]*y + w[3]*x*y + w[4]*x*x + w[5]*y*y

#En este caso la función no es lineal, por lo que el método que usaremos será dibujar líneas de contorno
#Como hemos obtenido una función de dos variables generamos 100 valores equiespaciados entre -1 y 1
t1 = np.linspace(-1,1, 100)
t2 = np.linspace(-1,1, 100)

#CReamos una matriz de ceros de dimensión (len(t1), len(t2))
z= np.zeros((len(t1), len(t2)))


for i  in range(len(t1)):
    for j in range(len(t2)):
        z[i,j] = h(t1[i],t2[j],w) #Rellenamos la matriz con el valor de la función evaluada en el elemento i del vector t1 y j del vector t2
        
#Una vez hecho esto debemos pasar como parámetro la matriz z transpuesta        
plt.contour(t1,t2, np.transpose(z),0, label='Función obtenida', linewidths=2) #Finalmente usando la función contour de pyplot dibujamos las líneas de contorno, y especifico que quiero que aparezca solo una
plt.legend();
plt.title("Ejercicio2.2 apartado c) SGD")

plt.figure()
plt.show()

w=np.ones(x.shape[1]) #Inicializo w a un vector de unos del tamaño de las columnas de x
w=w.reshape(-1,1) #Lo transformo en un vector columna
w = pseudoinverse(X,y,w)
print ('Bondad del resultado para pseudoinversa:\n')
print('Uso eta=0.1, error=1e-14 , max_iteraciones=200 y w inicializado a [1. ... 1. 1.]')
print('w final: ', w)
print ("Ein: ", Err(X,y,w))
input("\n--- Pulsar tecla para continuar ---\n")
#Procedemos como en 2.1
y0 = np.where(y == -1)
y1 = np.where(y == 1)
x_2 = np.array([X[y0[0]],X[y1[0]]])

#Representamos la nube de puntos
plt.scatter(x_2[0][:, 1], x_2[0][:, 2],  c = 'purple', label = 'Valores con etiqueta 1')
plt.scatter(x_2[1][:, 1], x_2[1][:, 2],  c = 'yellow', label = 'Valores con etiqueta -1')


def h(x,y,w):
    return w[0] + w[1]*x + w[2]*y + w[3]*x*y + w[4]*x*x + w[5]*y*y

#En este caso la función no es lineal, por lo que el método que usaremos será dibujar líneas de contorno
#Como hemos obtenido una función de dos variables generamos 100 valores equiespaciados entre -1 y 1
t1 = np.linspace(-1,1, 100)
t2 = np.linspace(-1,1, 100)

#CReamos una matriz de ceros de dimensión (len(t1), len(t2))
z= np.zeros((len(t1), len(t2)))


for i  in range(len(t1)):
    for j in range(len(t2)):
        z[i,j] = h(t1[i],t2[j],w) #Rellenamos la matriz con el valor de la función evaluada en el elemento i del vector t1 y j del vector t2
        
#Una vez hecho esto debemos pasar como parámetro la matriz z transpuesta        
plt.contour(t1,t2, np.transpose(z),0, label='Función obtenida', linewidths=2) #Finalmente usando la función contour de pyplot dibujamos las líneas de contorno, y especifico que quiero que aparezca solo una
plt.legend();
plt.title("Ejercicio2.2 apartado c) Pseudoinversa")

plt.figure()
plt.show()
############################################# Apartado d) #########################################################################
##################################Experimento con 1000 iteraciones##################################
Ein=0
E_out=0

for i in range(1000):
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
    w = sgd(X,y,eta,num_iterations,error2get, tam_Minibatch)
    Ein+=Err(X,y,w)
   
    
    ###########################PREPARAMOS EL TEST SET ###########################
    X=simula_unif(1000,2,1) #Generamos 1000 datos nuevos 
    y=[]
    for i in X:
        y.append(f1(i[0],i[1])) #Generamos las etiquetas para los nuevos datos
    x1x2=X[:,0]*X[:,1] #multiplicación de las dos columnas elemento a elemento
    x1x2=x1x2.reshape(-1,1)
    x1_cuadrado=X[:,0]*X[:,0] 
    x1_cuadrado=x1_cuadrado.reshape(-1,1)
    x2_cuadrado=X[:,1]*X[:,1] 
    x2_cuadrado=x2_cuadrado.reshape(-1,1)
    X=np.concatenate((unos,X,x1x2,x1_cuadrado,x2_cuadrado),axis=1) #Unimos por columnas todo
        
    X_=pd.DataFrame(data=X); #Convierto la matriz X en un Dataframe de Pandas, que es más cómodo de usar 
    X_=X_.sample(frac=0.10,random_state=1); #Hacemos que tome un 10% de los datos de forma aleatoria

    for i in X_.index:
        y[i]=y[i]*-1 #Cambio el signo de esos elementos
        
    y=np.array(y)
    y=y.reshape(-1,1) #convertimos y en un vector columna 
    E_out += Err(X,y,w)

print ('Tras mil iteraciones repitiendo el ejemplo anterior:\n')
print ("Ein medio: ", Ein/1000.0)   
print ("Eout medio: ", E_out/1000.0)   
#Ein medio:  0.5791197645316455
#Eout medio:  0.5867079177511026
input("\n--- Pulsar tecla para continuar ---\n")
