# -*- coding: utf-8 -*-
"""
TRABAJO 1. 
Nombre Estudiante: 
"""

import numpy as np
import matplotlib.pyplot as plt

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
    N=1.0 #En este caso N vale 1 porque el vector de etiquetas y solo contiene un elemento que es el 0

    while Err>error and iterations<num_iterations:
       h_x=E(w[0],w[1])
       partial_derivative=gradE(w[0],w[1])
       w=w - (2/N)*(eta*np.transpose(partial_derivative))
       iterations=iterations + 1 
       Err=calc_coste1(w) 
       print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')
       print ('Error: ', Err)
    
        
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
        Err=calc_coste2(w)
        
        
        
def calc_coste1(w):
    Err=(E(w[0],w[1]))**2
    return Err 
    
def calc_coste2(X,y,w): #X tienen que ser np.matrix, y np.array, w np.array
    N=len(y) #Calculo el número de filas de y
    Err=(1/N)*np.transpose(X*w-y)*(X*w-y)
    return Err.item()



eta = 0.01 
maxIter = 10000000000
error2get = 1e-14
initial_point = np.array([1.0,1.0])
w, it = gradient_descent(initial_point, eta,maxIter, error2get)


print('Funcion a minimizar: E(u,v)=(u^3*e^(v-2)-2*v^2*e^(-u))^2')
print('Gradiente: [2*(e^(v-2)*u^3-2*v^2*e^(-u))*(2*v^2*e^(-u)+3*e^(v-2)*u^2), 2*(u^3*e^(v-2)-4*e^(-u)*v)*(u^3*e^(v-2)-2*e^(-u)*v^2)]')
print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')

input("\n--- Pulsar tecla para continuar ---\n")

#Ejercicio 1.2

def f(x,y):
    return  (x+2)**2 + 2*(y-2)**2 + 2*np.sin(2*np.pi*x)*np.sin(2*np.pi*y)

#Derivada parcial de f con respecto a x
def dfx(x,y):
    return 4*np.pi*np.sin(2*np.pi*y)*cos(2*np.pi*x)+2*(x+2)
    
#Derivada parcial de f con respecto a y
def dfy(x,y):
    return 4*np.pi*np.sin(2*np.pi*x)*cos(2*np.pi*y)+4*(y-2)

#Gradiente de f
def gradf(x,y):
    return np.array([dfx(x,y), dfy(x,y)])


# DISPLAY FIGURE
from mpl_toolkits.mplot3d import Axes3D
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

input("\n--- Pulsar tecla para continuar ---\n")

#Seguir haciendo el ejercicio...






'''
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
    return 

# Gradiente Descendente Estocastico
def sgd(?):
    #
    return w

# Pseudoinversa	
def pseudoinverse(?):
    #
    return w


# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy')
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy')


w = sgd(?)
print ('Bondad del resultado para grad. descendente estocastico:\n')
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

def f(x1, x2):
	return sign(?) 

#Seguir haciendo el ejercicio...

'''
