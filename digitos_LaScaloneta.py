#%%
""" 
    Nombre del grupo: La Scaloneta
    Integrantes: 
    Axel Belbrun
    Mauricio Enrich
    Giovanni Paredes """
    
    #%%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import tree
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from random import randint
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
#%%
#Ejercicio 1
df = pd.read_csv('mnist_desarrollo.csv')

#Cuántas filas hay? (Considerando que "un dato" es una imagen, y cada imagen es una fila)
print(df.count(axis=1))
#Cuántos atributos hay? (Considerando que cada atributo es una columna de nuestro dfframe)
print(df.count(axis=0))
#Hay 10 clases de nuestra variable de interés (1,2,3,4,5,6,7,8,9,0)
#Como cada columna representa a un pìxel, entonces la cantidad de pìxeles por imagen es igual a la
#cantidad de columnas del dataset - 1 (excluyendo la primer columna que representa al dìgito)
cantidad_pixeles = len(df.columns)
pixeles_relevantes = np.zeros(cantidad_pixeles)
#Renombramos la columna que representa al dìgito como "digito":
df.rename(columns = {'5' : 'digito'}, inplace = True)

#Iteramos sobre cada columna del dataset, y hacemos dos cosas: la primera es 
#tomar la cantidad de pixeles relevantes (con intensidad mayor a 100) de la columna, y luego,
#reescalar los datos en un intervalo de 0 a 1. 
 
for pixel in range (1,cantidad_pixeles):
    pixeles_relevantes[pixel] = (df.iloc[:,pixel] > 100).sum()
    df.iloc[:,pixel]=df.iloc[:,pixel]/255
    
plt.plot(range(0,785),pixeles_relevantes)

""" Graficamos entonces la cantidad de elementos con intensidad mayor
a 100 que tiene cada columna, lo que nos indica qué tan importante es, ya que cuando una columna solo tiene
valores màs bajos nos aporta info. Consideramos atributos (o sea,columnas) importantes a las columnas que tienen más 
de 100 elementos distintos de cero"""

                                                                          
#%%
#Ejercicio 2
df_01 = df[df['digito'].isin([0,1])]                                                                                                   

#%%
#Ejercicio 3
df_01['digito'].value_counts()   


#Viendo que tenemos 6742 "1" + 5923 "0", tenemos un total de 12665 muestras, de las cuáles un 53% son "1" y el 47% son "0"

#%%
#Ejercicio 4


cero_arquetipico = df_01[df_01['digito']== 0].mean()
uno_arquetipico = df_01[df_01['digito']== 1].mean()

promedio=np.zeros(len(cero_arquetipico))
promedio[0]=cero_arquetipico[0]
for pixel in range(1,len(cero_arquetipico)):
    promedio[pixel]=abs(cero_arquetipico[pixel]-uno_arquetipico[pixel])


#Creamos un DataFrame que se queda solo con los píxeles que, en al menos 5000 imágenes, tienen valores
# mayores o iguales a 50. O sea, nos quedamos con los "pìxeles que suelen ser claros"

dfa = pd.DataFrame()
dfa['5'] = df_01['5']

for i in range (1,785):
    if (df_01.iloc[:,i] >= 50).sum() > 5000:
        n = '0.' + str(i)
        dfa[n] = df_01.iloc[:,i]

score_promedio_c0 = np.zeros(20)
score_promedio_s0 = np.zeros(20)


#Probamos knn para distintas cantidades de atributos, tomando solo aquellos que caen dentro de nuestro criterio de elección

for m in range (1,21):
    score = np.zeros(5)
    for i in range (0,5):
        X = pd.DataFrame()
        #Hacemos un slice que devuelve m atributos
        X = dfa.iloc[:,1::(105//m)]
        Y = pd.DataFrame()
        Y['5'] = dfa['5']
        model = KNeighborsClassifier(n_neighbors = 5)
        model.fit(X,Y)
        Y_pred = model.predict(X)
        score[i] = metrics.accuracy_score(Y, Y_pred)
    score_promedio_s0[m-1] = np.sum(score)/len(score)

#Ahora hacemos lo mismo pero tomando valores aleatorios que pueden incluir aquellos píxeles que siempre son 0
for m in range (1,21):
    score = np.zeros(5)
    for i in range (0,5):
        X = pd.DataFrame()
        #Hacemos un slice que devuelve m atributos
        X = df_01.iloc[:,1::(785//m)]
        Y = pd.DataFrame()
        Y['5'] = df_01['5']
        model = KNeighborsClassifier(n_neighbors = 5)
        model.fit(X,Y)
        Y_pred = model.predict(X)
        score[i] = metrics.accuracy_score(Y, Y_pred)
    score_promedio_c0[m-1] = np.sum(score)/len(score)

y = np.arange(1,21)
plt.plot(y,score_promedio_c0)
plt.plot(y,score_promedio_s0)
plt.show()
plt.close()
#%%

#Ejercicio 5


dfa = shuffle(dfa)
y = dfa['5']
X = pd.DataFrame()
cross_validation_n_atributos = []
for t in range(5,20):
    X = dfa.iloc[:,1::(105//t)]

    
    #Ahora, con la cantidad de atributos t, pruebo modelos con r vecinos
    cross_validation_score=np.zeros(20)
    for r in range (5,25):
        score = np.zeros(5)
        m = 0
        for i in range(0,12665,2533):
            x_train = X.iloc[i:(i+2532)]
            x_test = X.drop(x_train.index)
            y_train = y.iloc[i:(i+2532)]
            y_test = y.drop(y_train.index)
            model = KNeighborsClassifier(n_neighbors = r)
            model.fit(x_train,y_train)
            y_pred = model.predict(x_test)
            score[m] = metrics.accuracy_score(y_test,y_pred)
            m = m+1
        cross_validation_score[r-5] = np.sum(score)/len(score)
    cross_validation_n_atributos.append(cross_validation_score)

resultados = []
for array in cross_validation_n_atributos:
    mejor_score = np.max(array)
    mejor_vecinos = np.argmax(array) + 5
    tupla = (mejor_score,mejor_vecinos)
    resultados.append(tupla)
max_score = 0
max_vecinos = 0
mejor_atributos = 0
for i in  range(len(resultados)):
    if max_score < resultados[i][0]:
        max_score = resultados[i][0]
        max_vecinos = resultados[i][1]
        mejor_atributos = i+5
        


#%%
#EJERCICIO 6

exactitud = []
cant_atributos = []
r=5
while r <= 750:
   
    X= pd.DataFrame()
   
    #Decidimos agrupar las columnas mediante la sumatoria de estas mismas
    #Utilizamos un rango de cada 1000 para obtener los indices en los cuales
    #la sumatoria de las columnas valian el valor de t
    valores = []
    t=100
    v=1/r
    while t <=45000:
       valores.append(t)
       t= t + v*45000
    #print(valores)
    def find_closest(array, valor):
        aux = []
        for i in valor:
           idx = np.abs(array - i).argmin()
           aux.append(idx)
        return aux
    #print(find_closest(contador_ceros_dfa, valores))
    indices = find_closest(contador_ceros_dfa, valores)
    #indices = atributos
    for i in range(0,len(indices)):
        k=indices[i]
        X[i]= dfa.iloc[:,k]
   
    Y = df['5']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3) # 70% para train y 30% para test
   
    arbol = tree.DecisionTreeClassifier(criterion = "entropy", max_depth= 10)
    arbol = arbol.fit(X_train,Y_train)
    Y_pred = arbol.predict(X_test)
    exactitud.append(metrics.accuracy_score(Y_test, Y_pred))
    cant_atributos.append(len(indices))
    if r <= 50:
        r = r + 5
    elif r <= 150:
        r = r + 10
    elif r <= 450:
        r = r + 20
    else :
        r = r + 50
    #print("Exactitud del modelo:", metrics.accuracy_score(Y_test, Y_pred))
print(cant_atributos)
print(exactitud)
sns.scatterplot(x = cant_atributos, y = exactitud)
plt.title('grafico de exactitud en base a la cantidad atributos')
plt.xlabel('cantidad de atributos')
plt.ylabel('exactitud')
plt.show()



sns.scatterplot(x = exactitud, y = cant_atributos)
plt.title('grafico de accuracy y atributos')
plt.xlabel('exactitud')
plt.ylabel('cant de atributos')
plt.show()


#nos quedamos con la que tiene aprox 150 atributos

X= pd.DataFrame()
   
    #Decidimos agrupar las columnas mediante la sumatoria de estas mismas
    #Utilizamos un rango de cada 1000 para obtener los indices en los cuales
    #la sumatoria de las columnas valian el valor de t
valores = []
t=100

while t <=44000:
    valores.append(t)
    t = t + (1/100) * 44000
#print(valores)
def find_closest(array, valor):
    aux = []
    for i in valor:
        idx = np.abs(array - i).argmin()
        aux.append(idx)
    return aux
#print(find_closest(contador_ceros_dfa, valores))
indices = find_closest(contador_ceros_dfa, valores)
#indices = atributos
for i in range(0,len(indices)):
    k=indices[i]
    X[i]= dfa.iloc[:,k]
   
Y = df['5']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3) # 70% para train y 30% para test

accuracy_arbol=[]
profundidad_arbol=[]

# aca probamos con distintos tamaño del arbol
i = 4
while i <= 30:
    arbol = tree.DecisionTreeClassifier(criterion = "entropy", max_depth= i)
    arbol = arbol.fit(X_train,Y_train)
    Y_pred = arbol.predict(X_test)
    accuracy_arbol.append(metrics.accuracy_score(Y_test, Y_pred))
    profundidad_arbol.append(i)
    if i<=15:
        i = i + 1
    else:
        i = i + 2
   
sns.scatterplot(x = profundidad_arbol , y = accuracy_arbol)
plt.title('grafico de exactitud en base de la profundidad del arbol')
plt.xlabel('profundidad arbol')
plt.ylabel('exactitud')
plt.show()
    
#%%
# EJERCICIO 7  
# creo una lista para guardar la profundidad del arbol y otra para el promedio por usar cross validation
profundidad_arbol = []
promedio_score = []

#aca hago un while que va de 1 en 1 de 4 a 10 y luego de 2 en 2 hasta llegar a 20 
i = 4
while i <= 20:
    # k fold 5
    k_folds = KFold(n_splits = 5)
    arbol = tree.DecisionTreeClassifier(criterion = "entropy", max_depth= i)
    # X es el data que sale de agarrar 100 atributos por particion usando contador_ceros_dfa
    # Y es la columna de los numeros
    # k_folds en cuantas particiones hago cross validation
    scores = cross_val_score(arbol, X, Y, cv = k_folds)
    # voy juntando en una lista los promedios de hacer cross validation
    promedio_score.append(scores.mean())
    profundidad_arbol.append(i)
    print("Promedio CV Score: ", scores.mean())
    if i < 10:
        i = i + 1
    else :
        i = i + 2
    
#grafico de las listas
sns.scatterplot(x = profundidad_arbol , y = promedio_score)
plt.title('grafico de promedio de cross validation en base de la profundidad del arbol entropy')
plt.xlabel('profundidad arbol')
plt.ylabel('promedio cross validation k = 5')
plt.show()



profundidad_arbol = []
promedio_score = []

#aca hago un while que va de 1 en 1 de 4 a 10 y luego de 2 en 2 hasta llegar a 20 
i = 4
while i <= 20:
    # k fold 5
    k_folds = KFold(n_splits = 5)
    arbol = tree.DecisionTreeClassifier(criterion = "gini", max_depth= i)
    # X es el data que sale de agarrar 100 atributos por particion usando contador_ceros_dfa
    # Y es la columna de los numeros
    # k_folds en cuantas particiones hago cross validation
    scores = cross_val_score(arbol, X, Y, cv = k_folds)
    # voy juntando en una lista los promedios de hacer cross validation
    promedio_score.append(scores.mean())
    profundidad_arbol.append(i)
    print("Promedio CV Score: ", scores.mean())
    if i < 10:
        i = i + 1
    else :
        i = i + 2
    
#grafico de las listas
sns.scatterplot(x = profundidad_arbol , y = promedio_score)
plt.title('grafico de promedio de cross validation en base de la profundidad del arbol gini')
plt.xlabel('profundidad arbol')
plt.ylabel('promedio cross validation k = 5')
plt.show()

# viendo el grafico agarro el arbol de 12 de profundidad



promedio_score = []
cant_atributos = []
r=5
while r <= 750:
   
    X= pd.DataFrame()

    valores = []
    t=100
    v=1/r
    while t <=45000:
       valores.append(t)
       t= t + v*45000
    #print(valores)
    def find_closest(array, valor):
        aux = []
        for i in valor:
           idx = np.abs(array - i).argmin()
           aux.append(idx)
        return aux
    #print(find_closest(contador_ceros_dfa, valores))
    indices = find_closest(contador_ceros_dfa, valores)
    #indices = atributos
    for i in range(0,len(indices)):
        k=indices[i]
        X[i]= dfa.iloc[:,k]
   
    Y = df['5']
    k_folds = KFold(n_splits = 5)
    arbol = tree.DecisionTreeClassifier(criterion = "entropy", max_depth= 12)
    scores = cross_val_score(arbol, X, Y, cv = k_folds)
    promedio_score.append(scores.mean())
    cant_atributos.append(len(indices))
    if r <= 50:
        r = r + 5
    elif r <= 150:
        r = r + 10
    elif r <= 450:
        r = r + 20
    else :
        r = r + 50
    #print("Exactitud del modelo:", metrics.accuracy_score(Y_test, Y_pred))
print(cant_atributos)
print(exactitud)
sns.scatterplot(x = cant_atributos, y = promedio_score)
plt.title('grafico de promedio al utilizar cross validation en base a la cantidad atributos')
plt.xlabel('cantidad de atributos')
plt.ylabel('promedio cross validation kfold = 5')
plt.show()

#%%

#Ejercicio 8:

#KNN:
datos_test = pd.read_csv('/home/axel/Descargas/mnist_test_binario.csv')
model = KNeighborsClassifier(n_neighbors = max_vecinos)
X = pd.DataFrame()
X = dfa.iloc[:,1::(105//mejor_atributos)]
y = dfa['5']
x_test = datos_test[X.columns]
y_test = datos_test['1']  

model.fit(X, y)
y_pred = model.predict(x_test)
print(metrics.accuracy_score(y_test,y_pred))

#Árbol:
datos_test = pd.read_csv('/home/axel/Descargas/mnist_test.csv')
arbol = tree.DecisionTreeClassifier(criterion = "entropy", max_depth= 12)
X = df[:,1::]
y = df['5']
x_test = datos_test.iloc[:,1::]
y_test = datos_test['1']
arbol.fit(X,y)
y_pred = arbol.predict(x_test)
print(metrics.accuracy_score(y_test, y_pred))
