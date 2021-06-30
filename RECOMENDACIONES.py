import pandas as pd

df = pd.read_csv('D:/Anaconda/datasets/ml-100k/u.data.csv',sep='\t',header=None)

print(df.shape)

#CAMBIO NOMBRES A LAS COLUMNAS

df.columns = ['UserId','ItemId','Rating','TimeStamp']

#ANALISIS DE LOS ITEMS

import matplotlib.pyplot as plt

plt.hist(df.Rating)
plt.hist(df.TimeStamp)

print(df.groupby(['Rating'])['UserId'].count())


print(df.groupby(['ItemId'])['ItemId'].count())


plt.hist(df.groupby(['ItemId'])['ItemId'].count())

#REPRESENTACION EN FORMA MATRICIAL
import numpy as np

#VEO CUANTOS USUARIOS HAY EN EL DATASER
n_users = df.UserId.unique().shape[0]

#VEO CUANTAS PELICULAS HAY EN EL DATA
n_items = df.ItemId.unique().shape[0]

ratings = np.zeros([n_users,n_items])

#ARMO UNA MATRIZ RELACIONANDO CADA USUARIO CON CADA PELICULA

for row in df.itertuples():
    ratings[row[1]-1,row[2]-1] = row [3]

#SPARSITY ES EL PORCENTAJE DE DATOS QUE TENEMOS DE LA MATRIZ, EL RESTO ES 0

sparsity = float(len(ratings.nonzero()[0]))

sparsity/=(ratings.shape[0]*ratings.shape[1])
sparsity*=100

#CREO CONJUNTOS DE ENTRENAMIENTO Y VALIDACION

from sklearn.model_selection import train_test_split
ratings_train, ratings_test = train_test_split(ratings, test_size = 0.3, random_state = 42)


#CREO UNA MATRIZ DE SIMILARIDAD DE USUARIOS
#PREDECIR LA VALORACION DESCONOCIDA DE UN ITEM PARA UN USUARIO 
#BASANDOME EN LA SUMA DE TODAS LAS VALORACIONES 
#DEL RESTO DE USUARIOS PARA DICHO ITEM

import sklearn as sk

#HAGO UNA MATRIZ PARA VER QUE TAN SIMILARES SON LOS USUARIOS ENTRE SI
sim_matrix = 1 - sk.metrics.pairwise.cosine_distances(ratings_train)

user_predictions = sim_matrix.dot(ratings_train)/np.array([np.abs(sim_matrix).sum(axis=1)]).T

print(user_predictions)

from sklearn.metrics import mean_squared_error

def get_mse(preds,actuals):
    preds = preds[actuals.nonzero()].flatten()
    actuals = actuals[actuals.nonzero()].flatten()
    return mean_squared_error(preds, actuals)

print(get_mse(user_predictions, ratings_train))
print(get_mse(user_predictions, ratings_test))

#MODELO KNN

from sklearn.neighbors import NearestNeighbors

k = 5

neighbors = NearestNeighbors(k,'cosine')

neighbors.fit(ratings_train)

#CALCULO LOS USUARIOS MAS CERCANOS A CADA UNO Y SU DISTANCIA CON C/U

top_k_distances, top_k_users = neighbors.kneighbors(ratings_train, return_distance=True)

users_predicts_k = np.zeros(ratings_train.shape)

for i in range(ratings_train.shape[0]):
    users_predicts_k[i,:] = top_k_distances[i].T.dot(ratings_train[top_k_users][i]) / np.array([np.abs(top_k_distances[i].T)])
    
