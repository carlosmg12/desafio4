import keras
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from keras.engine import input_spec
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, multilabel_confusion_matrix, precision_score
import seaborn as sns
import matplotlib.pyplot as plt

#Se carga el dataset
dataset = 'https://raw.githubusercontent.com/OptativoPUCV/Fashion-DataSet/master/fashion-1.csv'
df = pd.read_csv(dataset)
df.head()

#Clasificacion de las diferentes tipos de ropas 
Y = pd.get_dummies(df['label'])
X = df.drop(columns=['label'])
X = (X-X.mean())/X.std()

#Dividir los datos en entrenamiento 0.7 y prueba 0.3
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=2)
print(X_train)
print(Y_train)

#Se define la red neuronal con 320 neuronas en la capa de entrada
#4 capas ocultas de 320,160,80,30 neuronas respectivamente 
#10 neuronas de capa de salida
#Se usa relu como funcion de activacion en todas las capas menos en la capa de salida que se usa un softmax
modelo = Sequential()
modelo.add(Dense(320, input_dim=784, activation="relu"))
modelo.add(Dense(160, activation="relu"))
modelo.add(Dense(80, activation="relu"))
modelo.add(Dense(30, activation="relu"))
#Funcion softmax normaliza todos los elementos para que puedan ser operados como probabilidades
modelo.add(Dense(10, activation="softmax"))

#para la perdida se ocupa regresion del error cuadratico medio mse y adam como optimizador de descenso de gradiente estocastico
modelo.compile(loss='mse', optimizer='adam')

#Define variables de regresion lineal
history = modelo.fit(X_train, Y_train, epochs=20, batch_size=400, validation_split=0.2)

#Se muestra por pantalla el progreso del aprendizaje y testeo mediante un grafico
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Función de coste')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#Metricas de Rendimiento del modelo
Y_pred = modelo.predict(X_test)
Y_exp = Y_test.to_numpy()
for i in range(len(Y_pred)):
  Y_pred[i] = np.array([1 if Y_pred[i][j] == np.max(Y_pred[i]) else 0 for j in range(len(Y_pred[i]))])
print(accuracy_score(Y_exp, Y_pred))
print(classification_report(Y_exp, Y_pred))
matriz_confusion = multilabel_confusion_matrix(Y_exp, Y_pred)

#10 tipos diferentes de prendas
clases = ['0', '1', '2','3','4','5','6','7','8','9']

for i, matrix in enumerate(matriz_confusion):
  labels = [f'True Neg\n{matrix[0][0]}',f'False Pos\n{matrix[0][1]}',f'False Neg\n{matrix[1][0]}',f'True Pos\n{matrix[1][1]}']

  labels = np.asarray(labels).reshape(2,2)

  ax = sns.heatmap(matriz_confusion[2], annot=labels, fmt='', cmap='rocket_r')

  ax.set_title(f'Matriz de Confusión para {clases[i]}\n\n');
  ax.set_xlabel('\nPredicted Values')
  ax.set_ylabel('Actual Values ');

  ## Ticket labels - List must be in alphabetical order
  ax.xaxis.set_ticklabels(['False','True'])
  ax.yaxis.set_ticklabels(['False','True'])

  ## Display the visualization of the Confusion Matrix.
  plt.show()