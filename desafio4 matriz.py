import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
import pandas as pd
from sklearn.metrics import mean_absolute_error,accuracy_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

dataset = 'https://raw.githubusercontent.com/OptativoPUCV/Fashion-DataSet/master/fashion-1.csv'
df = pd.read_csv(dataset)
df.head()
p=784

X = df.drop(columns=['label'])
X = (X-X.mean())/X.std()
X = X.to_numpy()
Y = df['label'].tolist()
Y2 = np.array(Y)
Y = Y2[:, np.newaxis]

def activation(x):
  return ((1/(1+np.e**(-x))) , (x * (1-x)))

class Capa():
  def __init__(self, n_conexiones: int, n_neuronas: int, activation):
    self.activation = activation
    self.W = np.random.rand(n_conexiones, n_neuronas) * 2 - 1

def crear_red(topologia: list, activation):
  red = []
  #for i in range(len(topologia) - 1):
  for l, capa in enumerate(topologia[:-1]):
    red.append( Capa(topologia[l], topologia[l+1], activation) )
  return red

def forward(red, X):
  out = [(None, X)]
  for l, capa in enumerate(red):
    z = out[-1][1] @ red[l].W # Multiplicación de matrices
    a = red[l].activation(z)[0] # La función de activación retorna el valor activado y el derivado, necesitamos el activado para el forward
    out.append((z, a)) # Guardamos todas las combinaciones para poder usar la misma función en el backpropagation
  return out

def coste(Ypred, Yesp):
  return (np.mean((Ypred - Yesp) ** 2), (Ypred - Yesp))

def train(red, X, Y, coste, learning_rate=0.001):
  # forward 
  out = forward(red, X)

  # backward pass
  delta = []
  #for i in range(len(red)-1, -1, -1): # recorrer hacie atrás del largo a 0
  for i in reversed(range(0,len(red))):
    z = out[i+1][0]
    a = out[i+1][1]
    if i == len(red)-1:
        #delta última capa
        delta.insert(0, coste(a, Y)[1] * red[i].activation(a)[1] ) # delta 0 = derivada del coste (osea Ypred - Yesp) * derivada de activación de la capa
    else:
        # delta respecto al anterior
        delta.insert(0, delta[0] @ aux_W.T * red[i].activation(a)[1]) # delta n = delta(n+1) x W(n+1).T * derivada de activación de la capa 
    aux_W = red[i].W
    # Descenso del gradiente
    red[i].W = red[i].W - out[i][1].T @ delta[0] * learning_rate # nuevoW[i] = actualW[i] - salida[i].T x delta * learning_rate

  return out[-1][1]

topologia = [p, 320, 80, 30, 1]
red = crear_red(topologia, activation)
loss = []
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=2)
for i in range(100):
  pY = train(red, X_train, Y_train, coste, learning_rate=0.05)
  if i % 25 == 0:
    costo = coste(pY, Y_train)[0]
    print(f'Coste iteración {i}: {costo}')
    loss.append(costo)
    

    
plt.plot(range(len(loss)), loss)
plt.show()

prediccion = forward(red, X_test)[-1][1]

print(mean_absolute_error(Y_test, np.trunc(prediccion)))
print(accuracy_score(Y_test, np.trunc(prediccion)))