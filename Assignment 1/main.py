import pandas as pd
import numpy as np
import matplotlib.cm as cm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import pandas as pd
import random


class Data:

  def __init__(self, X, Y, Type):
    self.X = X
    # self.X[:,12] = X[:,12]/8
    self.Y = Y
    self.Type = Type


  def fitPCA(self, dim = 2):
    self.pca = PCA(n_components=dim)
    self.pca.fit(self.X)


  def transformPCA(self):
    self.pcaX = self.pca.transform(self.X)
    print(self.pcaX.shape)
    return self.pcaX


  def distanceMatrix(self, weights):
    n = self.X.shape[0]
    D = np.zeros((n,n))
    Xweighted = self.X * weights

    for i in range(n):
      for j in range(i,n):
        D[i,j] = np.linalg.norm(Xweighted[i,:]-Xweighted[j,:]) ** 2
        D[j,i] = D[i,j]

    self.D = D


  def similarityMatrix(self, D):
    n = self.X.shape[0]
    ones = np.ones(D.shape)
    rowMean = (1/n) * np.dot(D, ones)
    colMean = (1/n) * np.dot(ones, D)
    totMean = (1/n**2) * np.dot(ones, np.dot(colMean, ones))

    self.S = -1/2 * (D - rowMean - colMean + totMean)


  def eigenDecomp(self):
    eigenValues = np.real(np.linalg.eig(self.S)[0])
    eigenVectors = np.real(np.linalg.eig(self.S)[1])
    indices = np.argsort(-eigenValues)

    self.eigenValues = eigenValues[indices]
    self.eigenVectors = eigenVectors[:,indices]


  def MDS(self):
    k = 2
    self.Lambda = np.diag(np.sqrt(np.real(self.eigenValues[0:k])))
    self.U_k = np.real(self.eigenVectors[:,0:k])
    self.mdsX = np.dot(self.Lambda, np.transpose(self.U_k))


def runPCA(data, dim):

  data.fitPCA(dim)

  data.transformPCA()

  print("PCA | " " Dim = " + str(dim))

  plotPCA(data)


def runMDS(data, mode = "noWeights"):
  weights = getWeights(data, mode)

  data.distanceMatrix(weights)

  data.similarityMatrix(data.D)

  data.eigenDecomp()

  data.MDS()

  print("MDS | " + mode)

  plotMDS(data)


def plot2D(data, X):
  offset = [0.002,0.002]

  types = np.unique(data.Type)

  colors = cm.rainbow(np.linspace(0, 1, len(types)))

  for (j,type_) in enumerate(types):

    indices = np.where(data.Type==type_)[0]

    plt.scatter(X[indices,0],X[indices,1],linewidths=1,s=25,color=colors[j],marker='o',alpha=0.75)
  
  for i in range(len(data.Y)):
    plt.annotate(data.Y[i] + " : " + str(data.Type[i]), X[i,:]+offset)


  plt.show()


def plotPCA(data):
  plot2D(data, data.pcaX)


def plotMDS(data):
  plot2D(data, np.transpose(data.mdsX))


def getData():
  dataFrame = pd.read_csv('zoo.data', delimiter=",", header=None)
  return Data(dataFrame.values[:,1:17], dataFrame.values[:,0], dataFrame.values[:,17])


def isomap(data, k, mode = "noWeights"):
  nNeighbors = k

  weights = getWeights(data, mode)

  weightedX = data.X * weights

  notNeighbor = 10 ** 15 # Non neighbors get assigned distance 10^15

  graphDistance = kneighbors_graph(weightedX, nNeighbors, mode="distance").toarray() # Distances

  graphNeighbor = notNeighbor*(1-kneighbors_graph(weightedX, nNeighbors).toarray()-np.identity(data.X.shape[0])) # Non neighbors

  graph = graphDistance + graphNeighbor
  
  for k in range(len(graph)):
    for i in range(len(graph)):
      for j in range(len(graph)):
        graph[i,j] = min(graph[i,j], graph[i,k] + graph[k,j])  

  graph = graph ** 2 # Squaring the distances

  data.similarityMatrix(graph)

  data.eigenDecomp()

  data.MDS()

  print("Isomap | " + mode + " | " + str(nNeighbors))

  plotMDS(data)


def entropyGain(data):
  types = np.unique(data.Type)
  n = len(data.Type)
  indices = np.arange(n)

  startEntropy = entropy(data, types, indices)
  entropies = np.zeros(data.X.shape[1])

  for i in range(data.X.shape[1]):
    values = np.unique(data.X[:,i])
    entropies[i] = startEntropy
    for value in values:
      indices = np.where(data.X[:,i]==value)[0]
      entropies[i] -= len(indices)/n * entropy(data, types, indices)

  return entropies


def entropy(data, types, indices):
  n = len(indices)
  p = []

  for i in range(len(types)):
    index = np.where(data.Type[indices]==types[i])[0]
    p.append(len(index)/n)

  entropy = sum([-x * np.log2(max(x, 10 ** (-10))) for x in p])

  return entropy


def getWeights(data, mode = "noWeights"):
  rf = RandomForestClassifier(n_estimators=100)
  y = [str(x) for x in data.Type]

  if mode == "noWeights":
    weights = np.ones(data.X.shape[1])
  else:
    if mode == "entropy":
      weights = entropyGain(data)


    elif mode == "rfImportance":
      rf.fit(data.X, y)
      weights = rf.feature_importances_

    elif mode == "permRfImportance":
      nRepitions = 10
      weights = np.zeros(data.X.shape[1])
      for i in range(nRepitions):
        X_train, X_test, y_train, y_test = train_test_split(data.X, y, test_size=0.3, random_state = i)
        rf.fit(X_train, y_train)
        weights += permImportance(X_test, y_test, rf, accuracy, 10) / nRepitions

    weights = weights / np.linalg.norm(weights)

  return weights


def accuracy(y_pred, y):
  return 100/len(y)* (y == y_pred).sum()


def permImportance(X, y, rf, metric, num_iterations=100):
    baseline_metric=metric(y, rf.predict(X))
    scores=np.zeros(X.shape[1])
    for c in range(X.shape[1]):
        X1=X.copy()
        for i in range(num_iterations):
            temp=X1[c]
            random.seed(i)
            random.shuffle(temp)
            X1[c]=temp
            score=metric(y, rf.predict(X1))
            scores[c] += (baseline_metric-score) / num_iterations
    return scores
  

def printWeights(data, mode, order = True):
  weights = getWeights(data, "mode")
  list_ = np.argsort(-weights)
  if order:
    print(" & ".join([str(x,2) for x in list_]))
  else:
    print(" & ".join([str(round(x,2)) for x in weights]))


def main():

  data = getData()

  runPCA(data, 2)

  for mode in ["noWeights", "entropy", "rfImportance", "permRfImportance"]:
    runMDS(data, mode)

  for k in [28, 32, 40, 60, 80, 100]:
    isomap(data, k)
  
  isomap(data, 38, "rfImportance")

main()