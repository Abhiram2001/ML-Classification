import numpy as np
import csv
X = []
y = []
fa = []
f = open('IRIS.csv')
csv_f  = csv.reader(f)
for Row in csv_f:
    if(Row[0] != "sepal.length"):
        X.append([float(Row[0]),float(Row[1]),float(Row[2]),float(Row[3])])
        if (Row[4] == "Setosa"):
            y.append((1,0,0))
        elif (Row[4] == "Versicolor"):
            y.append((0,1,0))
        else:
            y.append((0,0,1))
f.close()
X = X/np.amax(X, axis=0)
class NeuralNetwork(object):
    def __init__(self):
        self.lrate = 0.13
        self.inputsize = 4
        self.outputsize = 3
        self.h1size = 6
        self.h2size = 4
        self.hd = [0]*(2)
        self.wih1 = np.random.randn(self.inputsize , self.h1size)
        self.wh1h2 = np.random.randn(self.h1size , self.h2size)
        self.wh2o = np.random.randn(self.h2size , self.outputsize)
    def sigmoid(self, s, deriv=False):
        if (deriv == True):
            return s * (1 - s) * self.lrate
        return 1/(1 + np.exp(-s))
    def feedForward(self,X):
        self.hd[0] = self.sigmoid(np.dot(X,self.wih1))
        self.hd[1] = self.sigmoid(np.dot(self.hd[0],self.wh1h2))
        self.output = self.sigmoid(np.dot(self.hd[1],self.wh2o))
        return self.output
    def backWard(self,X,Y,output): 
        self.o_error = y - output
        self.o_delta = self.o_error * self.sigmoid(output, deriv = True)
        self.h2_error = self.o_delta.dot(self.wh2o.T)
        self.h2_delta = self.h2_error * self.sigmoid(self.hd[1], deriv=True)
        self.h1_error = self.h2_delta.dot(self.wh1h2.T)
        self.h1_delta = self.h1_error * self.sigmoid(self.hd[0], deriv=True)
        self.wih1 += X.T.dot(self.h1_delta)
        self.wh1h2 += self.hd[0].T.dot(self.h2_delta)
        self.wh2o += self.hd[1].T.dot(self.o_delta)
    def train(self, X, y):
        output = self.feedForward(X)
        self.backWard(X, y, output)
NN = NeuralNetwork()
for i in range(10000): #trains the NN 1000 times
    NN.train(X, y)
print("Loss: " + str(np.mean(np.square(y - NN.feedForward(X)))))
print("\n")
Z = NN.feedForward(X)
for i in range(0,len(X)):
    Z[i] =  Z[i]/np.amax(Z[i])
for item in Z:
    c =int(len(item))
    for temp in range(c):
        if(item[temp]!= 1.0):
            item[temp] = 0
        else:
            item[temp]= 1
#print(Z)
for item in Z:
    i = ""
    if(item[0] == 1):
        i = "Setosa"
    elif(item[1] == 1):
        i = "Versicolor"
    else:
        i = "Virginica"
    fa.append(i)
size = len(fa)
acc = 0.00000
j=0
f = open('IRIS.csv')
csv_f  = csv.reader(f)
print("predicted  -  Actual")
for Row in csv_f:
    if(Row[0] != "sepal.length"):
        print(fa[j],"  -  ",Row[4])
        if(Row[4]==fa[j]):
            acc+=1
        j+=1
f.close()
acc = (acc/len(X))
for i in range (0,size):
    print(fa[i],y[i])
print(acc)
