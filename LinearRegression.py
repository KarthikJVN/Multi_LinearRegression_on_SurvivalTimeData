from sklearn import preprocessing, svm, neighbors
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, svm, utils
from scipy.io import loadmat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


ST = [624,46,64,1350,280,10,1024,39,730,136,836,60]

MS = [1.32,0.61,1.89,0.87,1.12,2.76,1.13,1.38,0.96,1.62,1.58,0.60]

Age = [51.0,42.5,54.6,54.1,49.5,55.3,43.4,42.8,58.4,52.0,45.0,64.5]

beta = []

features = np.transpose([Age,MS])
Class = np.transpose(np.log10(ST))

#features are X and Labels are Y
x = np.array(features)

#scaling the data (normalizing)
x = preprocessing.scale(x)

y = np.array(Class)

print(x.shape,y.shape)

#testing and training

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.0)

clf = LinearRegression(n_jobs=-1)
clf.fit(x_train, y_train)
# accuracy = clf.score(x_test, y_test)
# print(accuracy)

beta.append(clf.intercept_)
beta.append(clf.coef_[0])
beta.append(clf.coef_[1])

print('Coefficients: \n', clf.coef_)
print('intercept: \n', clf.intercept_)
#print(beta)
Ymodel = []
for row in features:
    Ymodel.append(beta[0]+(beta[1]*row[0])+(beta[2]*row[1]))

error = y - Ymodel
print('error vector: \n',error)
print(np.var(error))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(Age, MS, np.transpose(Class), c='r', marker='*')

ax.set_xlabel('Age')
ax.set_ylabel('MS')
ax.set_zlabel('ST')
# create x,y
xx, yy = np.meshgrid(range(65), range(5))

# calculate corresponding z
# z = (beta[1] * xx + beta[2] * yy + beta[0])
z = (-0.499 * yy + -0.011 * xx + 3.4836)

# plot the surface

ax.plot_surface(xx, yy, z)
plt.show()
