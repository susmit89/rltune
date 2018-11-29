
import pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

with open("actions.pickle", "rb") as fp:
     (x,y) = pickle.load(fp)
len = 10
ind = y.argsort()[-10:][::-1]
y_axis = np.take(y, ind)
x_axis = [x[i] for i in ind]
print y_axis
print x_axis

plt.xlabel('indexes')
plt.ylabel('action count')
plt.xticks(range(10), x_axis, fontsize=7)
plt.bar(np.arange(10), y_axis)
plt.show()
