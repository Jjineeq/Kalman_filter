import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

xbuf = []
firstRun = True

input_mat = pd.read_csv("C:/Users/User/Desktop/measurmens.csv")
test_mat = pd.read_csv("C:/Users/User/Desktop/groundTruth.csv")

def MovAvgFilter_batch(x):
    global n, xbuf, firstRun
    if firstRun:
        n = 5
        xbuf = x * np.ones(n)
        firstRun = False
    else:
        for i in range(n-1):
            xbuf[i] = xbuf[i+1]
        xbuf[n-1] = x
    avg = np.sum(xbuf) / n
    return avg


y1 = input_mat.iloc[:,[1]] 
x1 = input_mat.iloc[:,[0]] 
y1 = y1.to_numpy()
x1 = x1.to_numpy()
y2 = np.ravel(y1,order = 'c')
x2 = np.ravel(x1,order ='c')


first = True
for i in range(len(y2)):
    if first == True:
        Count = np.array([0])
        a = MovAvgFilter_batch(y2[i])
        first = False
    else:
        a = np.append(a,np.array(MovAvgFilter_batch(y2[i])))
        Count = np.append(Count, np.array([i]))
        i +=1
    


plt.plot(Count, y2, 'b--', label='Measured')
plt.plot(Count, a, 'r', label='Average')
plt.legend(loc='upper left')
plt.ylabel('y')
plt.xlabel('x')
plt.show()