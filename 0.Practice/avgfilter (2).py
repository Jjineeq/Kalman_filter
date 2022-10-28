import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

prevAvg = 0
k = 1

input_mat = pd.read_csv("C:/Users/Jang/measurmens.csv")
test_mat = pd.read_csv("C:/Users/Jang/groundTruth.csv")

def AvgFilter(x):
    global k, prevAvg
    alpha = (k-1) / k
    avg = alpha * prevAvg + (1 - alpha)*x
    prevAvg = avg
    k += 1
    return avg

y1 = input_mat.iloc[:,[1]] # y 추출
x1 = input_mat.iloc[:,[0]] # x 추출
y1 = y1.to_numpy()
x1 = x1.to_numpy()
y2 = np.ravel(y1,order = 'c')
x2 = np.ravel(x1,order ='c')


first = True
for i in range(len(y2)):
    if first == True:
        Count = np.array([0])
        a = AvgFilter(y2[i])
        first = False
    else:
        a = np.append(a,np.array(AvgFilter(y2[i])))
        Count = np.append(Count, np.array([i]))
        i +=1
    


plt.plot(Count, y2, 'b--', label='Measured')
plt.plot(Count, a, 'r', label='Average')
plt.legend(loc='upper left')
plt.ylabel('y')
plt.xlabel('x')
plt.show()