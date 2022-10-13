import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

prevAvg = 0
k = 1

input_mat = pd.read_csv("C:/Users/User/Desktop/measurmens.csv")
test_mat = pd.read_csv("C:/Users/User/Desktop/groundTruth.csv")

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


for i in y1:
    first = True
    Count = pd.DataFrame([0])
    if first == True:
        a = AvgFilter(y1[i])
        first == False
    else:
        a.append.AvgFilter(y1[i])
        Count.loc[i]=[i]
    


plt.plot(Count, y1, 'b*--', label='Measured')
plt.plot(Count, a, 'ro', label='Average')
plt.legend(loc='upper left')
plt.ylabel('y')
plt.xlabel('x')
plt.show()