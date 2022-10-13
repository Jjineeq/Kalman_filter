import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

input_mat = pd.read_csv("C:/Users/User/Desktop/measurmens.csv")
test_mat = pd.read_csv("C:/Users/User/Desktop/groundTruth.csv")

y1 = input_mat.iloc[:,[1]] 
x1 = input_mat.iloc[:,[0]] 
y1 = y1.to_numpy()
x1 = x1.to_numpy()
y2 = np.ravel(y1,order = 'c')
x2 = np.ravel(x1,order ='c')

First = True
firstRun = True
X, P = 0, 0 
A, H, Q, R = 0, 0, 0, 0


def SimpleKalman(z):
    global firstRun
    global A, Q, H, R
    global X, P
    if firstRun:
        A, Q = 5,20
        H, R = 1,10

        X = 500
        P = 30
        firstRun = False

    Xp = A * X 
    Pp = A * P * A + Q 

    K = (Pp * H) / (H*Pp*H + R) 

    X = Xp + K*(z - H*Xp) 
    P = Pp - K*H*Pp 
    return X, P, K

X_esti = np.zeros([len(y2), 3])
Z_saved = np.zeros(len(y2))


for i in range(len(y2)):
    if First:
        Count = np.array([0])
        First = False
    else:
        Z = y2[i]
        Xe, Cov, Kg = SimpleKalman(Z)
        X_esti[i] = [Xe, Cov, Kg]
        Z_saved[i] = Z
        Count = np.append(Count, np.array([i]))


plt.plot(Count, y2, 'b.', label='Measurements') # real data
plt.plot(Count, X_esti[:,0], 'r', label='Kalman Filter') # 노이즈 제거 안됨
plt.legend(loc='upper right')
plt.ylabel('y')
plt.xlabel('x')
#plt.show()


t = test_mat.iloc[:,[1]] 
q = test_mat.iloc[:,[0]] 
t = t.to_numpy()
q = q.to_numpy()
t1 = np.ravel(t,order ='c')
q1 = np.ravel(q,order = 'c')

second = True
for k in range(len(t1)):
    if second:
        z = np.array([0])
        second = False
    else:
        z = np.append(z, np.array([k]))

plt.figure()
plt.plot(Count, X_esti[:,0], 'r', label='Kalman Filter')
plt.plot(z, t1, 'g.', label='groundTruth')
plt.plot(Count, y2, 'b.', label='Measurements')
plt.legend(loc='upper right')
plt.ylabel('y')
plt.xlabel('x')
plt.show()