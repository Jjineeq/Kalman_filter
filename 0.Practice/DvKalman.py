import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy.linalg import inv

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


def DvKalman(z):
    global firstRun
    global A, Q, H, R
    global X, P
    if firstRun:
        dt = 0.1
        A = np.array([[1, dt], [0, 1]])
        H = np.array([[1, 0]])
        Q = np.array([[1, 0], [0, 3]])
        R = np.array([10])

        X = np.array([0, 20]).transpose()
        P = 5 * np.eye(2)
        firstRun = False
    else:
        Xp = A @ X # Xp : State Variable Prediction
        Pp = A @ P @ A.T + Q # Error Covariance Prediction

        K = (Pp @ H.T) @ inv(H@Pp@H.T + R) # K : Kalman Gain

        X = Xp + K@(z - H@Xp) # Update State Variable Estimation
        P = Pp - K@H@Pp # Update Error Covariance Estimation

    pos = X[0]
    vel = X[1]

    return pos, vel

X_esti = np.zeros([len(y2), 3])
Z_saved = np.zeros(len(y2))


for i in range(len(y2)):
    if First:
        Count = np.array([0])
        First = False
    else:
        Z = y2[i]
        z, pos_true = DvKalman(Z)
        X_esti[i] = [pos, vel]
        Z_saved[i] = [pos_ture, vel_true]
        Count = np.append(Count, np.array([i]))


plt.plot(Count, y2, 'b.', label='Measurements') # real data
plt.plot(Count, X_esti[:,0], 'r', label='Kalman Filter') # 노이즈 제거 안됨
plt.legend(loc='upper right')
plt.ylabel('y')
plt.xlabel('x')
plt.show()


#t = test_mat.iloc[:,[1]] 
#q = test_mat.iloc[:,[0]] 
#t = t.to_numpy()
#q = q.to_numpy()
#t1 = np.ravel(t,order ='c')
#q1 = np.ravel(q,order = 'c')

#second = True
#for k in range(len(t1)):
#    if second:
#        z = np.array([0])
#        second = False
#    else:
#        z = np.append(z, np.array([k]))

#plt.figure()
#plt.plot(Count, X_esti[:,0], 'r', label='Kalman Filter')
#plt.plot(z, t1, 'g', label='groundTruth')
#plt.plot(Count, y2, 'b.', label='Measurements')
#plt.legend(loc='upper right')
#plt.ylabel('y')
#plt.xlabel('x')
#plt.show()