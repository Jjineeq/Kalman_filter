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

def DeDvKalman(z):
    global firstRun2
    global A2, Q2, H2, R2
    global X2, P2
    if firstRun2:
        dt = 0.1
        A2 = np.array([[1, dt], [0, 1]])
        H2 = np.array([[1, 0]])
        Q2 = np.array([[1, 0], [0, 3]])
        R2 = np.array([10])

        X2 = np.array([0, 20]).transpose()
        P2 = 5 * np.eye(2)
        firstRun2 = False
    else:
        Xp = A2 @ X2 # Xp : State Variable Prediction
        Pp = A2 @ P2 @ A2.T + Q2 # Error Covariance Prediction

        #K = (Pp @ H.T) @ inv(H@Pp@H.T + R) # K : Kalman Gain
        K = 1/(np.array(Pp[0,0]) + R2) * np.array([Pp[0,0], Pp[1,0]]).transpose()
        K = K[:, np.newaxis] # maintain axis

        X2 = Xp + K@(z - H@Xp) # Update State Variable Estimation
        P2 = Pp - K@H2@Pp # Update Error Covariance Estimation

    pos = X2[0]
    vel = X2[1]

    return pos, vel

X_esti = np.zeros([len(y2), 3])
Z_saved = np.zeros(len(y2))


for i in range(len(y2)):
    if First:
        Count = np.array([0])
        First = False
    else:
        Z = y2[i]
        Xe, Cov, Kg = (Z)
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
plt.plot(z, t1, 'g', label='groundTruth')
plt.plot(Count, y2, 'b.', label='Measurements')
plt.legend(loc='upper right')
plt.ylabel('y')
plt.xlabel('x')
plt.show()