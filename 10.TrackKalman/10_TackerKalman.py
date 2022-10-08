'''
 Filename: 10_TackerKalman.py
 Created on: April,3, 2021
 Author: dhpark
'''
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import cv2
from skimage.metrics import structural_similarity
np.random.seed(0)

firstRun = True
X, P, A, H, Q, R = 0, 0, 0, 0, 0, 0

def GetBallPos(iimg=0):
    # Read images.
    imageA = cv2.imread('C:/Users/User/github/Kalman_filter/10.TrackKalman/Img/bg.jpg')
    imageB = cv2.imread('C:/Users/User/github/Kalman_filter/10.TrackKalman/Img/{}.jpg'.format(iimg+1))

    # Convert the images to grayscale.
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    # Compute the Structural Similarity Index (SSIM) between the two images,
    # ensuring that the difference image is returned.
    _, diff = structural_similarity(grayA, grayB, full=True)
    diff = (diff * 255).astype('uint8')

    # Threshold the difference image, followed by finding contours to
    # obtain the regions of the two input images that differ
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    M = cv2.moments(contours[0])
    xc = int(M['m10'] / M['m00'])  # center of x as true position.
    yc = int(M['m01'] / M['m00'])  # center of y as true position.

    v = np.random.normal(0, 15)  # v: measurement noise of position.

    xpos_meas = xc + v  # x_pos_meas: measured position in x (observable).
    ypos_meas = yc + v  # y_pos_meas: measured position in y (observable).

    return np.array([xpos_meas, ypos_meas])

def TrackKalman(xm, ym):
    global firstRun
    global A, Q, H, R
    global X, P
    if firstRun:
        dt = 1
        A = np.array([[1,dt,0,0], [0,1,0,0], [0,0,1,dt],[0,0,0,1]])
        H = np.array([[1,0,0,0],[0,0,1,0]])
        Q = np.eye(4)
        R = np.array([[50, 0],[0, 50]])

        X = np.array([0,0,0,0]).transpose()
        P = 100 * np.eye(4)
        firstRun = False

    Xp = A @ X # Xp : State Variable Prediction
    Pp = A @ P @ A.T + Q # Error Covariance Prediction

    K = (Pp @ H.T) @ inv(H@Pp@H.T + R) # K : Kalman Gain

    z = np.array([xm, ym]).transpose()
    X = Xp + K@(z - H@Xp) # Update State Variable Estimation
    P = Pp - K@H@Pp # Update Error Covariance Estimation

    xh = X[0]
    yh = X[2]

    return xh, yh

NoOfImg = 24
Xmsaved = np.zeros([NoOfImg,2])
Xhsaved = np.zeros([NoOfImg,2])

for k in range(NoOfImg):
    xm, ym = GetBallPos(k)
    xh, yh = TrackKalman(xm, ym)

    Xmsaved[k] = [xm, ym]
    Xhsaved[k] = [xh, yh]

plt.figure()
plt.plot(Xmsaved[:,0], Xmsaved[:,1], '*', label='Measured')
plt.plot(Xhsaved[:,0], Xhsaved[:,1], 's', label='Kalman Filter')
plt.legend(loc='upper left')
plt.ylabel('Vertical [pixel]')
plt.xlabel('Horizontal [pixel]')
plt.ylim([0, 250])
plt.xlim([0, 350])
plt.gca().invert_yaxis()
plt.show()