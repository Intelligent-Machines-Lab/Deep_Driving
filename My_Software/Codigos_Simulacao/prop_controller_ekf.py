import numpy as np
import cv2
import os
import vrep
import math
import time
import dan_lib
from math import cos, sin, sqrt
import matplotlib.pyplot as plt

'''
Program writen by Daniel based on https://github.com/priya-dwivedi/CarND/tree/master/CarND-Advanced%20Lane%20Finder-P4 to control a simulated vehicle for lane keeping recording its camera's image as a data-base for neural network training.

'''

track = 1

vrep.simxFinish(-1)  # just in case, close all opened connections
clientID = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)  # Connect to V-REP

teste =vrep.simxStartSimulation(clientID, vrep.simx_opmode_oneshot_wait)  # Starting simulation
print(teste)
ErrorCode, steeringLeft = vrep.simxGetObjectHandle(clientID, 'nakedCar_steeringLeft', vrep.simx_opmode_oneshot_wait)
ErrorCode, car = vrep.simxGetObjectHandle(clientID, 'nakedAckermannSteeringCar', vrep.simx_opmode_oneshot_wait)
ErrorCode, steeringRight = vrep.simxGetObjectHandle(clientID, 'nakedCar_steeringRight', vrep.simx_opmode_oneshot_wait)
ErrorCode, motorLeft = vrep.simxGetObjectHandle(clientID, 'nakedCar_motorLeft', vrep.simx_opmode_oneshot_wait)
ErrorCode, motorRight = vrep.simxGetObjectHandle(clientID, 'nakedCar_motorRight', vrep.simx_opmode_oneshot_wait)
ErrorCode, vision_sensor = vrep.simxGetObjectHandle(clientID, 'Vision_sensor', vrep.simx_opmode_oneshot_wait)

desiredWheelRotSpeed = 0

vrep.simxSetJointTargetVelocity(clientID, motorLeft, desiredWheelRotSpeed, vrep.simx_opmode_streaming)
vrep.simxSetJointTargetVelocity(clientID, motorRight, desiredWheelRotSpeed, vrep.simx_opmode_streaming)

# Defining steering angles as constant 0s for speed diferential control instead of steering control
steeringAngleLeft = 0
steeringAngleRight = 0

vrep.simxSetJointTargetPosition(clientID, steeringLeft, steeringAngleLeft, vrep.simx_opmode_streaming)
vrep.simxSetJointTargetPosition(clientID, steeringRight, steeringAngleRight, vrep.simx_opmode_streaming)

# wait for simulation
time.sleep(2)

res, resolution, image = vrep.simxGetVisionSensorImage(clientID, vision_sensor, 0, vrep.simx_opmode_streaming)
ErrorCode, posL = vrep.simxGetJointPosition(clientID, motorLeft, vrep.simx_opmode_streaming)
ErrorCode, posR = vrep.simxGetJointPosition(clientID, motorRight, vrep.simx_opmode_streaming)
ErrorCode, pos = vrep.simxGetObjectPosition(clientID, car, -1, vrep.simx_opmode_streaming)
ErrorCode, euler = vrep.simxGetObjectOrientation(clientID, car, -1, vrep.simx_opmode_streaming)

vrep.simxSetJointTargetPosition(clientID, motorLeft, 0, vrep.simx_opmode_buffer)
vrep.simxSetJointTargetPosition(clientID, motorRight, 0, vrep.simx_opmode_buffer)
time.sleep(1)

desiredSteeringAngle = 0

j = 0
k = 0
i = 0
save = False


#Variables for EKF
width = 0.44
radius = 0.075
probMatrix = np.zeros((3,3))
np.fill_diagonal(probMatrix,np.array([[25],[25],[25]]))
ErrorCode, pos = vrep.simxGetObjectPosition(clientID, car, -1, vrep.simx_opmode_buffer)
ErrorCode, euler = vrep.simxGetObjectOrientation(clientID, car, -1, vrep.simx_opmode_buffer)
initialStates = np.array([[pos[0]], [pos[1]],[ euler[1] - math.pi / 2]])
Q = 0.001
Rgps = np.array([[sqrt(0.5)], [sqrt(0.5)], [sqrt(0.0001)]])
R = np.zeros((3,3))
np.fill_diagonal(R,Rgps)
car_EKF = dan_lib.EKF(vehicleWidth=width,initialStates= initialStates,probMatrix= probMatrix,Q= Q)
#print(initialStates)
try:
    global cur
    aux = None
    mean_time = []
    vmax = 6
    aux_m = 1
    start = vrep.simxGetLastCmdTime(clientID)
    start_pred = start
    tempo = time.process_time()
    nL = 0
    nR = 0
    still_neg_R = 0
    still_neg_L = 0
    auxR = 0
    auxL = 0
    iniR = 0
    iniL = 0
    index_pred = 0

    while (True):

        # Read sensor values for prior analysis
        ErrorCode, posR = vrep.simxGetJointPosition(clientID, motorRight, vrep.simx_opmode_buffer)
        ErrorCode, posL = vrep.simxGetJointPosition(clientID, motorLeft, vrep.simx_opmode_buffer)
        ErrorCode, pos = vrep.simxGetObjectPosition(clientID, car, -1, vrep.simx_opmode_buffer)
        ErrorCode, euler = vrep.simxGetObjectOrientation(clientID, car, -1, vrep.simx_opmode_buffer)
        res, resolution, image = vrep.simxGetVisionSensorImage(clientID, vision_sensor, 0, vrep.simx_opmode_buffer)
        aux_gpsx = np.random.normal(loc=0,scale=Rgps[0],size=1)
        aux_gpsy = np.random.normal(loc=0, scale=Rgps[1], size=1)
        aux_gpstheta = np.random.normal(loc=0, scale=Rgps[2], size=1)
        a = pos[0]+aux_gpsx
        b = pos[1]+aux_gpsy
        c = euler[1]+aux_gpstheta

        pos_GPS = np.array([a, b, c]).reshape(3,1)

        r_odom = np.random.normal(loc=0, scale=0.01)
        if posL <-5e-3 and still_neg_L == 0:
            nL=nL+1
            still_neg_L=1
        if posL >0:
            still_neg_L = 0
        posL_total = posL + 2*nL*math.pi + r_odom
        if posR <-5e-3 and still_neg_R == 0:
            nR=nR+1
            still_neg_R=1
        if posR >0:
            still_neg_R = 0
        posR_total = posR + 2*nR*math.pi + r_odom
        deltaR = abs(posR_total - iniR)
        deltaL = abs(posL_total - iniL)

        auxR = auxR + radius*deltaR
        auxL = auxL + radius*deltaL

        iniR = posR_total
        iniL = posL_total

        now = vrep.simxGetLastCmdTime(clientID)

        if now - start >= 250:
            index_pred = index_pred + 1
            car_EKF.predict(odom = np.array([auxR,auxL]), dT = (now - start)/1000)
            start = vrep.simxGetLastCmdTime(clientID)
            auxR = 0
            auxL = 0

            if index_pred == 4:
                car_EKF.update(gps=pos_GPS, R=R, dT= (now - start_pred)/1000)
                start_pred = start
                index_pred = 0

        img = np.array(image, dtype=np.uint8)
        img.resize([480, 640, 3])
        img = np.rot90(img, 2)
        img = np.fliplr(img)
        #tempo = time.process_time()
        undist, sxbinary, s_binary, combined_binary1, warped_im, Minv = dan_lib.lane_detector(img)
        left_fit, left_fit_null, right_fit, right_fit_null, out_img = dan_lib.fit_lines(warped_im, plot=False)
        left_cur, right_cur, center = dan_lib.curvature(left_fit, left_fit_null, right_fit, right_fit_null, warped_im,
                                                        print_data=False)

        lspeed, rspeed, vb = dan_lib.dif_speed(left_cur, right_cur, center, vmax)
        cur = (left_cur + right_cur) / 2
        #mean_time.append(time.process_time() - tempo)
        if lspeed > rspeed:
            if (lspeed != 0):
                razao = rspeed / lspeed
            else:
                razao = 0
        else:
            if (rspeed != 0):
                razao = -lspeed / rspeed
            else:
                razao = 0

        if save == True:
            if k < 2000 and abs(razao) > 0.98 and time.time() - start > 0.3:
                os.chdir('/home/daniel-lmi/treinamento_dcontrol2/')
                k = k + 1
                start = time.time()
                i = i + 1
            elif abs(razao) < 0.98 and abs(razao) > 0.8 and time.time() - start > 0.3:
                os.chdir('/home/daniel-lmi/treinamento_dcontrol2/')
                start = time.time()
                i = i + 1
            elif abs(razao) < 0.8 and time.time() - start > 0.3:
                os.chdir('/home/daniel-lmi/treinamento_dcontrol2/')
                start = time.time()
                i = i + 1

        desiredSteeringAngle = 0

        vrep.simxSetJointTargetVelocity(clientID, motorLeft, int(lspeed), vrep.simx_opmode_streaming)
        vrep.simxSetJointTargetVelocity(clientID, motorRight, int(rspeed), vrep.simx_opmode_streaming)



except KeyboardInterrupt:
    print("Parando a simulacao e liberando arquivos")
    end = vrep.simxGetLastCmdTime(clientID)
    print(end)
    cv2.destroyAllWindows()
    vrep.simxStopSimulation(clientID, vrep.simx_opmode_oneshot)


plt.plot(car_EKF.attributeX)

comando = False
error = 0
if save == True:
    f1.close()
    f.close()
