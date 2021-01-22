import numpy as np
import cv2
import os
import vrep
import math
import time
import dan_lib
from math import cos, sin, sqrt, radians, degrees
import matplotlib.pyplot as plt
import tensorflow as tf

'''
Program writen by Daniel based on https://github.com/priya-dwivedi/CarND/tree/master/CarND-Advanced%20Lane%20Finder-P4 to control a simulated vehicle for lane keeping recording its camera's image as a data-base for neural network training.

'''
plt.rcParams.update({'font.size':16})

track = 1
plot_EKF = True
record = False
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
speedx,speedy,speedz = vrep.simxGetObjectVelocity(clientID,car,vrep.simx_opmode_streaming)
vrep.simxSetJointTargetPosition(clientID, motorLeft, 0, vrep.simx_opmode_buffer)
vrep.simxSetJointTargetPosition(clientID, motorRight, 0, vrep.simx_opmode_buffer)
time.sleep(1)

euler[1] = -euler[1]
euler[2] = -euler[2]
if euler[2] <= 0:
    euler[1] = math.pi - euler[1]
c = euler[1] +math.pi/2
aux_gpsthetaant = c
desiredSteeringAngle = 0

j = 0
k = 0
i = 0
os.chdir('/home/daniel-lmi/')
#model = tf.keras.models.load_model('my_model_manual2.h5')
model = tf.keras.models.load_model('my_model_prop.h5')

#Variables for EKF
width = 0.44
radius = 0.075
probMatrix = np.zeros((3,3))
np.fill_diagonal(probMatrix,np.array([[0.7],[0.7],[0.7]]))
ErrorCode, pos = vrep.simxGetObjectPosition(clientID, car, -1, vrep.simx_opmode_buffer)
ErrorCode, euler = vrep.simxGetObjectOrientation(clientID, car, -1, vrep.simx_opmode_buffer)
initialStates = np.array([[pos[0]], [pos[1]],[ c]])
initialStates_filter = np.array([[pos[0]], [pos[1]],[c]])
Q = 0.7*0.7
Rgps = np.array([[sqrt(0.07)], [sqrt(0.05)], [sqrt(0.0009)]])
#Rgps = np.array([[sqrt(0.7)], [sqrt(0.8)], [sqrt(0.0009)]])
R = np.zeros((3,3))
np.fill_diagonal(R,Rgps)
true_pos = np.array([[],[],[]])
car_EKF = dan_lib.EKF(vehicleWidth=width,initialStates= initialStates_filter,probMatrix= probMatrix,Q= Q)

true_pos = np.append(true_pos,initialStates.reshape(3,1),axis = 1)

#file_real = open("teste_real.txt","a")
#file_estimado = open("teste_estimado.txt","a")
#file_gps = open("teste_gps.txt","a")


try:
    global cur
    aux = None
    mean_time = []
    vmax = 5
    aux_m = 25
    k = 0
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
    #true_pos = np.array([[],[],[]])
    gps_r = np.array([[],[],[]])
    gen_index = 0
    tgps = []
    t_pred=[]
    t_pred.append(0)
    aux_lane = 320
    start_rec = time.process_time()
    ant_left = 0
    ant_right = 0
    vel = np.array([])
    mean_time = np.array([])
    n=0
    while (True):

        # Read sensor values for prior analysis
        ErrorCode, posR = vrep.simxGetJointPosition(clientID, motorRight, vrep.simx_opmode_buffer)
        ErrorCode, posL = vrep.simxGetJointPosition(clientID, motorLeft, vrep.simx_opmode_buffer)
        ErrorCode, pos = vrep.simxGetObjectPosition(clientID, car, -1, vrep.simx_opmode_buffer)
        ErrorCode, euler = vrep.simxGetObjectOrientation(clientID, car, -1, vrep.simx_opmode_buffer)
        speedx,speedy,speedz=vrep.simxGetObjectVelocity(clientID,car,vrep.simx_opmode_buffer)
        print(speedy)
        res, resolution, image = vrep.simxGetVisionSensorImage(clientID, vision_sensor, 0, vrep.simx_opmode_buffer)
        aux_gpsx = np.random.normal(loc=0,scale=0.65,size=1) #0.3
        aux_gpsy = np.random.normal(loc=0, scale=0.45, size=1) #0.3
        aux_gpstheta = np.random.normal(loc=0, scale=0.001, size=1) #0.0001
        
        a = pos[0]+aux_gpsx
        b = pos[1]+aux_gpsy
        euler[1] = -euler[1]
        euler[2] = -euler[2]
        if euler[2] < 0:
            euler[1] = math.pi - euler[1]
        c = euler[1] +math.pi/2 +aux_gpstheta
        
        er_theta = c - aux_gpsthetaant
        if abs(er_theta) > math.pi/2:
            n = n - math.copysign(1,er_theta)
            
                
        aux_gpsthetaant = c
        c = c + n*2*math.pi
        euler[1] = euler[1] + n*2*math.pi
        #print("angle",math.degrees(c[0]), n)
        pos_GPS = np.array([a, b, c]).reshape(3,1)

        r_odom1 = np.random.normal(loc=0, scale=0.047) #0.049
        r_odom2 = np.random.normal(loc=0, scale=0.047) #0.069
        if posL <-5e-3 and still_neg_L == 0:
            nL=nL+1
            still_neg_L=1
        if posL >0:
            still_neg_L = 0
        posL_total = posL + 2*nL*math.pi
        if posR <-5e-3 and still_neg_R == 0:
            nR=nR+1
            still_neg_R=1
        if posR >0:
            still_neg_R = 0
        posR_total = posR + 2*nR*math.pi
        deltaR = abs(posR_total - iniR)
        deltaL = abs(posL_total - iniL)

        auxR = auxR + radius*deltaR
        auxL = auxL + radius*deltaL

        iniR = posR_total
        iniL = posL_total

        now = vrep.simxGetLastCmdTime(clientID)

        if now - start >= 250:
            index_pred = index_pred + 1
            aux = np.array([[pos[0]],[pos[1]],[euler[1]+math.pi/2]])
            true_pos = np.append(true_pos,aux.reshape(3,1),axis = 1)
            car_EKF.predict(odom = np.array([auxR,auxL]), dT = (now - start)/1000)
            start = vrep.simxGetLastCmdTime(clientID)
            auxR = 0
            auxL = 0
            gen_index = gen_index + 0.25
            t_pred.append(gen_index)
            if index_pred == 4:
                car_EKF.update(gps=pos_GPS, R=R, dT= (now - start_pred)/1000)
                start_pred = start
                index_pred = 0
                tgps.append(gen_index)
                #gps_r.append(pos_GPS)
                gps_r = np.append(gps_r, pos_GPS, axis = 1)

        img = np.array(image, dtype=np.uint8)
        img.resize([480, 640, 3])
        img = np.rot90(img, 2)
        img = np.fliplr(img)
        img2 = img
        cv2.imshow('teste', img)
        cv2.waitKey(16)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img = cv2.resize(img,(160,120),cv2.INTER_AREA)


        #img = np.reshape(img, [-1,1,120 , 160, 3]) #lstm
        img = np.reshape(img, [-1,120 , 160, 3]) #cnn
        print("To antes")
        img = img.astype('float32') / 255
        time_cont = time.process_time() 
        with tf.device('/cpu:0'):
            desiredSteeringAngle = model.predict(img)
        delta = desiredSteeringAngle[0][0]
        print("After")
        mean_time = np.append(mean_time, time.process_time()- time_cont)
        #vb = 1
        #vb = 1
        vb = vmax*(1 - math.exp(-0.5*(1-abs(delta))))/(1 - math.exp(-0.5))
        if vb < 1:
            vb = 1

 

        if delta >= 0:
            rspeed = (1-delta)*vb
            lspeed = vb
        elif delta <0:
            rspeed = vb
            lspeed = (1 +delta)*vb
        #print(lspeed,rspeed)
        #cv2.imwrite('teste.jpg',img)
        #cv2.waitKey(10)
        vel = np.append(vel,vb)

        if lspeed > rspeed:
            if (lspeed != 0):
                #if rspeed == 0:
                 #   rspeed = 0.001
                razao = 1-rspeed / lspeed
            else:
                razao = 0
        else:
            if (rspeed != 0):
                #if lspeed == 0:
                #    lspeed = 0.001
                razao = -1 + lspeed / rspeed
            else:
                razao = 0

        #theta_p = (rspeed-lspeed)/width
        if record and time.process_time() - start_rec > 0.1:
            os.chdir('/home/daniel-lmi/propt/')
            cv2.imwrite('look'+str(k)+'r'+str(rspeed)+ 'l'+ str(lspeed)+'.jpg', img2)
            k = k+1
            start_rec = time.process_time()
        desiredSteeringAngle = 0

        vrep.simxSetJointTargetVelocity(clientID, motorLeft, int(lspeed), vrep.simx_opmode_streaming)
        vrep.simxSetJointTargetVelocity(clientID, motorRight, int(rspeed), vrep.simx_opmode_streaming)



except KeyboardInterrupt:
    print("Parando a simulacao e liberando arquivos")
    end = vrep.simxGetLastCmdTime(clientID)
    print(end)
    cv2.destroyAllWindows()
    vrep.simxStopSimulation(clientID, vrep.simx_opmode_oneshot)
print(np.mean(mean_time))
#file_real.write(true_pos)
#file_estimado.write(car_EKF.attributeX)
#file_gps.write(gps_r)

np.savetxt("teste_real.txt", true_pos, delimiter = ',')
np.savetxt("teste_estimado.txt", car_EKF.attributeX, delimiter = ',')
np.savetxt("teste_gps.txt", gps_r,delimiter = ',')
np.savetxt("teste_dpx.txt", car_EKF.stdDev[0, :],delimiter = ',')
np.savetxt("teste_dpy.txt", car_EKF.stdDev[1, :],delimiter = ',')
np.savetxt("teste_dptheta.txt", car_EKF.stdDev[2, :],delimiter = ',')
np.savetxt("teste_vel.txt", vel,delimiter = ',')

print(vel)
d = np.array([])
xd = np.array([])
xrd = np.array([])
for i in range(0,len(gps_r[2, :])):
    d = np.append(d, degrees(gps_r[2,i]))

for i in range(0,len(car_EKF.attributeX[2, :])):
    xd = np.append(xd, degrees(car_EKF.attributeX[2, i]))
    xrd = np.append(xrd, degrees(true_pos[2, i]))


if plot_EKF:
    fig, axs = plt.subplots(1,1)

    axs.plot(car_EKF.attributeX[0,0], car_EKF.attributeX[1,0], marker ='d', label = 'inicio', color = 'b',markersize = 15)
    axs.plot(car_EKF.attributeX[0,len(car_EKF.attributeX[1,:])-1], car_EKF.attributeX[1,len(car_EKF.attributeX[1,:])-1], marker ='p', label = 'fim', color = 'r',markersize = 15)
    axs.plot(car_EKF.attributeX[0,:],car_EKF.attributeX[1,:], label = 'Estado estimado')
    axs.plot(true_pos[0,:],true_pos[1,:], label = 'Posicao real')
    axs.scatter(gps_r[0,:],gps_r[1,:],marker = '*', label = 'Leitura GPS', color = 'g')
    axs.grid(True)
    axs.legend()
    axs.set_title('Trajetória do robô')
    axs.set_ylabel('Posição y (m)')
    axs.set_xlabel('Posição x (m)')
    axs.axis('equal')
    print('teste')


    plt.show()

    fig, axs = plt.subplots(1, 1)

    axs.plot(t_pred,car_EKF.stdDev[0, :], color='r', label='Desvio padrão')
    axs.plot(t_pred,-car_EKF.stdDev[0, :], color='r')
    axs.plot(t_pred,car_EKF.attributeX[0, :] - true_pos[0, :], label='Erro de estimação em x')
    axs.grid(True)
    #axs[0].set_title('Erro de estimativa em X (m)')
    #axs[0].set_ylabel('Desvio padrão (m)')
    #axs.set_xlabel('Tempo (s)')
    plt.show()
    #axs.legend()
    fig, axs = plt.subplots(1, 1)

    axs.plot(t_pred,car_EKF.stdDev[1, :], color='r', label = 'Desvio padrão')
    axs.plot(t_pred,-car_EKF.stdDev[1, :], color='r')
    axs.plot(t_pred,car_EKF.attributeX[1, :] - true_pos[1, :],label = 'Erro de estimação em y')
    axs.grid(True)
    #axs[1].set_title('Erro e desvio padrão do estado Y')
    #axs[1].set_ylabel('Erro de estimativa nos estados')
    #axs.set_xlabel('Tempo (s)')
    #axs[1].legend()
    plt.show()

    fig, axs = plt.subplots(1, 1)

    axs.plot(t_pred,car_EKF.stdDev[2, :], color='r', label = 'Desvio padrão')
    axs.plot(t_pred,-car_EKF.stdDev[2, :], color='r')
    axs.plot(t_pred,car_EKF.attributeX[2, :] - true_pos[2, :], label = 'Erro de estimação na orientação')
    axs.grid(True)
    #axs.legend()
    #axs[2].set_xlabel('Tempo (s)')
    #axs[2].set_ylabel('Erro de estimativa de orientação (°)')
    #axs.set_title('Erro e desvio padrão do estado theta')

   
    plt.show()

    fig, axs = plt.subplots(1, 1)

    axs.plot(t_pred,car_EKF.attributeX[0, :], label='Estado estimado')
    axs.plot(t_pred,true_pos[0, :], label='Posicao real')
    axs.scatter(tgps, gps_r[0, :], marker='*', label='Leitura GPS', color='g')
    axs.grid(True)
    #axs.legend()
    #axs.set_title('Estado x')
    ##axs.set_ylabel('Posição (m)')
    #axs.set_xlabel('Tempo (s)')
    #axs.axis('equal')

    plt.show()

    fig, axs = plt.subplots(1, 1)

    axs.plot(t_pred,car_EKF.attributeX[1, :], label='Estado estimado')
    axs.plot(t_pred,true_pos[1, :], label='Posicao real')
    axs.scatter(tgps, gps_r[1, :], marker='*', label='Leitura GPS', color='g')
    axs.grid(True)
    #axs.legend()
    #axs.set_title('Estado y')
    #axs.set_ylabel('Posição (m)')
    #axs.set_xlabel('Tempo (s)')
    #axs.axis('equal')

    plt.show()

    fig, axs = plt.subplots(1, 1)

    axs.plot(t_pred,xd, label='Orientação estimada')
    axs.plot(t_pred,xrd, label='Orientação real')
    axs.scatter(tgps, d, marker='*', label='Leitura Bússola', color='g')
    axs.grid(True)
    #axs.legend()
    #axs.set_title('Estado theta')
    #axs.set_ylabel('Ângulo (graus)')
    #axs.set_xlabel('Tempo (s)')
    #axs.axis('equal')

    plt.show()







comando = False
error = 0
