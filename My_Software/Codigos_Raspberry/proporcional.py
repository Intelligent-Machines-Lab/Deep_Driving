import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2 as cv
sys.path.append(ros_path)
from multiprocessing import Process
import time
import math
import serial
global move
global lstick
global rstick
global lspeed
global rspeed
import dan_lib
import tensorflow as tf
import numpy as np
from math import sqrt,exp,sin,cos,tan,radians
from scipy.spatial import distance
from imutils.video import WebcamVideoStream

calibrated = False
ser = serial.Serial("/dev/serial0", 9600, timeout=0.1)
ser1 = serial.Serial("/dev/ttyUSB0", 9600, timeout=0.1)

# variavel para controle das threads
comando = True
stop = False
save_images = True
show_images = False
move = False
tol = 1
calibrated = False
ack = False
left = 0
right = 0
#r = 0

def msg(left,right):
    aux = "s," + str(left) + "," + str(right) + ","
    aux = aux + str(len(aux) + 4) + ",e"
    return(aux)


def lat2ret(lat,long,h):
    # variables for WSG84
    a = 6378137
    b = 6356752.314245
    f = (a - b) / a
    e = sqrt(f*(2 - f))
    
    lat = radians(lat)
    long = radians(long)
    
    N = a / (sqrt(1 - (e**2) * (sin(lat)) ** 2))
    x = (N+h)*cos(lat)*cos(long)
    y = (N + h)*cos(lat)*sin(long)
    return x,y



def run_cam():
    try:
          #model_prop = tf.keras.models.load_model('my_model_realteste.h5')
        vs = WebcamVideoStream(src=0).start()
            #cap = cv.VideoCapture(0)
            #cap.set(3,320)
            #cap.set(4,240)
        i = 0
        r = 0
        vmax = 60
        global stop
        global left
        global right
        lower = np.array([0,0,200])
        upper = np.array([179,17,255])
            #print("Antes do while")
        while not stop:
            img = vs.read()
            img = cv.resize(img,(320,240))
            #img = img[0:230,0:320]
            img2 = img.copy()
            #print("before")
            right,left,vb = dan_lib.prop_control(img,vmax)
            #print("after")
            if show_images:
                cv.imshow('cam', img2)
                cv.waitKey(16)
            if save_images:
                cv.imwrite('teste' + '_' + str(i) + '.jpg', img2)
                i = i + 1
    except KeyboardInterrupt:
        stop = True



def run_control():
        # variavel global
    global comando
    global stop
    global calibrated
    global move
    global left
    global right

    init = time.process_time()
    ack = 'nack'
    try:
        while comando:
                if calibrated:
                    if move:
                        aux = msg(left,right)
                    else:
                        aux = msg(0,0)

                    while ack != 'ack':
                            #print("aqui")
                        ser.write(aux.encode('ascii'))
                        ser.flush()
                        mensagem = ser.readline().decode("utf-8")
                        mensagem = str(mensagem)
                        #print(mensagem)
                        #print(aux)
                        if mensagem.find("s", 0, 2) != -1 and mensagem.find("e", len(mensagem) - 3,len(mensagem)) != -1 and len(mensagem) > 1:
                            if mensagem.find("ack", 0, len(mensagem)) != -1 or mensagem.find("nack", 0, len(mensagem)) != -1:
                                list = mensagem.split(',')
                                resp_aux = list[1]
                                checksum = list[2]
                                if checksum == str(len(mensagem)):
                                    ack = str(resp_aux)
                        time.sleep(0.15)
                    ack = 'nack'
                    init = time.process_time()
                if comando == False:
                    vs.stop()
                    print("Finalizando thread de controle")
                    break

    except:
            # writeNumber([200])
        stop = True
        print("Saindo por emergencia")

#cam = Cam()
#cam.start()

#time.sleep(500000)
#controle = Control()
#controle.start()
if __name__=='__main__':
    p = Process(target=run_control)
    p1 = Process(target=run_cam)
    p.start()
    print("main")
    p1.start()


    r = 0
    n = 0
#Kalman Filter variables
    width = 0.44
    radius = 0.075
    probMatrix = np.zeros((3,3))
    np.fill_diagonal(probMatrix,np.array([[25],[25],[25]]))
    Q = 0.05
    Rgps = np.array([[sqrt(0.07)], [sqrt(0.05)], [sqrt(0.1)]])
    R = np.zeros((3,3))
    np.fill_diagonal(R,Rgps)
    xg,yg = lat2ret(23,50,600)
    thetag = math.radians(90)
    final_position = np.array([[xg],[yg],[thetag]])
    sec_leg = False
    gps_r = np.array([[],[],[]])
    odom_r = np.array([[],[]])
    comando = True
    calibrated = False
    move = False
    try:
        i = 0
        q = input("Digite CAL para calibracao:")
        if q == 'CAL':
            aux = msg(-999,-999)
            while(ack != 'ack'):
                mensagem = ser.readline().decode("utf-8")
                mensagem = str(mensagem)
                if mensagem.find("s", 0, 2) != -1 and mensagem.find("e", len(mensagem) - 3, len(mensagem)) != -1:
                    if mensagem.find("ack", 0, len(mensagem)) != -1 or mensagem.find("nack", 0, len(mensagem)) != -1:
                        list = mensagem.split(',')
                        resp_aux = list[1]
                        checksum = list[2]
                        if checksum == str(len(mensagem)):
                            ack = resp_aux

        calibrated = True
        move = True
        valid = False
        while valid != True:
            mensagem = ser1.readline().decode("utf-8")
            print(mensagem)
            if (mensagem.find("s",0,2) != -1 and mensagem.find("e",len(mensagem)-3,len(mensagem)) != -1):
                list = str(mensagem).split(',')
            #print(list)
                lat = float(list[1]) / 10000000
                long = float(list[2]) / 10000000
                h = float(list[3]) / 1000
                theta = float(list[4])
                dleft_ant = float(list[5]) / 100
                dright_ant = float(list[6]) / 100
                checksum = list[7]
                if checksum == str(len(mensagem)):
                    valid = True
        theta_ant = theta
        x0,y0 = lat2ret(lat,long,h)
        theta0 = math.radians(theta) - math.pi / 2
        valid = False
        car_EKF = dan_lib.EKF(vehicleWidth=width, initialStates = np.array([[x0], [y0],[ math.radians(theta) - math.pi / 2]]), probMatrix=probMatrix, Q=Q)
        start = time.process_time()
        vmax = 70

        while not stop:
            #print("principal")
            now = time.process_time()
            if now - start > 0.25:
                while valid != True:
                    mensagem = ser1.readline().decode("utf-8")
                    mensagem = str(mensagem)
                    print(mensagem)
                    if (mensagem.find("s", 0, 2) != -1 and mensagem.find("e", len(mensagem) - 3, len(mensagem)) != -1):
                        list = str(mensagem).split(',')
                        lat = float(list[1]) / 10000000
                        long = float(list[2]) / 10000000
                        h = float(list[3]) / 1000
                        theta = float(list[4])
                        dleft = float(list[5]) / 100
                        dright = float(list[6]) / 100
                        checksum = list[7]
                        if checksum == str(len(mensagem)):
                            valid = True
                if theta - theta_ant > math.pi/2:
                        n = n - math.copysign(1,theta - theta_ant)
                theta_ant = theta
                theta = theta + 2*n*math.pi
                x,y = lat2ret(lat,long,h)
                gps_r = np.append(gps_r, np.array([[x], [y],[ math.radians(theta) - math.pi / 2]]), axis = 1)
                odom_r = np.append(odom_r, np.array([[dright-dright_ant],[dleft-dleft_ant]]), axis = 1)
                car_EKF.predict(odom=np.array([dright-dright_ant, dleft-dleft_ant]), dT=(now - start))
                car_EKF.update(gps= np.array([[x], [y],[ math.radians(theta) - math.pi / 2]]), R=R, dT=(now - start))
                valid = False
                start = time.process_time()
                dright_ant = dright
                dleft_ant = dleft

    except KeyboardInterrupt:
        stop = True
    
    

    print("Liberando a camera")

    # seta variavel para matar as outras threads
    comando = False
    stop = True
#aux = "," + str(0) + "," + str(0) + "," + "\n"
    aux = msg(0,0)
    ser.write(aux.encode('ascii'))
    time.sleep(0.5)
    ser.write(aux.encode('ascii'))
    time.sleep(0.5)
    ser.write(aux.encode('ascii'))
    time.sleep(0.5)
    ser.write(aux.encode('ascii'))
# libera arquivos e fecha janelas
#cap.release()
    cv.destroyAllWindows()
    print("Fiz tudo que deveria vou esperar as outras threads terminarem")

    print(aux)
# espera threads terminarem
    np.savetxt("state.txt", car_EKF.attributeX, delimiter =",")
    np.savetxt("stddev.txt", car_EKF.stdDev, delimiter = ",")
    np.savetxt("gps.txt", gps_r, delimiter = ",")
    np.savetxt("odom.txt", odom_r,delimiter = ",")
    p.join()
    p1.join()
