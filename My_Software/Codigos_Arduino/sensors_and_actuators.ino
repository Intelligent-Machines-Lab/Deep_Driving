#include <SparkFunLSM9DS1.h>
#include <Wire.h>
#include "SparkFun_Ublox_Arduino_Library.h"




//pinos necessarios para comunicacao e controle de motores
int direcao1 = 8; //marrom
int direcao2 = 9; //vermelho
int direcao3 = 10; //laranja
int direcao4 = 11; //amarelo
int pwm1 = 7;
int pwm2 = 6;
int int1 = 2;
int int2 = 3;
int pulses_left = 0;
int pulses_right = 0;
int pulses_left_ant = 0;
int pulses_right_ant = 0;
bool move_bool = true;
bool gps_bool = true;
bool imu_bool = true;
int pwm_left = 0;
int pwm_right = 0;
int pwm_left_ant = 0;
int pwm_right_ant = 0;
int dir_left = 1;
int dir_right = 1;

  /*
   * frente LOW/HIGH/LOW/HIGH
   * direcao 1-2 esquerda
   */

//instanciando os sensores
LSM9DS1 imu;
SFE_UBLOX_GPS myGPS;

//variaveis para IMU e GPS
float heading, headingDegrees = 0,contador;
float declinationAngle;
//int mx_bias = -190, my_bias = 4105, mx_ganhoescala = 1104, my_ganhoescala = 1262;
int mx_bias = 479, my_bias = 3440, mx_ganhoescala = 1085, my_ganhoescala = 1069;

float magnetom_x = 0;
float magnetom_y = 0;
int N_CALIB_DATA = 255;
long lastTime = 0;
String mensagem = "";
String resposta = "s,";

float latitude, longitude;


//variaveis para encoder
float d_left,d_right;

void setup() {
  //Iniciando WIRE  
  Wire.begin();
  
  //instanciando pinos necessarios como OUTPUT
  pinMode(direcao1, OUTPUT);
  pinMode(direcao2, OUTPUT);
  pinMode(direcao3, OUTPUT);
  pinMode(direcao4, OUTPUT);
  pinMode(pwm1, OUTPUT);
  pinMode(pwm2, OUTPUT);
  pinMode(int1, INPUT_PULLUP);
  pinMode(int2, INPUT_PULLUP);
  //configurando interrupcoes externas
  attachInterrupt(digitalPinToInterrupt(int1),counter_left,RISING);
  attachInterrupt(digitalPinToInterrupt(int2),counter_right,RISING);
  
  
  Serial.begin(9600);
  Serial1.begin(9600);
  if (gps_bool){
  if (myGPS.begin() == false) //Connect to the Ublox module using Wire port
  {
    Serial.println("Ublox GPS not detected at default I2C address. Please check wiring. Freezing.");
    while (1);
  }
  
  //GPS configuration
  myGPS.setI2COutput(COM_TYPE_UBX); //Set the I2C port to output UBX only (turn off NMEA noise)
  myGPS.setNavigationFrequency(4);
  myGPS.saveConfiguration(); //Save the current settings to flash and BBR
  }
  //Serial.println("Start");

  if (imu_bool){
  imu.begin();

  }
  
  
}  


char buf;
float leit_left = 0;
float leit_right = 0;
int message_length;
int len;
float right;
float left;
bool valid = false;
int command_ant = 0;
int command;
bool done = false;
float odoml_ant = 0;
float odomr_ant = 0;
float auxr = 0;
float auxl = 0;

void loop() {
  leituraCOMP();
  if(Serial1.available()>0){
  while (!done){
    if(Serial1.available()>0){
      buf = char(Serial1.read());
      mensagem = mensagem + buf;
      if (buf == 'e'){
        done = true;
      }
    }    
  }
  }
  done = false;
  message_length = mensagem.length();
  getValue(mensagem,',',0);
  pwm_left = getValue(mensagem, ',', 1).toInt();
  pwm_right = getValue(mensagem, ',',2).toInt();
  if (odomr_ant == encoder(pulses_right - pulses_right_ant)){
    auxr++;
  }else{
    auxr=0;
  }
  
  if (odoml_ant == encoder(pulses_left - pulses_left_ant)){
    auxl++;
  }else{
    auxl=0;
  }
  if (pwm_left > pwm_right){
    pwm_left = pwm_left + 70+auxl;
    pwm_right = pwm_right -70-auxr;
  }
  if (pwm_left < pwm_right){
    pwm_right = pwm_right+70+auxr;
    pwm_left = pwm_left-70-auxl;
  }

  
  len = getValue(mensagem, ',',3).toInt();
  if (message_length > 1){          
  if (len == message_length){
    valid = true;
    resposta = "s,";
    resposta = resposta + "ack" + ",";
    resposta = resposta + String(resposta.length()+ 6) + ",";
    resposta = resposta + "e\r\n";
    Serial1.print(resposta);
    delay(10);
  
    pwm_left_ant = pwm_left;
    pwm_right_ant = pwm_right;
  }else{
    valid = false;
    pwm_left = pwm_left_ant;
    pwm_right = pwm_right_ant;
    resposta = "s,";
    resposta = resposta + "nack" + ",";
    resposta = resposta + String(resposta.length()+ 6) + ",";
    resposta = resposta + "e\r\n";
    Serial1.print(resposta);
    delay(10);
  }
  }else{
    pwm_left = pwm_left_ant;
    pwm_right = pwm_right_ant;
  }
  mensagem = "";  
  
  if (pwm_left != -999){
    move_motors(pwm_left,pwm_right);
  }else{
    calibraMAG();
  }


  odomr_ant = encoder(pulses_left - pulses_left_ant);
  odoml_ant = encoder(pulses_right - pulses_right_ant);
  resposta = "s,";
  resposta = resposta + String(myGPS.getLatitude()) + ",";
  resposta = resposta + String(myGPS.getLongitude()) + ",";
  resposta = resposta + String(myGPS.getAltitude()) + ",";
  resposta = resposta + String(headingDegrees) + ",";
  resposta = resposta + String(odomr_ant) + ","; 
  resposta = resposta + String(odoml_ant) + ",";
  resposta = resposta + String(resposta.length()+ 6) + ",";
  resposta = resposta + "e\r\n";
  pulses_left_ant = 0;
  pulses_right_ant = 0;
  Serial.print(resposta);

}
      
      
    
    
    



/*************************************
 *************************************
 **  FUNÇÃO DE CALCULAR ENCODER********
 *************************************
 *************************************/
float encoder(int pulses){
    return(2*PI*7.5*(pulses)*11.25/360);
}




/*************************************
 *************************************
 **  FUNÇÃO DE SEPARAR STRINGS********
 *************************************
 *************************************/

String getValue(String data, char separator, int index)
{
    int found = 0;
    int strIndex[] = { 0, -1 };
    int maxIndex = data.length() - 1;

    for (int i = 0; i <= maxIndex && found <= index; i++) {
        if (data.charAt(i) == separator || i == maxIndex) {
            found++;
            strIndex[0] = strIndex[1] + 1;
            strIndex[1] = (i == maxIndex) ? i+1 : i;
        }
    }
    return found > index ? data.substring(strIndex[0], strIndex[1]) : "";
}


/*************************************
 *************************************
 **  FUNÇÃO DE MOVIMENTO DOS MOTORES *
 *************************************
 *************************************/
//dir = 0 means straight foward, dir = 1 means left and dir = 2 means right

void move_motors(int pwm_left, int pwm_right){
 

  if (pwm_left >= 0){
    digitalWrite(direcao1, LOW);
    digitalWrite(direcao2, HIGH);
  }else{
    digitalWrite(direcao1, HIGH);
    digitalWrite(direcao2, LOW);  
    
  }

  if (pwm_right >=  0){
    digitalWrite(direcao3, LOW);
    digitalWrite(direcao4, HIGH);  
    
  }else{
    digitalWrite(direcao3, HIGH);
    digitalWrite(direcao4, LOW);  
    
  }
  analogWrite(abs(pwm1), abs(pwm_left));
  analogWrite(abs(pwm2), abs(pwm_right));
}

/*************************************
 *************************************
 **  FUNÇÃO CALIBRAÇÃO MAGNETÔMETRO **
 *************************************
 *************************************/
void calibraMAG()
{
  unsigned int iter = 0;
  int max_mx, min_mx, max_my, min_my;
  contador = 0;

  // Efetua a leitura dos dados do magnetometro (x,y e z)
  imu.readMag();

  max_mx = imu.mx; min_mx = max_mx;
  max_my = imu.my; min_my = max_my;
  delay(5000);
  Serial.println("Iniciando rotina de calibragem do magnetometro. Por favor gire o veículo em 360 graus.");
  delay(5000);
  // Laço para calcular bias e ganho de escala do magnetômetro
  for (iter = 0; iter < N_CALIB_DATA; iter++)
  {
    Serial.println(iter);
    move_motors(0,0);
    contador = 0;
    //Serial.println(contador);
    imu.readMag();
    magnetom_x = imu.mx;
    magnetom_y = imu.my;

    if (magnetom_x > max_mx) max_mx = magnetom_x;
    if (magnetom_x < min_mx) min_mx = magnetom_x;
    if (magnetom_y > max_my) max_my = magnetom_y;
    if (magnetom_y < min_my) min_my = magnetom_y;
    delay(250);
  }
  move_motors(0,0);
  // Calcula o bias e o ganho de escala das leituras do magnetômetro
  // cálculo do bias
  mx_bias = (max_mx + min_mx) / 2;
  my_bias = (max_my + min_my) / 2;
  // cálculo do ganho de escala
  mx_ganhoescala = (max_mx - min_mx) / 2;
  my_ganhoescala = (max_my - min_my) / 2;
  Serial.print(mx_bias);
  Serial.print(" ");
  Serial.print(my_bias);
  Serial.print(" ");
  Serial.print(mx_ganhoescala);
  Serial.print(" ");
  Serial.println(my_ganhoescala);
  
  Serial.print("Done");
  Serial.print("\r");
  delay(5);
}



/*************************************
 *************************************
 *****    LEITURA DA BÚSSOLA     *****
 *************************************
 *************************************/
void leituraCOMP()
{
  imu.readMag();
  magnetom_x = imu.mx;
  magnetom_y = imu.my;
  magnetom_x = ((magnetom_x - mx_bias) * 100) / mx_ganhoescala; // medidas corrigidas do magnetometro usando bias e ganho de escala
  magnetom_y = ((magnetom_y - my_bias) * 100) / my_ganhoescala; // 100 significa que a escala do magnetometro agora vai de -100 a 100

  //Pequeno processamento dos dados
  heading = atan2(-magnetom_y, -magnetom_x); //devido a orientacao

  //Compensando a declinação magnética
  //Declinação magnética no ITA -22o21'
  declinationAngle = -(22 * PI / 180 + (21 / 60) * PI / 180);
  heading += (declinationAngle);

  // Correção do ângulo de guinada
  if (heading < 0)
    heading += 2 * PI;
  if (heading > 2 * PI)
    heading -= 2 * PI;

  //Conversão para graus e sentido crescente anti-horário
  headingDegrees = 360 - heading * 180 / PI;
  //Serial.println(headingDegrees);

}

unsigned long start_left = millis();
unsigned long start_right = start_left;
//ISR para encoders
void counter_left()
{
  if (millis() - start_left> 20){
    if (dir_left == 1){
        pulses_left++;
    }else if (dir_left == 2){
        pulses_left--;
    }
    start_left = millis();
  }
}

void counter_right()
{
  if (millis() - start_right> 20){
    
    if (dir_right == 1){
        pulses_right++;
    }else if (dir_right == 2){
        pulses_right--;
    }
    start_right = millis();
  }

}
