import matplotlib
import argparse
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from scipy.fftpack import fft, ifft, fftfreq, rfft, irfft
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import soundfile as sf


# Umbralización de LP a partir de extracelular, PD a partir de intrcelular. Calcular los intervalos en cada ciclo, realizar regresiones,
# y definir los diferentes modos de sonificación

# Registros de prueba
path1="./Registros/datos_parciales/5.dat"
path2="./Registros/02-Mar-2022/15h37m14s-02-Mar-2022.dat"
path3="./Registros/02-Mar-2022/15h36m02s-02-Mar-2022.dat"
path4="./Registros/02-Mar-2022/15h41m53s-02-Mar-2022.dat"
path5="./Registros/02-Mar-2022/18h48m19s-02-Mar-2022.dat"
path6="./Registros/02-Mar-2022/15h42m45s-02-Mar-2022.dat"

##########################################################
#### Argumentos
##########################################################
# Parseamos los argumentos
parser = argparse.ArgumentParser(description='Sonification')

# División de datos
parser.add_argument('--file_path',
                    type=str,
                    default=path2,
                    help='Path del archivo del registro')
parser.add_argument('--sonification_type',
                    type=int,
                    default=0,
                    help='0 Para tono variable en frecuencia, 1 para tono variable en amplitud y 2 para aviso sonoro')
parser.add_argument('--sonification_version',
                    type=int,
                    default=0,
                    help='0 para variabilidad, 1 invariantes dinámicos')                    
parser.add_argument('--S',
                    type=int,
                    default=4,
                    help='Frontera de decisión para sonification_type == 2 y sonification_version == 1')
parser.add_argument('--interval',
                    type=int,
                    default=0,
                    help='Intervalo a sonificar. 0 = LPPD interval. 1 = LPPD delay. 2 = PDLP interval. 3 = PDLP delay. 4 = BDLP. 5 = BDPD.')                    
parser.add_argument('--Fs', 
                    type = int, 
                    default = 10000,
                    help ='Frecuencia de muestreo a la que han sido tomados los registros')
parser.add_argument('--sound', 
                    type = bool, 
                    default = False,
                    help ='Escuchar el audio')
parser.add_argument('--show_spectrogram', 
                    type = bool, 
                    default = True,
                    help ='Ver el espectograma')
parser.add_argument('--show_intervals', 
                    type = bool, 
                    default = False,
                    help ='Ver los intervalos')
parser.add_argument('--write', 
                    type = bool, 
                    default = False,
                    help ='Escribir archivo .wav con los resultados')
parser.add_argument('--wav_name', 
                    type = str, 
                    default = 'Audio.wav',
                    help ='Nombre del .wav a escribir')                              
parser.add_argument('--umbral', 
                    type = float, 
                    default = 0.8,
                    help ='Umbral de detección de LP')
parser.add_argument('--umbralPD', 
                    type = float, 
                    default = 0.5,
                    help ='Umbral de detección de PD')                                    
args = parser.parse_args()


if args.sonification_type not in [0,1,2]:
    print('Sonificación no válida')
    exit()
if args.interval not in [0,1,2,3,4,5]:
    print('Intervalo no válido')
    exit()
if args.sonification_version not in [0,1]:
    args.sonification_version = 0
    print('No se ha dado una versión válida, se toma 0')

#Cargamos el registro seleccionado y guardamos las señales necesarias.
datos = np.loadtxt(args.file_path, dtype='f')
PD = datos[:,3]
PD = PD - PD.mean()
senal = datos[:,2]
senalIntra = datos[:,4]
t=(datos[:,0]-datos[0,0])/1000
t = np.linspace(t[0], t[-1], num=t.size)
print('Carga de datos completada')
# Definir las variables con las que se operará
if args.file_path == path1:
    umbral = 0.5 # umbral deseado
    umbralPD = 0 # umbral deseado PD
    borrado = 800 # Muestras a borrar a fuerza
    distanciaMax_PD = 1000 # cuantas muestras de distancia tiene que estar el anterior valor para considerar que pertenecen a la misma activación    
elif args.file_path == path2:  
    umbral = 0.5 # umbral deseado
    umbralPD = 0.8 # umbral deseado PD       
    borrado = 1000 # Muestras a borrar a fuerza
    distanciaMax_PD = 1200 # cuantas muestras de distancia tiene que estar el anterior valor para considerar que pertenecen a la misma activación        
else:
    borrado = 1000 # Muestras a borrar a fuerza
    distanciaMax_PD = 1200 # cuantas muestras de distancia tiene que estar el anterior valor para considerar que pertenecen a la misma activación  


# Aplicar filtro paso bajo a la señal PD para que la umbralización tenga mejores resultados

W = fftfreq(PD.size, d=1/args.Fs)
PD_f=rfft(PD)
b,a = signal.butter(2,20, 'low', fs=args.Fs, analog= False)
PD_lowpass = signal.filtfilt(b, a, PD)
PD_lowpass_f=rfft(PD_lowpass)

# Mostrar los cambios introducidos en el filtrado
'''
plt.figure(2)
ax1=plt.subplot(221)
ax1.plot( PD )
ax2=plt.subplot(222)
ax2.plot(W, PD_f )
ax3=plt.subplot(223 , sharex=ax1 )
ax3.plot( PD_lowpass)
ax4=plt.subplot(224,sharex=ax2 )
ax4.plot(W, PD_lowpass_f)
plt.show(block=False)
'''

# Umbralización de la señal extracelular para obtener la LP, la llamaremos LPUmb, y el inicio y fin de las regiones de activación de esta.

margen = 20 # Número de muestras a la izqda y dcha que también contamos como pertenecientes a la activación
distanciaMax = 800 # Muestras de distancia tiene que estar el anterior valor para considerar que pertenecen a la misma activación
distanciaMin = 3000 # Muestras tiene que haber entre dos activaciones LP para considerar que dicha actividad no pertenece a la LP
umbralv=np.zeros(senal.shape)
regionLPv=np.zeros(senal.shape)
regionPDv=np.zeros(senal.shape)
maximos=np.where(senal>umbral)
p=maximos[0][0]
umbralv[p]=1
regionLP=np.empty([1,2])
regionLP[0,:]=p
j=0
for i in maximos[0][1:]:
    umbralv[i-margen:i+margen]=1
    if i-1!=p and i-p <distanciaMax:
        umbralv[p:i]=1 #np.ones(i-p)
    p=i
    if i-maximos[0][j] > distanciaMax:
        regionLP=np.concatenate((regionLP, [[i,i]]))
    try:
        if maximos[0][j+2] - i > distanciaMax:
            regionLP[regionLP.shape[0]-1][1]=i
    except:
        regionLP[regionLP.shape[0]-1][1]=i
    j=j+1
for i in range(np.shape(regionLP)[0]): 
    if i>1 and i < regionLP.shape[0]:
        if regionLP[i][0] - regionLP[i-1][1] < distanciaMin:
            regionLP=np.delete(regionLP, i, 0) 

LPUmb = np.multiply(senal,umbralv)
LPUmbInv = np.multiply(senal, (umbralv-1)*-1)
regionLP=regionLP[:regionLP.shape[0]-1,:]

for i in regionLP:
    regionLPv[int(i[0]):int(i[1])] = 1

# Mostramos las regiones de activación obtenidas de la LP junto a la señal extracelular original.
'''
plt.figure(4)
ax1=plt.subplot(211)
ax1.plot( senal )
ax2=plt.subplot(212, sharex=ax1)
ax2.plot(regionLPv)
plt.show(block = False)  
''' 
# Similarmente al caso de la LP obtenemos los intervalos de activación de PD a partit de la señal intracelular de PD.

umbralvPD = np.zeros(senal.shape)
regionesActivas = np.where(PD_lowpass>umbralPD)
j=0
regionPD=np.empty([0,2])
for i in regionesActivas[0]:   
    umbralvPD[i]=1
    if regionesActivas[0][j-1]!=i-1:  
        regionPD=np.concatenate((regionPD, [[i,i]]))
    try:    
        if regionesActivas[0][j+1]!=i+1 and regionPD[regionPD.shape[0]-1][0]>1:     
            regionPD[regionPD.shape[0]-1][1]=i
    except:
        regionPD[regionPD.shape[0]-1][1]=i
    j=j+1
PDumb = np.multiply(PD_lowpass,umbralvPD)
regionPD=regionPD[:regionPD.shape[0]-1,:]
for p in regionPD:
    regionPDv[int(p[0]):int(p[1])]=1
regionLPv=np.multiply((regionPDv-1)*-1,regionLPv)

# Mostramos las regiones LP y PD obtenidas junto a la señal extracelular
'''
plt.figure(3)
ax1=plt.subplot(311)
ax1.plot(t, senal )
ax2=plt.subplot(312, sharex=ax1)
ax2.plot(t,regionLPv)
ax3=plt.subplot(313, sharex=ax1)
ax3.plot(t, regionPDv)
plt.show(block = True) 
'''
# Mostramos la región de activación PD obtenida junto a la señal LP filtrada paso bajo
  
'''
plt.figure(4)
ax1=plt.subplot(211)
ax1.plot( PD_lowpass )
ax2=plt.subplot(212, sharex=ax1)
ax2.plot(regionPDv)
plt.show(block = False)   
'''

# Calcular los intervalos. Periodo y todo el resto
# .
if regionPD[0,0] < regionLP[0,0]: # Forzar que la primera región sea LP, eliminando la primera PD si esta ocurre antes
    regionPD = np.delete(regionPD, 0, 0)
if regionLP.shape[0] > regionPD.shape[0]:
    regionLP=regionLP[:regionPD.shape[0],:]

periodo = np.empty([regionPD.shape[0]])
regionLP_shifted = regionLP[1:]
periodo = regionLP_shifted[:,0]-regionLP[:-1,0]
LPPD_inter=regionPD[:-1,0]-regionLP[:-1,0]
LPPD_delay=regionPD[:-1,0]-regionLP[:-1,1]
BDLP = regionLP[:-1,1]-regionLP[:-1,0]
BDPD = regionPD[:-1,1]-regionPD[:-1,0]
PDLP_inter=regionLP_shifted[:,0]-regionPD[:-1,0]
PDLP_delay=regionLP_shifted[:,0]-regionPD[:-1,1]

#  Obtenemos estos intervalos en segundos
LPPD_delay_s = LPPD_delay/args.Fs
LPPD_inter_s = LPPD_inter/args.Fs
PDLP_delay_s = PDLP_delay/args.Fs
PDLP_inter_s = PDLP_inter/args.Fs
BDLP_s = BDLP/args.Fs
BDPD_s = BDPD/args.Fs
periodo_s = periodo/args.Fs
#print('Periodo max', periodo.max()/Fs, 'Periodo min', periodo.min()/Fs)


# Apicación de un modelo de regresión lineal sobre los datos. Con esto podemos comprobar la invarianza de estos y calcular la desviación típica de estos.
reg_LPPD_delay = LinearRegression() 
reg_LPPD_delay.fit(periodo_s.reshape(-1,1), (LPPD_delay_s)) 
reg_LPPD_inter = LinearRegression() 
reg_LPPD_inter.fit(periodo_s.reshape(-1,1), (LPPD_inter_s)) 
reg_PDLP_delay = LinearRegression() 
reg_PDLP_delay.fit(periodo_s.reshape(-1,1), (PDLP_delay_s)) 
reg_PDLP_inter = LinearRegression() 
reg_PDLP_inter.fit(periodo_s.reshape(-1,1), (PDLP_inter_s)) 
reg_BDLP = LinearRegression() 
reg_BDLP.fit(periodo_s.reshape(-1,1), (BDLP_s)) 
reg_BDPD = LinearRegression() 
reg_BDPD.fit(periodo_s.reshape(-1,1), (BDPD_s)) 

linea_LPPD_delay = reg_LPPD_delay.predict(periodo_s.reshape(-1,1))
linea_LPPD_inter = reg_LPPD_inter.predict(periodo_s.reshape(-1,1))
linea_PDLP_delay = reg_PDLP_delay.predict(periodo_s.reshape(-1,1))
linea_PDLP_inter = reg_PDLP_inter.predict(periodo_s.reshape(-1,1))
linea_BDLP = reg_BDLP.predict(periodo_s.reshape(-1,1))
linea_BDPD = reg_BDPD.predict(periodo_s.reshape(-1,1))


LPPD_inter_std = (LPPD_inter_s-periodo_s*reg_LPPD_inter.coef_).std()
LPPD_delay_std = (LPPD_delay_s-periodo_s*reg_LPPD_delay.coef_).std()
PDLP_inter_std = (PDLP_inter_s-periodo_s*reg_PDLP_inter.coef_).std()
PDLP_delay_std = (PDLP_delay_s-periodo_s*reg_PDLP_delay.coef_).std()
BDLP_std = (BDLP_s-periodo_s*reg_BDLP.coef_).std()
BDPD_std = (BDPD_s-periodo_s*reg_BDPD.coef_).std()

LPPD_inter_coef = LPPD_inter_s.std()/abs(LPPD_inter_s.mean())*100
LPPD_delay_coef = LPPD_delay_s.std()/abs(LPPD_delay_s.mean())*100
PDLP_inter_coef = PDLP_inter_s.std()/abs(PDLP_inter_s.mean())*100
PDLP_delay_coef = PDLP_delay_s.std()/abs(PDLP_delay_s.mean())*100
BDLP_coef = BDLP_s.std()/abs(BDLP_s.mean())*100
BDPD_coef = BDPD_s.std()/abs(BDPD_s.mean())*100

print('LPPD interval coef =', LPPD_inter_coef)
print('LPPD delay coef =', LPPD_delay_coef)
print('PDLP interval coef =', PDLP_inter_coef)
print('PDLP delay coef =', PDLP_delay_coef)
print('BDLP coef =', BDLP_coef)
print('BDPD coef =', BDPD_coef)

# Mostramos los datos linealizados para comprobar que se ha realizado bien el proceso
'''
plt.figure(6)
plt.plot(periodo_s, LPPD_inter_s-periodo_s*regresion_lineal_inter.coef_,'x', color='red')
plt.ylim([-1, 0])
plt.show(block = False)
plt.figure(7)
plt.plot(periodo_s, LPPD_delay_s-periodo_s*regresion_lineal_delay.coef_,'o', color='black')
plt.ylim([-1, 0])
plt.show(block = False)
'''

# Mostramos los intervalos y la regresión lineal calculada, con los márgenes elegidos según x * sigma
times_sigma = 1
if args.show_intervals:
    plt.figure(11)

    plt.plot(periodo_s, LPPD_inter_s,'x', color='red', label = 'LPPD interval')

    plt.plot(periodo_s, linea_LPPD_inter+LPPD_inter_std*times_sigma, color='gray')
    plt.plot(periodo_s, linea_LPPD_inter, color='red', label = 'Regresión lineal LPPD interval')
    plt.plot(periodo_s, linea_LPPD_inter-LPPD_inter_std*times_sigma, color='gray')

    plt.plot(periodo_s, LPPD_delay_s,'o', color='black', label = 'LPPD delay')

    plt.plot(periodo_s, linea_LPPD_delay+LPPD_delay_std*times_sigma, color='gray')
    plt.plot(periodo_s, linea_LPPD_delay, color='black', label = 'Regresión lineal LPPD delay')
    plt.plot(periodo_s, linea_LPPD_delay-LPPD_delay_std*times_sigma, color='gray', label = 'Desviación típica')

    plt.legend(fontsize=20)
    plt.ylabel('Tiempo (s)', fontsize=20)
    plt.xlabel('Periodo instantáneo (s)', fontsize = 20)
    plt.show(block=False)

    plt.figure(12)

    plt.plot(periodo_s, PDLP_inter_s,'x', color='red')
    plt.plot(periodo_s, linea_PDLP_inter, color='red', label = 'PDLP interval')


    plt.plot(periodo_s, PDLP_delay_s,'o', color='black')
    plt.plot(periodo_s, linea_PDLP_delay, color='black', label = 'PDLP delay')


    plt.plot(periodo_s, BDLP_s,'+', color='blue')
    plt.plot(periodo_s, linea_BDLP, color='blue', label = 'BDLP')


    plt.plot(periodo_s, BDPD_s,'*', color='green')
    plt.plot(periodo_s, linea_BDPD, color='green', label = 'BDPD')

    plt.legend(fontsize=20)
    plt.ylabel('Tiempo (s)', fontsize=20)
    plt.xlabel('Periodo instantáneo (s)', fontsize = 20)
    plt.show(block=False)

    etiquetas = ['LPPD_interval','LPPD_delay','PDLP_interval','PDLP_delay','BDLP','BDPD']
    valores = [LPPD_inter_coef, LPPD_delay_coef, PDLP_inter_coef, PDLP_delay_coef, BDLP_coef, BDPD_coef]
    plt.figure(13)
    plt.bar(etiquetas, valores)
    plt.title('Coeficientes de variación', fontsize = 20)
    plt.show(block = False)
    print('LPPD interval standard deviation', LPPD_inter_std)
    print('LPPD delay standard deviation', LPPD_delay_std)

# Se seleccionan los datos del intervalo seleccionado
if args.interval == 1:
    linea = linea_LPPD_delay
    std = LPPD_delay_std
    intervalo_s = LPPD_delay_s
    intervalo = LPPD_delay
    coef = LPPD_delay_coef
    pend = reg_LPPD_delay.coef_
elif args.interval == 2:
    linea = linea_PDLP_inter
    std = PDLP_inter_std
    intervalo_s = PDLP_inter_s
    intervalo = PDLP_inter  
    coef = PDLP_inter_coef  
    pend = reg_PDLP_inter.coef_
elif args.interval == 3:
    linea = linea_PDLP_delay
    std = PDLP_delay_std
    intervalo_s = PDLP_delay_s
    intervalo = PDLP_delay
    coef = PDLP_delay_coef 
    pend = reg_PDLP_delay.coef_
elif args.interval == 4:
    linea = linea_BDLP
    std = BDLP_std
    intervalo_s = BDLP_s
    intervalo = BDLP
    coef = BDLP_coef
    pend = reg_BDLP.coef_
elif args.interval == 5:
    linea = linea_BDPD
    std = BDPD_std
    intervalo_s = BDPD_s
    intervalo = BDPD
    coef = BDPD_coef
    pend = reg_BDPD.coef_
else: 
    linea = linea_LPPD_inter
    std = LPPD_inter_std
    intervalo_s = LPPD_inter_s
    intervalo = LPPD_inter
    coef = LPPD_inter_coef
    pend = reg_LPPD_inter.coef_

k = abs(pend)
if k > 1:
    k = (periodo_s[2]-periodo_s[1])/(linea[2]-linea[1])
k1 = k - 1
k1 = abs(k1)*4 + 1
k2 = k *4 + 1
k3 = k*3
# Tres maneras de crear una señal sonora a partir de la varianza de los datos
# Primera forma. Se modula un tono en frecuencia.
if args.sonification_type == 0:
    f_tono = 100
    tono=np.zeros(np.shape(senal)[0])

    funcion_atenuante = np.arange(1000,-1,-1)/1000
    funcion_atenuante_2 = np.arange(0,1000,1)/1000
    funcion_atenuante = np.square(funcion_atenuante)
    funcion_atenuante_2 = np.square(funcion_atenuante_2)
    
    #Para reducir variación de los tonos factor_tono = 2, para aumentar = 3
    for i in range(linea.size):
        dur = periodo[i]
        t2 = t[:int(dur)]
        esperado=linea[i]  
        if args.sonification_version == 0:
            f_tono_2 = f_tono * (np.log2((np.abs(intervalo_s[i]-esperado)*np.square(coef))+1)+1)
        else:
            f_tono_2 = f_tono * f_tono * k1 + 10 * np.log2((np.abs(intervalo_s[i]-esperado)/std)+1)

        if f_tono_2>5000:
            f_tono_2=5000
        #print('f calculada =',f_tono_2)
        tono_temp = np.sin(2*np.pi*f_tono_2*t2)
        tono_temp[-1001:]   = tono_temp[-1001:]*funcion_atenuante
        tono_temp[:1000]   = tono_temp[:1000]*funcion_atenuante_2
        tono[int(regionLP[i][0]):int(regionLP[i+1][0])] = tono_temp

# Segunda forma. Modulando la amplitud de un tono de 100 Hz.
elif args.sonification_type == 1:
    f_tono = 220
    funcion_atenuante = np.arange(1000,-1,-1)/1000
    funcion_atenuante_2 = np.arange(0,1000,1)/1000
    funcion_atenuante = np.square(funcion_atenuante)
    funcion_atenuante_2 = np.square(funcion_atenuante_2)    
    t2 = t[:30000]
    tono_fuente=np.sin(2*np.pi*f_tono*t2)
    tono = np.zeros(np.shape(senal)[0])
    amp2 = np.zeros(linea.size)
    for i in range(linea.size):
        dur = regionLP[i+1][0]-regionLP[i][0]    
        esperado=linea[i] 
        if args.sonification_version == 0:
            amp = np.abs(intervalo_s[i]-esperado)*np.square(coef)+1
        else:
            amp2[i] = np.abs(intervalo_s[i]-esperado)/(2*std)
            amp = k2 + amp2[i]
        #print('Amplitud calculada =', amp)
        tono_temp = np.copy(tono_fuente[:int(dur)])
        tono_temp[-1001:]  = tono_temp[-1001:]*funcion_atenuante
        tono_temp[:1000]  = tono_temp[:1000]*funcion_atenuante_2
        tono_temp = tono_temp * amp
        tono[int(regionLP[i][0]):int(regionLP[i+1][0])] = tono_temp
    if args.sonification_version == 0:
        tono = tono = tono /tono.max()
    else:
        tono = tono /(5+ amp2.max())
   
# Tercera forma. Creando un aviso sonoro en forma de tono.
elif args.sonification_type == 2:
    tono=np.zeros(np.shape(senal)[0])
    f_tono=220
    tono_temp_parcial_1= t[:30000]*0
    tono_temp_parcial_2=np.sin(2*np.pi*f_tono*t[:30000])
    funcion_atenuante = np.arange(1500,-1,-1)/1500
    funcion_atenuante_2 = np.arange(0,1000,1)/1000
    number_of_outliers = 0
    if args.sonification_version == 0:
        criterio = np.abs((intervalo_s - linea) * coef )> 0.1
    else:
        criterio = k2 + np.abs(intervalo_s - linea)/std  > args.S
    #print(criterio)
    for i in range(linea.size):
        dur = periodo[i]
        outlier = 0
        #print(np.abs((intervalo_s[i] - linea[i]) * coef ))
        if  criterio[i]:    
            number_of_outliers = number_of_outliers + 1
            tono_temp_atenuado  = np.copy(tono_temp_parcial_2[:int(dur)])
            tono_temp_atenuado[-1501:]   = tono_temp_atenuado[-1501:]*funcion_atenuante
            tono_temp_atenuado[:1000]   = tono_temp_atenuado[:1000]*funcion_atenuante_2
            tono[int(regionLP[i][0]):int(regionLP[i+1][0])] = tono_temp_atenuado[:int(dur)]
        else:
            tono[int(regionLP[i][0]):int(regionLP[i+1][0])] = tono_temp_parcial_1[:int(dur)] 

    #print('Nº de intervalos:',linea.size ,'Nº de intervalos sonificados:',number_of_outliers,'Tasa:', number_of_outliers/(linea.size) * 100,'%' )


#Mostrar LP obtenida por umbralización y PD sacada del registro
'''
plt.figure(6)
ax1=plt.subplot(311)
ax1.plot( senal )
ax2=plt.subplot(312, sharex=ax1)
ax2.plot(LPUmb)
ax3=plt.subplot(313, sharex=ax1)
ax3.plot(PD)
plt.show(block = True)    
'''
#Mostrar umbralización
'''
plt.figure(7)
ax1=plt.subplot(411)
ax1.plot( senal )
ax2=plt.subplot(412, sharex=ax1)
ax2.plot(umbralv)
ax3=plt.subplot(413, sharex=ax1)
ax3.plot(LPUmb)
ax4=plt.subplot(414, sharex=ax1)
ax4.plot(LPUmbInv) 
plt.show(block = False)    
'''


# Mostrar el Audio creado 
'''   
plt.figure(9)
plt.plot(t,tono)
plt.xlabel('Segundos (s)', fontsize = 20)
plt.ylabel('Amplitud', fontsize = 20)
plt.show(block = False)
'''

if args.sound:
    sd.play(tono, args.Fs, blocking = False)

if args.show_spectrogram:
    plt.figure(8)
    plt.specgram(tono, NFFT = 1024, Fs= args.Fs, window=np.hamming(1024), cmap = 'jet')
    plt.ylim([0, 800])
    plt.xlabel('Tiempo (s)', fontsize = 20)
    plt.ylabel('Frecuencia (Hz)',  fontsize = 20)
    cb=plt.colorbar()
    cb.set_label('Amplitud [dB]' ,fontdict={'fontsize': 20})
    plt.show(block = False)	

if args.write:
    sf.write(args.wav_name, tono, args.Fs)

print('Introducir cualquier tecla para finalizar ejecución')
input()
print('Ejecución finalizada')


