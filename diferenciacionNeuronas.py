import matplotlib
import argparse
from scipy.fftpack import fft, ifft, fftfreq, rfft, irfft
import scipy.signal.windows as wd
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import soundfile as sf

# En este script realizamos la umbralización del LP. A partir del resto de la señal umbralizamos y recortamos la PD 

# Paths de prueba
path1="./Registros/datos_parciales/5.dat"
path2="./Registros/02-Mar-2022/15h37m14s-02-Mar-2022.dat"
path3="./Registros/02-Mar-2022/15h36m02s-02-Mar-2022.dat"
path4="./Registros/02-Mar-2022/15h41m53s-02-Mar-2022.dat"
path5="./Registros/02-Mar-2022/18h48m19s-02-Mar-2022.dat"



##########################################################
#### Argumentos
##########################################################
parser = argparse.ArgumentParser(description='Sonification')

# Declaración de los argumentos
parser.add_argument('--file_path',
                    type=str,
                    default=path1,
                    help='Path del archivo del registro')
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

#Definición de algunas variables
sd.default.samplerate = args.Fs
sd.default.channels = 2
fontsize=20
fontsize2 = 16

# Cargar señal
datos = np.loadtxt(args.file_path, dtype='f')
senal = datos[:,2]
senalIntra = datos[:,3]
t=(datos[:,0]-datos[0,0])/1000
t = np.linspace(t[0], t[-1], num=t.size)


# Definición de parámetros adaptados a cada señal
if args.file_path == path1:
    umbral = 0.5 # umbral deseado
    umbral_PD = 0.1 # umbral deseado PD
    borrado = 1400 # Muestras a borrar a fuerza
    distanciaMax_PD = 1000 # cuantas muestras de distancia tiene que estar el anterior valor para considerar que pertenecen a la misma activación
elif args.file_path == path2:
    umbral = 0.5 # umbral deseado
    umbral_PD = 0.1 # umbral deseado PD       
    borrado = 1400 # Muestras a borrar a fuerza
    distanciaMax_PD = 1200 # cuantas muestras de distancia tiene que estar el anterior valor para considerar que pertenecen a la misma activación     
else:
    borrado = 1400 # Muestras a borrar a fuerza
    distanciaMax_PD = 1200 # cuantas muestras de distancia tiene que estar el anterior valor para considerar que pertenecen a la misma activación    

# Umbralizar LP
margen = 40 # cuantas muestras a la izqda y dcha también contamos como pertenecientes a la señal
distanciaMax = 1000 # cuantas muestras de distancia tiene que estar el anterior valor para considerar que pertenecen a la misma activación
umbralv=np.zeros(senal.shape)
maximos=np.where(senal>umbral)
p=maximos[0][0]
umbralv[p]=1
for i in maximos[0][1:]:
    umbralv[i-margen:i+margen]=1
    if i-1!=p and i-p <distanciaMax:
        umbralv[p:i]=1 #np.ones(i-p)
    p=i

umbralv2 = np.copy(umbralv)
temp_np= np.where(umbralv ==1) 
attenuation_size = 200
attenuation_array = np.arange(0,attenuation_size,1)/attenuation_size
attenuation_array2 = np.arange(attenuation_size-1,-1,-1)/attenuation_size 
for j in range(temp_np[0].size-1):
    if  temp_np[0][j]+1 != temp_np[0][j+1]:
        umbralv2[temp_np[0][j+1]-attenuation_size: temp_np[0][j+1]] =  attenuation_array
        umbralv2[temp_np[0][j]: temp_np[0][j]+attenuation_size] =  attenuation_array2
    if j == 0:
        try:
            umbralv2[temp_np[0][j]-attenuation_size: temp_np[0][j]] =  attenuation_array
        except:
            continue    
    elif j == temp_np[0].size-2:
        try:    
            umbralv2[temp_np[0][j]: temp_np[0][j]+attenuation_size] =  attenuation_array2
        except:
            continue

senalUmb = np.multiply(senal,umbralv2)
senalUmbInv = np.multiply(senal, (umbralv-1)*-1)
senalUmb = senalUmb - senalUmb.mean()
senalUmbInv = senalUmbInv - senalUmbInv.mean()

# Umbralizamos PD a partir de la señal remanente de umbralizar LP
margen_PD = 20 # Cuantas muestras a la izqda y dcha también contamos como pertenecientes a la señal

umbralvPD=np.zeros(senal.shape)
maximos_PD=np.where(np.abs(senalUmbInv)>umbral_PD)
p=maximos_PD[0][0]
umbralvPD[p]=1
regionPD=np.empty([1,2])
regionPD[0,:]=p
j=0
for i in maximos_PD[0][1:]:
    umbralvPD[i-margen_PD:i+margen_PD]=1
    if i-1!=p and i-p <distanciaMax_PD:
        umbralvPD[p:i]=1 #np.ones(i-p)
    p=i
    if i-maximos_PD[0][j] > distanciaMax_PD:
        regionPD=np.concatenate((regionPD, [[i,i]]))
    try:
        if maximos_PD[0][j+2] - i > distanciaMax_PD:
            regionPD[regionPD.shape[0]-1][1]=i
    except:
        regionPD[regionPD.shape[0]-1][1]=i
    j=j+1



for i in regionPD[:,0]:
    try:
        umbralvPD[int(i)-20:int(i)+borrado]=0
    except:
        umbralvPD[int(i):int(i)+borrado]=0

temp_np= np.where(umbralvPD ==1) 
umbralvPD2 = np.copy(umbralvPD)
for j in range(temp_np[0].size-1):
    if  temp_np[0][j]+1 != temp_np[0][j+1]:
        umbralvPD2[temp_np[0][j+1]-attenuation_size: temp_np[0][j+1]] =  attenuation_array
        umbralvPD2[temp_np[0][j]: temp_np[0][j]+attenuation_size] =  attenuation_array2
    if j == 0:
        try:
            umbralvPD2[temp_np[0][j]-attenuation_size: temp_np[0][j]] =  attenuation_array
        except:
            continue    
    elif j == temp_np[0].size-2:
        try:
           umbralvPD2[temp_np[0][j]: temp_np[0][j]+attenuation_size] =  attenuation_array2
        except:
            continue

senalPDUmb = np.multiply(senalUmbInv,umbralvPD2)

# Visualización de las umbralizaciones resultantes
'''
plt.figure(4)
ax1=plt.subplot(311)
ax1.plot(t, senal)
ax1.set_title('Señal CPG', fontdict={'fontsize': fontsize})
ax1.set_xlabel('Tiempo (s)', loc='right')
ax1.set_ylabel('Amplitud', fontdict={'fontsize': fontsize2})
ax2=plt.subplot(312, sharex=ax1)
ax2.plot(t,umbralv)
ax2.set_title('Umbralización', fontdict={'fontsize': fontsize})
ax2.set_xlabel('Tiempo (s)', loc='right')
ax2.set_ylabel('Decisión', fontdict={'fontsize': fontsize2})
ax3=plt.subplot(313, sharex=ax1)
ax3.plot(t,senalUmb)
ax3.set_title('LP umbralizada', fontdict={'fontsize': fontsize})
ax3.set_xlabel('Tiempo (s)', loc='right')
ax3.set_ylabel('Amplitud', fontdict={'fontsize': fontsize2})
plt.show(block = False) 


plt.figure(5)
ax1=plt.subplot(311)
ax1.plot(t, senalUmbInv)
ax1.set_title('Remanente umbralización LP', fontdict={'fontsize': fontsize})
ax1.set_xlabel('Tiempo (s)', loc='right')
ax1.set_ylabel('Amplitud', fontdict={'fontsize': fontsize2})
ax2=plt.subplot(312, sharex=ax1)
ax2.plot(t, umbralvPD2)
ax2.set_title('Umbralización', fontdict={'fontsize': fontsize})
ax2.set_xlabel('Tiempo (s)', loc='right')
ax2.set_ylabel('Decisión', fontdict={'fontsize': fontsize2})
ax3=plt.subplot(313, sharex=ax1)
ax3.plot(t, senalPDUmb)
ax3.set_title('PD umbralizada', fontdict={'fontsize': fontsize})
ax3.set_ylabel('Amplitud', fontdict={'fontsize': fontsize2})
ax3.set_xlabel('Tiempo (s)', loc='right')
plt.show(block = False) 
'''


# Desplazado en frecuencia de la señal correspondiente a LP para diferenciarla auditivamente de la correspondiente a la PD

desplazamiento_f = 1000
cut_f = 2000 + desplazamiento_f
adaptacion = int(senalUmb.size/args.Fs) 
desplazamiento = desplazamiento_f*adaptacion
W = fftfreq(senalUmb.size, 1/args.Fs)
senal = senal- senal.mean()
senal_f=rfft(senal)
senalU_f=rfft(senalUmb)
senalU_f_shifted = np.zeros(senalU_f.size)
senalU_f_shifted[desplazamiento :desplazamiento+cut_f*adaptacion] = senalU_f[:cut_f*adaptacion]
senalU_f_shifted[int(senalU_f_shifted.size/2):] = 0
senalUmb_shifted = np.multiply(irfft(senalU_f_shifted),umbralv2)


# Visualización del desplazamiento en frecuencia
'''
plt.figure(8)
ax1=plt.subplot(411)
ax1.plot(t, senalUmb)
ax1.set_xlabel('Tiempo (s)', loc = 'right')
ax1.set_title('Señal LP umbralizada',fontdict={'fontsize': 18})
ax1.set_ylabel('Amplitud', fontdict={'fontsize': 14})
bx1=plt.subplot(412)
bx1.plot(W[:int(W.size/2)], senalU_f[:int(senalU_f.size/2)])
bx1.set_xlabel('Frecuencia (Hz)', loc = 'right')
bx1.set_title('Espectro de LP umbralizada',fontdict={'fontsize': 18})
bx1.set_ylabel('Amplitud', fontdict={'fontsize': 14})
ax2=plt.subplot(413, sharex = ax1)
ax2.plot(t, senalUmb_shifted)
ax2.set_xlabel('Tiempo (s)', loc = 'right')
ax2.set_title('Señal LP umbralizada desplazada',fontdict={'fontsize': 18})
ax2.set_ylabel('Amplitud', fontdict={'fontsize': 14})
bx2=plt.subplot(414, sharex = bx1, sharey = bx1)
bx2.plot(W[:int(W.size/2)], senalU_f_shifted[:int(senalU_f.size/2)])
bx2.set_xlabel('Frecuencia (Hz)', loc = 'right')
bx2.set_title('Espectro de LP umbralizada desplazada',fontdict={'fontsize': 18})
bx2.set_ylabel('Amplitud', fontdict={'fontsize': 14})
plt.show(block = False)
'''

# Normalizar volúmenes de ambas señales
senalUmb_shifted_descending = np.sort(np.abs(senalUmb_shifted))[::-1]
corte = senalUmb_shifted_descending[int(np.shape(senalUmb_shifted)[0]*0.01)]
senalUmb_shifted = (senalUmb_shifted-senalUmb_shifted.mean())/corte
senalUmb_shifted2 = senalUmb_shifted
senalUmb_shifted = np.tanh(senalUmb_shifted)

PDUmb_descending = np.sort(np.abs(senalPDUmb))[::-1]
corte = PDUmb_descending[int(np.shape(senalPDUmb)[0]*0.01)]
senalPDUmb = (senalPDUmb-senalPDUmb.mean())/corte
senalPDUmb2 = senalPDUmb
senalPDUmb = np.tanh(senalPDUmb)

senalPDUmb = senalPDUmb*0.8

stereo = np.column_stack((senalUmb_shifted, senalPDUmb-senalPDUmb[0]))

#Visualización de la señal estéreo producida
'''
plt.figure(9)
plt.plot(t, stereo[:,0], label = 'Canal 1')
plt.plot(t, stereo[:,1], label = 'Canal 2')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.title('Sonificación estereofónica', fontsize = fontsize)
plt.legend(fontsize=fontsize)
plt.show(block=False)
'''

if args.show_spectrogram:
    plt.figure(10)
    ax1=plt.subplot(211)
    plt.figure()
    plt.specgram(senalUmb + senalPDUmb, NFFT = 512, Fs= args.Fs, window=np.hamming(512), cmap = 'jet')
    plt.xlabel('Tiempo (s)', fontsize= fontsize)
    plt.ylabel('Frecuencia (Hz)', fontsize= fontsize)
    cb=plt.colorbar()
    cb.set_label('Amplitud [dB]' ,fontdict={'fontsize': fontsize})
    plt.show(block=False)
if args.sound:
    sd.play(stereo, blocking=False)
if args.write:
    sf.write(args.wav_name, stereo, args.Fs)
    
print('Introducir cualquier tecla para finalizar ejecución')
input()
print('Ejecución finalizada')



