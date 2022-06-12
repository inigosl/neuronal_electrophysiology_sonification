import matplotlib
import argparse
from scipy.fftpack import fft, ifft, fftfreq, rfft, irfft
import scipy.signal.windows as wd
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from scipy import signal
import soundfile as sf
# En este script realizamos la umbralización del LP a partir de la extracelular. Umbralizamos la Pd a partir del registro intracelular.

# Scripts de prueba
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

# Carga de datos
datos = np.loadtxt(args.file_path, dtype='f')
senal = datos[:,2]
PD = datos[:,3]
PD = PD - PD.mean()

t=(datos[:,0]-datos[0,0])/1000
t = np.linspace(t[0], t[-1], num=t.size)

# Crear eje de frecuencias
W = fftfreq(senal.size, 1/args.Fs)


# Definición de parámetros según el registro
if args.file_path== path1:
    umbral = 0.5 # umbral deseado
    umbralPD = 0.3 # umbral deseado PD
elif args.file_path == path2:
    umbral = 0.5 # umbral deseado
    umbralPD = 0.8 # umbral deseado PD       

# Umbralizado de LP
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

senalUmb = np.multiply(senal,umbralv)
senalUmbInv = np.multiply(senal, (umbralv-1)*-1)
senalUmb = senalUmb - senalUmb.mean()
senalUmbInv = senalUmbInv - senalUmbInv.mean()

# Umbralizamos PD a patir del registro intracelular

# Filtrado paso bajo para facilitar umbralización
PD_f=rfft(PD)
b,a = signal.butter(2,20, 'low', fs=args.Fs, analog= False)
PD_lowpass = signal.filtfilt(b, a, PD)
PD_lowpass_f=rfft(PD_lowpass)

regionPDv=np.zeros(PD.shape)
umbralvPD = np.zeros(PD.shape)
regionesActivas = np.where(PD_lowpass>umbralPD)
j=0
regionPD=np.empty([0,2])
for i in regionesActivas[0]:   
    umbralvPD[i]=1
    j=j+1

temp_np= np.where(umbralvPD ==1) 
attenuation_size = 200
attenuation_array = np.arange(0,attenuation_size,1)/attenuation_size
attenuation_array2 = np.arange(attenuation_size-1,-1,-1)/attenuation_size 
for j in range(temp_np[0].size-1):
    if  temp_np[0][j]+1 != temp_np[0][j+1]:
        umbralvPD[temp_np[0][j+1]-attenuation_size: temp_np[0][j+1]] =  attenuation_array
        umbralvPD[temp_np[0][j]: temp_np[0][j]+attenuation_size] =  attenuation_array2
    if j == 0:
        try:
            umbralvPD[temp_np[0][j]-attenuation_size: temp_np[0][j]] =  attenuation_array
        except:
            continue    
    elif j == temp_np[0].size-2:
        try:
           umbralvPD[temp_np[0][j]: temp_np[0][j]+attenuation_size] =  attenuation_array2
        except:
            continue        

PDUmb = np.multiply(PD,umbralvPD)

# Mostrar resultado del filtrado paso bajo 
'''
plt.figure(2)
ax1=plt.subplot(211)
ax1.plot(t, PD)
ax1.set_title('Registro PD intracelular', fontdict={'fontsize': fontsize})
ax1.set_xlabel('Tiempo (s)', loc='right')
ax1.set_ylabel('Amplitud', fontdict={'fontsize': 14})
ax2=plt.subplot(212, sharex=ax1)
ax2.plot(t,PD_lowpass)
ax2.set_title('PD filtrado paso bajo', fontdict={'fontsize': fontsize})
ax2.set_ylabel('Amplitud', fontdict={'fontsize': 14})
ax2.set_xlabel('Tiempo (s)', loc='right')
plt.show(block = False) 
'''
# Mostrar resultado de la umbralización
'''
plt.figure(3)
ax3=plt.subplot(211, sharex=ax1)
ax3.plot(t,umbralvPD)
ax3.set_title('Umbralización', fontdict={'fontsize': fontsize})
ax3.set_xlabel('Tiempo (s)', loc='right')
ax3.set_ylabel('Decisión', fontdict={'fontsize': 14})
ax4=plt.subplot(212, sharex=ax1)
ax4.plot(t, PDUmb)
ax4.set_title('Señal PD tras umbralización', fontdict={'fontsize': fontsize})
ax4.set_ylabel('Amplitud', fontdict={'fontsize': 14})
ax4.set_xlabel('Tiempo (s)', loc='right')
plt.show(block = False) 
'''

#Desplazar en frecuencia la señal correspondiente a LP. Casi seguro que se escuchara mejor recortando las altas frecuencias de ambas señales

desplazamiento_f = 1000
cut_f = 2000 + desplazamiento_f
adaptacion = int(senalUmb.size/args.Fs) 
desplazamiento = desplazamiento_f*adaptacion
W = fftfreq(senalUmb.size, 1/args.Fs)

senalU_f=rfft(senalUmb)
senalU_f_shifted = np.zeros(senalU_f.size)

senalU_f_shifted[desplazamiento :] = senalU_f[:-desplazamiento]
senalU_f_shifted[int(senalU_f_shifted.size/2):] = 0
senalUmb_shifted = np.multiply(irfft(senalU_f_shifted),umbralv2)

# Mostrar resultado de desplazado en frecuencia
'''
plt.figure(4)
ax1=plt.subplot(411)
ax1.plot(t, senalUmb)
ax1.set_xlabel('Tiempo (s)', loc = 'right')
ax1.set_ylabel('Amplitud')
ax1.set_title('Señal LP umbralizada')
bx1=plt.subplot(412)
bx1.plot(W, senalU_f)
bx1.set_xlabel('Frecuencia (Hz)', loc = 'right')
bx1.set_ylabel('Amplitud')
bx1.set_title('Espectro de LP umbralizada')
ax2=plt.subplot(413, sharex = ax1)
ax2.plot(t, senalUmb_shifted)
ax2.set_xlabel('Tiempo (s)', loc = 'right')
ax2.set_ylabel('Amplitud')
ax2.set_title('Señal LP umbralizada desplazada')
bx2=plt.subplot(414, sharex = bx1, sharey = bx1)
bx2.plot(W, senalU_f_shifted)
bx2.set_xlabel('Frecuencia (Hz)', loc = 'right')
bx2.set_ylabel('Amplitud')
bx2.set_title('Espectro de LP umbralizada desplazada')

plt.show(block = False)
'''

# Normalización de potencias de las señales

senalUmb_shifted_descending = np.sort(np.abs(senalUmb_shifted))[::-1]
corte = senalUmb_shifted_descending[int(np.shape(senalUmb_shifted)[0]*0.01)]
senalUmb_shifted = (senalUmb_shifted-senalUmb_shifted.mean())/corte
senalUmb_shifted2 = senalUmb_shifted
senalUmb_shifted = np.tanh(senalUmb_shifted)

PDUmb_descending = np.sort(np.abs(PDUmb))[::-1]
corte = PDUmb_descending[int(np.shape(PDUmb)[0]*0.01)]
PDUmb = (PDUmb-PDUmb.mean())/corte
PDUmb2 = PDUmb
PDUmb = np.tanh(PDUmb)

PDUmb = (PDUmb-0.3)*2
stereo = np.column_stack((senalUmb_shifted, (PDUmb+1)/2))

# Mostrar señal estéreo
'''
plt.figure(5)
plt.plot(t, stereo[:,0], label = 'Canal 1')
plt.plot(t, stereo[:,1], label = 'Canal 2')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.title('Sonificación estereofónica', fontsize = fontsize)
plt.legend(fontsize=fontsize)
plt.show(block=False)
'''


if args.sound:
    sd.play(stereo, blocking=False)

if args.show_spectrogram:
    plt.figure()
    ax1 = plt.subplot(211)
    ax1.specgram(senalUmb + PDUmb, Fs= args.Fs, window=np.hamming(256))
    ax1.set_xlabel('Tiempo (s)', fontdict={'fontsize': fontsize})
    ax1.set_ylabel('Frecuencia (Hz)', fontdict={'fontsize': fontsize})
    ax2 = plt.subplot(212, sharex=ax1, sharey=ax1)   
    ax2.specgram(senalUmb_shifted + PDUmb, Fs= args.Fs, window=np.hamming(256))
    ax2.set_xlabel('Tiempo (s)', fontdict={'fontsize': fontsize})
    ax2.set_ylabel('Frecuencia (Hz)', fontdict={'fontsize': fontsize})
    plt.show(block = False)	    

if args.write:
    sf.write(args.wav_name, stereo, args.Fs)

print('Introducir cualquier tecla para finalizar ejecución')
input()
print('Ejecución finalizada')


