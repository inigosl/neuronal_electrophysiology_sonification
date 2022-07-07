import ffmpeg
from scipy.io import wavfile

############################################
# Esta funcion combina el audio y el video #
############################################

# Si no se encuentra ffmpeg al ejecutar en Windows, puede funcionar descargar un build adecuado y añadir su path a las variables de entorno

segundos_de_desfase = 2 # Normalmente es necesario adelantar el audio del orden de 1.5-2.5 segundos para que video y audio estén correctamente sincronizados
Fs = 10000
path_audio = './paralaprueba.wav'
path_video_original = "./paralaprueba video.mp4"
path_video_destino = 'C:/Users/Íñigo/Desktop/TFG/Diferenciación video audio.mp4'
samplerate, data1 = wavfile.read(path_audio)
data = data1[segundos_de_desfase * Fs:,:] 
wavfile.write('temp.wav', samplerate,data)
video = ffmpeg.input(path_video_original)
audio = ffmpeg.input("./temp.wav")
ffmpeg.concat(video, audio, v=1, a=1).output(path_video_destino).run()
