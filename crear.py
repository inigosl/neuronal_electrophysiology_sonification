import invariante_video_functions_1
import numpy as np
path1="./Registros/datos_parciales/5.dat"
path2="./Registros/02-Mar-2022/15h37m14s-02-Mar-2022.dat"
path3="./Registros/02-Mar-2022/15h36m02s-02-Mar-2022.dat"
path4="./Registros/02-Mar-2022/15h41m53s-02-Mar-2022.dat"
path5="./Registros/02-Mar-2022/18h48m19s-02-Mar-2022.dat"
path6="./Registros/02-Mar-2022/15h42m45s-02-Mar-2022.dat"
path7 = "./Registros/02-Mar-2022/15h36m02s-02-Mar-2022.dat"

datos = np.loadtxt(path7, dtype='f')


# Se carga el eje de tiempos desde el proprio registro
t=(datos[:300000,0]-datos[0,0])/1000
t = np.linspace(t[0], t[-1], num=t.size)
senal = datos[:,1]

# El resto de varaibles, los periodos, regiones de activación y señales LP y PD han sido calculadas previamente y guardadas en ficheros .txt al ejecutar la función sonificaciónInvariantes.py
# Se puede ver como se han escrito estos ficheros en las líneas comentadas 287-292 de la función mencionada, para entender qe información contiene cada uno
# Sería bastante más inteligente llamar a una función que devuelva las variables necesarias.
periodo = np.loadtxt('periodo.txt').tolist()
regionLP = np.loadtxt('regionLP.txt')
regionPD = np.loadtxt('regionPD.txt')
intervalo = np.loadtxt('intervalo.txt').tolist()
LP = np.loadtxt('LP.txt').tolist() *10
PD = np.loadtxt('PD.txt').tolist()
print('/n /n longitud manito   ', len(t)- len(LP))
t_model_first_1 = regionLP[:,0].tolist() #inicio de las activaciones LP
t_model_last_1 = regionLP[:,1].tolist() #fin de las activaciones LP
t_living_first_1 = regionPD[:,0].tolist() #inicio de las activaciones PD
t_living_last_1 = regionPD[:,1].tolist() #fin de las activaciones PD

invariante_video_functions_1.start_video(t1_= t, v1_ = LP, c1_ = 0, t2_ = t, v2_ = PD, c2_ = 0, t_model_first_ = t_model_first_1, t_model_last_ = t_model_last_1, t_living_first_ = t_living_first_1, t_living_last_ = t_living_last_1, periodoLP_ = periodo, PDtoLP_ = intervalo)