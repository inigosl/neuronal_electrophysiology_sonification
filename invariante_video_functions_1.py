#########################
# IMPORTS AND ARGUMENTS #
#########################
import sys
from xml.dom import minicompat
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse
import time
import datetime
from sklearn.linear_model import LinearRegression
#############
# VARIABLES #
#############

#### Corrientes si o no
c_on = False

#### Configuración del programa
segundos_ventana = 2   # Tiempo que se verá en el plot, con 2 funciona otros nu se xD

#### No se recomienda tocar, valores internos
modo  = 1      # Modo de visualización, va por parámetro, el 2 da mejor rendimiento
freq  = 10000  # Frecuencia de muestreo, va por parámetro
fps   = 30     # Frames por segundo
dpi   = 100    # DPI. Puntos por pulgada

size_plot = 4.8                           # Variable que controla el tamaño del plot
relacion_plot = 4.0                       # Controla la relacion entre las dimensiones x / y
ini_ventana=0                             # Inicialización origen de la ventana
pts_ventana = int(segundos_ventana*freq)  # Ventana de datos a tener en cuenta
interval = 1000/fps                       # Cada cuantos milisegundos se refresca la pantalla
index_event = 0                           # Variable de uso interno para controlar el evento mostrado
pts_avance = int(freq/fps)                # Puntos que se deben avanzar en cada refresco

ml=0

################
# EVENT HANDLE #
################
def onKey(event):
    global pts_avance
    v_max=1200
    if abs(pts_avance) >= v_max:
        print("Max Speed") 
    if event.key == ' ':
        if pts_avance!=0:
            pts_avance = 0
        else:
            pts_avance = 150
    elif event.key == 'escape':
        exit(0)
    elif event.key == 'up' and pts_avance<v_max:
        pts_avance = pts_avance+150 
    elif event.key == 'down' and pts_avance>-v_max:
        pts_avance = pts_avance-150 
    elif event.key == 'right' and pts_avance<v_max:
        pts_avance = pts_avance+50   
    elif event.key == 'left' and pts_avance>-v_max:
        pts_avance = pts_avance-50 

###################
# PLOT CREATION 1 #
###################

########General
fig = plt.figure(figsize=(size_plot*relacion_plot, size_plot), dpi=dpi)
#print('fig size: {0} DPI, size in inches {1}'.format(fig.get_dpi(), fig.get_size_inches()))
fig.canvas.mpl_connect('key_press_event', onKey)

if c_on == True:
    ax_i = plt.subplot2grid((2, 3), (0, 2), rowspan=2)
    ax_v = plt.subplot2grid((2, 3), (0, 0), colspan=2)
    ax_c = plt.subplot2grid((2, 3), (1, 0), colspan=2)
    #ax_v.xaxis.set_visible(False)
    #ax_v.grid(True)
    #ax_v.set_axisbelow(True)
    #ax_v.xaxis.grid(color='gray', linestyle='dashed')
else:
    ax_i = plt.subplot2grid((1, 4), (0, 3))
    ax_v = plt.subplot2grid((1, 4), (0, 0), colspan=3)

########Voltage
if c_on == True:
    ax_v_m, = ax_v.plot([], [], label="Model neuron" , linewidth=0.4)
    ax_v_l, = ax_v.plot([], [], label="Living neuron", linewidth=0.4)
else: 
    ax_v_m, = ax_v.plot([], [], label="LP neuron" , linewidth=0.4)
    ax_v_l, = ax_v.plot([], [], label="PD neuron", linewidth=0.4)
#ax_v.set_title("Voltage time series")
#ax_v.legend(loc=2, ncol=2)
ax_v.set_yticklabels([])
ax_v.set_ylabel("Voltage (10mV/div)")
#ax_v.set_ylim(v_min, v_max)
ax_v.xaxis.set_animated(True)

########Current
if c_on == True:
    ax_c_l, = ax_c.plot([], [], label="Current to model" , linewidth=0.4)
    ax_c_m, = ax_c.plot([], [], label="Current to cell", linewidth=0.4)

    ax_c.legend(loc=2, ncol=2)
    ax_c.set_xlabel("Time (s)")
    ax_c.set_ylabel("Current")
    #ax_c.set_ylim(c_min, c_max)
    ax_c.xaxis.set_animated(True)
else:
    ax_v.set_xlabel("Time (s)")

########Invariant

#Último dato se pintará resaltado en estos ax
ax_i_red_last,  = ax_i.plot([], [], 'go', markersize='10.0')
ax_i_blue_last, = ax_i.plot([], [], 'go', markersize='10.0')

if modo==1:
    #Los datos se añaden progresivamente, declaramos las listas a las cuales se añadira y los ax
    list_inv_red_x, list_inv_red_y, list_inv_blue_x, list_inv_blue_y  = [], [], [], []
    ax_i_blue_progress, = ax_i.plot([], [], 'bo', markersize='1.0')
    ax_i_red_progress,  = ax_i.plot([], [], 'ro', markersize='1.0')

if modo==2:
    #Todos los datos desde el principio y coloreados, no hace falta ni guardar la scatter
    ax_i.scatter(periodoLP, PDtoLP, marker='o', c=t_model_first, cmap='Blues', s=1)

#########
# START #
#########
def start_video(t1_, v1_, c1_, t2_, v2_, c2_, t_model_first_, t_model_last_, t_living_first_, t_living_last_, periodoLP_, PDtoLP_):
    
    #### Variables pa quien las quiera
    v3 = np.copy(v1_) * 12
    v3 = v3.tolist() 
    v1_ = v3
    v4 = np.copy(v2_) * 2
    v4 = v4.tolist() 
    v2_ = v4
    global t1, v1, c1
    global t2, v2, c2
    t1 = t1_
    v1 = v1_
    c1 = c1_
    t2 = t2_
    v2 = v2_
    c2 = c2_

    global masimo, minismo
    masimo = max(v1)*1.2
    minismo = min(v1)*1.3
    print(PDtoLP_[10], periodoLP_[10])
    PDtoLP_ = np.divide(PDtoLP_,10)
    periodoLP_ = np.divide(periodoLP_,10)
    print(PDtoLP_[10], periodoLP_[10])
   
    global t_model_first, t_model_last
    global t_living_first, t_living_last
    t_model_first  = t_model_first_
    t_model_last   = t_model_last_
    t_living_first = t_living_first_
    t_living_last  = t_living_last_

    global periodoLP, PDtoLP
    periodoLP = periodoLP_
    PDtoLP    = PDtoLP_

    global num_points, num_events
    num_points = len(t1)
    num_events = len(PDtoLP_)

    global v_min, v_max, c_min, c_max, v_range
    v_min = min (min(v1), min(v2)) - .8
    v_max = max (max(v1), max(v2)) +  4
    #c_min = min (min(c1), min(c2)) - .0
    #c_max = max (max(c1), max(c2)) + .6
    v_range = v_max - v_min

    global pos1, pos2, pos3
    pos1  = v_min + (v_range*0.7)
    pos2  = v_min + (v_range*0.8)
    pos3  = v_min + (v_range*0.1)

    ###################
    # PLOT CREATION 2 #
    ###################

    #ax_v.set_ylim(v_min, v_max)
    if c_on == True:
        ax_c.set_ylim(c_min, c_max)
    ax_i.set_xlim(min(periodoLP)-0.01, max(periodoLP)+0.01)
    ax_i.set_ylim(min(PDtoLP)-0.01, max(PDtoLP)+0.01)

    #Barras en el plot del voltage
    #Los valores y son definitivos (pos), para que no de error meto valores x dummies [0,0]
    global ax_v_event_blue, ax_v_event_red, ax_v_event_black
    list_barra_blue_x, list_barra_red_x, list_barra_black_x = [], [], []
    ax_v_event_blue,  = ax_v.plot( [0,0], [pos1, pos1], 'b', marker=6, linestyle='-', label='Activación LP')
    ax_v_event_red,   = ax_v.plot( [0,0], [pos2, pos2], 'r', marker=6, linestyle='-')
    ax_v_event_black, = ax_v.plot( [0,0], [pos3, pos3], 'k', marker=6, linestyle='-', label='Activación PD')
    #ax_v_event_green, = ax_v.plot( [0,0], [pos1, pos1], 'y', marker=6, linestyle='-')
    ax_v.legend(loc=2, ncol=4)

    ##### Fit Blue
    fit    = np.polyfit( periodoLP, PDtoLP, 1)
    fit_fn = np.poly1d(fit) 
    m      = fit[0] #pendiente
    yhat   = fit_fn( periodoLP)                         
    ybar   = np.sum( PDtoLP) / len(PDtoLP)    
    ssreg  = np.sum( (yhat-ybar)**2)   
    sstot  = np.sum( (PDtoLP - ybar)**2)   
    R2     = ssreg / sstot
    #ax_i.plot(events.fPD_fPD, fit_fn(events.fPD_fPD), linewidth= 0.5, c="midnightblue", label="R2={0:.4f}\tm={1:.4f}".format(R2, m).expandtabs())
    #ax_i.plot(periodoLP, fit_fn(periodoLP), linewidth= 0.5, c="midnightblue", label="LPPD interval\tR2={0:.4f}".format(R2).expandtabs())
    reg_LPPD_delay = LinearRegression() 
    reg_LPPD_delay.fit(np.copy(periodoLP).reshape(-1, 1), np.copy(PDtoLP).reshape(-1, 1)) 
    linea_LPPD_delay = reg_LPPD_delay.predict(np.copy(periodoLP).reshape(-1, 1))
    ax_i.plot(periodoLP, linea_LPPD_delay, linewidth= 0.5, c="midnightblue")

    ##### Estetica plot fit
    #ax_i.legend()
    ax_i.set_title  ("Dynamical invariant")
    ax_i.set_xlabel ("Period (ms)")
    ax_i.set_ylabel ("LPPD interval (ms)")

    plt.tight_layout()

    ########
    # MAIN #
    ########
    print("Duración señal   = ", int(len(t1)/freq), "s")
    print("DPI              = ", dpi)
    print("Resolucion       = ", dpi*size_plot*relacion_plot, "x", dpi*size_plot)
    print("FPS              = ", fps)

    save = True
    name_file = "paralaprueba video"

    if save == False:
        anim = FuncAnimation(fig, update, interval=interval, repeat=False, blit=True, init_func=init)
        plt.show()

    else:
        frames = (len(t1) - pts_ventana) / pts_avance # Calculo de cuantos avances hay que producir para llegar al final de la señal
        print("Frames           = ", int(frames))
        print("Est. t dpi100    = ", int(frames*180/1381/60)+1, "min")
        #print("Est. t dpi100    = ", int(frames*150/42349)+1, "min")
        #print("Est. t dpi300    = ", int(frames*230/42349)+1, "min")
        anim = FuncAnimation(fig, update, interval=interval, repeat=False, blit=True, init_func=init, frames=int(frames))
        print("Inicio           =  " + '{:%H:%M:%S}'.format(datetime.datetime.now()))
        anim.save(name_file+".mp4", dpi=dpi, writer= 'ffmpeg', bitrate=-1) 
        print("Fin              =  " + '{:%H:%M:%S}'.format(datetime.datetime.now()))


           ###############################################################
     ###########################################################################
  #################################################################################
         ###################################################################

#########
# PATCH #
#########
'''
def _blit_draw_old(self, artists, bg_cache):
    # Handles blitted drawing, which renders only the artists given instead
    # of the entire figure.
    updated_ax = []
    for a in artists:
        # If we haven't cached the background for this axes object, do
        # so now. This might not always be reliable, but it's an attempt
        # to automate the process.
        if a.axes not in bg_cache:
            # bg_cache[a.axes] = a.figure.canvas.copy_from_bbox(a.axes.bbox)
            # change here
            bg_cache[a.axes] = a.figure.canvas.copy_from_bbox(a.axes.figure.bbox)
        a.axes.draw_artist(a)
        updated_ax.append(a.axes)

    # After rendering all the needed artists, blit each axes individually.
    for ax in set(updated_ax):
        # and here
        # ax.figure.canvas.blit(ax.bbox)
        ax.figure.canvas.blit(ax.figure.bbox)
'''

def _blit_draw(self, artists):
    # Handles blitted drawing, which renders only the artists given instead
    # of the entire figure.
    updated_ax = {a.axes for a in artists}
    # Enumerate artists to cache axes' backgrounds. We do not draw
    # artists yet to not cache foreground from plots with shared axes
    for ax in updated_ax:
        # If we haven't cached the background for the current view of this
        # axes object, do so now. This might not always be reliable, but
        # it's an attempt to automate the process.
        cur_view = ax._get_view()
        view, bg = self._blit_cache.get(ax, (object(), None))
        if cur_view != view:
            self._blit_cache[ax] = (
                # cur_view, ax.figure.canvas.copy_from_bbox(ax.bbox))
                cur_view, ax.figure.canvas.copy_from_bbox(ax.figure.bbox)) # Change 1
    # Make a separate pass to draw foreground.
    for a in artists:
        a.axes.draw_artist(a)
    # After rendering all the needed artists, blit each axes individually.
    #for ax in updated_ax:
    #    ax.figure.canvas.blit(ax.bbox)
    updated_ax.pop().figure.canvas.blit(updated_ax.pop().figure.bbox) # Change 2

# MONKEY PATCH!!
matplotlib.animation.Animation._blit_draw = _blit_draw

########
# INIT #
########
def init():
    # Aqui se pone todo lo que implique crear
    ax_v_m.set_data( [], [] )
    ax_v_l.set_data( [], [] )
    if c_on == True:
        ax_c_m.set_data( [], [] )
        ax_c_l.set_data( [], [] )
    ax_v.set_ylim( pos3-0.8, pos1+0.8 )
    ax_i_red_last.set_data ( [], [] )
    ax_i_blue_last.set_data( [], [] )

    ax_v_event_blue.set_data  ( [t_model_first[0]/10000, t_model_last[0]/10000], [pos1, pos1] )
    ax_v_event_red.set_data   ( [0,0])
    ax_v_event_black.set_data ( [t_living_first[0]/10000,t_living_last[0]/10000], [pos3, pos3] )
    #ax_v_event_green.set_data ( [0,0], [pos1, pos1] )

    if modo==1:
        ax_i_red_progress.set_data  ( [], [] )
        ax_i_blue_progress.set_data ( [], [] )
        if c_on == True:
            return (ax_v.xaxis, ax_c.xaxis, ax_v_m, ax_v_l, ax_c_m, ax_c_l, ax_i_red_last, ax_i_blue_last, ax_v_event_blue, ax_v_event_red, ax_v_event_black, ax_i_red_progress, ax_i_blue_progress)
        else:
            return (ax_v.xaxis, ax_v_m, ax_v_l, ax_i_red_last, ax_i_blue_last, ax_v_event_blue, ax_v_event_red, ax_v_event_black, ax_i_red_progress, ax_i_blue_progress)
    else:
        if c_on == True:
            return (ax_v.xaxis, ax_c.xaxis, ax_v_m, ax_v_l, ax_c_m, ax_c_l, ax_i_red_last, ax_i_blue_last, ax_v_event_blue, ax_v_event_red, ax_v_event_black)
        else:
            return (ax_v.xaxis, ax_v_m, ax_v_l, ax_i_red_last, ax_i_blue_last, ax_v_event_blue, ax_v_event_red, ax_v_event_black)

##########
# UPDATE #
##########
def update(i):
    global ini_ventana, pts_avance, index_event, pts_ventana
    ini_ventana += pts_avance
    fin =  ini_ventana+pts_ventana

    # Limites de tiempo alcanzados o velocidad parada
    if ini_ventana<0:
        pts_avance = 0
        ini_ventana = 0
    elif fin>num_points:
        pts_avance = 0
        ini_ventana = num_points - pts_ventana -1
        fin = ini_ventana + pts_ventana
    elif pts_avance == 0:
        if modo == 1:
            if c_on == True:
                return (ax_v.xaxis, ax_c.xaxis, ax_v_m, ax_v_l, ax_c_m, ax_c_l, ax_i_red_last, ax_i_blue_last, ax_v_event_blue, ax_v_event_red, ax_v_event_black, ax_i_red_progress, ax_i_blue_progress)
            else:
                return (ax_v.xaxis, ax_v_m, ax_v_l, ax_i_red_last, ax_i_blue_last, ax_v_event_blue, ax_v_event_red, ax_v_event_black, ax_i_red_progress, ax_i_blue_progress)
        else:
            if c_on == True:
                return (ax_v.xaxis, ax_c.xaxis, ax_v_m, ax_v_l, ax_c_m, ax_c_l, ax_i_red_last, ax_i_blue_last, ax_v_event_blue, ax_v_event_red, ax_v_event_black)
            else:
                return (ax_v.xaxis, ax_v_m, ax_v_l, ax_i_red_last, ax_i_blue_last, ax_v_event_blue, ax_v_event_red, ax_v_event_black)

    # Actualizamos las señales V y C a los rangos nuevos
    ax_v_m.set_data( t1[ini_ventana:fin], v1[ini_ventana:fin] )
    ax_v_l.set_data( t1[ini_ventana:fin], v2[ini_ventana:fin] )
    ax_v.set_xlim( t1[ini_ventana], t2[fin-1] )
    
    if c_on == True:
        ax_c_m.set_data( t1[ini_ventana:fin], c1[ini_ventana:fin] )
        ax_c_l.set_data( t1[ini_ventana:fin], c2[ini_ventana:fin] )
        ax_c.set_xlim( t1[ini_ventana], t1[fin-1] )

    #### EVENTOS
    ref_ini = t1[ini_ventana]
    ref_fin = t2[ini_ventana+pts_ventana-1]
    try:
        ref_event = t_living_first[index_event]/10000 #El evento de referencia
    except:
        print('pos ha pasao')
    #print(index_event)
    # Avanzando hacia delante AND eventos pendientes
    if pts_avance>0 and index_event<num_events and ref_event <= ref_fin:
        update_events(index_event)
        index_event+=1
            
    # Avanzando hacia atras AND se han recorrido eventos
    elif pts_avance<0 and index_event!=0 and ref_event>=ref_fin:
        index_event-=1
        update_events(index_event)

    if modo==1:
        if c_on == True:
            return (ax_v.xaxis, ax_c.xaxis, ax_v_m, ax_v_l, ax_c_m, ax_c_l, ax_i_red_last, ax_i_blue_last, ax_v_event_blue, ax_v_event_red, ax_v_event_black, ax_i_red_progress, ax_i_blue_progress)
        else: 
            return (ax_v.xaxis, ax_v_m, ax_v_l, ax_i_red_last, ax_i_blue_last, ax_v_event_blue, ax_v_event_red, ax_v_event_black, ax_i_red_progress, ax_i_blue_progress)
    else:
        if c_on == True:
            return (ax_v.xaxis, ax_c.xaxis, ax_v_m, ax_v_l, ax_c_m, ax_c_l, ax_i_red_last, ax_i_blue_last, ax_v_event_blue, ax_v_event_red, ax_v_event_black)
        else:
            return (ax_v.xaxis, ax_v_m, ax_v_l, ax_i_red_last, ax_i_blue_last, ax_v_event_blue, ax_v_event_red, ax_v_event_black)

def update_events(index_event):
    global ml

    ### Evitando problemas de ruido
    # El modelo no disparo
    if t_living_first[index_event] < t_model_first[index_event-ml]:
        # El modelo no disparo, no hacemos nada
        ml+=1
        return

    #### Funcion que actualiza los eventos
    #print(events.firstLP [index_event+1] - events.lastPD [index_event]) 

    #### Dato resaltado en verde en los puntos del invariante
    #ax_i_red_last.set_data  ( events.fPD_fPD[index_event], events.lLD_fPD[index_event] )
    ax_i_blue_last.set_data ( periodoLP[index_event],PDtoLP[index_event] )

    #### Actualización de las tres barras sobre el voltaje
    ax_v_event_blue.set_xdata  ( [t_model_first[index_event]/10000,t_model_last[index_event]/10000])
    ax_v_event_red.set_xdata   ( [0,0 ])#events.lastLP  [index_event], events.firstPD [index_event] ] )
    ax_v_event_black.set_xdata ( [ t_living_first[index_event]/10000, t_living_last[index_event]/10000 ] )
    #ax_v_event_green.set_xdata ( [ events.lastPD [index_event], events.firstLP [index_event+1] ] )
    #print(index_event)
    #fix
    if t_living_first[index_event+1] < t_model_first[index_event+1-ml]:
        ax_v_event_black.set_xdata ( [ t_living_first[index_event], t_living_first[index_event+2] ] )
    
    if modo==1:
        # En este modo los puntos del invariante nuevos no estan

        # Añadimos valores a las listas
        #list_inv_red_x.append ( events.fPD_fPD [index_event] )
        #list_inv_red_y.append ( events.lLD_fPD [index_event] )
        list_inv_blue_x.append( periodoLP [index_event] )
        list_inv_blue_y.append( PDtoLP [index_event] )

        # Las listas al objeto axe correspondiente
        ax_i_red_progress.set_data  ( list_inv_red_x,  list_inv_red_y  )
        ax_i_blue_progress.set_data ( list_inv_blue_x, list_inv_blue_y )
