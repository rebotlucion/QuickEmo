#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Created on Sat Jul 11 20:13:42 2020

@author: fco_p
"""
"""
Using

https://github.com/tyiannak/pyAudioAnalysis

@article{giannakopoulos2015pyaudioanalysis,
  title={pyAudioAnalysis: An Open-Source Python Library for Audio Signal Analysis},
  author={Giannakopoulos, Theodoros},
  journal={PloS one},
  volume={10},
  number={12},
  year={2015},
  publisher={Public Library of Science}
}


In this case, is exactly the same burt avoding working with files and passing as arguments directly arrays
"""
"""
El programa "PREDICTOR", debería de tener lo siguiente:
- Separador de canales. Identificar a cada uno de los interlocutores
- Establecer el género de los intelocutores
- Establecer intervalo de edad de los interlocutores
- Esto para cada "evento de entrada", en "chunks" de tiempo
- para cada chunk además pasar a texto (solapar con anterior y posterior para poder construir palabras)
- para cada chunk comparar los interlocutores identificados con el anterior y conectar con los ya identificados o generar uno nuevo
- rehacer los chunks para mapearlos con las palabras identificadas. Decidir si se hace palabra a palabra, de dos en dos, etc....)
- Inicialmente en modo comando o algo así, y finalmente realizar un GUI para interactuar con la aplicación


El modelo habría que hacerlo así:
- Hacer una primera versión, muy simple, con cualquier libreria
- Entrenar el modelo con una sola fuente
- A ser posible, consolidar diferentes fuentes y entrenar con todas (sería lo último después de tener un primer entregable)
     - ¿hay diferencias con actores y con gente real?
     - ¿hay diferencias entre varios idiomas y culturas?
     - ¿ayuda o es contraproducente mezlar idiomas?¿y culturas?

- Despues del primer entregable completo, si puedo, sofisticar esta primera versión, con otras librerias y otras funcionalidades
- ¿se podría identificar la personalidad de uno de los interlocutores? esto es muy subjetivo e improbable supongo


hacer pruebas y validaciones con personaas
- primero conmigo mismo
- luego con Ana y la familia
- luego con amigos por teléfono o skype o a saber
- probar con una conversación grabada entera de algún call centre

PROBLEMAS A RESOLVER
- ¿Como estar "escuchando" en tiempo real una conversación y procesándola en paralelo?

"""

### General imports ###
#from __future__ import division
import numpy as np
#import pandas as pd
#import time
#import re
import os
from __future__ import print_function
#import timeit

#from collections import Counter


### Audio imports ###
#from library.speech_emotion_recognition import *

### Text imports ###
#from library.text_emotion_recognition import *
#from library.text_preprocessor import *
#from nltk import *
#from tika import parser
#from werkzeug.utils import secure_filename
#import tempfile

"""
Cuando tenga el GUI hecho, el control del flujo de la herramienta se llevará desde el GUI, por lo que habrá que lanzar el gui aquí

if __name__ == '__main__':
    app.run(debug=True)

"""

if __name__ == "__main__":
    print("Vamos a ello")


### Audio imports ###
import pyAudioAnalysis
from QE_audioUtils import QE_speaker_diarization, QE_assemble, QE_speech_Identify
from QE_VoiceTextModel import QE_Voice_Text_Model
#import wave  ¿?

#### INICIALIZAR CLASES ESPECÍFICAS DE LA APLICACIÓN
VT_model = QE_Voice_Text_Model # carga el modelo que tenemos entrenado. 
    #Es un volcado de los parámetros del modelo que hemos conseguido entrenar, tiene tanto la parte de speech como la de text


 #### a ser posible transformar las listas siguientes en ARRAYS que será más rápido en CPU, creo   
interlocutors = [] # lista de identificadores de interlocutores, que son ternas (id_interlocutor, género interlocutor, rango edad interlocutor)
                        # aún no se lo que es un "id_interlocutor", tiene que permitir la identificación de interlocutores en el tiempo
                            
emociones = [] # lista ordenada de emociones asociadas a cada interlocutor en cada momento de la conversación, ternas (emoción, probabilidad, tiempo o id_palabra donde sucede)
               # la emoción más reciente de cada interlocutor, la que está sucendiendo en el momento de hablar, está en un extremo de la lista (¿principio? 
               # ¿más antiguos hacia el final?)
#palabras = [] # palabras pronunciandose de cada interlocutor o corte de audio de palabra incompleta (aún no identificada) por interlocutor
        #ya no, ahora tanto lo identificado como lo pendiente est´en current Ones, pero hay un índice de por donde va en cada interlocutor
mensajes = [] # lista de mensajes de cada uno de los interlocutores
                # un mensaje es una lista ordenada de pares (texto palabra, audio palabra)
current_ones = [] # estructura con los canales de audio a medida que se van registrando, uno por cada interlocutor. 
                  # son duplas  (    audio,  recognized_index,   interlocutor(id_interlocutor, género interlocutor, rango edad interlocutor)      )
                        # recognized_index, is the point of audio to which the words already have been identified, and the point from 
                        # recognition must continue

# pyAudioAnalysis diarization models
# In order to avoid a recurrent load of the models, they are loaded more globally, here, only once and then also passed as arguments
# to the pyAudioAnalysis functions
# ¿transformar esto en una clase y crear un objeto que luego haga las cosas sin perder el modelo?
# Se podría llamar QE_diarization_model
base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 
                            "data/models")#copiado en el sitio correspondiente, creo
    classifier_all, mean_all, std_all, class_names_all, _, _, _, _, _ = \
        at.load_model_knn(os.path.join(base_dir, "knn_speaker_10"))
    classifier_fm, mean_fm, std_fm, class_names_fm, _, _, _, _,  _ = \
        at.load_model_knn(os.path.join(base_dir, "knn_speaker_male_female"))


# INICIALIZAR "ESCUCHADOR" DE AUDIO
CHUNK = 88200 # Number of data points to read at a time, Initially 4096 SAMPLES, lets see wether 88200 =>  2 seconds is possible
RATE = 44100  # time resolution of the recording device (Hz), RECORD AT 44100 SAMPLES PER SECOND

# Abrir un "escuchador" que va registrando el audio y generando "chunks" o cortes de audio que
# vamos procesando poco a poco en tiempo real según suceden
p=pyaudio.PyAudio() # start the PyAudio class. 
print('Recording')

stream=p.open(format=pyaudio.paInt16,
                channels=1, #ya capturamos en mono
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK) #uses default input device

while 1 : # Infinite listening loop , en realidad vendrá determinado por el loop del GUI
    corte = stream.read(chunk)  #corte = data, data = corte
     
    interlocutors_labels = QE_speaker_diarization(sampling_rate = RATE, signal = corte , numSpeakers=1, mid_window=0.4, mid_step=0.05, # defaults are mid_window=2.0  and mid_step=0.2
                        classifier_all, mean_all, std_all, class_names_all, classifier_fm, mean_fm, std_fm, class_names_fm # Load models from avove
                        short_window=0.05, lda_dim=0, plot_res=False):   # separa los interlocutores, 
                        # y lo que devuelve es un ID del speaker de cada subsegmento, para cada window o step le asigna un label de speaker
                        """
                            mid_window creo que son segundos
                            mid_step creo que son segundos
                            si corte durase 2 segunos, se podrían hacer 33 ventanas solapadas  de 400ms con saltos de 50ms
                        """
                        # ¿y que quería yo? una lista de audios separadas por cada interlocutor, en forma
                         # de  duplas (    audio,    interlocutor(id_interlocutor, género interlocutor, rango edad interlocutor)      )
                         # ¿tengo que pasar todas las veces todo el audio hsata el momento?¿como mantengo la identificación de interlocutores a lo largo del tiempo?
    current_ones = QE_assemble(interlocutors_labels, corte) #. "fusiona" o concatena los cortes de cada interlocutor para ir construyendo cada canal. # faltaría el género y la edad
                    # problema, ¿como mantener la choerencia de los interlocutores identificados a lo largo del tiempo y de las sucesivas llamadas a diarization
                    # ¿tal vez una clase que vaya registrando las características , mfcc , y tal de los cortes que se van evaluando, ¿generar un hash o algo así de los 
                    # clusters?¿distancia a los centroides?¿pasar los centroides como argumento y recalcularlos a medida que se llama a diarization pero incorporando la info anteior?
                    # De momento asumir que está resuelto y si esto se complica mucho, fuerzo a que haya dos canales de alguna forma.
    mensajes, flag_nuevo, current_ones = QE_speech_Identify(mensajes, current_ones)  #¿se chace una copia de current_Ones? no debería de hacerse, hay 
    #que trabajar sobre el mismo array en memoria, ver como pasar los argumentos para que sea así, o usando en su lugar una clase que mantiene el objeto, o jugando con punteros.
            # Para cada interlocutor intenta identificar nuevas palabras mapeándolas con su segmento de audio correspondiente
            # las palabras incompletas; intenta reconocerlas, y si lo consigue, añade a "mensajes" las nuevas palabras en cada interlocutor y 
            # devuelve flag_nuevo=true, y además modifica los INDEX de current_ones
    if flag_nuevo:
        emociones = VT_model.evaluate(mensajes) # intenta reconocer la emoción que hay en el par (texto palabra, audio palabra) nuevo que hay en 
            # los diferentes mensajes,  con una probabilidad. 
            # Tambíen revisa las palabras anteriores con las nuevas, en cada mensaje, para ver si puede modificar la emoción anterior con una 
            # mejor probabilidad gracias a la nueva palabra  # ¿evaluar palabra a palabra? ¿en pares, ternas....?
    
    visualizar_emociones(emociones, mensajes) # visualiza, por una parte la emoción actual de cada interlocutor con la probabilidad o certidumbre del modelo y
            # por otra parte la secuencia de emociones, con sus probabilidades, por las que ha ido pasando de más nueva a más antigua.
            # Se mostrará en la forma en que se decida, por GUI o como sea...

    if evento_final == true: # esto en realidad vendrá determinado por el GUI         
        d.close()
   

# timeit.timeit(while_loop) 
# ¿?no tiene mucho sentido medir la duración del bucle dado que será el tiempo real que tengo la aplicación activa. Lo que habría que medir es el tiempo que tarda el bucle en dar el salto atrás

# Cerramos el escuchador
stream.stop_stream()
stream.close()
p.terminate()

print('Finished recording')

# Save the recorded data as a WAV file
wf = wave.open(filename, 'wb')
wf.setnchannels(channels)
wf.setsampwidth(p.get_sample_size(sample_format))
wf.setframerate(fs)
wf.writeframes(b''.join(frames))
wf.close()



 ####################################################################################
 #                                                                                  # 
 #   ¿  Y   QUE     PROBLEMA    RESUELVE    ESTO    ?                               #
 #   vender la moto bien bien                                                       #
 #                                                                                  # 
 ####################################################################################

"""
from goto import with_goto
@with_goto  # Decorador necesario.
def f():
    label .get_input  # Definir porción del código.
    age = raw_input("Edad: ")
    try:
        age = int(age)
    except ValueError:
        goto .get_input  # Regresar a get_input.
f()
"""