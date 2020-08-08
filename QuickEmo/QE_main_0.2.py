sss#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Created on Sat Jul 11 20:13:42 2020

@author: fco_p
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
from __future__ import division
import numpy as np
import pandas as pd
import time
import re
import os
from collections import Counter
import altair as alt

### Audio imports ###
from library.speech_emotion_recognition import *

### Text imports ###
from library.text_emotion_recognition import *
from library.text_preprocessor import *
from nltk import *
from tika import parser
from werkzeug.utils import secure_filename
import tempfile


    def emotion_label(emotion) :
        if emotion == 0 :
            return "Angry"
        elif emotion == 1 :
            return "Disgust"
        elif emotion == 2 :
            return "Fear"
        elif emotion == 3 :
            return "Happy"
        elif emotion == 4 :
            return "Sad"
        elif emotion == 5 :
            return "Surprise"
        else :
            return "Neutral"

    ### Altair Plot
    df_altair = pd.read_csv('static/js/db/prob.csv', header=None, index_col=None).reset_index()
    df_altair.columns = ['Time', 'Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    
    angry = alt.Chart(df_altair).mark_line(color='orange', strokeWidth=2).encode(
       x='Time:Q',
       y='Angry:Q',
       tooltip=["Angry"]
    )

    disgust = alt.Chart(df_altair).mark_line(color='red', strokeWidth=2).encode(
        x='Time:Q',
        y='Disgust:Q',
        tooltip=["Disgust"])


    fear = alt.Chart(df_altair).mark_line(color='green', strokeWidth=2).encode(
        x='Time:Q',
        y='Fear:Q',
        tooltip=["Fear"])


    happy = alt.Chart(df_altair).mark_line(color='blue', strokeWidth=2).encode(
        x='Time:Q',
        y='Happy:Q',
        tooltip=["Happy"])


    sad = alt.Chart(df_altair).mark_line(color='black', strokeWidth=2).encode(
        x='Time:Q',
        y='Sad:Q',
        tooltip=["Sad"])


    surprise = alt.Chart(df_altair).mark_line(color='pink', strokeWidth=2).encode(
        x='Time:Q',
        y='Surprise:Q',
        tooltip=["Surprise"])


    neutral = alt.Chart(df_altair).mark_line(color='brown', strokeWidth=2).encode(
        x='Time:Q',
        y='Neutral:Q',
        tooltip=["Neutral"])


    chart = (angry + disgust + fear + happy + sad + surprise + neutral).properties(
    width=1000, height=400, title='Probability of each emotion over time')

    chart.save('static/CSS/chart.html')
    
    return render_template('video_dash.html', emo=emotion_label(emotion), emo_other = emotion_label(emotion_other), prob = emo_prop(df_2), prob_other = emo_prop(df))

# Audio Recording
def audio_recording():

    # Instanciate new SpeechEmotionRecognition object
    SER = speechEmotionRecognition()

    # Voice Recording
    rec_duration = 16 # in sec
    rec_sub_dir = os.path.join('tmp','voice_recording.wav')
    SER.voice_recording(rec_sub_dir, duration=rec_duration)

# Audio Emotion Analysis
def audio_dash():

    # Sub dir to speech emotion recognition model
    model_sub_dir = os.path.join('Models', 'audio.hdf5')

    # Instanciate new SpeechEmotionRecognition object
    SER = speechEmotionRecognition(model_sub_dir)

    # Voice Record sub dir
    rec_sub_dir = os.path.join('tmp','voice_recording.wav')

    # Predict emotion in voice at each time step
    step = 1 # in sec
    sample_rate = 16000 # in kHz
    emotions, timestamp = SER.predict_emotion_from_file(rec_sub_dir, chunk_step=step*sample_rate)

    # Export predicted emotions to .txt format
    SER.prediction_to_csv(emotions, os.path.join("static/js/db", "audio_emotions.txt"), mode='w')
    SER.prediction_to_csv(emotions, os.path.join("static/js/db", "audio_emotions_other.txt"), mode='a')

    # Get most common emotion during the interview
    major_emotion = max(set(emotions), key=emotions.count)

    # Calculate emotion distribution
    emotion_dist = [int(100 * emotions.count(emotion) / len(emotions)) for emotion in SER._emotion.values()]

    # Export emotion distribution to .csv format for D3JS
    df = pd.DataFrame(emotion_dist, index=SER._emotion.values(), columns=['VALUE']).rename_axis('EMOTION')
    df.to_csv(os.path.join('static/js/db','audio_emotions_dist.txt'), sep=',')

    # Sleep
    time.sleep(0.5)#  ¿yo lo necesito?

## En mi caso lo que necesito es sacar el texto del audio anterior

global df_text

tempdirectory = tempfile.gettempdir()

# para entrevistas de trabajo puede estar bien, ¿puede servir a mi modelo que 
# como parte de la predicción vayamos construyendo un modelo de personalidad 
# según progresa la interacción?
def get_personality(text):
    try:
        pred = predict().run(text, model_name = "Personality_traits_NN")
        return pred
    except KeyError:
        return None



def text_1():
    
    text = request.form.get('text')
    traits = ['Extraversion', 'Neuroticism', 'Agreeableness', 'Conscientiousness', 'Openness']
        # estas etiquetas son distintas a voz, ¿debería de estar alineado o son complementarios?
        # ¿fusionar o dar información complementaria y diferente
    probas = get_personality(text)[0].tolist()
    
    df_text = pd.read_csv('static/js/db/text.txt', sep=",")
    df_new = df_text.append(pd.DataFrame([probas], columns=traits))
    df_new.to_csv('static/js/db/text.txt', sep=",", index=False)
    
    perso = {}
    perso['Extraversion'] = probas[0]
    perso['Neuroticism'] = probas[1]
    perso['Agreeableness'] = probas[2]
    perso['Conscientiousness'] = probas[3]
    perso['Openness'] = probas[4]
    
    df_text_perso = pd.DataFrame.from_dict(perso, orient='index')
    df_text_perso = df_text_perso.reset_index()
    df_text_perso.columns = ['Trait', 'Value']
    
    df_text_perso.to_csv('static/js/db/text_perso.txt', sep=',', index=False)
    
    means = {}
    means['Extraversion'] = np.mean(df_new['Extraversion'])
    means['Neuroticism'] = np.mean(df_new['Neuroticism'])
    means['Agreeableness'] = np.mean(df_new['Agreeableness'])
    means['Conscientiousness'] = np.mean(df_new['Conscientiousness'])
    means['Openness'] = np.mean(df_new['Openness'])
    
    probas_others = [np.mean(df_new['Extraversion']), np.mean(df_new['Neuroticism']), np.mean(df_new['Agreeableness']), np.mean(df_new['Conscientiousness']), np.mean(df_new['Openness'])]
    probas_others = [int(e*100) for e in probas_others]
    
    df_mean = pd.DataFrame.from_dict(means, orient='index')
    df_mean = df_mean.reset_index()
    df_mean.columns = ['Trait', 'Value']
    
    df_mean.to_csv('static/js/db/text_mean.txt', sep=',', index=False)
    trait_others = df_mean.loc[df_mean['Value'].idxmax()]['Trait']
    
    probas = [int(e*100) for e in probas]
    
    data_traits = zip(traits, probas)
    
    session['probas'] = probas
    session['text_info'] = {}
    session['text_info']["common_words"] = []
    session['text_info']["num_words"] = []
    
    preprocessed_text = preprocess_text(text)
    common_words, num_words, counts = get_text_info(preprocessed_text)
    
    session['text_info']["common_words"].append(common_words)
    session['text_info']["num_words"].append(num_words)
    
    trait = traits[probas.index(max(probas))]
    
    with open("static/js/db/words_perso.txt", "w") as d:
        d.write("WORDS,FREQ" + '\n')
        for line in counts :
            d.write(line + "," + str(counts[line]) + '\n')
        d.close()
    
    with open("static/js/db/words_common.txt", "a") as d:
        for line in counts :
            d.write(line + "," + str(counts[line]) + '\n')
        d.close()

    df_words_co = pd.read_csv('static/js/db/words_common.txt', sep=',', error_bad_lines=False)
    df_words_co.FREQ = df_words_co.FREQ.apply(pd.to_numeric)
    df_words_co = df_words_co.groupby('WORDS').sum().reset_index()
    df_words_co.to_csv('static/js/db/words_common.txt', sep=",", index=False)
    common_words_others = df_words_co.sort_values(by=['FREQ'], ascending=False)['WORDS'][:15]

    df_words_perso = pd.read_csv('static/js/db/words_perso.txt', sep=',', error_bad_lines=False)
    common_words_perso = df_words_perso.sort_values(by=['FREQ'], ascending=False)['WORDS'][:15]

    return render_template('text_dash.html', traits = probas, trait = trait, trait_others = trait_others, probas_others = probas_others, num_words = num_words, common_words = common_words_perso, common_words_others=common_words_others)

ALLOWED_EXTENSIONS = set(['pdf'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def text_pdf():
    f = request.files['file']
    f.save(secure_filename(f.filename))
    
    text = parser.from_file(f.filename)['content']
    traits = ['Extraversion', 'Neuroticism', 'Agreeableness', 'Conscientiousness', 'Openness']
    probas = get_personality(text)[0].tolist()
    
    df_text = pd.read_csv('static/js/db/text.txt', sep=",")
    df_new = df_text.append(pd.DataFrame([probas], columns=traits))
    df_new.to_csv('static/js/db/text.txt', sep=",", index=False)
    
    perso = {}
    perso['Extraversion'] = probas[0]
    perso['Neuroticism'] = probas[1]
    perso['Agreeableness'] = probas[2]
    perso['Conscientiousness'] = probas[3]
    perso['Openness'] = probas[4]
    
    df_text_perso = pd.DataFrame.from_dict(perso, orient='index')
    df_text_perso = df_text_perso.reset_index()
    df_text_perso.columns = ['Trait', 'Value']
    
    df_text_perso.to_csv('static/js/db/text_perso.txt', sep=',', index=False)
    
    means = {}
    means['Extraversion'] = np.mean(df_new['Extraversion'])
    means['Neuroticism'] = np.mean(df_new['Neuroticism'])
    means['Agreeableness'] = np.mean(df_new['Agreeableness'])
    means['Conscientiousness'] = np.mean(df_new['Conscientiousness'])
    means['Openness'] = np.mean(df_new['Openness'])
    
    probas_others = [np.mean(df_new['Extraversion']), np.mean(df_new['Neuroticism']), np.mean(df_new['Agreeableness']), np.mean(df_new['Conscientiousness']), np.mean(df_new['Openness'])]
    probas_others = [int(e*100) for e in probas_others]
    
    df_mean = pd.DataFrame.from_dict(means, orient='index')
    df_mean = df_mean.reset_index()
    df_mean.columns = ['Trait', 'Value']
    
    df_mean.to_csv('static/js/db/text_mean.txt', sep=',', index=False)
    trait_others = df_mean.ix[df_mean['Value'].idxmax()]['Trait']
    
    probas = [int(e*100) for e in probas]
    
    data_traits = zip(traits, probas)
    
    session['probas'] = probas
    session['text_info'] = {}
    session['text_info']["common_words"] = []
    session['text_info']["num_words"] = []
    
    preprocessed_text = preprocess_text(text)
    common_words, num_words, counts = get_text_info(preprocessed_text)
    
    session['text_info']["common_words"].append(common_words)
    session['text_info']["num_words"].append(num_words)
    
    trait = traits[probas.index(max(probas))]
    
    with open("static/js/db/words_perso.txt", "w") as d:
        d.write("WORDS,FREQ" + '\n')
        for line in counts :
            d.write(line + "," + str(counts[line]) + '\n')
        d.close()
    
    with open("static/js/db/words_common.txt", "a") as d:
        for line in counts :
            d.write(line + "," + str(counts[line]) + '\n')
        d.close()

    df_words_co = pd.read_csv('static/js/db/words_common.txt', sep=',', error_bad_lines=False)
    df_words_co.FREQ = df_words_co.FREQ.apply(pd.to_numeric)
    df_words_co = df_words_co.groupby('WORDS').sum().reset_index()
    df_words_co.to_csv('static/js/db/words_common.txt', sep=",", index=False)
    common_words_others = df_words_co.sort_values(by=['FREQ'], ascending=False)['WORDS'][:15]

    df_words_perso = pd.read_csv('static/js/db/words_perso.txt', sep=',', error_bad_lines=False)
    common_words_perso = df_words_perso.sort_values(by=['FREQ'], ascending=False)['WORDS'][:15]

    return render_template('text_dash.html', traits = probas, trait = trait, trait_others = trait_others, probas_others = probas_others, num_words = num_words, common_words = common_words_perso, common_words_others=common_words_others)

"""
Cuando tenga el GUI hecho, el control del flujo de la herramienta se llevará desde el GUI, por lo que habrá que lanzar el gui aquí

if __name__ == '__main__':
    app.run(debug=True)

"""

if __name__ == "__main__":
    print("Vamos a ello")


import modeloVozTexto

escucahdor = orejas.crear() # pendiente de resolver. Abrir un "escuchador" que va registrando el audio y generando "chunks" de audio que vamos procesando poco a poco en tiempo real según suceden
modeloVT = modeloVozTexto.crear() # carga el modelo que tenemos entrenado. Es un volcado de los parámetros del modelo que hemos conseguido entrenar, tiene tanto la parte de speech como la de text

interlocutores = [] # lista de identificadores de interlocutores, que son ternas (id_interlocutor, género interlocutor, rango edad interlocutor)
emociones = [] # lista ordenada de emociones asociadas a cada interlocutor en cada momento de la conversación, ternas (emoción, probabilidad, tiempo o id_palabra donde sucede)
                # la emoción más reciente de cada interlocutor, la que está sucendiendo en el momento de hablar, está en un extremo de la lista (¿principio? ¿más antiguos hacia el final?)
palabras = [] # palabras pronunciandose de cada interlocutor o corte de audio de palabra incompleta (aún no identificada) por interlocutor
mensajes = [] # lista de mensajes de cada uno de los interlocutores
                # un mensaje es una lista ordenada de pares (texto palabra, audio palabra)

with escuchador.open() as esc:
    for corte in esc.sonido_in :
        actuales = audioX.diarization(corte) # separa los interlocutores, devuelve una lista de audios separadas por cada interlocutor, en forma
                # de  duplass (    audio,    interlocutor(id_interlocutor, género interlocutor, rango edad interlocutor)      )
        interlocutores = audioX.incorporar(actuales) #
        palabras, mensajes, flag_nuevo = audioX.pegar(palabras, actuales) # incorpora audios en "actuales" por cada interlocutor a las palabras incompletas, intenta reconocerlas, 
                # y si lo consigue, añade a "mensajes" las nuevas palabras en cada interlocutor y devuelve flag_nuevo=true, y además elimina de "palabras" los segmentos de audio que han sido identificados
        if flag_nuevo
            modeloVT.evaluar_emociones(mensajes) # intenta reconocer la emoción que hay en el par (texto palabra, audio palabra) nuevo que hay en los diferentes mensajes,  con una probabilidad. 
                # Tambíen revisa las palabras anteriores con las nuevas, en cada mensaje, para ver si puede modificar la emoción anterior con una mejor probabilidad gracias a la nueva palabra
                # ¿evaluar palabra a palabra? ¿en pares, ternas....?
        
        visualizarEmociones(mensajes) # visualiza, por una parte la emoción actual de cada interlocutor con la probabilidad o certidumbre del modelo y
                # por otra parte la secuendia de emociones, con sus probabilidades, por las que ha ido pasando de más nueva a más antigua.
                # Se mostrará en la forma en que se decida, por GUI o como sea...

        if evento_final == true
        
        d.close()
   
 ####################################################################################
 #                                                                                  # 
 #   ¿  Y   QUE     PROBLEMA    RESUELVE    ESTO    ?                               #
 #   vender la moto bien bien                                                       #
 #                                                                                  # 
 ####################################################################################
    