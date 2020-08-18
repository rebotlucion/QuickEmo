# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 20:13:42 2020

@author: fco_p
"""

from __future__ import print_function
import os
import numpy
#import glob
#import matplotlib.pyplot as plt
from pyAudioAnalysis import audioFeatureExtraction as aF
from pyAudioAnalysis import audioTrainTest as aT
from pyAudioAnalysis import audioSegmentation as aS
from pyAudioAnalysis import audioVisualization as aV
from pyAudioAnalysis import audioBasicIO
import scipy.io.wavfile as wavfile
import matplotlib.patches

3#QE_SpeechTextModel # Es un paquete para almacenar un volcado de los parámetros del modelo que hemos conseguido entrenar, tiene tanto la parte de speech como la de text


## Basics ##
import time
import os
import numpy as np

## Audio Preprocessing ##
import pyaudio
import wave
import librosa
from scipy.stats import zscore

## text processing
#import sentiment-analysis-spanish
from sentiment_analysis import SentimentAnalysisSpanish
    
"""
## Time Distributed CNN ##
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, TimeDistributed
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten
from tensorflow.keras.layers import LSTM
"""

'''
Speech Emotion Recognition
'''

# Es un paquete para almacenar un volcado de los parámetros del modelo que hemos conseguido 
# entrenar, tiene tanto la parte de speech como la de text

class QE_Voice_Text_Model:

    '''
    Voice recording function
    '''
    def __init__(self, subdir_model=None):

        # voice = Load modelo de voz
        if subdir_model is not None:
            self._model = self.build_model()
            self._model.load_weights(subdir_model)
        # Emotion encoding
        self._emotion = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Neutral', 5:'Sad', 6:'Surprise'}

        # text = Load modelo de texto
        #
        #
        #

    def voice (self, bigran1, bigram2):
        # Split audio signals into chunks
        chunks = self.frame(y.reshape(1, 1, -1), chunk_step, chunk_size)
        # Reshape chunks
        chunks = chunks.reshape(chunks.shape[1],chunks.shape[-1])
        # Z-normalization
        y = np.asarray(list(map(zscore, chunks)))
        # Compute mel spectrogram
        mel_spect = np.asarray(list(map(self.mel_spectrogram, y))) # EN DIARIZATION YA LOS HABÍA CALCULADO, 
        #¿NO PUEDO SOLAPAR DE ALGUNA FORMA AMBAS COSAS?
        # TAL VEZ DEBO DE INVENTAR UNA CLASE QUE VAYA REGISTRANDO LOS MEL Y OTRAS CARACTERÍSTICAS, Y ARMONIZAR EL TAMAÑO DE LAS VENTANAS
        # Time distributed Framing
        mel_spect_ts = self.frame(mel_spect)
        # Build X for time distributed CNN
        X = mel_spect_ts.reshape(mel_spect_ts.shape[0],
                                    mel_spect_ts.shape[1],
                                    mel_spect_ts.shape[2],
                                    mel_spect_ts.shape[3],
                                    1)
        # Predict emotion
        #if predict_proba is True:
        #    predict = self._model.predict(X)
        #else:
        #    predict = np.argmax(self._model.predict(X), axis=1)
        #    predict = [self._emotion.get(emotion) for emotion in predict]
        predict = self._model.predict(X)
        # Clear Keras session
        K.clear_session()
        # Predict timestamp
        #timestamp = np.concatenate([[chunk_size], np.ones((len(predict) - 1)) * chunk_step]).cumsum()
        #timestamp = np.round(timestamp / sample_rate)
        return predict # [predict, timestamp]

    def text (self, text):
        #pip install sentiment-analysis-spanish
        sentiment = SentimentAnalysisSpanish()
        return sentiment.sentiment(text)
        """> 0.5, "Sentiment should be possitive"
           < 0.5, "Sentiment should be negative"
        """

    def evaluate(self, mensajes_interlocutores, current):
        # mensajes:  lista de mensajes de cada uno de los interlocutores
        # un mensaje es una lista ordenada (según la secuencia de tiempo ) de
        # pares (texto palabra, audio palabra)

        # otros posibles argumentos 
        # chunk_step=16000, chunk_size=49100, predict_proba=False, sample_rate=16000):

        # get audio file
        # y, sr = audio_from_mensajes(mensajes)#, sr=sample_rate, offset=0.5)

        
        emotions_interlocutors = []
        
        for mensajes in mensajes_interlocutores:
            pending_segment = len(mensajes)-current
            emotions = []
            emotions_vector = []

            for i in range(pending_segment-1):
                # evaluar de dos en dos con saltos de 1
                # almacenar emocion en la primera palabra de cada 2

                emotions = voice(self, mensajes[i][1],mensajes[i+1][1]) #o mensajes[current](1))
                prob_text = text(self, mensajes[i][0],mensajes[i+1][0]) #o mensajes[current](0))

                prob_text_avg = prob_text/2

                #fusionar la emoción en el vector agregado, palabra a palabra, partiendo de 
                # lo que se ha conseguido intentificar en voz y ya veremos si el texto 
                # tiene 7 emociones o solo algo postivo o algo negativo o neutro
                if prob_text > 0.6: # potenciar probabilidad de las positivas y reducir las negativas
                                    # a otro nivel de la jerarquía más alto, quedarse con lo más frecuente
                    emotions[0] = emotions[0]-emotions[0]*prob_text_avg
                    emotions[1] = emotions[1]-emotions[1]*prob_text_avg
                    emotions[2] = emotions[2]-emotions[2]*prob_text_avg
                    emotions[3] = emotions[3] + (1-emotions[3])*prob_text_avg
                    emotions[4] = emotions[4]-emotions[4]*prob_text_avg
                    emotions[5] = emotions[5]-emotions[5]*prob_text_avg
                    emotions[6] = emotions[6 + (1-emotions[6])*prob_text_avg
                # ¿como argumento el mapeo de hemociones realizado?
                
                if prob_text < 0.4: # potenciar probabilidad de las positivas y reducir las negativas
                                    # a otro nivel de la jerarquía más alto, quedarse con lo más frecuente
                    emotions[0] = emotions[0] +(1-emotions[0])*(1-prob_text)/2
                    emotions[1] = emotions[1] +(1-emotions[1])*(1-prob_text)/2
                    emotions[2] = emotions[2] +(1-emotions[2])*(1-prob_text)/2
                    emotions[3] = emotions[3] - emotions[3]*(1-prob_text)/2
                    emotions[4] = emotions[4] +(1-emotions[4])*(1-prob_text)/2
                    emotions[5] = emotions[5] +(1-emotions[5])*(1-prob_text)/2
                    emotions[6] = emotions[6] - emotions[6]*(1-prob_text)/2

                emotions_vector.append( (np.argmax(emotions, axis=1), la prob del máximos )

                emotions_weights = [0,0,0,0,0,0,0]
                emotions_weights[0] += emotions_vector[x](i,) if i==0
                emotions_weights[1] += emotions_vector[x](i,) if i==0
                emotions_weights[2] += emotions_vector[x](i,) if i==0
                emotions_weights[3] += emotions_vector[x](i,) if i==0
                emotions_weights[4] += emotions_vector[x](i,) if i==0
                emotions_weights[5] += emotions_vector[x](i,) if i==0
                emotions_weights[6] += emotions_vector[x](i,) if i==0
                emotions_consens = np.argmax(emotions, axis=1)
        
                emotions_interlocutors.append((emotions_consens, emotions_weights))
        
        #predict = [self._emotion.get(emotion) for emotion in emotions]
        
        return emotions_interlocutors
        # lista ordenada de emociones asociadas a cada interlocutor en cada momento de la conversación, ternas (emoción, probabilidad, tiempo o id_palabra donde sucede)
               # la emoción más reciente de cada interlocutor, la que está sucendiendo en el momento de hablar, está en un extremo de la lista (¿principio? 
               # ¿más antiguos hacia el final?)
               # almacenar y continuar por donde lo dejé????

    '''
    Voice recording function
    '''
    """
    def voice_recording(self, filename, duration=5, sample_rate=16000, chunk=1024, channels=1):

        # Start the audio recording stream
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
                        channels=channels,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=chunk)

        # Create an empty list to store audio recording
        frames = []

        # Determine the timestamp of the start of the response interval
        print('* Start Recording *')
        stream.start_stream()
        start_time = time.time()
        current_time = time.time()

        # Record audio until timeout
        while (current_time - start_time) < duration:

            # Record data audio data
            data = stream.read(chunk)

            # Add the data to a buffer (a list of chunks)
            frames.append(data)

            # Get new timestamp
            current_time = time.time()

        # Close the audio recording stream
        stream.stop_stream()
        stream.close()
        p.terminate()
        print('* End Recording * ')

        # Export audio recording to wav format
        wf = wave.open(filename, 'w')
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))
        wf.close()
"""

    '''
    Mel-spectogram computation
    '''
    def mel_spectrogram(self, y, sr=16000, n_fft=512, win_length=256, hop_length=128, window='hamming', n_mels=128, fmax=4000):

        # Compute spectogram
        mel_spect = np.abs(librosa.stft(y, n_fft=n_fft, window=window, win_length=win_length, hop_length=hop_length)) ** 2

        # Compute mel spectrogram
        mel_spect = librosa.feature.melspectrogram(S=mel_spect, sr=sr, n_mels=n_mels, fmax=fmax)

        # Compute log-mel spectrogram
        mel_spect = librosa.power_to_db(mel_spect, ref=np.max)

        return np.asarray(mel_spect)


    '''
    Audio framing
    '''
    def frame(self, y, win_step=64, win_size=128):

        # Number of frames
        nb_frames = 1 + int((y.shape[2] - win_size) / win_step)

        # Framming
        frames = np.zeros((y.shape[0], nb_frames, y.shape[1], win_size)).astype(np.float16)
        for t in range(nb_frames):
            frames[:,t,:,:] = np.copy(y[:,:,(t * win_step):(t * win_step + win_size)]).astype(np.float16)

        return frames


    '''
    Time distributed Convolutional Neural Network model
    '''
    def build_model(self):

        # Clear Keras session
        K.clear_session()

        # Define input
        input_y = Input(shape=(5, 128, 128, 1), name='Input_MELSPECT')

        # First LFLB (local feature learning block)
        y = TimeDistributed(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same'), name='Conv_1_MELSPECT')(input_y)
        y = TimeDistributed(BatchNormalization(), name='BatchNorm_1_MELSPECT')(y)
        y = TimeDistributed(Activation('elu'), name='Activ_1_MELSPECT')(y)
        y = TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'), name='MaxPool_1_MELSPECT')(y)
        y = TimeDistributed(Dropout(0.2), name='Drop_1_MELSPECT')(y)

        # Second LFLB (local feature learning block)
        y = TimeDistributed(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same'), name='Conv_2_MELSPECT')(y)
        y = TimeDistributed(BatchNormalization(), name='BatchNorm_2_MELSPECT')(y)
        y = TimeDistributed(Activation('elu'), name='Activ_2_MELSPECT')(y)
        y = TimeDistributed(MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same'), name='MaxPool_2_MELSPECT')(y)
        y = TimeDistributed(Dropout(0.2), name='Drop_2_MELSPECT')(y)

        # Third LFLB (local feature learning block)
        y = TimeDistributed(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same'), name='Conv_3_MELSPECT')(y)
        y = TimeDistributed(BatchNormalization(), name='BatchNorm_3_MELSPECT')(y)
        y = TimeDistributed(Activation('elu'), name='Activ_3_MELSPECT')(y)
        y = TimeDistributed(MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same'), name='MaxPool_3_MELSPECT')(y)
        y = TimeDistributed(Dropout(0.2), name='Drop_3_MELSPECT')(y)

        # Fourth LFLB (local feature learning block)
        y = TimeDistributed(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same'), name='Conv_4_MELSPECT')(y)
        y = TimeDistributed(BatchNormalization(), name='BatchNorm_4_MELSPECT')(y)
        y = TimeDistributed(Activation('elu'), name='Activ_4_MELSPECT')(y)
        y = TimeDistributed(MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same'), name='MaxPool_4_MELSPECT')(y)
        y = TimeDistributed(Dropout(0.2), name='Drop_4_MELSPECT')(y)

        # Flat
        y = TimeDistributed(Flatten(), name='Flat_MELSPECT')(y)

        # LSTM layer
        y = LSTM(256, return_sequences=False, dropout=0.2, name='LSTM_1')(y)

        # Fully connected
        y = Dense(7, activation='softmax', name='FC')(y)

        # Build final model
        model = Model(inputs=input_y, outputs=y)

        return model


    '''
    Predict speech emotion over time from an audio file
    '''
    def predict_emotion_from_file(self, filename, chunk_step=16000, 
    chunk_size=49100, predict_proba=False, sample_rate=16000):

        # Read audio file
        y, sr = librosa.core.load(filename, sr=sample_rate, offset=0.5)

        # Split audio signals into chunks
        chunks = self.frame(y.reshape(1, 1, -1), chunk_step, chunk_size)

        # Reshape chunks
        chunks = chunks.reshape(chunks.shape[1],chunks.shape[-1])

        # Z-normalization
        y = np.asarray(list(map(zscore, chunks)))

        # Compute mel spectrogram
        mel_spect = np.asarray(list(map(self.mel_spectrogram, y)))

        # Time distributed Framing
        mel_spect_ts = self.frame(mel_spect)

        # Build X for time distributed CNN
        X = mel_spect_ts.reshape(mel_spect_ts.shape[0],
                                    mel_spect_ts.shape[1],
                                    mel_spect_ts.shape[2],
                                    mel_spect_ts.shape[3],
                                    1)

        # Predict emotion
        if predict_proba is True:
            predict = self._model.predict(X)
        else:
            predict = np.argmax(self._model.predict(X), axis=1)
            predict = [self._emotion.get(emotion) for emotion in predict]

        # Clear Keras session
        K.clear_session()

        # Predict timestamp
        timestamp = np.concatenate([[chunk_size], np.ones((len(predict) - 1)) * chunk_step]).cumsum()
        timestamp = np.round(timestamp / sample_rate)

        return [predict, timestamp]

    '''
    Export emotions predicted to csv format
    '''
    """
    def prediction_to_csv(self, predictions, filename, mode='w'):

        # Write emotion in filename
        with open(filename, mode) as f:
            if mode == 'w':
                f.write("EMOTIONS"+'\n')
            for emotion in predictions:
                f.write(str(emotion)+'\n')
            f.close()
"""