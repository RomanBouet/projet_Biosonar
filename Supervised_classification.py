import soundfile as sf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import configparser
import numpy as np
from obspy import UTCDateTime, read, Trace, Stream
from obspy.signal.trigger import classic_sta_lta,plot_trigger
from scipy.fft import fft, ifft
import librosa.feature as feat
from tqdm import tqdm
from scipy import signal
import soundfile as sf
#########################################################
#                       Fonctions                       #
#########################################################


def load_soundfile(file_path, name_file):

    Path = file_path+"\\"+name_file
    sig, n_sample = sf.read(Path)
    sos = signal.butter(6, [5000, 100000], 'bandpass', fs=n_sample, output='sos')
    sig = signal.sosfiltfilt(sos, sig)
    trace = Trace(sig)
    trace.stats.sampling_rate = 256000
    trace.stats.station = name_file[0:-3]
    return trace, n_sample



def make_features(sig):
    # J'initialise mes variables qui vont contenir les features

    STA_LTA_MAX = 0

    # Je réalise une fenêtre glissante
    iter = 0
    T=0.2
    fech = sig.stats.sampling_rate
    window_length = 512/fech
    step = window_length/2
    for windowed_tr in tqdm(sig.slide(window_length=window_length, step=step)):


    # Je calcule le STA/LTA pour chercher dans l'extrait audio l'impulsion qui m'interresse

        STA_LTA = classic_sta_lta(windowed_tr,nsta=256,nlta=512)
        if STA_LTA.max() > STA_LTA_MAX:

            windowed_tr_computation = windowed_tr.copy()
            STA_LTA_MAX = STA_LTA.max()




    # CALCUL DES FEATURES :
    rms = feat.rms(y=windowed_tr_computation.data, frame_length=256,  hop_length=64) 
    sc = feat.spectral_centroid(y=windowed_tr_computation.data, sr=fech,n_fft=256, hop_length=64)
    sb = feat.spectral_bandwidth(y=windowed_tr_computation.data, sr=fech,n_fft=256, hop_length=64)
    sf = feat.spectral_flatness(y=windowed_tr_computation.data, n_fft=256, hop_length=64)

    features = [[np.mean(rms), np.std(rms), np.min(rms), np.max(rms),\
            np.mean(sc), np.std(sc), np.min(sc), np.max(sc),\
            np.mean(sb), np.std(sb), np.min(sb), np.max(sb),\
            np.mean(sf), np.std(sf), np.min(sf), np.max(sf),\
            STA_LTA_MAX]]

    return features

#########################################################
#               Chargement des paramètres               #
#########################################################


#load du fichier configparser

config = configparser.ConfigParser()
config.sections()
config.read("Config_parametre.ini",encoding="utf-8")

#ajout des paramètres

dir_data = config["Paramètres"]["dir_data"]
file_path_data_information =  config["Paramètres"]["file_path_data_information"]


#########################################################
#                     Programme                         #
#########################################################



# Je charge le fichier excel_id contenant l'information des id des fichiers audio

df_train_info = pd.read_csv(file_path_data_information,header=0)

# Je calcul pour chaque fichier audio les features et je les enregistres dans un fichiers np.

features = np.empty((1,17))  # ATTENTION IL FAUT SUPPRIMER LE TOUT PREMIER ECHANTILLONS LORS DU CHARGEMENT
                             # DES DONNEES TEST A CAUSE DE CETTE LIGNE

for name_file in tqdm(df_train_info['id']):
    sig, n_sample = load_soundfile(dir_data,name_file)
    new_features = make_features(sig)
    features = np.concatenate((features,new_features), axis=0)


np.save('data_Test',features)
