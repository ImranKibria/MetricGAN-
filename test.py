"""
Specify Target Metric & corresponding model path for testing
"""

import os
import keras
import torch 
import scipy
import shutil
import librosa
import warnings
import numpy as np
import soundfile as sf
import tensorflow as tf
import tensorflow_datasets as tfds
from joblib import Parallel, delayed
from DNSMOS.dnsmos_local import dnsmos
from compute_metrics import compute_metrics
from SpectralNormalizationKeras import DenseSN, ConvSN2D
from torchmetrics.functional.audio.snr import scale_invariant_signal_noise_ratio
from torchmetrics.functional.audio.stoi import short_time_objective_intelligibility
from torchmetrics.functional.audio.pesq import perceptual_evaluation_speech_quality

warnings.filterwarnings("ignore")
print(tf.config.list_physical_devices('GPU'))

###
# Global Variables
###

TargetMetric = 'dnsmos' 
print(f'*** Testing for objective {TargetMetric} ***')

mask_min = 0.05
maxv = np.iinfo(np.int16).max 
output_path = os.path.join(f'objective_{TargetMetric}', 'output')

###
# Functions to Compute Metric Scores
###

def read_stoi(clean_root, enhanced_file, sr):
    wave_name = enhanced_file.split('/')[-1]
    clean_file = os.path.join(clean_root, wave_name)

    clean_wav, _ = librosa.load(clean_file, sr=sr)     
    en_wav, _ = librosa.load(enhanced_file, sr=sr)
    
    stoi_score = short_time_objective_intelligibility(torch.tensor(en_wav), torch.tensor(clean_wav), fs=sr, extended=True).numpy()
    return stoi_score

def read_batch_stoi(clean_root, enhanced_list):
    stoi_scores = Parallel(n_jobs=32)(delayed(read_stoi)(clean_root, en, 16000) for en in enhanced_list)
    return stoi_scores, enhanced_list

def read_pesq(clean_root, enhanced_file, sr):
    wave_name = enhanced_file.split('/')[-1]
    clean_file = os.path.join(clean_root, wave_name)

    clean_wav, _ = librosa.load(clean_file, sr=sr)     
    en_wav, _ = librosa.load(enhanced_file, sr=sr)
    
    pesq_score = perceptual_evaluation_speech_quality(torch.tensor(en_wav), torch.tensor(clean_wav), fs=sr, mode='wb').numpy()
    return pesq_score

def read_batch_pesq(clean_root, enhanced_list):
    pesq_scores = Parallel(n_jobs=32)(delayed(read_pesq)(clean_root, en, 16000) for en in enhanced_list)
    return pesq_scores, enhanced_list

def read_batch_dnsmos(clean_root, enhanced_list):
    enhanced_dir = os.path.dirname(enhanced_list[0])
    dnsmos_fnames, dnsmos_scores = dnsmos(enhanced_dir)
    dnsmos_scores = np.float32(dnsmos_scores)
    return dnsmos_scores, dnsmos_fnames

def read_si_snr(clean_root, enhanced_file, sr):
    wave_name = enhanced_file.split('/')[-1]
    clean_file = os.path.join(clean_root, wave_name)

    clean_wav, _ = librosa.load(clean_file, sr=sr)     
    en_wav, _ = librosa.load(enhanced_file, sr=sr)
    
    si_snr_score = scale_invariant_signal_noise_ratio(torch.tensor(en_wav), torch.tensor(clean_wav)).numpy()
    return si_snr_score

def read_batch_si_snr(clean_root, enhanced_list):
    si_snr_scores = Parallel(n_jobs=32)(delayed(read_si_snr)(clean_root, en, 16000) for en in enhanced_list)
    return si_snr_scores, enhanced_list

###
# Load Test Dataset
###

root_dir = '/fs/ess/PAS2301/Data/Speech/datasets_voicebank/noisy-vctk-16k'
test_clean_dir = os.path.join(root_dir, 'clean_testset_wav_16k')

def spect_2_wav(mag, phase, sig_len):
    Rec = np.multiply(mag , np.exp(1j*phase))
    result = librosa.istft(Rec,
                           hop_length=256,
                           win_length=512,
                           window=scipy.signal.hamming, length=sig_len)
    return result 

dataset_name = f'vctk/default:3.0.0'
(ds_train, ds_test), dataset_info = tfds.load(dataset_name, split=['TRAIN', 'TEST'],
  data_dir='/fs/scratch/PAS2301/Imran', shuffle_files=True, with_info=True,)
print(dataset_info)

num_test_samples = ds_test.cardinality().numpy()
print(f'Samples in test data = {num_test_samples}')

###
# Load Generator Model
###

@keras.saving.register_keras_serializable()
class Learnable_Sigmoid(tf.keras.layers.Layer):
    def __init__(self):
        super(Learnable_Sigmoid, self).__init__()

    def build(self, input_shape):
        self.alpha = self.add_weight(
            shape = (input_shape[-1]),
            initializer = "ones",
            trainable = True, 
            constraint=tf.keras.constraints.MaxNorm(max_value=3.5))

    def call(self, x):
        return 1.2 * keras.activations.sigmoid(self.alpha * x)

@keras.saving.register_keras_serializable()
class Generator_Model(keras.Model):
    def __init__(self):
        super(Generator_Model, self).__init__()
        
        self.BLSTM_1 = keras.layers.Bidirectional(keras.layers.LSTM(200, return_sequences=True), merge_mode='concat', name='BLSTM_1')
        self.BLSTM_2 = keras.layers.Bidirectional(keras.layers.LSTM(200, return_sequences=True), merge_mode='concat', name='BLSTM_2')

        self.linear_1 = keras.layers.TimeDistributed(keras.layers.Dense(300), name='Linear_1')
        self.actv_ftn = keras.layers.LeakyReLU(name='LeakyReLU')
        self.dropout_1 = keras.layers.Dropout(0.05, name='Dropout')

        self.linear_2 = keras.layers.TimeDistributed(keras.layers.Dense(257), name='Linear_2')
        self.out_actv = keras.layers.TimeDistributed(Learnable_Sigmoid(), name='Learnable_sigmoid')
  
    def call(self, inputs, training=False):
        x = inputs
        x = self.BLSTM_1(x)
        x = self.BLSTM_2(x)
        
        x = self.linear_1(x)
        x = self.actv_ftn(x)

        x = self.dropout_1(x, training=training)
        
        x = self.linear_2(x)
        x = self.out_actv(x)      
        return x

    def get_config(self):
        base_config = super().get_config()
        return {**base_config}

generator_model_path = '/users/PAS2301/kibria5/Research/quality_enhancement/metricgan+/objective_dnsmos/models/generator@135.keras'
generator_model = keras.saving.load_model(generator_model_path)
print(generator_model.summary())

###
# Enhance & Store Audio
###

shutil.rmtree(output_path+'/generator_testing', ignore_errors=True)
os.makedirs(output_path+'/generator_testing', exist_ok=True)

Test_en_Name = []
for record in ds_test:
    noisy_LP = tf.expand_dims(tf.math.log1p(record['deg_f_abs']), axis=0)
    gen_input = tf.transpose(noisy_LP, perm=[0, 2, 1])
    deg_phase = record['deg_f_phase']
    
    IRM = generator_model.predict(gen_input)            # shape: (1, None, 257)
    mask = tf.transpose(IRM, perm=[0, 2, 1])            # shape: (1, 257, None)
    E = noisy_LP * np.maximum(mask, mask_min)           # shape: (1, 257, None)                   
    
    enhanced_wav = spect_2_wav(mag = tf.math.expm1(np.squeeze(E)), 
                                phase = np.float32(deg_phase), 
                                sig_len = int(record['deg_len']))
            
    wave_name = record['name'].numpy().decode('utf-8') 
    enhanced_name = os.path.join(output_path, 'generator_testing', wave_name)
    sf.write(enhanced_name, np.int16(enhanced_wav* maxv), 16000)
    
    Test_en_Name.append(enhanced_name)

###
# Determine Metric Results
###

pesq_scores, _ = read_batch_pesq(test_clean_dir, Test_en_Name)    
print('\nEnhanced PESQ: ', np.mean(pesq_scores))

estoi_scores, _ = read_batch_stoi(test_clean_dir, Test_en_Name)    
print('\nEnhanced ESTOI: ', np.mean(estoi_scores))

dnsmos_scores, _ = read_batch_dnsmos(test_clean_dir, Test_en_Name)    
print('\nEnhanced DNSMOS: ', np.mean(dnsmos_scores))

si_snr_scores, _ = read_batch_si_snr(test_clean_dir, Test_en_Name)    
print('\nEnhanced SI-SNR: ', np.mean(si_snr_scores))

Csig, Cbak, Covl = np.zeros(num_test_samples), np.zeros(num_test_samples), np.zeros(num_test_samples)
for index in range(num_test_samples):
    enhanced_file = Test_en_Name[index]
    wave_name = enhanced_file.split('/')[-1]
    clean_file = os.path.join(test_clean_dir, wave_name)

    clean_wav, _ = librosa.load(clean_file, sr=16000)     
    en_wav, _ = librosa.load(enhanced_file, sr=16000)
    
    _, Csig[index], Cbak[index], Covl[index], _, _ = compute_metrics(clean_wav, en_wav, 16000, 0)

print('\nEnhanced CSIG: ', np.mean(Csig))
print('\nEnhanced CBAK: ', np.mean(Cbak))
print('\nEnhanced COVL: ', np.mean(Covl))