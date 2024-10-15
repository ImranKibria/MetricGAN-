"""
Ammendments: 
1) input spectrum to generator is not normalized. 
2) generator objective is 2-fold, metric optimize & match ref.
3) input to generator/discriminator is log magnitude spectrogam.
4) replay buffer consists of enhanced audio only, not ref & deg.
5) learnable sigmoid has an upper bound to avoid binary masking. 

6) check if I am writing files okay, like do I need normalizing before writing?
"""

import os
import time 
import keras
import torch 
import scipy
import shutil
import random
import librosa
import warnings
import numpy as np
import soundfile as sf
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from joblib import Parallel, delayed
from DNSMOS.dnsmos_local import dnsmos
from SpectralNormalizationKeras import DenseSN, ConvSN2D
from torchmetrics.functional.audio.stoi import short_time_objective_intelligibility
from torchmetrics.functional.audio.pesq import perceptual_evaluation_speech_quality

warnings.filterwarnings("ignore")
print(tf.config.list_physical_devices('GPU'))


###
# Global Variables
###


TargetMetric = 'pesq' 
models_path = os.path.join(f'objective_{TargetMetric}', 'models')
output_path = os.path.join(f'objective_{TargetMetric}', 'output')
shutil.rmtree(models_path, ignore_errors=True)
shutil.rmtree(output_path, ignore_errors=True)

BATCH_SIZE = 1
mask_min = 0.05
total_iterations = 750          # 600 iterations for pesq/stoi
num_train_samples = 100         # training samples in an iteration
num_valid_samples = 824         # validation samples in an iteration
maxv = np.iinfo(np.int16).max 


###
# Global Functions
###

  
def read_stoi(clean_root, enhanced_file, sr):
    wave_name = enhanced_file.split('/')[-1]
    clean_file = os.path.join(clean_root, wave_name)

    clean_wav, _ = librosa.load(clean_file, sr=sr)     
    en_wav, _ = librosa.load(enhanced_file, sr=sr)
    
    stoi_score = short_time_objective_intelligibility(torch.tensor(en_wav), torch.tensor(clean_wav), fs=sr, extended=True).numpy()
    return stoi_score

def read_batch_stoi(clean_root, enhanced_list):
    """
    Parallel computing for accelerating    
    """
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
    """
    Parallel computing for accelerating    
    """
    pesq_scores = Parallel(n_jobs=64)(delayed(read_pesq)(clean_root, en, 16000) for en in enhanced_list)
    return pesq_scores, enhanced_list

def read_batch_dnsmos(clean_root, enhanced_list):
    """
    Parallel computing for accelerating    
    """
    enhanced_dir = os.path.dirname(enhanced_list[0])
    dnsmos_fnames, dnsmos_scores = dnsmos(enhanced_dir)
    dnsmos_scores = np.float32(dnsmos_scores)
    return dnsmos_scores, dnsmos_fnames

def plot_metric_graphs(metric, epochs, Test_en_array, Test_deg_array, Test_ref_array):
    plt.figure()
    plt.plot(range(1,epochs+1), Test_en_array, 'k', label='Enhanced')
    plt.plot(range(1,epochs+1), Test_deg_array, 'r', label='Noisy')
    plt.plot(range(1,epochs+1), Test_ref_array, 'b', label='Clean')
    plt.legend(loc="upper left")
    plt.xlim([1,epochs])
    plt.xlabel('Iterations (x100)')
    plt.ylabel(metric)
    plt.title('Corpus-VoiceBank')
    plt.grid(True)
    plt.savefig(f'{output_path}/Test_{metric}.png', dpi=150)

if TargetMetric == 'estoi':
    metric_minimum = tf.constant(0, dtype=tf.float32, shape=[1,])
    metric_maximum = tf.constant(1, dtype=tf.float32, shape=[1,])
    read_batch_quality = read_batch_stoi
elif TargetMetric == 'pesq':
    # metric_minimum = tf.constant(1.04269, dtype=tf.float32, shape=[1,])
    # metric_maximum = tf.constant(4.64388, dtype=tf.float32, shape=[1,])
    # metric_minimum = tf.constant(-0.5, dtype=tf.float32, shape=[1,])
    # metric_maximum = tf.constant(4.5, dtype=tf.float32, shape=[1,])
    metric_minimum = tf.constant(1.0, dtype=tf.float32, shape=[1,])
    metric_maximum = tf.constant(5.0, dtype=tf.float32, shape=[1,])
    read_batch_quality = read_batch_pesq
elif TargetMetric == 'dnsmos':
    metric_minimum = tf.constant(1, dtype=tf.float32, shape=[1,])
    metric_maximum = tf.constant(5, dtype=tf.float32, shape=[1,])
    read_batch_quality = read_batch_dnsmos  

def create_folder(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)
        
###
# Dataset & processing functions
###


root_dir = '/fs/ess/PAS2301/Data/Speech/datasets_voicebank/noisy-vctk-16k'
train_clean_dir = os.path.join(root_dir, 'clean_trainset_28spk_wav_16k')
test_clean_dir = os.path.join(root_dir, 'clean_testset_wav_16k')

def wav_2_spect(signal, normalize=False):
    signal_length = signal.shape[0]
    
    n_fft = 512
    y_pad = librosa.util.fix_length(data=signal, size=signal_length + n_fft // 2)
    F = librosa.stft(y_pad, n_fft=512, hop_length=256, win_length=512, window=scipy.signal.hamming)
    
    Lp = np.abs(F)
    phase = np.angle(F)
    
    if normalize:    
        meanR = np.mean(Lp, axis=1).reshape((257,1))
        stdR = np.std(Lp, axis=1).reshape((257,1)) + 1e-12
        NLp = (Lp - meanR) / stdR
    else:
        NLp = Lp
    
    return NLp, phase, signal_length

def spect_2_wav(mag, phase, sig_len):
    Rec = np.multiply(mag , np.exp(1j*phase))
    result = librosa.istft(Rec,
                           hop_length=256,
                           win_length=512,
                           window=scipy.signal.hamming, length=sig_len)
    return result 

def Generator_Mapping(record):
    noisy_LP = tf.expand_dims(tf.math.log1p(record['deg_f_abs']), axis=0)      # shape: (1, 257, None)
    clean_LP = tf.expand_dims(tf.math.log1p(record['ref_f_abs']), axis=0)      # shape: (1, 257, None) 
    
    mask = mask_min * tf.ones_like(noisy_LP)                    # since mask is to be applied on noisy_LP
    Target_score = tf.constant(1.0, dtype=tf.float32, shape=[1,]) 

    return (noisy_LP, clean_LP, mask), Target_score

@tf.autograph.experimental.do_not_convert
def Discriminator_Mapping(record, argument):
    clean_LP = tf.expand_dims(tf.math.log1p(record['ref_f_abs']), axis=0)          # shape: (1, 257, None)
    noisy_LP = tf.expand_dims(tf.math.log1p(record[argument+'_f_abs']), axis=0)    # shape: (1, 257, None)

    True_score = tf.reshape(record[argument+'_'+TargetMetric], [1,])
    True_score = (True_score - metric_minimum) / (metric_maximum - metric_minimum)
    
    return tf.stack([noisy_LP, clean_LP], axis=3), True_score       # shape: (1, 257, None, 2), ()

def Discriminator_Train_Dataset(score_list, enhanced_list):
    big_dataset = tf.data.Dataset.from_tensors((tf.zeros([1,257,2]), tf.constant(0.0))).take(0)
    for index in range(len(enhanced_list)):
        en_fpath = enhanced_list[index]
        en_wav, _ = librosa.load(en_fpath, sr=16000)
        enhanced_LP, _, _ = wav_2_spect(en_wav)
        
        wave_name = en_fpath.split('/')[-1]
        clean_wav, _ = librosa.load(os.path.join(train_clean_dir, wave_name), sr=16000) 
        clean_LP, _, _ = wav_2_spect(clean_wav)

        clean_LP = tf.expand_dims(tf.math.log1p(clean_LP), axis=0)              # shape (1, 257, None)
        enhanced_LP = tf.expand_dims(tf.math.log1p(enhanced_LP), axis=0)        # shape (1, 257, None)
        image = tf.stack([enhanced_LP, clean_LP], axis=3)                       # shape (1, 257, None, 2)
        
        score = tf.constant(score_list[index], dtype=tf.float32, shape=[1,])
        label = (score - metric_minimum) / (metric_maximum - metric_minimum)
        single_dataset = tf.data.Dataset.from_tensors((image, label))
        
        big_dataset = big_dataset.concatenate(single_dataset)
    return big_dataset

dataset_name = f'vctk/default:3.0.0'
(ds_train, ds_valid), dataset_info = tfds.load(dataset_name, split=['TRAIN', 'TEST'],
  data_dir='/fs/scratch/PAS2301/Imran', shuffle_files=True, with_info=True,)

print(dataset_info)
print(f'Samples in training data = {ds_train.cardinality().numpy()} ')
print(f'Samples in validation data = {ds_valid.cardinality().numpy()} ')


###
# Architecture & Compilation
###


class MyActivityRegularizer(keras.layers.Layer):
    def __init__(self, strength=1e-2):
        super(MyActivityRegularizer, self).__init__()
        self.strength = strength

    def call(self, Clean, Enhanced):
        regularizer = tf.norm(Clean - Enhanced, ord='euclidean')
        self.add_loss(self.strength * regularizer)
        return regularizer
    
    def get_config(self):
        return {'strength': self.strength}

@keras.saving.register_keras_serializable()
class Learnable_Sigmoid(tf.keras.layers.Layer):
    def __init__(self):
        super(Learnable_Sigmoid, self).__init__()

    def build(self, input_shape):
        self.alpha = self.add_weight(
            shape = (input_shape[-1]),
            initializer = "ones",
            trainable = True, 
            constraint=tf.keras.constraints.MinMaxNorm(min_value=0.5, max_value=3.5)
        )
        
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
        # self.out_actv = keras.layers.Activation('sigmoid')
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

@keras.saving.register_keras_serializable()
class Discriminator_Model(keras.Model):
    def __init__(self):
        super(Discriminator_Model, self).__init__()  
        self.batch_norm = keras.layers.BatchNormalization(axis=-1, name='Batch_Norm')
        
        self.conv2D_1 = ConvSN2D(15, (5,5), padding='valid',  data_format='channels_last', name='conv_1')
        self.actv_1 = keras.layers.LeakyReLU(name='LeakyReLU')
        
        self.conv2D_2 = ConvSN2D(15, (5,5), padding='valid',  data_format='channels_last', name='conv_2')
        self.actv_2 = keras.layers.LeakyReLU(name='LeakyReLU')
        
        self.conv2D_3 = ConvSN2D(15, (5,5), padding='valid',  data_format='channels_last', name='conv_3')
        self.actv_3 = keras.layers.LeakyReLU(name='LeakyReLU')
        
        self.conv2D_4 = ConvSN2D(15, (5,5), padding='valid',  data_format='channels_last', name='conv_4')
        self.actv_4 = keras.layers.LeakyReLU(name='LeakyReLU')

        self.avg_pool = keras.layers.GlobalAveragePooling2D(name='Average_Pooling')
        
        self.linear_1 = DenseSN(50, name='linear_1')
        self.actv_5 = keras.layers.LeakyReLU(name='LeakyReLU')
        
        self.linear_2 = DenseSN(10, name='linear_2')
        self.actv_6 = keras.layers.LeakyReLU(name='LeakyReLU')
        
        self.linear_3 = DenseSN(1, name='linear_3')
        
    def call(self, inputs, training=False):
        x = inputs
        x = self.batch_norm(x, training=training)
        
        x = self.conv2D_1(x)
        x = self.actv_1(x)
        x = self.conv2D_2(x)
        x = self.actv_2(x)
        x = self.conv2D_3(x)
        x = self.actv_3(x)
        x = self.conv2D_4(x)
        x = self.actv_4(x)
        
        x = self.avg_pool(x)
        
        x = self.linear_1(x)
        x = self.actv_5(x)
        x = self.linear_2(x)
        x = self.actv_6(x)
        x = self.linear_3(x)
        return x

    def get_config(self):
        base_config = super().get_config()
        return {**base_config}

@keras.saving.register_keras_serializable()
class MetricGAN_Model(keras.Model):
    def __init__(self, Generator, Discriminator):
        super(MetricGAN_Model, self).__init__()  

        self.generator = Generator
        self.discriminator = Discriminator 
        self.discriminator.trainable = False 

        # self.regularizer = MyActivityRegularizer(0)

    def call(self, inputs):
        [Noisy_LP, Clean_LP, Min_mask] = inputs 
        
        gen_inp = keras.layers.Permute((2, 1))(Noisy_LP)    # shape: (1, None, 257)
        gen_out = self.generator(gen_inp)                   # shape: (1, None, 257)
        gen_out = keras.layers.Permute((2, 1))(gen_out)     # shape: (1, 257, None)
        Mask = keras.layers.Maximum()([gen_out, Min_mask])

        Enhanced_LP = keras.layers.Multiply()([Mask, Noisy_LP]) 
        disc_inp = tf.stack([Enhanced_LP, Clean_LP], axis=3)    # shape (1, 257, None, 2)

        Predicted_score = self.discriminator(disc_inp) 
        # self.regularizer(Clean_LP, Enhanced_LP)
        return Predicted_score

print ('\nGenerator constructing...')
generator_model = Generator_Model()
generator_model.build(input_shape=(BATCH_SIZE, None, 257))
print(generator_model.summary())

print ('\nDiscriminator constructing...')
discriminator_model = Discriminator_Model() 
discriminator_model.build(input_shape=(BATCH_SIZE, 257, None, 2))
print(discriminator_model.summary())

discriminator_model.trainable = True 
discriminator_model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005))

print ('\nMetricGAN constructing...')
MetricGAN = MetricGAN_Model(generator_model, discriminator_model)
MetricGAN.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005))


###
# Training & Evaluation
###


prev_len = 0
Test_en_array, Test_deg_array, Test_ref_array = [], [], []
previous_dataset = tf.data.Dataset.from_tensors((tf.zeros([1,257,200,2]), tf.constant(0.0))).take(0)

start_time = time.time()
for iteration in np.arange(1, total_iterations+1):
    
    # Prepare directories
    os.makedirs(models_path, exist_ok=True)
    os.makedirs(output_path+'/discriminator_training', exist_ok=True)
    os.makedirs(output_path+'/generator_testing', exist_ok=True)
    
    tr_data_batch = ds_train.shuffle(1000).take(num_train_samples) 
    val_data_batch = ds_valid.shuffle(1000).take(num_valid_samples)
    
    ###
    # Generator Training 
    ###
    
    gen_tr_data = tr_data_batch.map(Generator_Mapping)
    print ('Generator training (with discriminator fixed)...') 
    if iteration >= 2: 
        Generator_hist = MetricGAN.fit(gen_tr_data, batch_size = BATCH_SIZE)

    ###
    # Generator Evaluation
    ###

    print ('Evaluate G by validation data ...')   
    
    Test_en_Name = []
    for record in val_data_batch:
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
        enhanced_name = os.path.join(output_path, 'generator_testing' , wave_name)
        sf.write(enhanced_name, np.int16(enhanced_wav * maxv), 16000)

        Test_en_Name.append(enhanced_name)

    test_en_array, _ = read_batch_quality(test_clean_dir, Test_en_Name)    
    
    Test_en_array.append(tf.math.reduce_mean(test_en_array)) 
    Test_deg_array.append(val_data_batch.map(lambda x: x[f'deg_{TargetMetric}']).reduce(0., tf.math.add) / num_valid_samples)
    Test_ref_array.append(val_data_batch.map(lambda x: x[f'ref_{TargetMetric}']).reduce(0., tf.math.add) / num_valid_samples)
    
    print('Noisy Metric Score: ', Test_deg_array[-1])
    print('Enhanced Metric Score: ', Test_en_array[-1])  
    print('Clean Metric Score: ', Test_ref_array[-1])

    plot_metric_graphs(TargetMetric, iteration, Test_en_array, Test_deg_array, Test_ref_array)

    ###
    # Store Generator Model
    ###
    
    model_name = os.path.join(models_path, f'generator@{iteration}.keras')
    if Test_en_array[-1] == max(Test_en_array):
        generator_model.save(model_name)

    ###
    # Process Audio for Discriminator
    ###
    
    print('Sample training data for discriminator training...')
    Enhanced_name = []    
    for record in tr_data_batch:
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
        enhanced_name = os.path.join(output_path, 'discriminator_training', wave_name)
        sf.write(enhanced_name, np.int16(enhanced_wav* maxv), 16000)
        
        Enhanced_name.append(enhanced_name)
    
    Enhanced_scores, Enhanced_name = read_batch_quality(train_clean_dir , Enhanced_name) 
    
    disc_tr_dset_en = Discriminator_Train_Dataset(Enhanced_scores, Enhanced_name)
    disc_tr_dset_ref = tr_data_batch.map(lambda x: Discriminator_Mapping(x, 'ref'))
    disc_tr_dset_deg = tr_data_batch.map(lambda x: Discriminator_Mapping(x, 'deg'))

    ###
    # Discriminator Training
    ###
    
    print ('Discriminator training...')  
    current_dataset = disc_tr_dset_deg.concatenate(disc_tr_dset_en).concatenate(disc_tr_dset_ref) 
    current_dataset = current_dataset.shuffle(1000)

    discr_history = discriminator_model.fit(current_dataset, batch_size = BATCH_SIZE) 
    
    # Training for current list + Previous list (like replay buffer in RL, optional)   
    tot_disc_dataset = disc_tr_dset_en.concatenate(previous_dataset.take(prev_len//5))
    tot_disc_dataset.shuffle(1000)   
    
    discr_history = discriminator_model.fit(tot_disc_dataset, batch_size = BATCH_SIZE)
    
    previous_dataset = disc_tr_dset_en.concatenate(previous_dataset)
    prev_len = previous_dataset.cardinality().numpy()
    # previous_dataset.shuffle(1000)

    # # Update the history list 
    # Training current list again (optional)   
    discr_history = discriminator_model.fit(current_dataset, batch_size = BATCH_SIZE)
        
    ###
    # Store Discriminator Model
    ###
    
    disc_model_name = os.path.join(models_path, f'discriminator@{iteration}.keras')
    if Test_en_array[-1] == max(Test_en_array):
        discriminator_model.save(disc_model_name)
    
    ###    
    # Prevent Out-of-Memory Errors
    ###
    shutil.rmtree(output_path + '/generator_testing')                    # to save hard disk memory
    shutil.rmtree(output_path+'/discriminator_training')
    
    if iteration >= 120:
        previous_dataset = previous_dataset.take(ds_train.cardinality().numpy())
    
end_time = time.time()
print ('The code for this file ran for %.2fm' % ((end_time - start_time) / 60.))

