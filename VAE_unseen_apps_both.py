import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
import os
import datetime
import argparse
from VAE_functions import *

import pickle
from scipy.stats import norm
from keras.utils.vis_utils import plot_model
from dtw import *
import logging

# Example
# python3 VAE_unseen_apps_both.py --appliance Fridge --gpu 0 --epoch 100 --batch_size 32 --lr 0.001 --ratio_train 0.3 --decay_steps 2 --val_data 1 --optimizer rmsprop --width 1024 --strides 256 --dataset ukdale --run 10 --patience 0 --save_best 0

parser = argparse.ArgumentParser()

parser.add_argument("--model", default="VAE")
parser.add_argument("--config", default=0, type=int)
parser.add_argument("--appliance", default="Fridge", help="Appliance to learn")
parser.add_argument("--gpu", default=0, type=int, help="Appliance to learn")
parser.add_argument("--batch_size", default=150, type=int)
parser.add_argument("--epoch", default=200, type=int)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--ratio_train", default=0.5, type=float)
parser.add_argument("--ratio_test", default=1, type=float)
parser.add_argument("--decay_steps", default=20, type=int)
parser.add_argument("--val_data", default=1, type=int)
parser.add_argument("--optimizer", default="adam")
parser.add_argument("--main_mean", default=0, type=int)
parser.add_argument("--main_std", default=1, type=int)
parser.add_argument("--app_mean", default=0, type=int)
parser.add_argument("--app_std", default=1, type=int)
parser.add_argument("--width", default=1024, type=int)
parser.add_argument("--strides", default=256, type=int)
parser.add_argument("--dataset", default="ukdale")
parser.add_argument("--run", default=1, type=int)
parser.add_argument("--patience", default=10, type=int)
parser.add_argument("--start_stopping", default=25, type=int)
parser.add_argument("--save_best", default=0, type=int)

a = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)

logging.getLogger('tensorflow').disabled = True

np.random.seed(123)

name = "VAE_unseen_{}".format(a.appliance)
time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

for r in range(1, int(a.run)):
    ###############################################################################
    # Load dataset
    ###############################################################################
    if a.model == "S2P":
        if a.dataset == "ukdale":
            ###############################################################################
            # UKDALE
            ###############################################################################
            ###############################################################################
            # Load train data
            ###############################################################################

            app_list = [a.appliance]
            width = int(a.width)*2-1
            stride = int(a.strides)

            def import_data(app_type, train_test="train"):
                x = np.load("Data/{}_main_{}.npy".format(app_type, train_test))
                y = np.load("Data/{}_appliance_{}.npy".format(app_type, train_test))

                return x, y

            def seq_dataset(x, y, width, stride):
                x_ = []
                y_ = []

                for t in range(0, x.shape[0]-width, stride):
                    x_.append(x[t:t+width])
                    y_.append(y[t:t+width])

                x_ = np.array(x_).reshape([-1, width, 1])
                y_ = np.array(y_).reshape([-1, width, 1])

                return x_, y_

            x_train = np.array([]).reshape(0, width, 1)
            y_train = np.array([]).reshape(0, width, 1)

            for i in range(len(app_list)):
                x, y = import_data(app_list[i], "train")
                x_, y_ = seq_dataset(x, y, width, stride)

                x_train = np.vstack([x_train, x_])
                y_train = np.vstack([y_train, y_])

            print(x_train.shape, y_train.shape)

            ###############################################################################
            # Load test data
            ###############################################################################

            app_list = [a.appliance]
            width = width
            stride = int(a.strides)

            for app_ind in range(len(app_list)):
                x_test = np.array([]).reshape(0, width, 1)
                y_test = np.array([]).reshape(0, width, 1)

                x, y = import_data(app_list[app_ind], "test")
                x_, y_ = seq_dataset(x, y, width, stride)

                x_test = np.vstack([x_test, x_])
                y_test = np.vstack([y_test, y_])
            
            if a.ratio_test == 0:
                x_test = np.copy(x_test[0:1,:,:])
                y_test = np.copy(y_test[0:1,:,:])
                
            print(x_test.shape, y_test.shape)
    
    else:
        if a.dataset == "ukdale":
            ###############################################################################
            # UKDALE
            ###############################################################################
            ###############################################################################
            # Load train data
            ###############################################################################

            app_list = [a.appliance] # ["Fridge", "WashingMachine", "Dishwasher", "Kettle", "Microwave"]
            width = int(a.width)
            stride = int(a.strides)

            def import_data(app_type, train_test="train"):
                x = np.load("Data/{}_main_{}.npy".format(app_type, train_test))
                y = np.load("Data/{}_appliance_{}.npy".format(app_type, train_test))

                return x, y

            def seq_dataset(x, y, width, stride):
                x_ = []
                y_ = []

                for t in range(0, x.shape[0]-width, stride):
                    #if np.diff(y[t:t+width]).max() > 20:
                    x_.append(x[t:t+width])
                    y_.append(y[t:t+width])

                x_ = np.array(x_).reshape([-1, width, 1])
                y_ = np.array(y_).reshape([-1, width, 1])

                return x_, y_

            x_train = np.array([]).reshape(0, width, 1)
            y_train = np.array([]).reshape(0, width, 1)
            c_train = np.array([]).reshape(0, width, len(app_list))

            for i in range(len(app_list)):
                x, y = import_data(app_list[i], "train")
                x_, y_ = seq_dataset(x, y, width, stride)
                c = np.zeros([x_.shape[0], width, len(app_list)])
                c[:, :, i] = 1

                x_train = np.vstack([x_train, x_])
                y_train = np.vstack([y_train, y_])
                c_train = np.vstack([c_train, c])

            print(x_train.shape, y_train.shape, c_train.shape)

            ###############################################################################
            # Load val data
            ###############################################################################

            app_list = [a.appliance] #, "WashingMachine", "Dishwasher", "Kettle", "Microwave"]
            width = width
            stride = int(a.strides)

            for app_ind in range(len(app_list)):
                x_test = np.array([]).reshape(0, width, 1)
                y_test = np.array([]).reshape(0, width, 1)
                c_test = np.array([]).reshape(0, width, len(app_list))

                x, y = import_data(app_list[app_ind], "test")
                x_, y_ = seq_dataset(x, y, width, stride)
                c = np.zeros([x_.shape[0], width, len(app_list)])
                c[:, :, app_ind] = 1

                x_test = np.vstack([x_test, x_])
                y_test = np.vstack([y_test, y_])
                c_test = np.vstack([c_test, c])

            print(x_test.shape, y_test.shape, c_test.shape)

        elif a.dataset == "refit":
            ###############################################################################
            # REFIT
            ###############################################################################
            def resample_u_r(x):
                from scipy import interpolate

                f = interpolate.interp1d(np.arange(0,x.shape[0]*6,6), x, bounds_error=False)
                x_re = f(np.arange(0,x.shape[0]*6,8))

                return x_re
            ###############################################################################
            # Load train data
            ###############################################################################
            dataset_path = "Data/REFIT/appliance_data/{}/".format(a.appliance)

            app_list = [a.appliance]
            width = int(a.width)
            stride = int(a.strides)
            
            def import_refit(app_type):
                x = np.load("Data/REFIT/appliance_data/{}/main_train_validation_all.npy".format(app_type))
                y = np.load("Data/REFIT/appliance_data/{}/appliance_train_validation_all.npy".format(app_type))

                x = x.reshape([-1])
                y = y.reshape([-1])
                
                return x, y

            def seq_dataset(x, y, width, stride):
                x_ = []
                y_ = []

                for t in range(0, x.shape[0]-width, stride):
                    x_.append(x[t:t+width])
                    y_.append(y[t:t+width])

                x_ = np.array(x_).reshape([-1, width, 1])
                y_ = np.array(y_).reshape([-1, width, 1])

                return x_, y_

            for app_ind in range(len(app_list)):
                x_train = np.array([]).reshape(0, width, 1)
                y_train = np.array([]).reshape(0, width, 1)

                x, y = import_refit(app_list[app_ind])
                x_, y_ = seq_dataset(x, y, width, stride)

                x_train = np.vstack([x_train, x_])
                y_train = np.vstack([y_train, y_])

            print(x_train.shape, y_train.shape)
            
            ###############################################################################
            # Load validation data
            ###############################################################################
            dataset_path = "Data/REFIT/appliance_data/{}/".format(a.appliance)

            app_list = [a.appliance]
            width = int(a.width)
            stride = int(a.strides)
            
            def import_refit(app_type):
                x = np.load("Data/REFIT/appliance_data/{}/main_test_all.npy".format(app_type))
                y = np.load("Data/REFIT/appliance_data/{}/appliance_test_all.npy".format(app_type))

                x = x.reshape([-1])
                y = y.reshape([-1])
                
                return x, y

            def seq_dataset(x, y, width, stride):
                x_ = []
                y_ = []

                for t in range(0, x.shape[0]-width, stride):
                    x_.append(x[t:t+width])
                    y_.append(y[t:t+width])

                x_ = np.array(x_).reshape([-1, width, 1])
                y_ = np.array(y_).reshape([-1, width, 1])

                return x_, y_

            for app_ind in range(len(app_list)):
                x_val = np.array([]).reshape(0, width, 1)
                y_val = np.array([]).reshape(0, width, 1)

                x, y = import_refit(app_list[app_ind])
                x_, y_ = seq_dataset(x, y, width, stride)

                x_val = np.vstack([x_val, x_])
                y_val = np.vstack([y_val, y_])

            print(x_val.shape, y_val.shape)
            
            ###############################################################################
            # Load test data
            ###############################################################################

            app_list = [a.appliance]
            width = int(a.width)
            stride = int(a.strides)
            
            def import_data(app_type, train_test="train"):
                x = np.load("Data/{}_main_{}.npy".format(app_type, train_test))
                y = np.load("Data/{}_appliance_{}.npy".format(app_type, train_test))

                return x, y

            def seq_dataset(x, y, width, stride):
                x_ = []
                y_ = []

                for t in range(0, x.shape[0]-width, stride):
                    x_.append(x[t:t+width])
                    y_.append(y[t:t+width])

                x_ = np.array(x_).reshape([-1, width, 1])
                y_ = np.array(y_).reshape([-1, width, 1])

                return x_, y_

            for app_ind in range(len(app_list)):
                x_test = np.array([]).reshape(0, width, 1)
                y_test = np.array([]).reshape(0, width, 1)

                x, y = import_data(app_list[app_ind], "test")
                
                x = resample_u_r(x)
                y = resample_u_r(y)
                
                x_, y_ = seq_dataset(x, y, width, stride)

                x_test = np.vstack([x_test, x_])
                y_test = np.vstack([y_test, y_])

            print(x_test.shape, y_test.shape)

        elif a.dataset == "house_2":
            ###############################################################################
            # Load dataset Train
            ###############################################################################

            app_list = [a.appliance]
            width = int(a.width)
            stride = int(a.strides)

            def import_data(app_type, train_test="train"):
                x = np.load("Data/House_2/{}_main_{}.npy".format(app_type, train_test))
                y = np.load("Data/House_2/{}_appliance_{}.npy".format(app_type, train_test))

                return x, y

            def seq_dataset(x, y, width, stride):
                x_ = []
                y_ = []

                for t in range(0, x.shape[0]-width, stride):
                    #if np.diff(y[t:t+width]).max() > 20:
                    x_.append(x[t:t+width])
                    y_.append(y[t:t+width])

                x_ = np.array(x_).reshape([-1, width, 1])
                y_ = np.array(y_).reshape([-1, width, 1])

                return x_, y_

            x_train = np.array([]).reshape(0, width, 1)
            y_train = np.array([]).reshape(0, width, 1)
            c_train = np.array([]).reshape(0, width, len(app_list))

            for i in range(len(app_list)):
                x, y = import_data(app_list[i], "train")
                x_, y_ = seq_dataset(x, y, width, stride)
                c = np.zeros([x_.shape[0], width, len(app_list)])
                c[:, :, i] = 1

                x_train = np.vstack([x_train, x_])
                y_train = np.vstack([y_train, y_])
                c_train = np.vstack([c_train, c])

            print(x_train.shape, y_train.shape, c_train.shape)
            
            ###############################################################################
            # Load dataset Test
            ###############################################################################

            app_list = [a.appliance]
            width = int(a.width)
            stride = int(a.strides)

            x_test = np.array([]).reshape(0, width, 1)
            y_test = np.array([]).reshape(0, width, 1)

            for i in range(len(app_list)):
                x, y = import_data(app_list[i], "test")
                x_, y_ = seq_dataset(x, y, width, stride)

                x_test = np.vstack([x_test, x_])
                y_test = np.vstack([y_test, y_])

            print(x_test.shape, y_test.shape)

    ###############################################################################
    # Build the model
    ###############################################################################

    if a.model == "VAE":
        if int(a.config) == 0:
            def KL_loss(y_true, y_pred):
                # Regularization term
                kl_loss = - .5 * tf.reduce_sum(1 + z_log_var - K.square(z_mu) - K.exp(z_log_var), axis=-1)

                #kl_loss = 0.5 * K.sum(K.exp(z_log_var) + K.square(z_mu) - 1. - z_log_var, axis=-1)

                return kl_loss

            def Recon_loss(data_orig, data_reconstructed):
                """Negative log likelihood (Bernouilli). """
            #     delta = 1e-10
            #     reconstruction_loss = K.sum(K.mean((y_true + delta)/(K.exp(y_pred) + delta) + y_pred, axis=-1), axis=-1)

                reconstruction_loss = tf.reduce_mean((data_orig - data_reconstructed)**2)

                return reconstruction_loss

            def vae_loss(data_orig, data_reconstructed):
                delta = 1e-10
                # Reconstruction term, equivalent to the IS divergence between x_orig and x_reconstructed
                #reconstruction_loss = K.sum(K.mean((data_orig + delta)/(K.exp(data_reconstructed) + delta) + data_reconstructed, axis=-1), axis=-1)
                reconstruction_loss = tf.reduce_mean((data_orig - data_reconstructed)**2)
                # Regularization term
                #kl_loss = - 0.5 * K.sum(z_log_var - K.square(z_mu) - K.exp(z_log_var), axis=-1)
                kl_loss = - .5 * tf.reduce_sum(1 + z_log_var - K.square(z_mu) - K.exp(z_log_var), axis=-1)

                #kl_loss = 0.5 * K.sum(K.exp(z_log_var) + K.square(z_mu) - 1. - z_log_var, axis=-1)

            #     print(z_log_var.shape)
            #     print(z_mu.shape)
                print(kl_loss.shape)
                print(reconstruction_loss.shape)

                return reconstruction_loss + kl_loss

            start_filter_num = 32
            kernel_size = 3
            latent_dim = 16

            x = tf.keras.Input(shape=(width,1))

            conv_seq1 = conv_block_seq_res(x, start_filter_num, kernel_size, 1, "conv_seq1")
            pool1 = tf.keras.layers.MaxPooling1D(name="pool1")(conv_seq1)

            conv_seq2 = conv_block_seq_res(pool1, int(start_filter_num*1.5), kernel_size, 1, "conv_seq2")
            pool2 = tf.keras.layers.MaxPooling1D(name="pool2")(conv_seq2)

            conv_seq3 = conv_block_seq_res(pool2, start_filter_num*3, kernel_size, 1, "conv_seq3")
            pool3 = tf.keras.layers.MaxPooling1D(name="pool3")(conv_seq3)

            conv_seq4 = conv_block_seq_res(pool3, int(start_filter_num*4.5), kernel_size, 1, "conv_seq4")
            pool4 = tf.keras.layers.MaxPooling1D(name="pool4")(conv_seq4)

            conv_seq5 = conv_block_seq_res(pool4, start_filter_num*6, kernel_size, 1, "conv_seq5")
            pool5 = tf.keras.layers.MaxPooling1D(name="pool5")(conv_seq5)

            conv_seq6 = conv_block_seq_res(pool5, int(start_filter_num*7.5), kernel_size, 1, "conv_seq6", In=False)
            pool6 = tf.keras.layers.MaxPooling1D(name="pool6")(conv_seq6)

            conv_seq7 = conv_block_seq_res(pool6, start_filter_num*9, kernel_size, 1, "conv_seq7", In=False)

            flatten1 = tf.keras.layers.Flatten()(conv_seq7)

            z_mu = tf.keras.layers.Dense(latent_dim, name="z_mu")(flatten1)
            z_log_var = tf.keras.layers.Dense(latent_dim, name="z_log_var")(flatten1)

            ###############################################################################
            # normalize log variance to std dev
            z_sigma = tf.keras.layers.Lambda(lambda t: K.exp(.5*t), name="z_sigma")(z_log_var)
            eps = tf.keras.Input(tensor=K.random_normal(shape=(K.shape(x)[0], latent_dim)), name="eps")

            z_eps = tf.keras.layers.Multiply(name="z_eps")([z_sigma, eps])
            z = tf.keras.layers.Add(name="z")([z_mu, z_eps])

            #latent_conv = tf.keras.layers.Dense(width//64, name="latent_conv")(z)
            reshape1 = tf.keras.layers.Reshape([width//64,1], name="reshape1")(z)

            ###############################################################################
            #New for conditional VAE
            dconv_seq4 = conv_block_seq_res(reshape1, start_filter_num*9, kernel_size, 1, "dconv_seq4", In=False)
            dconc5 = tf.keras.layers.concatenate([dconv_seq4, conv_seq7], name="dconc5")
            deconv1 = Conv1DTranspose(dconc5, start_filter_num*2, kernel_size=3, strides=2, padding='same')

            dconv_seq5 = conv_block_seq_res(deconv1, int(start_filter_num*7.5), kernel_size, 1, "dconv_seq5", In=False)
            dconc7 = tf.keras.layers.concatenate([dconv_seq5, conv_seq6], name="dconc7")
            deconv2 = Conv1DTranspose(dconc7, start_filter_num, kernel_size=3, strides=2, padding='same')

            dconv_seq6 = conv_block_seq_res(deconv2, start_filter_num*6, kernel_size, 1, "dconv_seq6", In=False)
            dconc9 = tf.keras.layers.concatenate([dconv_seq6, conv_seq5], name="dconc9")
            deconv3 = Conv1DTranspose(dconc9, start_filter_num, kernel_size=3, strides=2, padding='same')

            dconv_seq7 = conv_block_seq_res(deconv3, int(start_filter_num*4.5), kernel_size, 1, "dconv_seq7", In=False)
            dconc11 = tf.keras.layers.concatenate([dconv_seq7, conv_seq4], name="dconc11")
            deconv4 = Conv1DTranspose(dconc11, start_filter_num, kernel_size=3, strides=2, padding='same')

            dconv_seq8 = conv_block_seq_res(deconv4, start_filter_num*3, kernel_size, 1, "dconv_seq8", In=False)
            dconc13 = tf.keras.layers.concatenate([dconv_seq8, conv_seq3], name="dconc13")
            deconv5 = Conv1DTranspose(dconc13, start_filter_num, kernel_size=3, strides=2, padding='same')

            dconv_seq9 = conv_block_seq_res(deconv5, int(start_filter_num*1.5), kernel_size, 1, "dconv_seq9", In=False)
            dconc15 = tf.keras.layers.concatenate([dconv_seq9, conv_seq2], name="dconc15")
            deconv6 = Conv1DTranspose(dconc15, start_filter_num, kernel_size=3, strides=2, padding='same')

            dconv_seq10 = conv_block_seq_res(deconv6, start_filter_num, kernel_size, 1, "dconv_seq10", In=False)
            dconc17 = tf.keras.layers.concatenate([dconv_seq10, conv_seq1], name="dconc17")

            x_pred = tf.keras.layers.Conv1D(1, 3, padding="same", activation="relu", name="x_pred")(dconc17)

            model = tf.keras.Model(inputs=[x, eps], outputs=x_pred)
            model.summary()
            
        elif int(a.config) == 1:
            def KL_loss(y_true, y_pred):
                # Regularization term
                kl_loss = - .5 * tf.reduce_sum(1 + z_log_var - K.square(z_mu) - K.exp(z_log_var), axis=-1)

                #kl_loss = 0.5 * K.sum(K.exp(z_log_var) + K.square(z_mu) - 1. - z_log_var, axis=-1)

                return kl_loss

            def Recon_loss(data_orig, data_reconstructed):
                """Negative log likelihood (Bernouilli). """
            #     delta = 1e-10
            #     reconstruction_loss = K.sum(K.mean((y_true + delta)/(K.exp(y_pred) + delta) + y_pred, axis=-1), axis=-1)

                reconstruction_loss = tf.reduce_mean((data_orig - data_reconstructed)**2)

                return reconstruction_loss

            def vae_loss(data_orig, data_reconstructed):
                delta = 1e-10
                # Reconstruction term, equivalent to the IS divergence between x_orig and x_reconstructed
                #reconstruction_loss = K.sum(K.mean((data_orig + delta)/(K.exp(data_reconstructed) + delta) + data_reconstructed, axis=-1), axis=-1)
                reconstruction_loss = tf.reduce_mean((data_orig - data_reconstructed)**2)
                # Regularization term
                #kl_loss = - 0.5 * K.sum(z_log_var - K.square(z_mu) - K.exp(z_log_var), axis=-1)
                kl_loss = - .5 * tf.reduce_sum(1 + z_log_var - K.square(z_mu) - K.exp(z_log_var), axis=-1)

                #kl_loss = 0.5 * K.sum(K.exp(z_log_var) + K.square(z_mu) - 1. - z_log_var, axis=-1)

            #     print(z_log_var.shape)
            #     print(z_mu.shape)
                print(kl_loss.shape)
                print(reconstruction_loss.shape)

                return reconstruction_loss + kl_loss

            start_filter_num = 32
            kernel_size = 3
            latent_dim = 16

            x = tf.keras.Input(shape=(width,1))

            conv_seq1 = conv_block_seq_res(x, start_filter_num, kernel_size, 1, "conv_seq1")
            pool1 = tf.keras.layers.MaxPooling1D(name="pool1")(conv_seq1)

            conv_seq2 = conv_block_seq_res(pool1, int(start_filter_num*1.5), kernel_size, 1, "conv_seq2")
            pool2 = tf.keras.layers.MaxPooling1D(name="pool2")(conv_seq2)

            conv_seq3 = conv_block_seq_res(pool2, start_filter_num*3, kernel_size, 1, "conv_seq3")
            pool3 = tf.keras.layers.MaxPooling1D(name="pool3")(conv_seq3)

            conv_seq4 = conv_block_seq_res(pool3, int(start_filter_num*4.5), kernel_size, 1, "conv_seq4")
            pool4 = tf.keras.layers.MaxPooling1D(name="pool4")(conv_seq4)

            conv_seq5 = conv_block_seq_res(pool4, start_filter_num*6, kernel_size, 1, "conv_seq5")
            pool5 = tf.keras.layers.MaxPooling1D(name="pool5")(conv_seq5)

            conv_seq6 = conv_block_seq_res(pool5, int(start_filter_num*7.5), kernel_size, 1, "conv_seq6", In=False)
            pool6 = tf.keras.layers.MaxPooling1D(name="pool6")(conv_seq6)

            conv_seq7 = conv_block_seq_res(pool6, start_filter_num*9, kernel_size, 1, "conv_seq7", In=False)

            flatten1 = tf.keras.layers.Flatten()(conv_seq7)

            z_mu = tf.keras.layers.Dense(latent_dim, name="z_mu")(flatten1)
            z_log_var = tf.keras.layers.Dense(latent_dim, name="z_log_var")(flatten1)

            ###############################################################################
            # normalize log variance to std dev
            z_sigma = tf.keras.layers.Lambda(lambda t: K.exp(.5*t), name="z_sigma")(z_log_var)
            eps = tf.keras.Input(tensor=K.random_normal(shape=(K.shape(x)[0], latent_dim)), name="eps")

            z_eps = tf.keras.layers.Multiply(name="z_eps")([z_sigma, eps])
            z = tf.keras.layers.Add(name="z")([z_mu, z_eps])

            #latent_conv = tf.keras.layers.Dense(width//64, name="latent_conv")(z)
            reshape1 = tf.keras.layers.Reshape([width//64,1], name="reshape1")(z)

            ###############################################################################
            #New for conditional VAE
            dconv_seq4 = conv_block_seq_res(reshape1, start_filter_num*9, kernel_size, 1, "dconv_seq4", In=False)
            deconv1 = Conv1DTranspose(dconv_seq4, start_filter_num*2, kernel_size=3, strides=2, padding='same')

            dconv_seq5 = conv_block_seq_res(deconv1, int(start_filter_num*7.5), kernel_size, 1, "dconv_seq5", In=False)
            deconv2 = Conv1DTranspose(dconv_seq5, start_filter_num, kernel_size=3, strides=2, padding='same')

            dconv_seq6 = conv_block_seq_res(deconv2, start_filter_num*6, kernel_size, 1, "dconv_seq6", In=False)
            deconv3 = Conv1DTranspose(dconv_seq6, start_filter_num, kernel_size=3, strides=2, padding='same')

            dconv_seq7 = conv_block_seq_res(deconv3, int(start_filter_num*4.5), kernel_size, 1, "dconv_seq7", In=False)
            deconv4 = Conv1DTranspose(dconv_seq7, start_filter_num, kernel_size=3, strides=2, padding='same')

            dconv_seq8 = conv_block_seq_res(deconv4, start_filter_num*3, kernel_size, 1, "dconv_seq8", In=False)
            deconv5 = Conv1DTranspose(dconv_seq8, start_filter_num, kernel_size=3, strides=2, padding='same')

            dconv_seq9 = conv_block_seq_res(deconv5, int(start_filter_num*1.5), kernel_size, 1, "dconv_seq9", In=False)
            deconv6 = Conv1DTranspose(dconv_seq9, start_filter_num, kernel_size=3, strides=2, padding='same')

            dconv_seq10 = conv_block_seq_res(deconv6, start_filter_num, kernel_size, 1, "dconv_seq10", In=False)

            x_pred = tf.keras.layers.Conv1D(1, 3, padding="same", activation="relu", name="x_pred")(dconv_seq10)

            model = tf.keras.Model(inputs=[x, eps], outputs=x_pred)
            model.summary()
            
        elif int(a.config) == 2:
            def KL_loss(y_true, y_pred):
                # Regularization term
                kl_loss = - .5 * tf.reduce_sum(1 + z_log_var - K.square(z_mu) - K.exp(z_log_var), axis=-1)

                #kl_loss = 0.5 * K.sum(K.exp(z_log_var) + K.square(z_mu) - 1. - z_log_var, axis=-1)

                return kl_loss

            def Recon_loss(data_orig, data_reconstructed):
                """Negative log likelihood (Bernouilli). """
            #     delta = 1e-10
            #     reconstruction_loss = K.sum(K.mean((y_true + delta)/(K.exp(y_pred) + delta) + y_pred, axis=-1), axis=-1)

                reconstruction_loss = tf.reduce_mean((data_orig - data_reconstructed)**2)

                return reconstruction_loss

            def vae_loss(data_orig, data_reconstructed):
                delta = 1e-10
                # Reconstruction term, equivalent to the IS divergence between x_orig and x_reconstructed
                #reconstruction_loss = K.sum(K.mean((data_orig + delta)/(K.exp(data_reconstructed) + delta) + data_reconstructed, axis=-1), axis=-1)
                reconstruction_loss = tf.reduce_mean((data_orig - data_reconstructed)**2)
                # Regularization term
                #kl_loss = - 0.5 * K.sum(z_log_var - K.square(z_mu) - K.exp(z_log_var), axis=-1)
                kl_loss = - .5 * tf.reduce_sum(1 + z_log_var - K.square(z_mu) - K.exp(z_log_var), axis=-1)

                #kl_loss = 0.5 * K.sum(K.exp(z_log_var) + K.square(z_mu) - 1. - z_log_var, axis=-1)

            #     print(z_log_var.shape)
            #     print(z_mu.shape)
                print(kl_loss.shape)
                print(reconstruction_loss.shape)

                return reconstruction_loss + kl_loss

            start_filter_num = 32
            kernel_size = 3
            latent_dim = 16

            x = tf.keras.Input(shape=(width,1))

            conv_seq1 = conv_block_seq_res(x, start_filter_num, kernel_size, 1, "conv_seq1", ResCon=False)
            pool1 = tf.keras.layers.MaxPooling1D(name="pool1")(conv_seq1)

            conv_seq2 = conv_block_seq_res(pool1, int(start_filter_num*1.5), kernel_size, 1, "conv_seq2", ResCon=False)
            pool2 = tf.keras.layers.MaxPooling1D(name="pool2")(conv_seq2)

            conv_seq3 = conv_block_seq_res(pool2, start_filter_num*3, kernel_size, 1, "conv_seq3", ResCon=False)
            pool3 = tf.keras.layers.MaxPooling1D(name="pool3")(conv_seq3)

            conv_seq4 = conv_block_seq_res(pool3, int(start_filter_num*4.5), kernel_size, 1, "conv_seq4", ResCon=False)
            pool4 = tf.keras.layers.MaxPooling1D(name="pool4")(conv_seq4)

            conv_seq5 = conv_block_seq_res(pool4, start_filter_num*6, kernel_size, 1, "conv_seq5", ResCon=False)
            pool5 = tf.keras.layers.MaxPooling1D(name="pool5")(conv_seq5)

            conv_seq6 = conv_block_seq_res(pool5, int(start_filter_num*7.5), kernel_size, 1, "conv_seq6", In=False, ResCon=False)
            pool6 = tf.keras.layers.MaxPooling1D(name="pool6")(conv_seq6)

            conv_seq7 = conv_block_seq_res(pool6, start_filter_num*9, kernel_size, 1, "conv_seq7", In=False, ResCon=False)

            flatten1 = tf.keras.layers.Flatten()(conv_seq7)

            z_mu = tf.keras.layers.Dense(latent_dim, name="z_mu")(flatten1)
            z_log_var = tf.keras.layers.Dense(latent_dim, name="z_log_var")(flatten1)

            ###############################################################################
            # normalize log variance to std dev
            z_sigma = tf.keras.layers.Lambda(lambda t: K.exp(.5*t), name="z_sigma")(z_log_var)
            eps = tf.keras.Input(tensor=K.random_normal(shape=(K.shape(x)[0], latent_dim)), name="eps")

            z_eps = tf.keras.layers.Multiply(name="z_eps")([z_sigma, eps])
            z = tf.keras.layers.Add(name="z")([z_mu, z_eps])

            #latent_conv = tf.keras.layers.Dense(width//64, name="latent_conv")(z)
            reshape1 = tf.keras.layers.Reshape([width//64,1], name="reshape1")(z)

            ###############################################################################
            #New for conditional VAE
            dconv_seq4 = conv_block_seq_res(reshape1, start_filter_num*9, kernel_size, 1, "dconv_seq4", In=False, ResCon=False)
            dconc5 = tf.keras.layers.concatenate([dconv_seq4, conv_seq7], name="dconc5")
            deconv1 = Conv1DTranspose(dconc5, start_filter_num*2, kernel_size=3, strides=2, padding='same')

            dconv_seq5 = conv_block_seq_res(deconv1, int(start_filter_num*7.5), kernel_size, 1, "dconv_seq5", In=False, ResCon=False)
            dconc7 = tf.keras.layers.concatenate([dconv_seq5, conv_seq6], name="dconc7")
            deconv2 = Conv1DTranspose(dconc7, start_filter_num, kernel_size=3, strides=2, padding='same')

            dconv_seq6 = conv_block_seq_res(deconv2, start_filter_num*6, kernel_size, 1, "dconv_seq6", In=False, ResCon=False)
            dconc9 = tf.keras.layers.concatenate([dconv_seq6, conv_seq5], name="dconc9")
            deconv3 = Conv1DTranspose(dconc9, start_filter_num, kernel_size=3, strides=2, padding='same')

            dconv_seq7 = conv_block_seq_res(deconv3, int(start_filter_num*4.5), kernel_size, 1, "dconv_seq7", In=False, ResCon=False)
            dconc11 = tf.keras.layers.concatenate([dconv_seq7, conv_seq4], name="dconc11")
            deconv4 = Conv1DTranspose(dconc11, start_filter_num, kernel_size=3, strides=2, padding='same')

            dconv_seq8 = conv_block_seq_res(deconv4, start_filter_num*3, kernel_size, 1, "dconv_seq8", In=False, ResCon=False)
            dconc13 = tf.keras.layers.concatenate([dconv_seq8, conv_seq3], name="dconc13")
            deconv5 = Conv1DTranspose(dconc13, start_filter_num, kernel_size=3, strides=2, padding='same')

            dconv_seq9 = conv_block_seq_res(deconv5, int(start_filter_num*1.5), kernel_size, 1, "dconv_seq9", In=False, ResCon=False)
            dconc15 = tf.keras.layers.concatenate([dconv_seq9, conv_seq2], name="dconc15")
            deconv6 = Conv1DTranspose(dconc15, start_filter_num, kernel_size=3, strides=2, padding='same')

            dconv_seq10 = conv_block_seq_res(deconv6, start_filter_num, kernel_size, 1, "dconv_seq10", In=False, ResCon=False)
            dconc17 = tf.keras.layers.concatenate([dconv_seq10, conv_seq1], name="dconc17")

            x_pred = tf.keras.layers.Conv1D(1, 3, padding="same", activation="relu", name="x_pred")(dconc17)

            model = tf.keras.Model(inputs=[x, eps], outputs=x_pred)
            model.summary()
            
        elif int(a.config) == 3:
            def KL_loss(y_true, y_pred):
                # Regularization term
                kl_loss = - .5 * tf.reduce_sum(1 + z_log_var - K.square(z_mu) - K.exp(z_log_var), axis=-1)

                #kl_loss = 0.5 * K.sum(K.exp(z_log_var) + K.square(z_mu) - 1. - z_log_var, axis=-1)

                return kl_loss

            def Recon_loss(data_orig, data_reconstructed):
                """Negative log likelihood (Bernouilli). """
            #     delta = 1e-10
            #     reconstruction_loss = K.sum(K.mean((y_true + delta)/(K.exp(y_pred) + delta) + y_pred, axis=-1), axis=-1)

                reconstruction_loss = tf.reduce_mean((data_orig - data_reconstructed)**2)

                return reconstruction_loss

            def vae_loss(data_orig, data_reconstructed):
                delta = 1e-10
                # Reconstruction term, equivalent to the IS divergence between x_orig and x_reconstructed
                #reconstruction_loss = K.sum(K.mean((data_orig + delta)/(K.exp(data_reconstructed) + delta) + data_reconstructed, axis=-1), axis=-1)
                reconstruction_loss = tf.reduce_mean((data_orig - data_reconstructed)**2)
                # Regularization term
                #kl_loss = - 0.5 * K.sum(z_log_var - K.square(z_mu) - K.exp(z_log_var), axis=-1)
                kl_loss = - .5 * tf.reduce_sum(1 + z_log_var - K.square(z_mu) - K.exp(z_log_var), axis=-1)

                #kl_loss = 0.5 * K.sum(K.exp(z_log_var) + K.square(z_mu) - 1. - z_log_var, axis=-1)

            #     print(z_log_var.shape)
            #     print(z_mu.shape)
                print(kl_loss.shape)
                print(reconstruction_loss.shape)

                return reconstruction_loss + kl_loss

            start_filter_num = 32
            kernel_size = 3
            latent_dim = 16

            x = tf.keras.Input(shape=(width,1))

            conv_seq1 = conv_block_seq_res(x, start_filter_num, kernel_size, 1, "conv_seq1", ResCon=False)
            pool1 = tf.keras.layers.MaxPooling1D(name="pool1")(conv_seq1)

            conv_seq2 = conv_block_seq_res(pool1, int(start_filter_num*1.5), kernel_size, 1, "conv_seq2", ResCon=False)
            pool2 = tf.keras.layers.MaxPooling1D(name="pool2")(conv_seq2)

            conv_seq3 = conv_block_seq_res(pool2, start_filter_num*3, kernel_size, 1, "conv_seq3", ResCon=False)
            pool3 = tf.keras.layers.MaxPooling1D(name="pool3")(conv_seq3)

            conv_seq4 = conv_block_seq_res(pool3, int(start_filter_num*4.5), kernel_size, 1, "conv_seq4", ResCon=False)
            pool4 = tf.keras.layers.MaxPooling1D(name="pool4")(conv_seq4)

            conv_seq5 = conv_block_seq_res(pool4, start_filter_num*6, kernel_size, 1, "conv_seq5", ResCon=False)
            pool5 = tf.keras.layers.MaxPooling1D(name="pool5")(conv_seq5)

            conv_seq6 = conv_block_seq_res(pool5, int(start_filter_num*7.5), kernel_size, 1, "conv_seq6", In=False, ResCon=False)
            pool6 = tf.keras.layers.MaxPooling1D(name="pool6")(conv_seq6)

            conv_seq7 = conv_block_seq_res(pool6, start_filter_num*9, kernel_size, 1, "conv_seq7", In=False, ResCon=False)

            flatten1 = tf.keras.layers.Flatten()(conv_seq7)

            z_mu = tf.keras.layers.Dense(latent_dim, name="z_mu")(flatten1)
            z_log_var = tf.keras.layers.Dense(latent_dim, name="z_log_var")(flatten1)

            ###############################################################################
            # normalize log variance to std dev
            z_sigma = tf.keras.layers.Lambda(lambda t: K.exp(.5*t), name="z_sigma")(z_log_var)
            eps = tf.keras.Input(tensor=K.random_normal(shape=(K.shape(x)[0], latent_dim)), name="eps")

            z_eps = tf.keras.layers.Multiply(name="z_eps")([z_sigma, eps])
            z = tf.keras.layers.Add(name="z")([z_mu, z_eps])

            #latent_conv = tf.keras.layers.Dense(width//64, name="latent_conv")(z)
            reshape1 = tf.keras.layers.Reshape([width//64,1], name="reshape1")(z)

            ###############################################################################
            #New for conditional VAE
            dconv_seq4 = conv_block_seq_res(reshape1, start_filter_num*9, kernel_size, 1, "dconv_seq4", In=False, ResCon=False)
            deconv1 = Conv1DTranspose(dconv_seq4, start_filter_num*2, kernel_size=3, strides=2, padding='same')

            dconv_seq5 = conv_block_seq_res(deconv1, int(start_filter_num*7.5), kernel_size, 1, "dconv_seq5", In=False, ResCon=False)
            deconv2 = Conv1DTranspose(dconv_seq5, start_filter_num, kernel_size=3, strides=2, padding='same')

            dconv_seq6 = conv_block_seq_res(deconv2, start_filter_num*6, kernel_size, 1, "dconv_seq6", In=False, ResCon=False)
            deconv3 = Conv1DTranspose(dconv_seq6, start_filter_num, kernel_size=3, strides=2, padding='same')

            dconv_seq7 = conv_block_seq_res(deconv3, int(start_filter_num*4.5), kernel_size, 1, "dconv_seq7", In=False, ResCon=False)
            deconv4 = Conv1DTranspose(dconv_seq7, start_filter_num, kernel_size=3, strides=2, padding='same')

            dconv_seq8 = conv_block_seq_res(deconv4, start_filter_num*3, kernel_size, 1, "dconv_seq8", In=False, ResCon=False)
            deconv5 = Conv1DTranspose(dconv_seq8, start_filter_num, kernel_size=3, strides=2, padding='same')

            dconv_seq9 = conv_block_seq_res(deconv5, int(start_filter_num*1.5), kernel_size, 1, "dconv_seq9", In=False, ResCon=False)
            deconv6 = Conv1DTranspose(dconv_seq9, start_filter_num, kernel_size=3, strides=2, padding='same')

            dconv_seq10 = conv_block_seq_res(deconv6, start_filter_num, kernel_size, 1, "dconv_seq10", In=False, ResCon=False)

            x_pred = tf.keras.layers.Conv1D(1, 3, padding="same", activation="relu", name="x_pred")(dconv_seq10)

            model = tf.keras.Model(inputs=[x, eps], outputs=x_pred)
            model.summary()
        
    elif a.model == "DAE":
        sequence_len = int(a.width)
        
        model = tf.keras.models.Sequential()

        # 1D Conv
        model.add(tf.keras.layers.Conv1D(8, 4, activation="linear", input_shape=(sequence_len, 1), padding="same", strides=1))
        model.add(tf.keras.layers.Flatten())

        # Fully Connected Layers
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense((sequence_len-0)*8, activation='relu'))

        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(128, activation='relu'))

        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense((sequence_len-0)*8, activation='relu'))

        model.add(tf.keras.layers.Dropout(0.2))

        # 1D Conv
        model.add(tf.keras.layers.Reshape(((sequence_len-0), 8)))
        model.add(tf.keras.layers.Conv1D(1, 4, activation="linear", padding="same", strides=1))

    elif a.model == "S2P":
        """Specifies the structure of a seq2point model using Keras' functional API.

        Returns:
        model (tensorflow.keras.Model): The uncompiled seq2point model.

        """

#         input_layer = tf.keras.layers.Input(shape=(a.width,))
#         reshape_layer = tf.keras.layers.Reshape((1, a.width, 1))(input_layer)
#         conv_layer_1 = tf.keras.layers.Convolution2D(filters=30, kernel_size=(10, 1), strides=(1, 1), padding="same", activation="relu")(reshape_layer)
#         conv_layer_2 = tf.keras.layers.Convolution2D(filters=30, kernel_size=(8, 1), strides=(1, 1), padding="same", activation="relu")(conv_layer_1)
#         conv_layer_3 = tf.keras.layers.Convolution2D(filters=40, kernel_size=(6, 1), strides=(1, 1), padding="same", activation="relu")(conv_layer_2)
#         conv_layer_4 = tf.keras.layers.Convolution2D(filters=50, kernel_size=(5, 1), strides=(1, 1), padding="same", activation="relu")(conv_layer_3)
#         conv_layer_5 = tf.keras.layers.Convolution2D(filters=50, kernel_size=(5, 1), strides=(1, 1), padding="same", activation="relu")(conv_layer_4)
#         flatten_layer = tf.keras.layers.Flatten()(conv_layer_5)
#         label_layer = tf.keras.layers.Dense(1024, activation="relu")(flatten_layer)
#         output_layer = tf.keras.layers.Dense(1, activation="linear")(label_layer)

#         model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
        
        input_layer = tf.keras.layers.Input(shape=(a.width))
        reshape_layer = tf.keras.layers.Reshape((a.width, 1))(input_layer)
        conv_layer_1 = tf.keras.layers.Conv1D(filters=30, kernel_size=10, strides=1, padding="same", activation="relu")(reshape_layer)
        conv_layer_1 = tf.keras.layers.Dropout(0.5)(conv_layer_1)
        conv_layer_2 = tf.keras.layers.Conv1D(filters=30, kernel_size=8, strides=1, padding="same", activation="relu")(conv_layer_1)
        conv_layer_2 = tf.keras.layers.Dropout(0.5)(conv_layer_2)
        conv_layer_3 = tf.keras.layers.Conv1D(filters=40, kernel_size=6, strides=1, padding="same", activation="relu")(conv_layer_2)
        conv_layer_3 = tf.keras.layers.Dropout(0.5)(conv_layer_3)
        conv_layer_4 = tf.keras.layers.Conv1D(filters=50, kernel_size=5, strides=1, padding="same", activation="relu")(conv_layer_3)
        conv_layer_4 = tf.keras.layers.Dropout(0.5)(conv_layer_4)
        conv_layer_5 = tf.keras.layers.Conv1D(filters=50, kernel_size=5, strides=1, padding="same", activation="relu")(conv_layer_4)
        conv_layer_5 = tf.keras.layers.Dropout(0.5)(conv_layer_5)
        flatten_layer = tf.keras.layers.Flatten()(conv_layer_5)
        label_layer = tf.keras.layers.Dense(1024, activation="relu")(flatten_layer)
        label_layer = tf.keras.layers.Dropout(0.5)(label_layer)
        output_layer = tf.keras.layers.Dense(1, activation="linear")(label_layer)

        model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    
    ###############################################################################
    # Build the model
    ###############################################################################
        
    if a.dataset == "ukdale":
        ###############################################################################
        # Split dataset v1.0
        ###############################################################################
        # num_data = x_train.shape[0]

        # rand_ind = np.random.permutation(num_data)

        # ratio_training = float(a.ratio_train)
        # ratio_validation= 0.2

        # ind_train = rand_ind[0:int(ratio_training*num_data)]
        # ind_val = rand_ind[-int(ratio_validation*num_data):]

        ###############################################################################
        # Split dataset v1.1
        ###############################################################################
        num_data = x_train.shape[0]

        rand_ind = np.random.permutation(int(num_data*0.9))

        ind = list(rand_ind[0:int(float(a.ratio_train)*rand_ind.shape[0])])+list(np.arange(int(num_data*0.92), num_data))

        ind_train = np.array(ind)

    elif a.dataset == "refit":
        ratio_training = float(a.ratio_train)

        num_data_train = x_train.shape[0]
        rand_ind = np.random.permutation(num_data_train)
        ind_train = rand_ind[0:int(ratio_training*num_data_train)]

#         num_data_val = x_val.shape[0]
#         rand_ind = np.random.permutation(num_data_val)
#         ind_val = rand_ind[0:int(ratio_training*num_data_val)]

    elif a.dataset == "house_2":
        ratio_training = float(a.ratio_train)

        num_data_train = x_train.shape[0]
        rand_ind = np.random.permutation(num_data_train)
        ind_train = rand_ind[0:int(ratio_training*num_data_train)]


    ###############################################################################
    # Training parameters
    ###############################################################################

    epochs = int(a.epoch)
    batch_size = int(a.batch_size)

    STEPS_PER_EPOCH = ind_train.shape[0]//batch_size

    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
                    float(a.lr), 
                    decay_steps=STEPS_PER_EPOCH*int(a.decay_steps),
                    decay_rate=1,
                    staircase=False)

    ###############################################################################
    # Optimizer
    ###############################################################################

    def get_optimizer(opt):
        if opt == "adam":
            return tf.keras.optimizers.Adam(lr_schedule)
        else:
            return tf.keras.optimizers.RMSprop(lr_schedule)


    ###############################################################################
    # Initialize the VAE model
    ###############################################################################
    if a.model == "VAE":
        model.compile(optimizer=get_optimizer(a.optimizer), loss=vae_loss, metrics=[KL_loss, Recon_loss, "mean_absolute_error"])
    elif a.model == "DAE":
        model.compile(loss='mse', optimizer=get_optimizer(a.optimizer), metrics=["mean_absolute_error"])
    elif a.model == "S2P":
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=a.lr, beta_1=0.9, beta_2=0.999), loss="mse", metrics=["mean_absolute_error"])

    ###############################################################################
    # Callback checkpoint settings
    ###############################################################################
    
    list_callbacks = []
    
    # Create a callback that saves the model's weights
    if int(a.save_best) == 1:
        checkpoint_path = "{}/{}/{}/logs/model/{}/{}".format(name, a.dataset, a.model, time, r) +"/checkpoint.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)
        
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=0,
                                                         monitor="val_mean_absolute_error",
                                                         mode="min",
                                                         save_best_only=True)
    else:
        checkpoint_path = "{}/{}/{}/logs/model/{}/{}".format(name, a.dataset, a.model, time, r) +"/cp-{epoch:04d}.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)
        
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=0,
                                                         period=1)
        
    list_callbacks.append(cp_callback)

    class CustomStopper(tf.keras.callbacks.EarlyStopping):
        def __init__(self, monitor='val_loss',
                 min_delta=0, patience=0, verbose=0, mode='auto', start_epoch = 25): # add argument for starting epoch
            super(CustomStopper, self).__init__()
            self.start_epoch = start_epoch

        def on_epoch_end(self, epoch, logs=None):
            if epoch > self.start_epoch:
                super().on_epoch_end(epoch, logs)
    
    if int(a.patience) > 0:
        es_callback = CustomStopper(monitor='val_mean_absolute_error', patience=int(a.patience), start_epoch=int(a.start_stopping))
        
        list_callbacks.append(es_callback)
    
    class AdditionalValidationSets(tf.keras.callbacks.Callback):
        def __init__(self, validation_sets, verbose=0, batch_size=None):
            """
            :param validation_sets:
            a list of 3-tuples (validation_data, validation_targets, validation_set_name)
            or 4-tuples (validation_data, validation_targets, sample_weights, validation_set_name)
            :param verbose:
            verbosity mode, 1 or 0
            :param batch_size:
            batch size to be used when evaluating on the additional datasets
            """
            super(AdditionalValidationSets, self).__init__()
            self.validation_sets = validation_sets
            for validation_set in self.validation_sets:
                if len(validation_set) not in [2, 3]:
                    raise ValueError()
            self.epoch = []
            self.history = {}
            self.verbose = verbose
            self.batch_size = batch_size

        def on_train_begin(self, logs=None):
            self.epoch = []
            self.history = {}

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            self.epoch.append(epoch)

            # record the same values as History() as well
            for k, v in logs.items():
                self.history.setdefault(k, []).append(v)

            # evaluate on the additional validation sets
            for validation_set in self.validation_sets:
                if len(validation_set) == 3:
                    validation_data, validation_targets, validation_set_name = validation_set
                    sample_weights = None
                elif len(validation_set) == 4:
                    validation_data, validation_targets, sample_weights, validation_set_name = validation_set
                else:
                    raise ValueError()

                results = self.model.evaluate(x=validation_data,
                                              y=validation_targets,
                                              verbose=self.verbose,
                                              sample_weight=sample_weights,
                                              batch_size=self.batch_size)

                for i, result in enumerate(results):
                    if i == 0:
                        valuename = validation_set_name + '_loss'
                    else:
                        valuename = validation_set_name + '_' + str(self.model.metrics[i-1].name)
                    self.history.setdefault(valuename, []).append(result)
    
    def transform_s2p(x, y, stride=1):
        print("Shape before S2P data transformations : {}, {}".format(x.shape, y.shape))
        
        x_s2p, y_s2p = [], []
    
        for i in range(x.shape[0]):
            for t in range(0, a.width, stride):
                x_s2p.append(x[i,t:t+a.width,0])
                y_s2p.append(y[i,t+(a.width//2)+1,0])
            
        x_s2p = np.array(x_s2p)
        y_s2p = np.array(y_s2p)
        
        print("Shape after S2P data transformations : {}, {}".format(x_s2p.shape, y_s2p.shape))
        
        return x_s2p, y_s2p
    
    if a.dataset == "ukdale":
        main_mean = int(a.main_mean)
        main_std = int(a.main_std)
            
        app_mean = int(a.app_mean)
        app_std = int(a.app_std)
        
        if a.model == "S2P":
            x_test_s2p, y_test_s2p = transform_s2p(x_test, y_test, 4)
            
            history_cb = AdditionalValidationSets([((x_test_s2p-main_mean)/main_std, (y_test_s2p-app_mean)/app_std, 'House_2')], verbose=1)
        else:
            history_cb = AdditionalValidationSets([((x_test-main_mean)/main_std, (y_test-app_mean)/app_std, 'House_2')], verbose=1)
            
    elif a.dataset == "house_2":
        history_cb = AdditionalValidationSets([(x_test, y_test, 'House_2')], verbose=1)
    elif a.dataset == "refit":
        history_cb = AdditionalValidationSets([(x_test, y_test, 'House_2')], verbose=1)
    
    list_callbacks.append(history_cb)
    
    ###############################################################################
    # Summary of all parameters
    ###############################################################################
    print("###############################################################################")
    print("Summary")
    print("###############################################################################")
    print("Model : {}".format(a.model))
    print("Config : {}".format(a.config))
    print("GPU : {}".format(a.gpu))
    print("Dataset : {}".format(a.dataset))
    print("Appliance : {}".format(a.appliance))
    print("Main mean : {}".format(a.main_mean))
    print("Main std : {}".format(a.main_std))
    print("App mean : {}".format(a.app_mean))
    print("App std : {}".format(a.app_std))
    print("Width : {}".format(a.width))
    print("Strides : {}".format(a.strides))
    print("Epochs : {}".format(a.epoch))
    print("Batch size : {}".format(a.batch_size))
    print("Ratio data : {}".format(a.ratio_train))
    print("Optimizer : {}".format(a.optimizer))
    print("Patience : {}".format(a.patience))
    print("Start stopping epoch : {}".format(a.start_stopping))
    print("Learning rate : {}".format(a.lr))
    print("Decay steps : {}".format(a.decay_steps))
    print("Save best only : {}".format(a.save_best))
    print("Validation data : {}".format(a.val_data))
    print("Run number : {}/{}".format(r,a.run))
    print("###############################################################################")

    if not os.path.exists("{}/{}/{}/logs/model/{}".format(name, a.dataset, a.model, time)):
        os.makedirs("{}/{}/{}/logs/model/{}".format(name, a.dataset, a.model, time))

    file = open("{}/{}/{}/logs/model/{}/config.txt".format(name, a.dataset, a.model, time),"a")
    file.write("###############################################################################\n")
    file.write("Summary\n")
    file.write("###############################################################################\n")
    file.write("Model : {}\n".format(a.model))
    file.write("Config : {}\n".format(a.config))
    file.write("GPU : {}\n".format(a.gpu))
    file.write("Dataset : {}\n".format(a.dataset))
    file.write("Appliance : {}\n".format(a.appliance))
    file.write("Main mean : {}\n".format(a.main_mean))
    file.write("Main std : {}\n".format(a.main_std))
    file.write("App mean : {}\n".format(a.app_mean))
    file.write("App std : {}\n".format(a.app_std))
    file.write("Width : {}\n".format(a.width))
    file.write("Strides : {}\n".format(a.strides))
    file.write("Epochs : {}\n".format(a.epoch))
    file.write("Batch size : {}\n".format(a.batch_size))
    file.write("Ratio data : {}\n".format(a.ratio_train))
    file.write("Optimizer : {}\n".format(a.optimizer))
    file.write("Patience : {}\n".format(a.patience))
    file.write("Start stopping epoch : {}\n".format(a.start_stopping))
    file.write("Learning rate : {}\n".format(a.lr))
    file.write("Decay steps : {}\n".format(a.decay_steps))
    file.write("Save best only : {}\n".format(a.save_best))
    file.write("Validation data : {}\n".format(a.val_data))
    file.write("Run number : {}/{}\n".format(r,a.run))
    file.write("###############################################################################\n")
    file.close()

    if a.dataset == "ukdale":
        if int(a.val_data) == 2:
            ###############################################################################
            # Train the VAE model
            ###############################################################################

            history = model.fit(x_train[ind_train], y_train[ind_train], validation_data=(x_test, y_test), shuffle=True, 
                               epochs=epochs, batch_size=batch_size, callbacks=list_callbacks, verbose=1, initial_epoch=0)
        else:
            ###############################################################################
            # Real Validation
            ###############################################################################
            val_ratio = 0.2
            num_train = ind_train.shape[0]

            ind_tr = ind_train[int(num_train*(val_ratio)):]
            ind_vl = ind_train[0:int(num_train*(val_ratio))]

            if a.model == "S2P":
                x_train_s2p, y_train_s2p = transform_s2p(x_train[ind_tr], y_train[ind_tr], 4)
                x_val_s2p, y_val_s2p = transform_s2p(x_train[ind_vl], y_train[ind_vl], 4)
                
                history = model.fit((x_train_s2p-main_mean)/main_std, (y_train_s2p-app_mean)/app_std, validation_data=((x_val_s2p-main_mean)/main_std, (y_val_s2p-app_mean)/app_std), shuffle=True, 
                                    epochs=epochs, batch_size=batch_size, callbacks=list_callbacks, verbose=1, initial_epoch=0)
                
            else:    
                history = model.fit((x_train[ind_tr]-main_mean)/main_std, (y_train[ind_tr]-app_mean)/app_std, validation_data=((x_train[ind_vl]-main_mean)/main_std, (y_train[ind_vl]-app_mean)/app_std), shuffle=True, 
                                    epochs=epochs, batch_size=batch_size, callbacks=list_callbacks, verbose=1, initial_epoch=0)

        ###############################################################################
        # Save history
        ###############################################################################
        np.save("{}/{}/{}/logs/model/{}/{}/history_{}.npy".format(name, a.dataset, a.model, time, r, epochs), history.history)
        np.save("{}/{}/{}/logs/model/{}/{}/history_cb_{}.npy".format(name, a.dataset, a.model, time, r, epochs), history_cb.history)

        print("Fit finished!")

    elif a.dataset == "refit":
        ###############################################################################
        # Train the VAE model
        ###############################################################################

        history = model.fit(x_train[ind_train], y_train[ind_train], validation_data=(x_val, y_val), shuffle=True, 
                           epochs=epochs, batch_size=batch_size, callbacks=list_callbacks, verbose=1, initial_epoch=0)

        ###############################################################################
        # Save history
        ###############################################################################
        np.save("{}/{}/{}/logs/model/{}/{}/history_{}.npy".format(name, a.dataset, a.model, time, r, epochs), history.history)
        np.save("{}/{}/{}/logs/model/{}/{}/history_cb_{}.npy".format(name, a.dataset, a.model, time, r, epochs), history_cb.history)

        print("Fit finished!")

    elif a.dataset == "house_2":
        history = model.fit(x_train[ind_train], y_train[ind_train], validation_split=0.1, shuffle=True, 
                           epochs=epochs, batch_size=batch_size, callbacks=list_callbacks, verbose=1, initial_epoch=0)

        ###############################################################################
        # Save history
        ###############################################################################
        np.save("{}/{}/{}/logs/model/{}/{}/history_{}.npy".format(name, a.dataset, a.model, time, r, epochs), history.history)
        np.save("{}/{}/{}/logs/model/{}/{}/history_cb_{}.npy".format(name, a.dataset, a.model, time, r, epochs), history_cb.history)

        print("Fit finished!")

    else:
        print("Error in dataset name!")

