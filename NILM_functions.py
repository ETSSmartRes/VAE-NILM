import numpy as np
import tensorflow as tf
from VAE_functions import *

def load_data(model, appliance, dataset, width, strides, test_from=0, set_type="both"):

    def import_data(app_type, house):
        x = np.load("Data/UKDALE/{}_main_house_{}.npy".format(app_type, house))
        y = np.load("Data/UKDALE/{}_appliance_house_{}.npy".format(app_type, house))

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

    def select_ratio(x, y, ratio, set_type, test_from=0):
        num_data = x.shape[0]
        if set_type == "train":
            ind = np.random.permutation(num_data)
        else:
            ind = np.arange(num_data)
        
        min_data = int(num_data*test_from)
        max_data = int(num_data*ratio)

        if ratio == 0:
            return x[ind[0:1]], y[ind[0:1]]
        else:
            return x[ind[:max_data]], y[ind[:max_data]]

    def create_dataset(appliance, dataset, width, strides, set_type, test_from=0):
        x_tot = np.array([]).reshape(0, width, 1)
        y_tot = np.array([]).reshape(0, width, 1)

        for h, r in zip(dataset[set_type]["house"], dataset[set_type]["ratio"]):
            x, y = import_data(appliance, h) # Load complete dataset
            x_, y_ = seq_dataset(x, y, width, strides) # Divide dataset in window
            
            if set_type == "test":
                x_r, y_r = select_ratio(x_, y_, r, set_type, test_from=test_from)
            else:
                x_r, y_r = select_ratio(x_, y_, r, set_type)# Select the proportion needed

            print("Total house {} : x:{}, y:{}".format(h, x_.shape, y_.shape))
            print("Ratio house {} : {}, x:{}, y:{}".format(h, r, x_r.shape, y_r.shape))

            x_tot = np.vstack([x_tot, x_r])
            y_tot = np.vstack([y_tot, y_r])

        print("Complete dataset : x:{}, y{}".format(x_tot.shape, y_tot.shape))

        return x_tot, y_tot

    ###############################################################################
    # Load dataset
    ###############################################################################
    if dataset["name"] == "ukdale":
        if model == "S2P":
            width = int(width)*2-1
            stride = int(strides)
    
        elif model == "VAE":
            width = width
            stride = strides
        
        elif model == "S2S":
            width = width
            stride = strides
            
        elif model == "DAE":
            width = width
            stride = strides

        print("###############################################################################")
        if (set_type == "train") or (set_type == "both"):
            print("Create train dataset")
            x_train, y_train = create_dataset(appliance, dataset, width, strides, "train")
        
        if (set_type == "test") or (set_type == "both"):
            print("Create test dataset")
            x_test, y_test = create_dataset(appliance, dataset, width, strides, "test", test_from=test_from)

        if (set_type == "both"):
            return x_train, y_train, x_test, y_test
        elif (set_type == "train"):
            return x_train, y_train
        else:
            return x_test, y_test

#         elif a.dataset == "refit":
#             ###############################################################################
#             # REFIT
#             ###############################################################################
#             def resample_u_r(x):
#                 from scipy import interpolate

#                 f = interpolate.interp1d(np.arange(0,x.shape[0]*6,6), x, bounds_error=False)
#                 x_re = f(np.arange(0,x.shape[0]*6,8))

#                 return x_re
#             ###############################################################################
#             # Load train data
#             ###############################################################################
#             dataset_path = "Data/REFIT/appliance_data/{}/".format(a.appliance)

#             app_list = [a.appliance]
#             width = int(a.width)
#             stride = int(a.strides)
            
#             def import_refit(app_type):
#                 x = np.load("Data/REFIT/appliance_data/{}/main_train_validation_all.npy".format(app_type))
#                 y = np.load("Data/REFIT/appliance_data/{}/appliance_train_validation_all.npy".format(app_type))

#                 x = x.reshape([-1])
#                 y = y.reshape([-1])
                
#                 return x, y

#             def seq_dataset(x, y, width, stride):
#                 x_ = []
#                 y_ = []

#                 for t in range(0, x.shape[0]-width, stride):
#                     x_.append(x[t:t+width])
#                     y_.append(y[t:t+width])

#                 x_ = np.array(x_).reshape([-1, width, 1])
#                 y_ = np.array(y_).reshape([-1, width, 1])

#                 return x_, y_

#             for app_ind in range(len(app_list)):
#                 x_train = np.array([]).reshape(0, width, 1)
#                 y_train = np.array([]).reshape(0, width, 1)

#                 x, y = import_refit(app_list[app_ind])
#                 x_, y_ = seq_dataset(x, y, width, stride)

#                 x_train = np.vstack([x_train, x_])
#                 y_train = np.vstack([y_train, y_])

#             print(x_train.shape, y_train.shape)
            
#             ###############################################################################
#             # Load validation data
#             ###############################################################################
#             dataset_path = "Data/REFIT/appliance_data/{}/".format(a.appliance)

#             app_list = [a.appliance]
#             width = int(a.width)
#             stride = int(a.strides)
            
#             def import_refit(app_type):
#                 x = np.load("Data/REFIT/appliance_data/{}/main_test_all.npy".format(app_type))
#                 y = np.load("Data/REFIT/appliance_data/{}/appliance_test_all.npy".format(app_type))

#                 x = x.reshape([-1])
#                 y = y.reshape([-1])
                
#                 return x, y

#             def seq_dataset(x, y, width, stride):
#                 x_ = []
#                 y_ = []

#                 for t in range(0, x.shape[0]-width, stride):
#                     x_.append(x[t:t+width])
#                     y_.append(y[t:t+width])

#                 x_ = np.array(x_).reshape([-1, width, 1])
#                 y_ = np.array(y_).reshape([-1, width, 1])

#                 return x_, y_

#             for app_ind in range(len(app_list)):
#                 x_val = np.array([]).reshape(0, width, 1)
#                 y_val = np.array([]).reshape(0, width, 1)

#                 x, y = import_refit(app_list[app_ind])
#                 x_, y_ = seq_dataset(x, y, width, stride)

#                 x_val = np.vstack([x_val, x_])
#                 y_val = np.vstack([y_val, y_])

#             print(x_val.shape, y_val.shape)
            
#             ###############################################################################
#             # Load test data
#             ###############################################################################

#             app_list = [a.appliance]
#             width = int(a.width)
#             stride = int(a.strides)
            
#             def import_data(app_type, train_test="train"):
#                 x = np.load("Data/{}_main_{}.npy".format(app_type, train_test))
#                 y = np.load("Data/{}_appliance_{}.npy".format(app_type, train_test))

#                 return x, y

#             def seq_dataset(x, y, width, stride):
#                 x_ = []
#                 y_ = []

#                 for t in range(0, x.shape[0]-width, stride):
#                     x_.append(x[t:t+width])
#                     y_.append(y[t:t+width])

#                 x_ = np.array(x_).reshape([-1, width, 1])
#                 y_ = np.array(y_).reshape([-1, width, 1])

#                 return x_, y_

#             for app_ind in range(len(app_list)):
#                 x_test = np.array([]).reshape(0, width, 1)
#                 y_test = np.array([]).reshape(0, width, 1)

#                 x, y = import_data(app_list[app_ind], "test")
                
#                 x = resample_u_r(x)
#                 y = resample_u_r(y)
                
#                 x_, y_ = seq_dataset(x, y, width, stride)

#                 x_test = np.vstack([x_test, x_])
#                 y_test = np.vstack([y_test, y_])

#             print(x_test.shape, y_test.shape)

#         elif a.dataset == "house_2":
#             ###############################################################################
#             # Load dataset Train
#             ###############################################################################

#             app_list = [a.appliance]
#             width = int(a.width)
#             stride = int(a.strides)

#             def import_data(app_type, train_test="train"):
#                 x = np.load("Data/House_2/{}_main_{}.npy".format(app_type, train_test))
#                 y = np.load("Data/House_2/{}_appliance_{}.npy".format(app_type, train_test))

#                 return x, y

#             def seq_dataset(x, y, width, stride):
#                 x_ = []
#                 y_ = []

#                 for t in range(0, x.shape[0]-width, stride):
#                     #if np.diff(y[t:t+width]).max() > 20:
#                     x_.append(x[t:t+width])
#                     y_.append(y[t:t+width])

#                 x_ = np.array(x_).reshape([-1, width, 1])
#                 y_ = np.array(y_).reshape([-1, width, 1])

#                 return x_, y_

#             x_train = np.array([]).reshape(0, width, 1)
#             y_train = np.array([]).reshape(0, width, 1)
#             c_train = np.array([]).reshape(0, width, len(app_list))

#             for i in range(len(app_list)):
#                 x, y = import_data(app_list[i], "train")
#                 x_, y_ = seq_dataset(x, y, width, stride)
#                 c = np.zeros([x_.shape[0], width, len(app_list)])
#                 c[:, :, i] = 1

#                 x_train = np.vstack([x_train, x_])
#                 y_train = np.vstack([y_train, y_])
#                 c_train = np.vstack([c_train, c])

#             print(x_train.shape, y_train.shape, c_train.shape)
            
#             ###############################################################################
#             # Load dataset Test
#             ###############################################################################

#             app_list = [a.appliance]
#             width = int(a.width)
#             stride = int(a.strides)

#             x_test = np.array([]).reshape(0, width, 1)
#             y_test = np.array([]).reshape(0, width, 1)

#             for i in range(len(app_list)):
#                 x, y = import_data(app_list[i], "test")
#                 x_, y_ = seq_dataset(x, y, width, stride)

#                 x_test = np.vstack([x_test, x_])
#                 y_test = np.vstack([y_test, y_])

#             print(x_test.shape, y_test.shape)

def create_model(model, config, width, optimizer):
    
    config = "fixe_filter"
    
    if model == "VAE":
        def KL_loss(y_true, y_pred):
            # Regularization term
            kl_loss = - .5 * tf.reduce_sum(1 + z_log_var - K.square(z_mu) - K.exp(z_log_var), axis=-1)

            return kl_loss

        def Recon_loss(data_orig, data_reconstructed):
            reconstruction_loss = tf.reduce_mean((data_orig - data_reconstructed)**2)

            return reconstruction_loss

        def vae_loss(data_orig, data_reconstructed):
            reconstruction_loss = tf.reduce_mean((data_orig - data_reconstructed)**2)

            kl_loss = - .5 * tf.reduce_sum(1 + z_log_var - K.square(z_mu) - K.exp(z_log_var), axis=-1)

            print(kl_loss.shape)
            print(reconstruction_loss.shape)

            return reconstruction_loss + kl_loss
        
        if config == 0:
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
            
        if config == "fixe_filter":
            start_filter_num = 256
            kernel_size = 3
            latent_dim = 16

            x = tf.keras.Input(shape=(width,1))

            conv_seq1 = conv_block_seq_res_fixe(x, start_filter_num, kernel_size, 1, "conv_seq1", ResCon=False)
            pool1 = tf.keras.layers.MaxPooling1D(name="pool1")(conv_seq1)

            conv_seq2 = conv_block_seq_res_fixe(pool1, start_filter_num, kernel_size, 1, "conv_seq2")
            pool2 = tf.keras.layers.MaxPooling1D(name="pool2")(conv_seq2)

            conv_seq3 = conv_block_seq_res_fixe(pool2, start_filter_num, kernel_size, 1, "conv_seq3")
            pool3 = tf.keras.layers.MaxPooling1D(name="pool3")(conv_seq3)

            conv_seq4 = conv_block_seq_res_fixe(pool3, start_filter_num, kernel_size, 1, "conv_seq4")
            pool4 = tf.keras.layers.MaxPooling1D(name="pool4")(conv_seq4)

            conv_seq5 = conv_block_seq_res_fixe(pool4, start_filter_num, kernel_size, 1, "conv_seq5")
            pool5 = tf.keras.layers.MaxPooling1D(name="pool5")(conv_seq5)

            conv_seq6 = conv_block_seq_res_fixe(pool5, start_filter_num, kernel_size, 1, "conv_seq6", In=False)
            pool6 = tf.keras.layers.MaxPooling1D(name="pool6")(conv_seq6)

            conv_seq7 = conv_block_seq_res_fixe(pool6, start_filter_num, kernel_size, 1, "conv_seq7", In=False)

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
            dconv_seq4 = conv_block_seq_res_fixe(reshape1, start_filter_num, kernel_size, 1, "dconv_seq4", In=False, ResCon=False)
            dconc5 = tf.keras.layers.concatenate([dconv_seq4, conv_seq7], name="dconc5")
            deconv1 = Conv1DTranspose(dconc5, start_filter_num, kernel_size=3, strides=2, padding='same')

            dconv_seq5 = conv_block_seq_res_fixe(deconv1, start_filter_num, kernel_size, 1, "dconv_seq5", In=False)
            dconc7 = tf.keras.layers.concatenate([dconv_seq5, conv_seq6], name="dconc7")
            deconv2 = Conv1DTranspose(dconc7, start_filter_num, kernel_size=3, strides=2, padding='same')

            dconv_seq6 = conv_block_seq_res_fixe(deconv2, start_filter_num, kernel_size, 1, "dconv_seq6", In=False)
            dconc9 = tf.keras.layers.concatenate([dconv_seq6, conv_seq5], name="dconc9")
            deconv3 = Conv1DTranspose(dconc9, start_filter_num, kernel_size=3, strides=2, padding='same')

            dconv_seq7 = conv_block_seq_res_fixe(deconv3, start_filter_num, kernel_size, 1, "dconv_seq7", In=False)
            dconc11 = tf.keras.layers.concatenate([dconv_seq7, conv_seq4], name="dconc11")
            deconv4 = Conv1DTranspose(dconc11, start_filter_num, kernel_size=3, strides=2, padding='same')

            dconv_seq8 = conv_block_seq_res_fixe(deconv4, start_filter_num, kernel_size, 1, "dconv_seq8", In=False)
            dconc13 = tf.keras.layers.concatenate([dconv_seq8, conv_seq3], name="dconc13")
            deconv5 = Conv1DTranspose(dconc13, start_filter_num, kernel_size=3, strides=2, padding='same')

            dconv_seq9 = conv_block_seq_res_fixe(deconv5, start_filter_num, kernel_size, 1, "dconv_seq9", In=False)
            dconc15 = tf.keras.layers.concatenate([dconv_seq9, conv_seq2], name="dconc15")
            deconv6 = Conv1DTranspose(dconc15, start_filter_num, kernel_size=3, strides=2, padding='same')

            dconv_seq10 = conv_block_seq_res_fixe(deconv6, start_filter_num, kernel_size, 1, "dconv_seq10", In=False)
            dconc17 = tf.keras.layers.concatenate([dconv_seq10, conv_seq1], name="dconc17")

            x_pred = tf.keras.layers.Conv1D(1, 3, padding="same", activation="relu", name="x_pred")(dconc17)

            model = tf.keras.Model(inputs=[x, eps], outputs=x_pred)
            model.summary()
            
        elif config == 1:
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
            
        elif config == 2:
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
            
        elif config == 3:
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
            
        model.compile(optimizer=optimizer, loss=vae_loss, metrics=[KL_loss, Recon_loss, "mean_absolute_error"])
        
    elif model == "DAE":
        sequence_len = width
        
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
        
        model.compile(loss='mse', optimizer=optimizer, metrics=["mean_absolute_error"])

    elif model == "S2P":
        """Specifies the structure of a seq2point model using Keras' functional API.

        Returns:
        model (tensorflow.keras.Model): The uncompiled seq2point model.

        """
        
        input_layer = tf.keras.layers.Input(shape=(width))
        reshape_layer = tf.keras.layers.Reshape((width, 1))(input_layer)
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
        
        model.compile(optimizer=optimizer, loss="mse", metrics=["mean_absolute_error"])
        
    elif model == "S2S":
        """Specifies the structure of a seq2point model using Keras' functional API.

        Returns:
        model (tensorflow.keras.Model): The uncompiled seq2point model.

        """
        
        input_layer = tf.keras.layers.Input(shape=(width, 1))
        #reshape_layer = tf.keras.layers.Reshape((width, 1))(input_layer)
        conv_layer_1 = tf.keras.layers.Conv1D(filters=30, kernel_size=10, strides=1, padding="same", activation="relu")(input_layer)
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
        output_layer = tf.keras.layers.Dense(width, activation="linear")(label_layer)
        reshape_output_layer = tf.keras.layers.Reshape((width, 1))(output_layer)

        model = tf.keras.Model(inputs=input_layer, outputs=reshape_output_layer)
        
        model.compile(optimizer=optimizer, loss="mse", metrics=["mean_absolute_error"])
        
    return model

def transform_s2p(x, y, width, stride=1):
    print("Shape before S2P data transformations : {}, {}".format(x.shape, y.shape))

    x_s2p, y_s2p = [], []

    for i in range(x.shape[0]):
        for t in range(0, width, stride):
            x_s2p.append(x[i,t:t+width,0])
            y_s2p.append(y[i,t+(width//2)+1,0])

    x_s2p = np.array(x_s2p)
    y_s2p = np.array(y_s2p)

    print("Shape after S2P data transformations : {}, {}".format(x_s2p.shape, y_s2p.shape))

    return x_s2p, y_s2p

class CustomStopper(tf.keras.callbacks.EarlyStopping):
    def __init__(self, monitor='val_loss', min_delta=0.0001, patience=10, verbose=1, mode='auto', start_epoch=5):
        super().__init__(monitor=monitor, min_delta=min_delta, patience=patience, verbose=verbose, mode=mode)
        self.start_epoch = start_epoch

    def on_epoch_end(self, epoch, logs=None):
        print("On epoch End!")
        if epoch > self.start_epoch:
            super().on_epoch_end(epoch, logs)
            print("On epoch End after starting point!")
    

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
                
def acc_Power(x_pred, x_test, c_test=0, app_ratio=0, disaggregation=False):
    if disaggregation:
        Pest = np.sum(x_pred, axis=1).reshape([-1,1])

        Preal = np.sum(x_test, axis=1).reshape([-1,1])

        acc_P = ((np.abs(Pest - Preal)/(2*Preal))*-1)+1

        acc_P = np.nan_to_num(acc_P)
        
        acc_P_tot = np.mean(acc_P[acc_P>0])
        acc_P_app = acc_P.reshape(-1)
    else:
        M_ratio = (np.tile(app_ratio, [c_test.shape[0],1])*c_test)

        Pest = np.sum(x_pred, axis=1).reshape([-1,1])

        Pest = (np.tile(Pest, [1,c_test.shape[1]])*M_ratio)

        Preal = np.sum(x_test, axis=1).reshape([-1,1])

        Preal = (np.tile(Preal, [1,c_test.shape[1]])*M_ratio)

        acc_P = ((np.abs(Pest - Preal)/(2*Preal))*-1)+1

        acc_P = np.nan_to_num(acc_P)

        acc_P_tot = acc_P.sum(axis=1).mean()
        acc_P_app = acc_P.sum(axis=0)/c_test.sum(axis=0)

    print(acc_P_tot)
    
    return acc_P_tot, acc_P_app, acc_P

def MAE_metric(x_pred, x_test, c_test=0, app_ratio=0, disaggregation=False, only_power_on=False):
    
    if disaggregation:
        if only_power_on:
            MAE = np.zeros(x_pred.shape[0])
            for i in range(x_pred.shape[0]):
                ind = (x_pred[i,:])>0
                MAE[i] = np.mean(np.abs((x_test[i,ind]-x_pred[i,ind])))
            MAE = np.nan_to_num(MAE)
        else:
            MAE = np.mean(np.abs((x_test-x_pred)), axis=1).reshape([-1,1])
        MAE_app = MAE
        MAE_tot = np.mean(MAE[MAE>0])
    else:
        MAE = np.mean(np.abs((x_test-x_pred)), axis=1).reshape([-1,1])
        
        M_ratio = (np.tile(app_ratio, [c_test.shape[0],1])*c_test)
        MAE = np.tile(MAE, [1,c_test.shape[1]])*M_ratio

        MAE_tot = MAE.sum(axis=1).mean()
        MAE_app = MAE.sum(axis=0)/c_test.sum(axis=0)

    print(MAE_tot)
    
    return MAE_tot, MAE_app, MAE

# def SAE_metric(x_pred, x_test, window_size):
    
#     SAE = np.zeros(x_pred.shape[0])
    
#     for i in range(x_pred.shape[0]):
        
#         SAE_1d = []
        
#         for t in range(0, x_pred.shape[1], window_size):
#             SAE_1d.append(np.abs(x_pred[i,t:t+window_size].sum() - x_test[i,t:t+window_size].sum())/x_test[i,t:t+window_size].sum())
        
#         SAE[i] = np.mean(SAE_1d)
        
#     for s in SAE:
#         print(s)
    
#     return SAE

def SAE_metric(x_pred, x_test):
    
    SAE = np.zeros(x_pred.shape[0])
    
    for i in range(x_pred.shape[0]):
        SAE[i] = np.abs(x_pred[i,:].sum() - x_test[i,:].sum())/x_test[i,:].sum()
        
    print(SAE)
    
    return SAE

def EpD_metric(x_pred, x_test, sampling=6):
    
    sPerDay = (60//sampling)*60*24
    
    EpD = np.zeros(x_pred.shape[0])
    
    for i in range(x_pred.shape[0]):
        N_days = x_pred[i,:].shape[0]//sPerDay
        EpD[i] = np.mean(np.abs(np.sum(x_pred[i,0:N_days*sPerDay].reshape(N_days,-1), axis=-1)-np.sum(x_test[i,0:N_days*sPerDay].reshape(N_days,-1), axis=-1)))*sampling/3600
        
    print(EpD)
    
    return EpD

def F1_metric(x_pred, x_test, thr):
    from sklearn.metrics import f1_score as f1_score
    
    x_pred_b = np.copy(x_pred)
    x_pred_b[x_pred_b<thr] = 0
    x_pred_b[x_pred_b>=thr] = 1
    
    x_test_b = np.copy(x_test)
    x_test_b[x_test_b<thr] = 0
    x_test_b[x_test_b>=thr] = 1
    
    F1 = np.zeros(x_pred.shape[0])
    
    for i in range(x_pred.shape[0]):
        F1[i] = f1_score(x_test_b[i,:], x_pred_b[i,:])

    for s in F1:
        print(s)
        
    return F1

def RE_metric(x_pred, x_test, thr):
    from sklearn.metrics import recall_score
    
    x_pred_b = np.copy(x_pred)
    x_pred_b[x_pred_b<thr] = 0
    x_pred_b[x_pred_b>=thr] = 1
    
    x_test_b = np.copy(x_test)
    x_test_b[x_test_b<thr] = 0
    x_test_b[x_test_b>=thr] = 1
    
    RE = np.zeros(x_pred.shape[0])
    
    for i in range(x_pred.shape[0]):
        RE[i] = recall_score(x_test_b[i,:], x_pred_b[i,:])
        
    for s in RE:
        print(s)
        
    return RE

def PR_metric(x_pred, x_test, thr):
    from sklearn.metrics import precision_score
    
    x_pred_b = np.copy(x_pred)
    x_pred_b[x_pred_b<thr] = 0
    x_pred_b[x_pred_b>=thr] = 1
    
    x_test_b = np.copy(x_test)
    x_test_b[x_test_b<thr] = 0
    x_test_b[x_test_b>=thr] = 1
    
    PR = np.zeros(x_pred.shape[0])
    
    for i in range(x_pred.shape[0]):
        PR[i] = precision_score(x_test_b[i,:], x_pred_b[i,:])
        
    for s in PR:
        print(s)
        
    return PR

# def reconstruct(y, width, strides):
    
#     yr = np.zeros([width+(y.shape[0]-1)*strides])
#     zr = np.zeros([width+(y.shape[0]-1)*strides])
    
#     #print(yr.shape)
    
#     for i in range(y.shape[0]):
#         #print(i)
#         yr[i*strides:i*strides+width] += y[i,:,0]
#         zr[i*strides:i*strides+width] += np.ones([width])
        
#     yr /= zr
    
#     return yr

def reconstruct(y, width, strides, merge_type="mean"):
    
    len_total = width+(y.shape[0]-1)*strides
    depth = width//strides
    
    yr = np.zeros([len_total, depth])
    yr[:] = np.nan
    
    for i in range(y.shape[0]):
        for d in range(depth):
            yr[i*strides+(d*strides):i*strides+((d+1)*strides),d] = y[i,d*strides:(d+1)*strides,0]
    
    if merge_type == "mean":
        yr = np.nanmean(yr, axis=1)
    else:
        yr = np.nanmedian(yr, axis=1)
    
    return yr
