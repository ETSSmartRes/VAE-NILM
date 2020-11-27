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
from NILM_functions import *
import pickle
from scipy.stats import norm
from keras.utils.vis_utils import plot_model
from dtw import *
import logging
import json

ADD_VAL_SET = False

logging.getLogger('tensorflow').disabled = True

###############################################################################
# Config
###############################################################################
parser = argparse.ArgumentParser()

parser.add_argument("--gpu", default=0, type=int, help="Appliance to learn")
parser.add_argument("--config", default="", type=str, help="Path to the config file")
a = parser.parse_args()

# Select GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)

print("###############################################################################")
print("NILM DISAGREGATOR")
print("GPU : {}".format(a.gpu))
print("CONFIG : {}".format(a.config))
print("###############################################################################")

with open(a.config) as data_file:
    nilm = json.load(data_file)

np.random.seed(123)

name = "NILM_Disag_{}".format(nilm["appliance"])
time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

for r in range(1, nilm["run"]+1):
    ###############################################################################
    # Load dataset
    ###############################################################################
    x_train, y_train = load_data(nilm["model"], nilm["appliance"], nilm["dataset"], nilm["preprocessing"]["width"], nilm["preprocessing"]["strides"], set_type="train")
    
    main_mean = nilm["preprocessing"]["main_mean"]
    main_std = nilm["preprocessing"]["main_std"]

    app_mean = nilm["preprocessing"]["app_mean"]
    app_std = nilm["preprocessing"]["app_std"]

    ###############################################################################
    # Training parameters
    ###############################################################################
    epochs = nilm["training"]["epoch"]
    batch_size = nilm["training"]["batch_size"]

    STEPS_PER_EPOCH = x_train.shape[0]//batch_size

    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
                    float(nilm["training"]["lr"]), 
                    decay_steps=STEPS_PER_EPOCH*nilm["training"]["decay_steps"],
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
    # Create and initialize the model
    ###############################################################################
    if nilm["model"] == "VAE":
        model = create_model(nilm["model"], nilm["config"], nilm["preprocessing"]["width"], optimizer=get_optimizer(nilm["training"]["optimizer"]))
    elif nilm["model"] == "DAE":
        model = create_model(nilm["model"], nilm["config"], nilm["preprocessing"]["width"], optimizer="Adam")
    elif nilm["model"] == "S2P":
        model = create_model(nilm["model"], nilm["config"], nilm["preprocessing"]["width"], optimizer=tf.keras.optimizers.Adam(learning_rate=nilm["training"]["lr"], beta_1=0.9, beta_2=0.999))
    elif nilm["model"] == "S2S":
        model = create_model(nilm["model"], nilm["config"], nilm["preprocessing"]["width"], optimizer=tf.keras.optimizers.Adam(learning_rate=nilm["training"]["lr"], beta_1=0.9, beta_2=0.999))

    ###############################################################################
    # Callback checkpoint settings
    ###############################################################################
    
    list_callbacks = []
    
    # Create a callback that saves the model's weights
    if nilm["training"]["save_best"] == 1:
        checkpoint_path = "{}/{}/{}/logs/model/House_{}/{}/{}".format(name, nilm["dataset"]["name"], nilm["model"], nilm["dataset"]["test"]["house"][0], time, r) +"/checkpoint.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)
        
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=0,
                                                         monitor="val_mean_absolute_error",
                                                         mode="min",
                                                         save_best_only=True)
    else:
        checkpoint_path = "{}/{}/{}/logs/model/House_{}/{}/{}".format(name, nilm["dataset"]["name"], nilm["model"], nilm["dataset"]["test"]["house"][0], time, r) +"/cp-{epoch:04d}.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)
        
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=0,
                                                         period=1)
        
    list_callbacks.append(cp_callback)

    if nilm["training"]["patience"] > 0:
        patience = nilm["training"]["patience"]
        start_epoch = nilm["training"]["start_stopping"]
        
        print("Patience : {}, Start at : {}".format(patience, start_epoch))
        
        es_callback = CustomStopper(monitor='val_loss', patience=patience, start_epoch=start_epoch, mode="auto")
        
        list_callbacks.append(es_callback)
    
    ###############################################################################
    # Normalize Test Data and History Callback
    ###############################################################################
    if ADD_VAL_SET:
        if nilm["dataset"]["name"] == "ukdale":
            if nilm["model"] == "S2P":
                x_test_s2p, y_test_s2p = transform_s2p(x_test, y_test, nilm["preprocessing"]["width"], nilm["training"]["S2P_strides"])
                history_cb = AdditionalValidationSets([((x_test_s2p-main_mean)/main_std, (y_test_s2p-app_mean)/app_std, 'House_2')], verbose=1)
            else:
                history_cb = AdditionalValidationSets([((x_test-main_mean)/main_std, (y_test-app_mean)/app_std, 'House_2')], verbose=1)

        elif nilm["dataset"]["name"] == "house_2":
            history_cb = AdditionalValidationSets([(x_test, y_test, 'House_2')], verbose=1)
        elif nilm["dataset"]["name"] == "refit":
            history_cb = AdditionalValidationSets([(x_test, y_test, 'House_2')], verbose=1)

        list_callbacks.append(history_cb)
    
    ###############################################################################
    # Summary of all parameters
    ###############################################################################
    print("###############################################################################")
    print("Summary")
    print("###############################################################################")
    print("{}".format(nilm))
    print("Run number : {}/{}".format(r,nilm["run"]))
    print("###############################################################################")

    if not os.path.exists("{}/{}/{}/logs/model/House_{}/{}".format(name, nilm["dataset"]["name"], nilm["model"], nilm["dataset"]["test"]["house"][0], time)):
        os.makedirs("{}/{}/{}/logs/model/House_{}/{}".format(name, nilm["dataset"]["name"], nilm["model"], nilm["dataset"]["test"]["house"][0], time))

    with open("{}/{}/{}/logs/model/House_{}/{}/config.txt".format(name, nilm["dataset"]["name"], nilm["model"], nilm["dataset"]["test"]["house"][0], time), "w") as outfile:
        json.dump(nilm, outfile)

    ###############################################################################
    # Train Model
    ###############################################################################
    if nilm["dataset"]["name"] == "ukdale":
        ###############################################################################
        # Real Validation
        ###############################################################################
        if nilm["model"] == "S2P":
            x_train_s2p, y_train_s2p = transform_s2p(x_train, y_train, nilm["preprocessing"]["width"], nilm["training"]["S2P_strides"])
            
            history = model.fit((x_train_s2p-main_mean)/main_std, (y_train_s2p-app_mean)/app_std, validation_split=nilm["training"]["validation_split"], shuffle=True, 
                                epochs=epochs, batch_size=batch_size, callbacks=list_callbacks, verbose=1, initial_epoch=0)

        elif nilm["model"] == "VAE":    
            history = model.fit((x_train-main_mean)/main_std, (y_train-app_mean)/app_std, validation_split=nilm["training"]["validation_split"], shuffle=True, 
                                epochs=epochs, batch_size=batch_size, callbacks=list_callbacks, verbose=1, initial_epoch=0)
            
        elif nilm["model"] == "S2S":    
            history = model.fit((x_train-main_mean)/main_std, (y_train-app_mean)/app_std, validation_split=nilm["training"]["validation_split"], shuffle=True, 
                                epochs=epochs, batch_size=batch_size, callbacks=list_callbacks, verbose=1, initial_epoch=0)
            
        elif nilm["model"] == "DAE":    
            history = model.fit((x_train-main_mean)/main_std, (y_train-app_mean)/app_std, validation_split=nilm["training"]["validation_split"], shuffle=True, 
                                epochs=epochs, batch_size=batch_size, callbacks=list_callbacks, verbose=1, initial_epoch=0)

        ###############################################################################
        # Save history
        ###############################################################################
        np.save("{}/{}/{}/logs/model/House_{}/{}/{}/history.npy".format(name, nilm["dataset"]["name"], nilm["model"], nilm["dataset"]["test"]["house"][0], time, r), history.history)
        #np.save("{}/{}/{}/logs/model/{}/{}/history_cb_{}.npy".format(name, nilm["dataset"]["name"], nilm["model"], time, r, epochs), history_cb.history)

        print("Fit finished!")
    else:
        print("Error in dataset name!")

