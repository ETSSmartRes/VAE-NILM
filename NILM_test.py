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

logging.getLogger('tensorflow').disabled = True

###############################################################################
# Config
###############################################################################
parser = argparse.ArgumentParser()

parser.add_argument("--gpu", default=0, type=int, help="GPU to use")
parser.add_argument("--config", default="", type=str, help="Path to the config file")
parser.add_argument("--time", default="", type=str, help="Folder name containing runs")
parser.add_argument("--save_pred", default=False, type=bool, help="Save y_pred_all")
a = parser.parse_args()

# Select GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)

thr_house_2 = { "Fridge" : 50,
                "WashingMachine" : 20,
                "Dishwasher" : 100,
                "Kettle" : 100,
                "Microwave" : 200}

with open(a.config) as data_file:
    nilm = json.load(data_file)

np.random.seed(123)

name = "NILM_Disag_{}".format(nilm["appliance"])
epochs = nilm["training"]["epoch"]
start = nilm["training"]["start_stopping"]

if a.time == "":
    files_and_directories = os.listdir("{}/ukdale/{}/logs/model/".format(name, nilm["model"]))
    time = np.sort(files_and_directories)[-1]
else:
    time = a.time 
    
print("###############################################################################")
print("NILM DISAGREGATOR")
print("GPU : {}".format(a.gpu))
print("CONFIG : {}".format(a.config))
print("FOLDER : {}".format(time))
print("###############################################################################")

###############################################################################
# Load history files
###############################################################################
hist = []

for r in range(1, nilm["run"]+1):
    #hist.append(np.load("{}/ukdale/{}/logs/model/{}/{}/history_cb_{}.npy".format(name, nilm["model"], time, r, epochs), allow_pickle=True))
    hist.append(np.load("{}/ukdale/{}/logs/model/{}/{}/history_{}.npy".format(name, nilm["model"], time, r, epochs), allow_pickle=True))
    
MAE_run = []
for r in range(len(hist)):
    pos_val_min = np.argmin(hist[r].all()["val_mean_absolute_error"][start:epochs])
    MAE_run.append(hist[r].all()["val_mean_absolute_error"][pos_val_min+start])
    
print("Result : {} ± {}".format(np.mean(MAE_run), np.std(MAE_run)))

###############################################################################
# Create Model
###############################################################################

###############################################################################
# Optimizer
###############################################################################
def get_optimizer(opt):
    if opt == "adam":
        return tf.keras.optimizers.Adam(0.001)
    else:
        return tf.keras.optimizers.RMSprop(0.001)

if nilm["model"] == "VAE":
    model = create_model(nilm["model"], nilm["config"], nilm["preprocessing"]["width"], optimizer=get_optimizer(nilm["training"]["optimizer"]))
elif nilm["model"] == "DAE":
    model = create_model(nilm["model"], nilm["config"], nilm["preprocessing"]["width"], optimizer="adam")
elif nilm["model"] == "S2P":
    model = create_model(nilm["model"], nilm["config"], nilm["preprocessing"]["width"], optimizer=tf.keras.optimizers.Adam(learning_rate=nilm["training"]["lr"], beta_1=0.9, beta_2=0.999))
elif nilm["model"] == "S2S":
    model = create_model(nilm["model"], nilm["config"], nilm["preprocessing"]["width"], optimizer=tf.keras.optimizers.Adam(learning_rate=nilm["training"]["lr"], beta_1=0.9, beta_2=0.999))
        
app_list = nilm["appliance"]
width = nilm["preprocessing"]["width"]
stride = nilm["preprocessing"]["strides"]

main_mean = nilm["preprocessing"]["main_mean"]
main_std = nilm["preprocessing"]["main_std"]
app_mean = nilm["preprocessing"]["app_mean"]
app_std = nilm["preprocessing"]["app_std"]

def transform_s2p(x, y, width=199):
    x_s2p, y_s2p = [], []

    for i in range(x.shape[0]):
        for t in range(width):
            x_s2p.append(x[i,t:t+width,0])
            y_s2p.append(y[i,t+width//2+1,0])

    x_s2p = np.array(x_s2p)
    y_s2p = np.array(y_s2p)

    print("Shape before S2P data transformations : {}, {}".format(x.shape, y.shape))
    print("Shape after S2P data transformations : {}, {}".format(x_s2p.shape, y_s2p.shape))

    return x_s2p, y_s2p


MAE_run = []
ACC_run =[]
PR_run = []
RE_run = []
F1_run = []
SAE_run = []

#Load Data
nilm["dataset"]["test"]["ratio"] = [1]    
x_train, y_train, x_test, y_test = load_data(nilm["model"], nilm["appliance"], nilm["dataset"], nilm["preprocessing"]["width"], nilm["preprocessing"]["strides"])

for r in range(1, nilm["run"]+1):
    pos_val_min = np.argmin(hist[r-1].all()["val_mean_absolute_error"][start:epochs]) + start
    
    if nilm["training"]["save_best"]:
        model.load_weights("{}/ukdale/{}/logs/model/{}/{}/checkpoint.ckpt".format(name, nilm["model"], time, r))
    else:
        model.load_weights("{}/ukdale/{}/logs/model/{}/{}/cp-{epoch:04d}.ckpt".format(name, nilm["model"], time, r, epoch=pos_val_min+1))
    
    if nilm["model"] == "S2P":
        x_test_s2p, y_test_s2p = transform_s2p(x_test, y_test, width)
        y_pred = model.predict([(x_test_s2p-main_mean)/main_std], verbose=1)

        y_all_pred = y_pred.reshape([-1])*app_std+app_mean
        x_all = x_test_s2p.reshape([-1])
        y_all_true = y_test_s2p.reshape([-1])
    elif nilm["model"] == "VAE":
        y_pred = model.predict([(x_test-main_mean)/main_std])

        y_all_pred = reconstruct(y_pred[:]*app_std+app_mean, width, stride)
        x_all = reconstruct(x_test[:], width, stride)
        y_all_true = reconstruct(y_test[:], width, stride)
        
    elif nilm["model"] == "S2S":
        y_pred = model.predict([(x_test-main_mean)/main_std], verbose=1)

        y_all_pred = reconstruct(y_pred[:]*app_std+app_mean, width, stride)
        x_all = reconstruct(x_test[:], width, stride)
        y_all_true = reconstruct(y_test[:], width, stride)
        
    elif nilm["model"] == "DAE":
        y_pred = model.predict([(x_test-main_mean)/main_std], verbose=1)

        y_all_pred = reconstruct(y_pred[:]*app_std+app_mean, width, stride)
        x_all = reconstruct(x_test[:], width, stride)
        y_all_true = reconstruct(y_test[:], width, stride)
        
    #y_all_true[y_all_true<15] = 0
    y_all_pred[y_all_pred<15] = 0

    x_all = x_all.reshape([1,-1])
    y_all_pred = y_all_pred.reshape([1,-1])
    y_all_true = y_all_true.reshape([1,-1])
    
    if a.save_pred:
        np.save("{}/ukdale/{}/logs/model/{}/pred.npy".format(name, nilm["model"], time), [y_all_pred, y_all_true])

    print("Best Epoch : {}".format(pos_val_min+1))

    MAE_tot, MAE_app, MAE = MAE_metric(y_all_pred, y_all_true, disaggregation=True, only_power_on=False)
    acc_P_tot, acc_P_app, acc_P = acc_Power(y_all_pred, y_all_true, disaggregation=True)
    PR_app = PR_metric(y_all_pred, y_all_true, thr=thr_house_2[nilm["appliance"]])
    RE_app = RE_metric(y_all_pred, y_all_true, thr=thr_house_2[nilm["appliance"]])
    F1_app = F1_metric(y_all_pred, y_all_true, thr=thr_house_2[nilm["appliance"]])
    SAE_app = SAE_metric(y_all_pred, y_all_true)

    if np.isnan(acc_P_tot):
        print("Nan Detected")
    else:
        MAE_run.append(MAE_tot)
        ACC_run.append(acc_P_tot)
        PR_run.append(PR_app[0])
        RE_run.append(RE_app[0])
        F1_run.append(F1_app[0])
        SAE_run.append(SAE_app[0])

print(np.mean(MAE_run), np.std(MAE_run))
print(np.mean(ACC_run), np.std(ACC_run))
print(np.mean(PR_run), np.std(PR_run))
print(np.mean(RE_run), np.std(RE_run))
print(np.mean(F1_run), np.std(F1_run))
print(np.mean(SAE_run), np.std(SAE_run))

np.save("{}/ukdale/{}/logs/model/{}/results.npy".format(name, nilm["model"], time), [MAE_run, ACC_run, PR_run, RE_run, F1_run, SAE_run])
