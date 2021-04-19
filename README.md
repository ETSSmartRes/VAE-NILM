# Energy Disaggregation using Variational Autoencoders
This code implements the Variational Autoencoders model used in the paper : 

**Langevin, A., Carbonneau, M. A., Cheriet, M., & Gagnon, G. (2021). Energy Disaggregation using Variational Autoencoders. arXiv preprint arXiv:2103.12177.**

### Comparison methods:

Kelly, J., & Knottenbelt, W. (2015, November). Neural nilm: Deep neural networks applied to energy disaggregation. In Proceedings of the 2nd ACM international conference on embedded systems for energy-efficient built environments (pp. 55-64).

https://github.com/JackKelly/neuralnilm

Chaoyun Zhang, Mingjun Zhong, Zongzuo Wang, Nigel Goddard, and Charles Sutton. "Sequence-to-point learning with neural networks for nonintrusive load monitoring." Thirty-Second AAAI Conference on Artificial Intelligence (AAAI-18), Feb. 2-7, 2018.

https://github.com/MingjunZhong/seq2point-nilm

S2SSPan, Y., Liu, K., Shen, Z., Cai, X., & Jia, Z. (2020, May). Sequence-to-subsequence learning with conditional gan for power disaggregation. In ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 3202-3206). IEEE.

https://github.com/DLZRMR/seq2subseq


# Setup

1. Create your own environment with Python > 3.6
2. Configure deep learning environment with Tensorflow
3. Install others requirement packages
4. Clone this repository

# Datasets and preprocessing

1. Download UKDALE files and extract .dat files in each house folder.

Example:
```
Data/
|-- UKDALE/
|   |-- house_1
|   |   |-- channel1.dat
|   |   |-- channel2.dat
|   |   |-- ...
|   |-- house_2
|   |   |-- channel1.dat
|   |   |-- ...
|   |-- ...
```

2. Execute the preprocess code
```
python uk_dale_preprocess.py
```

It will generate these files for each house and the each appliance:
```
Data/
|-- UKDALE/
|   |-- Dishwasher_appliance_house_1
|   |-- Dishwasher_main_house_1
|   |-- Fridge_appliance_house_1
|   |-- Fridge_main_house_1
|   |-- ...
|   |-- Dishwasher_appliance_house_2
|   |-- Dishwasher_main_house_2
|   |-- Fridge_appliance_house_2
|   |-- Fridge_main_house_2
|   |-- ...
```

# Training and testing
The training is performed with the following command:
```
python NILM_disaggregation.py --gpu 0 --config Config/House_2/WashingMachine_VAE.json
```

Where --gpu is used to select a specific GPU, and --config to select the config file associated with the training to execute.

The test is performed with the following command:
```
python NILM_test.py --gpu 0 --config Config/House_2/WashingMachine_VAE.json
```
The script tests the last trained model of the selected configuration. It predicts the energy disaggregation on the test data e.g., house 2 and saves it in "pred_1.npy". It also prints the results for the metrics: MAE, ACC, PRECISION, RECALL, F1-SCORE, SAE and saves the scores in "results_median.npy".

Example:
```
Best Epoch : 82
6.366289849142183 # MAE
0.8244607666324364 # ACC
0.8333902355752817 # PREC
0.9463532832566028 # RECALL
0.8862867905689065 # F1-SCORE
[0.35107847] # SAE
```
