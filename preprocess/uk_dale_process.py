import sys
sys.path.append('preprocess')
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.pyplot import savefig
import numpy as np
import uk_dale_cfg
import os
import random
from os.path import join

#Source Code from : https://github.com/DLZRMR/seq2subseq


name = ['WashingMachine', 'Kettle', 'Microwave', 'Fridge', 'Dishwasher']
appliance_dict = {
    'WashingMachine': uk_dale_cfg.WashingMachine,
    'Kettle': uk_dale_cfg.Kettle,
    'Microwave': uk_dale_cfg.Microwave,
    'Fridge': uk_dale_cfg.Fridge,
    'Dishwasher': uk_dale_cfg.Dishwasher
}
base_path = '../Data/UKDALE'


# step1: convert .dat to .npy
def convertnpy():
    for house_id in range(1, 6):
        aggregate_path = join(base_path, 'house_%d/channel_%d.dat' % (house_id, 1))
        aggregate_data = []
        with open(aggregate_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            s = line.split()
            aggregate_data.append([int(s[0]), int(s[1])])
        np.save(join(base_path, 'house_%s/main.npy' % house_id), aggregate_data)
        print('House: %d finished!' % house_id)

    for appliance_name in name:
        for house_id, channel_id in appliance_dict[appliance_name].items():
            appliance_path = join(base_path, 'house_%d/channel_%d.dat' % (house_id, channel_id))
            print('Appliance: %s in house %s Load!' % (appliance_name, house_id))
            appliance_data = []

            with open(appliance_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    s = line.split()
                    appliance_data.append([int(s[0]), int(s[1])])
            np.save(join(base_path, 'house_%d/%s_appliance.npy' % (house_id, appliance_name)), appliance_data)
            print('Appliance: %s House: %d finished!' % (appliance_name, house_id))


# step2: align according to timestamp
def align_per_houses():
    for appliance_name in name:
        for house_id in [1,2,5]:
            aggregate_new = []
            appliance_new = []

            aggregate_data = np.load(join(base_path, 'house_%d/main.npy' % house_id))
            appliance_data = np.load(join(base_path, 'house_%d/%s_appliance.npy' % (house_id, appliance_name)))

            aggregate_index = 0
            appliance_index = 0
            if appliance_data[0][0] < aggregate_data[0][0]:
                print('Appliance time is ahead of aggregate time!')
                for i in range(0, len(appliance_data)):
                    if appliance_data[i][0] < aggregate_data[0][0]:
                        appliance_index = i

            for i in range(appliance_index, len(appliance_data)):
                appliance_time = appliance_data[i][0]
                front = aggregate_index
                behind = -1
                for j in range(aggregate_index, len(aggregate_data)):
                    if aggregate_data[j][0] <= appliance_time:
                        front = j
                    else:
                        behind = j
                        d_1 = abs(appliance_time - aggregate_data[front][0])
                        d_2 = abs(appliance_time - aggregate_data[behind][0])
                        if d_1 <= d_2:
                            aggregate_new.append(aggregate_data[front][1])
                            appliance_new.append(appliance_data[i][1])
                        else:
                            aggregate_new.append(aggregate_data[behind][1])
                            appliance_new.append(appliance_data[i][1])
                        aggregate_index = front
                        break
                if i % 10000 == 0:
                    print('Appliance: %s, House: %s, processing: %f' % (
                    appliance_name, house_id, i / len(appliance_data)))
            print("Appliance: {}, House: {} finished!".format(appliance_name, house_id))
            np.save(join(base_path, "{}_main_house_{}.npy".format(appliance_name, house_id)), aggregate_new)
            np.save(join(base_path, "{}_appliance_house_{}.npy".format(appliance_name, house_id)), appliance_new)
            
# step2: align according to timestamp
def align():
    for appliance_name in name:
        aggregate_new = []
        appliance_new = []
        for house_id, c in uk_dale_cfg.train[appliance_name].items():
            aggregate_data = np.load(join(base_path, 'house_%d/main.npy' % house_id))
            appliance_data = np.load(join(base_path, 'house_%d/%s_appliance.npy' % (house_id, appliance_name)))

            aggregate_index = 0
            appliance_index = 0
            if appliance_data[0][0] < aggregate_data[0][0]:
                print('Appliance time is ahead of aggregate time!')
                for i in range(0, len(appliance_data)):
                    if appliance_data[i][0] > aggregate_data[0][0]:
                        appliance_index = i

            for i in range(appliance_index, len(appliance_data)):
                appliance_time = appliance_data[i][0]
                front = aggregate_index
                behind = -1
                for j in range(aggregate_index, len(aggregate_data)):
                    if aggregate_data[j][0] <= appliance_time:
                        front = j
                    else:
                        behind = j
                        d_1 = abs(appliance_time - aggregate_data[front][0])
                        d_2 = abs(appliance_time - aggregate_data[behind][0])
                        if d_1 <= d_2:
                            aggregate_new.append(aggregate_data[front][1])
                            appliance_new.append(appliance_data[i][1])
                        else:
                            aggregate_new.append(aggregate_data[behind][1])
                            appliance_new.append(appliance_data[i][1])
                        aggregate_index = front
                        break
                if i % 10000 == 0:
                    print('Appliance: %s, House: %s, processing: %f' % (appliance_name, house_id, i/len(appliance_data)))
            print('Appliance: %s, House: %s finished!' % (appliance_name, house_id))
        np.save(join(base_path, '%s_main_train.npy' % appliance_name), aggregate_new)
        np.save(join(base_path, '%s_appliance_train.npy' % appliance_name), appliance_new)
        print('Appliance - train : %s finished!' % appliance_name)

        aggregate_new = []
        appliance_new = []
        house_id = 2
        aggregate_data = np.load(join(base_path, 'house_%d/main.npy' % house_id))
        appliance_data = np.load(join(base_path, 'house_%d/%s_appliance.npy' % (house_id, appliance_name)))

        aggregate_index = 0
        appliance_index = 0
        if appliance_data[0][0] < aggregate_data[0][0]:
            print('Appliance time is ahead of aggregate time!')
            for i in range(0, len(appliance_data)):
                if appliance_data[i][0] < aggregate_data[0][0]:
                    appliance_index = i

        for i in range(appliance_index, len(appliance_data)):
            appliance_time = appliance_data[i][0]
            front = aggregate_index
            behind = -1
            for j in range(aggregate_index, len(aggregate_data)):
                if aggregate_data[j][0] <= appliance_time:
                    front = j
                else:
                    behind = j
                    d_1 = abs(appliance_time - aggregate_data[front][0])
                    d_2 = abs(appliance_time - aggregate_data[behind][0])
                    if d_1 <= d_2:
                        aggregate_new.append(aggregate_data[front][1])
                        appliance_new.append(appliance_data[i][1])
                    else:
                        aggregate_new.append(aggregate_data[behind][1])
                        appliance_new.append(appliance_data[i][1])
                    aggregate_index = front
                    break
            if i % 10000 == 0:
                print('Appliance: %s, House: %s, processing: %f' % (
                appliance_name, house_id, i / len(appliance_data)))
        print('Appliance: %s, House: %s finished!' % (appliance_name, house_id))
        np.save(join(base_path, '%s_main_test.npy' % appliance_name), aggregate_new)
        np.save(join(base_path, '%s_appliance_test.npy' % appliance_name), appliance_new)
        print('Appliance - test : %s finished!' % appliance_name)


# step3: observe the total sequence
def view_total():
    if not os.path.exists(join(base_path, 'total_view')):
        os.mkdir(join(base_path, 'total_view'))
    for appliance_name in name:
        aggregate_data = np.load(join(base_path, '%s_main.npy' % appliance_name))
        appliance_data = np.load(join(base_path, '%s_appliance.npy' % appliance_name))

        plt.figure(('Appliance: %s' % appliance_name), figsize=(30, 20), dpi=200)
        plt.subplot(211)
        plt.plot(aggregate_data)
        plt.subplot(212)
        plt.plot(appliance_data)
        savefig('data/UK_DALE/total_view/%s' % appliance_name)


def separate():
    for appliance_name in name:
        window_width = uk_dale_cfg.window_width[appliance_name]
        scale_up = int(1024 / window_width)
        print('Scale up %d' % scale_up)
        data_path = '../data/UK_DALE'
        count = 0
        negative_ratio = uk_dale_cfg.negative_ratio[appliance_name]
        positive_negative_threshold = uk_dale_cfg.positive_negative_threshold[appliance_name]
        outlier_length = uk_dale_cfg.outlier_threshold[appliance_name]
        on_power_threshold = uk_dale_cfg.on_power_threshold[appliance_name]

        appliance_new = []
        main_new = []

        def exception_process(d):
            one = 0
            current_length = 0
            clip_index = 0
            negative = 1
            for i in range(len(d)):
                current_number = d[i]
                if current_number > on_power_threshold:
                    negative = 0
                if one == 0:
                    if current_number > 0:
                        one = 1
                        current_length += 1
                        clip_index = i
                else:
                    if current_number > 0:
                        current_length += 1
                    else:
                        if current_length <= outlier_length:
                            for j in range(clip_index, i):
                                d[j] = 0
                        one = 0
                        current_length = 0
            if negative == 1:
                for j in range(len(d)):
                    d[j] = 0
            return d

        appliance_data = np.load(join(base_path, '%s_appliance_train.npy' % appliance_name))
        appliance_data_test = np.load(join(base_path, '%s_appliance_test.npy' % appliance_name))
        main_data = np.load(join(data_path, '%s_main_train.npy' % appliance_name))
        main_data_test = np.load(join(data_path, '%s_main_test.npy' % appliance_name))
        print('Appliance: %s data load complete!' % appliance_name)

        # train process
        current_head = 1
        data_length = len(appliance_data)
        end = data_length - window_width - 1
        while current_head < end:
            temp_main = []
            temp_appliance = []
            t_a = []
            data_ = exception_process(appliance_data[current_head:current_head + window_width])
            for i in range(current_head, current_head+window_width):
                for k in range(scale_up):
                    temp_main.append(main_data[i])
                    temp_appliance.append(data_[i-current_head])
                    t_a.append(data_[i-current_head])
            current_head += int(window_width/2)
            sum_ = np.sum(t_a)
            if sum_ < positive_negative_threshold:
                r = random.random()
                if r > negative_ratio:
                    continue
            appliance_new.append(temp_appliance)
            main_new.append(temp_main)
            count += 1
            if count % 1000 == 0:
                print('Train - Type 1, Appliance: %s  processing: %f' % (appliance_name, (current_head / data_length)))

        data_length = len(appliance_data) - window_width-1
        random_clip = uk_dale_cfg.random_clip[appliance_name]
        for j in range(random_clip):
            r = random.random()
            start = int(r * data_length)
            temp_main = []
            temp_appliance = []
            t_a = []
            data_ = exception_process(appliance_data[start:start + window_width])
            for i in range(start, start+window_width):
                for k in range(scale_up):
                    temp_main.append(main_data[i])
                    temp_appliance.append(data_[i-start])
                    t_a.append(data_[i-start])
            sum_ = np.sum(t_a)
            if sum_ < positive_negative_threshold:
                r = random.random()
                if r > negative_ratio:
                    continue
            appliance_new.append(temp_appliance)
            main_new.append(temp_main)
            count += 1
            if count % 1000 == 0:
                print('Train - Type 2, Appliance: %s  processing: %f' % (appliance_name, (j / random_clip)))
        print('Appliance: %s complete!' % appliance_name)
        np.save(os.path.join(data_path, '%s/appliance_train.npy' % appliance_name), appliance_new)
        np.save(os.path.join(data_path, '%s/main_train.npy' % appliance_name), main_new)

        # test process
        appliance_new = []
        main_new = []
        
        current_head = 1
        data_length = len(appliance_data_test)
        end = data_length - window_width - 1
        while current_head < end:
            temp_main = []
            temp_appliance = []
            t_a = []
            data_ = exception_process(appliance_data_test[current_head:current_head + window_width])
            for i in range(current_head, current_head+window_width):
                for k in range(scale_up):
                    temp_main.append(main_data_test[i])
                    temp_appliance.append(data_[i-current_head])
                    t_a.append(data_[i-current_head])
            current_head += int(window_width/2)
            sum_ = np.sum(t_a)
            if sum_ < positive_negative_threshold:
                r = random.random()
                if r > negative_ratio:
                    continue
            appliance_new.append(temp_appliance)
            main_new.append(temp_main)
            count += 1
            if count % 1000 == 0:
                print('Test - Type 1, Appliance: %s  processing: %f' % (appliance_name, (current_head / data_length)))

#         data_length = len(appliance_data_test) - window_width-1
#         random_clip = uk_dale_cfg.random_clip[appliance_name]
#         for j in range(random_clip):
#             r = random.random()
#             start = int(r * data_length)
#             temp_main = []
#             temp_appliance = []
#             t_a = []
#             data_ = exception_process(appliance_data_test[start:start + window_width])
#             for i in range(start, start+window_width):
#                 for k in range(scale_up):
#                     temp_main.append(main_data_test[i])
#                     temp_appliance.append(data_[i-start])
#                     t_a.append(data_[i-start])
#             sum_ = np.sum(t_a)
#             if sum_ < positive_negative_threshold:
#                 r = random.random()
#                 if r > negative_ratio:
#                     continue
#             appliance_new.append(temp_appliance)
#             main_new.append(temp_main)
#             count += 1
#             if count % 1000 == 0:
#                 print('Test - Type 2, Appliance: %s  processing: %f' % (appliance_name, (j / random_clip)))
        print('Appliance: %s complete!' % appliance_name)
        np.save(os.path.join(data_path, '%s/appliance_test.npy' % appliance_name), appliance_new)
        np.save(os.path.join(data_path, '%s/main_test.npy' % appliance_name), main_new)


def convert512():
    for appliance_name in name:
        appliance_train = np.load(join(base_path, '%s/appliance_train.npy' % appliance_name))
        appliance_test = np.load(join(base_path, '%s/appliance_test.npy' % appliance_name))
        train_new = []
        test_new = []
        for i in range(len(appliance_train)):
            train_new.append(appliance_train[i][256: 768])
        for i in range(len(appliance_test)):
            test_new.append(appliance_test[i][256: 768])

        np.save(join(base_path, '%s/appliance_train_512.npy' % appliance_name), train_new)
        np.save(join(base_path, '%s/appliance_test_512.npy' % appliance_name), test_new)
        print('512: %s finished!' % appliance_name)


def generateBalancedDataset(thres, ratio):
    for appliance_name in name:
        appliance_data = np.load(join(base_path, '%s/appliance_train_512.npy' % appliance_name))
        main_data = np.load(join(base_path, '%s/main_train.npy' % appliance_name))
        print('Appliance: %s data load!' % appliance_name)

        appliance_positve = []
        main_positive = []
        appliance_negative = []
        main_negative = []
        appliance_new = []
        main_new = []

        for i in range(len(appliance_data)):
            if np.sum(appliance_data[i]) > thres:
                appliance_positve.append(appliance_data[i])
                main_positive.append(main_data[i])
            else:
                appliance_negative.append(appliance_data[i])
                main_negative.append(main_data[i])

        print('Appliance: %s positive: %d  negative: %d' %
              (appliance_name, len(appliance_positve), len(appliance_negative)))

        if len(appliance_positve)*ratio < len(appliance_negative):
            negative_length = len(appliance_positve)*ratio
        else:
            negative_length = len(appliance_negative)
        negative_index = np.linspace(0, negative_length-1, negative_length).astype(int)
        random.shuffle(negative_index)
        positive_index = np.linspace(0, len(appliance_positve)-1, len(appliance_positve)).astype(int)
        random.shuffle(positive_index)

        for i in positive_index:
            appliance_new.append(appliance_positve[i])
            main_new.append(main_positive[i])
        for i in negative_index:
            appliance_new.append(appliance_negative[i])
            main_new.append(main_negative[i])

        print('Appliance: %s length: %d' % (appliance_name, len(appliance_new)))
        np.save(join(base_path, '%s/appliance_train_balanced_512.npy' % appliance_name), appliance_new)
        np.save(join(base_path, '%s/main_train_balanced.npy' % appliance_name), main_new)
        print('Appliance: %s finished!' % appliance_name)


def shrink(scale):
    for appliance_name in name:
        appliance_train = np.load(join(base_path, '%s/appliance_train_balanced_512.npy' % appliance_name))
        appliance_test = np.load(join(base_path, '%s/appliance_test_512.npy' % appliance_name))
        main_train = np.load(join(base_path, '%s/main_train_balanced.npy' % appliance_name))
        main_test = np.load(join(base_path, '%s/main_test.npy' % appliance_name))
        atr_new = []
        ate_new = []
        mtr_new = []
        mte_new = []
        print('Scale - Appliance: %s data load!' % appliance_name)

        for i in range(len(appliance_train)):
            atr_temp = []
            mtr_temp = []
            for j in range(512):
                atr_temp.append(float(appliance_train[i][j]/scale))
            for j in range(1024):
                mtr_temp.append(float(main_train[i][j]/scale))
            atr_new.append(atr_temp)
            mtr_new.append(mtr_temp)

        for i in range(len(appliance_test)):
            ate_temp = []
            mte_temp = []
            for j in range(512):
                ate_temp.append(float(appliance_test[i][j]/scale))
            for j in range(1024):
                mte_temp.append(float(main_test[i][j]/scale))
            ate_new.append(ate_temp)
            mte_new.append(mte_temp)

        np.save(join(base_path, '%s/appliance_train_%d.npy' % (appliance_name, scale)), atr_new)
        np.save(join(base_path, '%s/main_train_%d.npy' % (appliance_name, scale)), mtr_new)
        np.save(join(base_path, '%s/appliance_test_%d.npy' % (appliance_name, scale)), ate_new)
        np.save(join(base_path, '%s/main_test_%d.npy' % (appliance_name, scale)), mte_new)

        print('Scale: %s finished! ' % appliance_name)


def clip_view():
    clip_number = 300
    for appliance_name in name:
        print('Clip - Appliance: %s' % appliance_name)
        if not os.path.exists(join(base_path, '%s/visual_train' % appliance_name)):
            os.mkdir(join(base_path, '%s/visual_train' % appliance_name))
        if not os.path.exists(join(base_path, '%s/visual_test' % appliance_name)):
            os.mkdir(join(base_path, '%s/visual_test' % appliance_name))

        appliance_data = np.load(join(base_path, '%s/appliance_train_1000.npy' % appliance_name))
        main_data = np.load(join(base_path, '%s/main_train_1000.npy' % appliance_name))
        print('Appliance length: %d, main length: %d' % (len(appliance_data), len(main_data)))

        for i in range(clip_number):
            fig = plt.figure(figsize=(16, 16), dpi=100)
            gs = gridspec.GridSpec(2, 1)
            gs.update(wspace=0.05, hspace=0.05)
            x = np.linspace(256, 768, 512)
            appliance_temp = []
            main_temp = []
            r = int(random.random()*len(appliance_data))
            for j in range(512):
                appliance_temp.append(int(appliance_data[r][j]*1000))
            for j in range(1024):
                main_temp.append(int(main_data[r][j]*1000))

            plt.subplot(2, 1, 1)
            plt.xlim(0, 1024)
            plt.plot(main_temp)

            plt.subplot(2, 1, 2)
            plt.xlim(0, 1024)
            plt.plot(x, appliance_temp)

            plt.savefig(join(base_path, '%s/visual_train/%d.jpg' % (appliance_name, i)))
            plt.close(fig)

        appliance_data = np.load(join(base_path, '%s/appliance_test_1000.npy' % appliance_name))
        main_data = np.load(join(base_path, '%s/main_test_1000.npy' % appliance_name))

        # test section
        for i in range(clip_number):
            fig = plt.figure(figsize=(16, 16), dpi=100)
            gs = gridspec.GridSpec(2, 1)
            gs.update(wspace=0.05, hspace=0.05)
            x = np.linspace(256, 768, 512)
            appliance_temp = []
            main_temp = []
            r = int(random.random()*len(appliance_data))
            for j in range(512):
                appliance_temp.append(int(appliance_data[r][j]*1000))
            for j in range(1024):
                main_temp.append(int(main_data[r][j]*1000))

            plt.subplot(2, 1, 1)
            plt.xlim(0, 1024)
            plt.plot(main_temp)

            plt.subplot(2, 1, 2)
            plt.xlim(0, 1024)
            plt.plot(x, appliance_temp)

            plt.savefig(join(base_path, '%s/visual_test/%d.jpg' % (appliance_name, i)))
            plt.close(fig)


if __name__ == '__main__':
    convertnpy()
    align_per_houses()
    #align()
    #separate()
    #convert512()
    #generateBalancedDataset(1500, 1)
    #shrink(1)
