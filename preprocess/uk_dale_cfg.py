#Source Code from : https://github.com/DLZRMR/seq2subseq

# House_id : Channel_id
Kettle = {
    1: 10, 2: 8, 3: 2, 5: 18
}
Microwave = {
    1: 13, 2: 15, 5: 23
}
Fridge = {
    1: 12, 2: 14, 4: 5, 5: 19
}
Dishwasher = {
    1: 6, 2: 13, 5: 22
}
WashingMachine = {
    1: 5, 2: 12, 5: 24
}

train = {
    'WashingMachine': {
        1: 5, 5: 24
    },
    'Kettle': {
        1: 10, 3: 2, 5: 18
    },
    'Fridge': {
        1: 12, 4: 5, 5: 19
    },
    'Dishwasher': {
        1: 6, 5: 22
    },
    'Microwave': {
        1: 13, 5: 23
    }
}

test = {
    'WashingMachine': {
        2: 12
    },
    'Kettle': {
        2: 8
    },
    'Fridge': {
        2: 14
    },
    'Dishwasher': {
        2: 13
    },
    'Microwave': {
        2: 15
    }
}

window_width = {
    'WashingMachine': 1024,
    'Kettle': 1024,
    'Fridge': 1024,
    'Dishwasher': 1024,
    'Microwave': 1024
}
negative_ratio = {
    'WashingMachine': 1,
    'Kettle': 1,
    'Fridge': 1,
    'Dishwasher': 1,
    'Microwave': 1
}

outlier_threshold = {
    'WashingMachine': 3,
    'Kettle': 3,
    'Fridge': 3,
    'Dishwasher': 3,
    'Microwave': 3
}

positive_negative_threshold = {
    'WashingMachine': 1500,
    'Kettle': 1500,
    'Fridge': 1500,
    'Dishwasher': 1500,
    'Microwave': 1500
}
on_power_threshold = {
    'WashingMachine': 20,
    'Kettle': 1000,
    'Fridge': 50,
    'Dishwasher': 10,
    'Microwave': 200
}
random_clip = {
    'WashingMachine': 5000,
    'Kettle': 5000,
    'Fridge': 5000,
    'Dishwasher': 5000,
    'Microwave': 5000
}
