from __future__ import division
import pandas as pd
import os

train = 'iben/trac2_iben_train.csv'
dev = 'iben/trac2_iben_dev.csv'


NAG = 0
CAG = 0
OAG = 0

G = 0
NG = 0


train_data = pd.read_csv(dev)
print(len(train_data))

for idx, row in train_data.iterrows():
    if row['Sub-task A'] == 'NAG':
        NAG += 1
    elif row['Sub-task A'] == 'OAG':
        OAG += 1
    else:
        CAG += 1


    if row['Sub-task B'] == 'GEN':
        G += 1
    else:
        NG += 1


print("# NAG: {}, ({})".format(NAG, NAG/len(train_data)))
print("# OAG: {}, ({})".format(OAG, OAG/len(train_data)))
print("# CAG: {}, ({})".format(CAG, CAG/len(train_data)))

print("\n# G: {}, ({})".format(G, G/len(train_data)))
print("# NG: {}, ({})".format(NG, NG/len(train_data)))