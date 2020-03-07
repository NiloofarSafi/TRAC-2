from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import numpy as np

df = pd.read_csv("eng/trac2_eng_train_final.csv", sep=',', encoding='utf-8')
y_train = []
for idx, row in df.iterrows():
    # y_train.append(row['Sub-task A'])
    y_train.append(row['Sub-task B'])


class_weights = compute_class_weight('balanced',
                                    np.unique(y_train),
                                    y_train)

print(class_weights)
print(np.unique(y_train))
# print(y_train)