import os
from os.path import join
from os.path import expanduser
import caffe
import pandas as pd
import numpy as np
import h5py
from sklearn import preprocessing,cross_validation
otto_root = join(expanduser("~"),'Dropbox','otto')

trainDir = join(otto_root,'train.csv')
train = pd.read_csv(trainDir)
labels = train.target.values
lbl_enc = preprocessing.LabelEncoder()
labels = lbl_enc.fit_transform(labels)

train = train.drop('id', axis=1)
train = train.drop('target', axis=1)

train=train.values

skf=cross_validation.StratifiedKFold(y=labels, n_folds=10,
                                         shuffle=True, random_state=1453)
train_index, test_index = next(iter(skf))
i=1
for train_index, test_index in skf:
    print("TRAIN:", train_index, "TEST:", test_index)
    train_train, train_test = train[train_index], train[test_index]
    labels_train, labels_test = labels[train_index], labels[test_index]

    print('batchSize of train: ' + str(np.shape(train_train)[0]))
    print('test size: ' + str(len(train_test)))
    train_filename = join(otto_root,'train'+str(i)+'.h5')
    test_filename= join(otto_root,'test'+str(i)+'.h5')
    
    i=i+1
    
    # HDF5DataLayer source should be a file containing a list of HDF5 filenames.
    # To show this off, we'll list the same data file twice.
    with h5py.File(train_filename, 'w') as f:
        f['data'] = train_train.astype(np.float64)
        f['label'] = labels_train.astype(np.float64)
    with open(os.path.join(otto_root, 'train.txt'), 'w') as f:
        print >>f,train_filename +'\n'
    
    with h5py.File(test_filename, 'w') as f:
        f['data'] = train_test.astype(np.float64)
        f['label'] = labels_test.astype(np.float64)
    with open(os.path.join(otto_root, 'test.txt'), 'w') as f:
        print >>f,test_filename +'\n'
        
    
    
del train
#del test



