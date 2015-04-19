from os.path import join
from os.path import expanduser
from numpy import genfromtxt
import pandas as pd
import numpy as np
import csv
import caffe

otto_root = join(expanduser("~"),'Dropbox','otto')
solverDir = join(otto_root,'solver.prototxt')

def learn_and_test():
    solver = caffe.SGDSolver(solverDir)
    solver.solve()
    accuracy=0
    loss=0
    test_iters=10
    for i in range(test_iters):
        solver.test_nets[0].forward()
        accuracy += solver.test_nets[0].blobs['accuracy'].data    
        loss+=solver.test_nets[0].blobs['loss'].data
    loss /= test_iters
    accuracy /= test_iters
    print 'accuracy: ' + str(accuracy)
    print 'loss: ' + str(loss)

learn_and_test()
