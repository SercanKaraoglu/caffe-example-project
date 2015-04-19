import caffe
import h5py
import numpy as np
import pandas as pd

otto_root='/home/sercan/Dropbox/otto/'

net_full_conv = caffe.Net(otto_root+'deploy_solver.txt',otto_root+'_iter_40000.caffemodel',caffe.TEST)

params_full_conv = ['conv1','conv2','conv3','conv4','conv5','fc6','fc7','fc8']
conv_params = {pr: (net_full_conv.params[pr][0].data, net_full_conv.params[pr][1].data) for pr in params_full_conv}
for conv in params_full_conv:
    print '{} weights are {} dimensional and biases are {} dimensional'.format(conv, conv_params[conv][0].shape, conv_params[conv][1].shape)
 
solverDir = otto_root+'solver.prototxt'
solver = caffe.SGDSolver(solverDir)
for pr in params_full_conv:
    solver.net.params[pr][0].data.flat = conv_params[pr][0].flat  # flat unrolls the arrays
    solver.net.params[pr][1].data.flat = conv_params[pr][1].flat
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