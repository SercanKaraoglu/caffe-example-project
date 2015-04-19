import caffe
import h5py
import numpy as np
import pandas as pd

otto_root='/home/sercan/Dropbox/otto/'

net_full_conv = caffe.Net(otto_root+'deploy_solver.txt',otto_root+'second_layer/_iter_20000.caffemodel',caffe.TEST)

params_full_conv = ['fc1','fc2','fc3']
#conv_params = {pr: (net_full_conv.params[pr][0].data, net_full_conv.params[pr][1].data) for pr in params_full_conv}
#for conv in params_full_conv:
 #   print '{} weights are {} dimensional and biases are {} dimensional'.format(conv, conv_params[conv][0].shape, conv_params[conv][1].shape)

conv_params=dict.fromkeys(['fc1','fc2','fc3'],[0,1])
conv_params['fc1'][0]=pd.read_csv(otto_root+'r1Weights.csv').values
conv_params['fc1'][1]=pd.read_csv(otto_root+'r1Bias.csv').values
conv_params['fc2'][0]=pd.read_csv(otto_root+'r2Weights.csv').values
conv_params['fc2'][1]=pd.read_csv(otto_root+'r2Bias.csv').values
conv_params['fc3'][0]=pd.read_csv(otto_root+'r3Weights.csv').values
conv_params['fc3'][1]=pd.read_csv(otto_root+'r3Bias.csv').values
   
solverDir = otto_root+'solver.prototxt'
solver = caffe.SGDSolver(solverDir)
for pr in params_full_conv:
    solver.net.params[pr][0].data.flat = conv_params[pr][0].flat  # flat unrolls the arrays
    solver.net.params[pr][1].data.flat = conv_params[pr][1].flat
solver.solve()
solver.net.save(otto_root+'second_layer/pretrained.caffemodel')
df=pd.read_hdf(otto_root+'test1.h5','r')

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