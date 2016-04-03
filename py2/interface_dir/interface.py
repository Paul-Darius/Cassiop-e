# v 1.0 by Paul Guelorget
#
#

import os
import sys
import time
import cPickle
import numpy
import theano
import theano.tensor as T
import Image
from theano.tensor.shared_randomstreams import RandomStreams

print
print ("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *")
print ("*                                                                                         *")
print ("* Welcome to the Deep Neural Network interface by Paul Guelorget and Paul-Darius Sarmadi. *")
print ("*                                                                                         *")
print ("* Step 1: we will choose a database for your neural network to learn from.                *")
print ("* Step 2: we will choose the parameters of your neural network.                           *")
print ("* Step 3: we will launch the learning. This will take a while.                            *")
print ("* Step 4: you will be able to test your neural network with any input.                    *")
print ("*                                                                                         *")
print ("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *")
print


###################################################################################################


#
# FONCTIONS PAR ORDRE D'APPEL 
#

def sigmoid(z):
  return (  1.0 / (1.0+numpy.exp(-1.0*z))  )



def load_data():
  print
  print
  print
  print ("          * * *  (1)  DATA LOADER   * * *")
  print ("Your file must contain at least 3 tuples of datasets.")
  print ("  - Any dataset tuple following the third tuple will be ignored.")
  print ("  - Each tuple must be composed of two tuples:")
  print ("    . an array of arrays of values (actual data)")
  print ("    . an array of integers (the dataset's labels)")
  print ("  - Your data's format will be checked as far as possible.")
  print ("  - Order of tuples must be :")
  print ("    1) training set")
  print ("    2) validation set")
  print ("    3) testing set")
  print
  run = 1
  while run == 1:
    dataset = raw_input("Enter dataset path [or 'quit' to abort]: ")
    if dataset == 'quit':
      return -1
    elif not os.path.isfile(dataset):
      print("! Invalid path")
    else:
      if dataset.endswith(".pkl.gz"):
        print ("dataset format is .pkl.gz")
        import gzip
        f = gzip.open(dataset, 'rb')
      else:
        f = file(dataset, 'rb')
      print("    ...loading dataset")
      sets = cPickle.load(f)
      f.close()
      print("    done!")
      print
      if len(sets) < 3:
        print(" /!\ It seems that your file contains less than 3 tuples.")
        print
      if len(sets) > 3:
        print(" ! " + str(len(sets)) + " tuples have been found in your file. Only tuples 1,2,3 will be used.")
      if len(sets) >=3:
        if len(sets[0])==len(sets[1])==len(sets[2])!=2:
          print("  - each set is composed of 2 tuples: ERROR")
          run = 0
        else:
          print("  - each set is composed of 2 tuples: OK")

        if run == 1 and len(sets[0][0])== len(sets[0][1]):
          print("  - training set's data and labels are of same size [" + str(len(sets[0][0])) + "]: OK")
        else:
          print("  - training set's data and labels are'nt of same size: ERROR")
          run = 0

        if run == 1 and len(sets[1][0])== len(sets[1][1]):
          print("  - validation set's data and labels are of same size [" + str(len(sets[1][0])) + "]: OK")
        else:
          print("  - valdation set's data and labels are'nt of same size: ERROR")
          run = 0

        if run == 1 and len(sets[2][0])== len(sets[2][1]):
          print("  - test set's data and labels are of same size [" + str(len(sets[2][0])) + "]: OK")
          data_shape = len(sets[0][0][0])
        else:        
          print("  - test set's data and labels are'nt of same size: ERROR")
          run = 0

        if run == 1:
          data_length = len(sets[0][0][0])
          for i in xrange(0,len(sets[0][0])):
            if len(sets[0][0][i]) != data_length:
              run = 0
              print("  - training set's data is not homogeneous in shape: ERROR")
          for i in xrange(0,len(sets[1][0])):
            if len(sets[1][0][i]) != data_length:
             run = 0
             print("  - validation set's data is not homogeneous in shape: ERROR")
          for i in xrange(0,len(sets[2][0])):
            if len(sets[2][0][i]) != data_length:
              run = 0
              print("  - test set's data is not homogeneous in shape: ERROR")
        if run == 1:
          print("  - all 3 sets' data are homegeneous in shape: OK")
          print("  DATASETS SUCCESSFULLY LOADED")

          return sets[0:3]

def shared_dataset(data_xy):
  data_x, data_y = data_xy
  shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX), borrow=True)
  shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX), borrow=True)
  return shared_x, T.cast(shared_y, 'int32')


###################################################################################################

# :REMINDER - RESULT ARRAY :
# [0] : RBM?
# [1] : RBM epochs
# [2]x10^-3 : RBM learning rate
# [10] : MLP hidden layers
# [11:20] : Size of each hidden layer
# [21] : Activation function (0=tanh / 1=sigmoid)
# [22] : Number of epochs
# [23] : Size of each minibatch
# [24]x10^-3 : MLP learning rate
# [25] : sizeof output


def param():
  result = numpy.zeros(26).astype(numpy.int32)
  string_input = ''
  print
  print
  print
  print ("          * * *  (2)  MACHINE LEARNING PARAMETERS   * * *")
  print ("We are going to set the parameters of the multilayer perceptron.")
  print ("Values between [brackets] are the default values. Use them if you have no clue.")
  print ("Make sure to type integers when needed.")
  print


# RBM?
  wrong_input = 1
  while wrong_input == 1:
    string_input = raw_input("Use Restricted Boltzmann Machine for pre-training? Y/[N] > ")
    if string_input == 'Y' or string_input == 'y':
      result[0] = 1
      wrong_input = 0
    elif string_input == 'N' or string_input == 'n' or string_input == '':
      result[0] = 0
      wrong_input = 0
    else:
      print("! wrong input")

# RBM epochs
  if result[0] == 1:
    wrong_input = 1
    while wrong_input == 1:
      string_input = raw_input("RBM's number of epochs ? [15] > ")
      if string_input.isdigit():
        int_input = int(string_input)
        if int_input > 0:
          wrong_input = 0
          result[1] = int_input
        else:
          print("! Must be > 0")
      elif string_input == '':
        result[1] = 15
        wrong_input = 0
      else:
        print("! Please enter a positive integer or leave blank.")

# RBM LEARNING RATE
    wrong_input = 1
    while wrong_input == 1:
      string_input = raw_input("RBM learning rate [10^-3 x100] > 10^-3 x")
      if string_input.isdigit():
        int_input = int(string_input)
        if int_input >0 and int_input <501:
          result[2] = int_input
          wrong_input = 0
        else:
          print("Must be in [1;500].")
      elif string_input == '':
        wrong_input = 0
        result[2] = 100
      else:
        print("! Please enter an integer or leave blank.")

# MLP hidden layers
  wrong_input = 1
  while wrong_input == 1:
    string_input = raw_input("Number of hidden layers in the neural network? [1]: ")
    if string_input.isdigit():
      int_input = int(string_input)
      if int_input > 0 and int_input < 11:
        result[10] = int_input 
        wrong_input = 0
      else:
        print("! Must be in [1;10].")
    elif string_input == '':
      wrong_input = 0
      result[10] = 1
    else:
      print("! Please enter an integer in [0;10] or leave blank.")

# Size of hidden layers
  if result[10] > 0:
    for i in range(0,int(result[10])):
      wrong_input = 1
      while wrong_input == 1:
        string_input = raw_input("Size of hidden layer number " + str(i + 1) + "["+ str(500/(5**i)) +"] > ")
        if string_input.isdigit():
          int_input = int(string_input)
          if int_input > 0:
            wrong_input = 0
            result[11+i] = int_input
          else:
            print("! Must be > 0.")
        elif string_input == '':
          result[11+i] = 500 / (5**i)
          wrong_input = 0
        else:
          print("! Please enter a positive integer or leave blank.")

# ACTIVATION FUNCTION
  if result[0]:
    print 'RBM has been chosen for pre-training. Activation function is set to sigmoid.'
    result[21] = 1
  else:
    wrong_input = 1
    while wrong_input == 1:
      string_input = raw_input("Activation function? [tanh]/sigmoid > ")
      if string_input == 'tanh' or string_input == '':
        result[21] = 0
        wrong_input = 0
      elif string_input == 'sigmoid':
        result[21] = 1
        wrong_input = 0
      else:
        print("! wrong input")

# NUMBER OF EPOCHS
  wrong_input = 1
  while wrong_input == 1:
    string_input = raw_input("Number of epochs in MultiLayer Perceptron? [1000] > ")
    if string_input.isdigit():
      int_input = int(string_input)
      if int_input > 0:
        wrong_input = 0
        result[22] = int_input
      else:
        print("! Must be > 0.")
    elif string_input == '':
      wrong_input = 0
      result[22] = 1000
    else:
      print("! Please enter a positive integer or leave blank.")

# SIZE OF MINI-BACTHES
  wrong_input = 1
  while wrong_input == 1:
    string_input = raw_input("Size of MultiLayer Perceptron (and RBM if asked for) mini-batches? [20] > ")
    if string_input.isdigit():
      int_input = int(string_input)
      if int_input > 0:
        wrong_input = 0
        result[23] = int_input
      else:
        print("! Must be > 0.")
    elif string_input == '':
      wrong_input = 0
      result[23] = 20
    else:
      print("! Please enter a positive integer or leave blank.")

# MLP LEARNING RATE
  wrong_input = 1
  while wrong_input == 1:
    string_input = raw_input("MLP learning rate [10^-3 x10] > 10^-3 x")
    if string_input.isdigit():
      int_input = int(string_input)
      if int_input >0 and int_input <501:
        result[24] = int_input
        wrong_input = 0
      else:
        print("Must be in [1;500].")
    elif string_input == '':
      wrong_input = 0
      result[24] = 10
    else:
      print("! Please enter an integer or leave blank.")

# OUTPUT SIZE / NUMBER OF CLASSES
  wrong_input = 1
  while wrong_input == 1:
    string_input = raw_input("How many classes / What ouput size? [10] > ")
    if string_input.isdigit():
      int_input = int(string_input)
      if int_input > 1:
        wrong_input = 0
        result[25] = int_input
      else:
        print("! Must be > 1.")
    elif string_input == '':
      wrong_input = 0
      result[25] = 10
    else:
      print ("! Please enter an integer or leave blank.")

  return result


###################################################################################################

def learning(parameters, datasets):
  print
  print
  print
  print ("          * * *  (3)  MACHINE LEARNING   * * *")
  print ("Parameters sum-up:")
  if parameters[0] == 1:
    print ("  - RBM for pre-training : YES")
    print ("  - RBM epochs: " + str(parameters[1]))
    print ("  - RBM learning rate:" + str(parameters[2]) + "x10^-3.")
  else:
    print ("  - RBM for pre-training : NO")
  print("  - MultiLayer Perceptron layers: "+ str(parameters[10]))
  for i in range(0,int(parameters[10])):
    print("  - Size of hidden layer number " + str(i+1) + ": " + str(parameters[11+i]) + ".")
  if parameters[21] == 0:
    print("  - Activation function: TANH")
  else:
    print("  - Activation function: SIGMOID")
  print("  - Number of epochs in MLP: " + str(parameters[22]) + ".")
  print("  - Size of MLP mini-batches: " + str(parameters[23]) + ".")
  print("  - MLP learning rate: " + str(parameters[24]) + "x10^-3.")

  train_set_x, train_set_y = shared_dataset(datasets[0])
  valid_set_x, valid_set_y = shared_dataset(datasets[1])
  test_set_x, test_set_y = shared_dataset(datasets[2])
  shared_datasets = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
  batch_size = parameters[23]

# This class represents a single layer Restricted Boltzmann Machine (RBM)
  class RBM(object):
    def __init__(self, input, n_in, n_out, W=None, hbias=None, vbias=None, numpy_rng=None, theano_rng=None):
      self.n_in = n_in   # size of input
      self.n_out = n_out # size of output
      print ('Creating a RBM object: n_in='+str(n_in)+', n_out='+str(n_out)+'.')
      if numpy_rng is None:
        # create a number generator
        numpy_rng = numpy.random.RandomState(1234)
      if theano_rng is None:
        theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
      if W is None:
        initial_W = numpy.asarray(rng.uniform(low=-4*numpy.sqrt(6. / (n_in + n_out)),
                                               high=4*numpy.sqrt(6. / (n_in + n_out)),
                                               size=(n_in, n_out)), dtype=theano.config.floatX)
        W = theano.shared(value=initial_W, name='W', borrow=True)
      if hbias is None:
        hbias = theano.shared(value=numpy.zeros(n_out, dtype=theano.config.floatX), name='hbias', borrow = True)
      if vbias is None:
        vbias = theano.shared(value=numpy.zeros(n_in,  dtype=theano.config.floatX), name='vbias', borrow=True)
      self.input = input
      self.W = W
      self.hbias = hbias
      self.vbias = vbias
      self.theano_rng = theano_rng
      self.params = [self.W, self.hbias, self.vbias]
    def free_energy(self, v_sample):
      wx_b = T.dot(v_sample, self.W) + self.hbias
      vbias_term = T.dot(v_sample, self.vbias)
      hidden_term = T.sum(T.log(1+T.exp(wx_b)), axis=1)
      return -hidden_term - vbias_term
    def propup(self, vis):
      pre_activation = T.dot(vis, self.W) + self.hbias
      if parameters[21] == 0:
        return [pre_activation, T.tanh(pre_activation)]
      else:
        return [pre_activation, T.nnet.sigmoid(pre_activation)]
    def sample_h_given_v(self, v0_sample):
      pre_activation_h1, h1_mean = self.propup(v0_sample)
      h1_sample = self.theano_rng.binomial(size=h1_mean.shape, n=1, p=h1_mean, dtype = theano.config.floatX)
      return [pre_activation_h1, h1_mean, h1_sample]
    def propdown(self, hid):
      pre_activation = T.dot(hid, self.W.T)+self.vbias
      if parameters[21] == 0:
        return [pre_activation, T.tanh(pre_activation)]
      else:
        return [pre_activation, T.nnet.sigmoid(pre_activation)]
    def sample_v_given_h(self, h0_sample):
      pre_activation_v1, v1_mean = self.propdown(h0_sample)
      v1_sample = self.theano_rng.binomial(size=v1_mean.shape, n=1, p=v1_mean, dtype = theano.config.floatX)
      return [pre_activation_v1, v1_mean, v1_sample]
    def gibbs_hvh(self, h0_sample):
      pre_activation_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
      pre_activation_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
      return [pre_activation_v1, v1_mean, v1_sample, pre_activation_h1, h1_mean, h1_sample]
    def gibbs_vhv(self, v0_sample):
      pre_activation_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
      pre_activation_v1, v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
      return [pre_activation_h1, h1_mean, h1_sample, pre_activation_v1, v1_mean, v1_sample]
    def get_cost_updates(self, lr=parameters[2]*0.001, persistent=None, k=1):
      pre_activation_ph, ph_mean, ph_sample = self.sample_h_given_v(self.input)
      if persistent is None:
        chain_start = ph_sample
      else:
        chain_start = persistent
      (
        [pre_activation_nvs, nv_mean, nv_samples, pre_aactivation_nhs, nh_mean, nh_samples],
        updates
      ) = theano.scan(self.gibbs_hvh, outputs_info=[None, None, None, None, None, chain_start],n_steps=k)
      chain_end = nv_samples[-1]
      cost = T.mean(self.free_energy(self.input)) - T.mean(self.free_energy(chain_end))
      gparams = T.grad(cost, self.params, consider_constant=[chain_end])
      for gparam, param in zip(gparams, self.params):
        updates[param] = param - gparam * T.cast(lr, dtype=theano.config.floatX)
      if persistent:
        updates[persistent] = nh_samples[-1]
        monitoring_cost = self.get_pseudo_likelihood_cost(updates)
      else:
        monitoring_cost = self.get_reconstruction_cost(updates, pre_activation_nvs[-1])
      return monitoring_cost, updates
    def get_pseudo_likelihood_cost(self, updates):
      bit_i_idx = theano.shared(value=0, name='bit_i_idx')
      xi = T.round(self.input)
      fe_xi = self.free_energy(xi)
      xi_flip = T.set_subtensor(xi[:, bit_i_idx], 1 - xi[:, bit_i_idx])
      fe_xi_flip = self.free_energy(xi_flip)
      cost = T.mean(self.n_in * T.log(T.nnet.sigmoid(fe_xi_flip - fe_xi)))
      updates[bit_i_idx] = (bit_i_idx+1)%self.n_in
      return cost
    def get_reconstruction_cost(self, updates, pre_activation_nv):
      activation = T.nnet.sigmoid
      return T.mean(T.sum(self.input * T.log(activation(pre_activation_nv)) + 
             (1 - self.input) * T.log(1 - activation(pre_activation_nv)), axis = 1 )) # cross entropy

# This class represents a single hidden layer of the MultiLayer Perceptron (MLP)
  class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, activation, rank):
      if not parameters[0]:  # if no pretraining
        W_values = numpy.asarray(rng.uniform(low=-numpy.sqrt(6. / (n_in + n_out)),
                                             high=numpy.sqrt(6. / (n_in + n_out)),
                                             size=(n_in, n_out)), dtype=theano.config.floatX)
        if activation == T.nnet.sigmoid:
          W_values *= 4
        W = theano.shared(value=W_values, name='W', borrow=True)
        b_values = numpy.zeros((n_out,),dtype=theano.config.floatX)
        b = theano.shared(value=b_values, name='b', borrow=True)
      else: # if pretraining
        f = file('RBMparams.save')
        RBMparamlist = cPickle.load(f)
        f.close()
        W = RBMparamlist[2*rank]
        b = RBMparamlist[2*rank+1]
      self.W = W
      self.b = b
      lin_output = T.dot(input, self.W) + self.b
      self.output = (lin_output if activation is None else activation(lin_output))
      self.params = [self.W, self.b]

# This class represents the last layer of the MLP a.k.a regression layer
  class RegressionLayer(object):
    def __init__(self, input, n_in, n_out):
      self.W = theano.shared(value=numpy.zeros((n_in, n_out), dtype=theano.config.floatX), name='W', borrow=True)
      self.b = theano.shared(value=numpy.zeros((n_out,),dtype=theano.config.floatX), name='b', borrow=True)
      self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
      self.y_pred = T.argmax(self.p_y_given_x, axis=1)
      self.params = [self.W, self.b]
    def negative_log_likelihood(self, y):
      return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
    def errors(self, y):
      if y.ndim != self.y_pred.ndim:
        raise TypeError('y should have the same shape as self.y_pred',('y', y.type, 'y_pred', self.y_pred.type))
      if y.dtype.startswith('int'):
        return T.mean(T.neq(self.y_pred, y))
      else:
        raise NotImplementedError()

# This class represents the whole MultiLayer Perceptron
  class MLP(object):
    def __init__(self, rng, input, n_in, n_out):
      if parameters[21] == 0:
        self.HiddenLayerList = [ HiddenLayer(rng=rng, input=input, n_in=n_in, n_out=parameters[11], activation=T.tanh, rank=0) ]
      else:
        self.HiddenLayerList = [ HiddenLayer(rng=rng, input=input, n_in=n_in, n_out=parameters[11], activation=T.nnet.sigmoid, rank=0) ]
      for i in range(1, parameters[10]):
        if parameters[21] == 0:
          self.HiddenLayerList.append( HiddenLayer(rng=rng,
                                                   input=self.HiddenLayerList[i-1].output,
                                                   n_in=parameters[10+i],
                                                   n_out=parameters[11+i],
                                                   activation=T.tanh, rank=i))
        else:
          self.HiddenLayerList.append( HiddenLayer(rng=rng,
                                                   input=self.HiddenLayerList[i-1].output,
                                                   n_in=parameters[10+i],
                                                   n_out=parameters[11+i],
                                                   activation=T.nnet.sigmoid, rank=i))
      self.logRegressionLayer = RegressionLayer(input=self.HiddenLayerList[-1].output,
                                                   n_in =parameters[10+parameters[10]],
                                                   n_out=n_out)

      self.L1 = abs(self.logRegressionLayer.W).sum()
      for i in xrange(0,parameters[10]):
        self.L1 += abs(self.HiddenLayerList[i].W).sum()

      self.L2_sqr = (self.logRegressionLayer.W ** 2).sum()
      for i in xrange(0,parameters[10]):
        self.L2_sqr = (self.HiddenLayerList[i].W ** 2).sum()

      self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood
      self.errors = self.logRegressionLayer.errors
      self.params = []
      for i in xrange(0,parameters[10]):
        self.params += self.HiddenLayerList[i].params
      self.params += self.logRegressionLayer.params

# RUNNING THE RBM. THIS CODE IS EXECUTED ONLY IF RBM AS BEEN SELECTED IN THE OPTIONS
  if parameters[0]:  # means "if RBM wanted"
    learning_rate = parameters[2]

    WBlist = []
    for i in xrange(0, parameters[10]):  # number of RBM layers

      n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
      index = T.lscalar()
      x = T.matrix('x')
      rng = numpy.random.RandomState(123)
      theano_rng = RandomStreams(rng.randint(2**30))
      persistent_chain = theano.shared(numpy.zeros((batch_size, parameters[11+i]), dtype=theano.config.floatX), borrow=True)

      # computing the RBM input
      X = theano.shared(value=train_set_x.get_value(), name='X', borrow=False)
      for layer in xrange(0,i):
        X.set_value = sigmoid(numpy.dot(X.get_value(), WBlist[2*layer].get_value())+WBlist[2*layer+1].get_value())

      # constructing the rbm object with the right input & output sizes, according to i
      if i==0: # first layer RBM
        rbm = RBM(input = x, n_in = len(datasets[0][0][0]), n_out=parameters[11+i],
                  numpy_rng=rng, theano_rng=theano_rng)
      else:   # other layer RBM
        rbm = RBM(input = x, n_in = parameters[10+i], n_out = parameters[11+i],
                  numpy_rng=rng, theano_rng=theano_rng)

      cost, updates = rbm.get_cost_updates(lr=learning_rate, persistent=persistent_chain, k=1)
      

      ####################
      # TRAINING THE RBM #
      ####################

      # disclaimer: the following function has no output. It only updates the RBM parameters

      train_rbm = theano.function([index], cost, updates = updates, givens=
                               {x: train_set_x[index * batch_size:(index+1)*batch_size]},
                                  name='train_rbm')

      plotting_time=0.
      start_time = time.clock()
      for epoch in xrange(parameters[1]): # parameters[1] = RBM epochs. Search for "REMINDER" in this file.
        mean_cost = []
        progress = 0
        for batch_index in xrange(n_train_batches):
          if batch_index*100/n_train_batches >= progress:
            print ('RBM progress: layer '+str(i+1)+'/'+str(parameters[10])+', epoch '+str(epoch+1)+'/'+str(parameters[1])+' : '+str(progress)+'%')
            progress += 5
          mean_cost += [train_rbm(batch_index)]
        print 'Training epoch %d, cost is ' % epoch, numpy.mean(mean_cost)

        # MISSING HERE : plotting filters as images
        # see original file rbm.py at deeplearning.net

      end_time = time.clock()
      pretraining_time = end_time - start_time
      print('Training took %f minutes' % (pretraining_time / 60.))

      WBlist.append(rbm.W)
      WBlist.append(rbm.hbias)


      #########################
      # SAMPLING FROM THE RBM #
      #########################

      # OMITTED ON PURPOSE
     
    #########################################
    # SAUVEGARDE DES POIDS ET BIAIS OBTENUS #
    #########################################

    ParamsFile = file('RBMparams.save', 'wb')
    cPickle.dump(WBlist, ParamsFile, protocol=cPickle.HIGHEST_PROTOCOL)
    ParamsFile.close()

# MLP
# COMPUTE NUMBER OF MINIBATCHES FOR TRAINING, VALIDATION AND TESTING
  n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
  n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
  n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

  ######################
  # BUILD ACTUAL MODEL #
  ######################
  print '   ... building the model'
  L1_reg = 0.00
  L2_reg = 0.0001
  index = T.lscalar()
  x = T.matrix('x')
  y = T.ivector('y')
  rng = numpy.random.RandomState(1234)
  classifier = MLP(rng=rng, input=x, n_in=len(datasets[0][0][0]), n_out=parameters[25])

  cost = classifier.negative_log_likelihood(y) + L1_reg*classifier.L1 + L2_reg*classifier.L2_sqr
  test_model = theano.function(
    inputs=[index],
    outputs=classifier.errors(y),
    givens={
      x: test_set_x[index * batch_size:(index+1) * batch_size],
      y: test_set_y[index * batch_size:(index+1) * batch_size]
    }
  )

  validate_model = theano.function(
    inputs=[index],
    outputs=classifier.errors(y),
    givens={
      x: test_set_x[index * batch_size:(index+1) * batch_size],
      y: test_set_y[index * batch_size:(index+1) * batch_size]
    }
  )

  gparams = [T.grad(cost, param) for param in classifier.params]
  updates = [(param, param - (parameters[24]*0.001).astype(theano.config.floatX)*gparam) for param, gparam in zip(classifier.params, gparams)]
  train_model = theano.function(
    inputs=[index],
    outputs=cost,
    updates=updates,
    givens={
      x: train_set_x[index*batch_size: (index+1)*batch_size],
      y: train_set_y[index*batch_size: (index+1)*batch_size]
    }
  )

  ###############
  # TRAIN MODEL #
  ###############
  print '   ...training'
  patience = 10000 # look at this many examples regardless
  patience_increase = 2 # wait this much longer when a new best is found
  improvement_threshold = 0.995 # a relative improvement of this much is considered significant
  validation_frequency = min(n_train_batches, patience / 2)
  # go through this many minibatche before checking the network on the validation set;
  # in this case we check every epoch

  best_validation_loss = numpy.inf
  best_iter = 0
  test_score = 0
  start_time = time.clock()

  epoch = 0
  done_looping = False

  while(epoch < parameters[22]) and (not done_looping):
    epoch = epoch + 1
    for minibatch_index in xrange(n_train_batches):
      minibatch_avg_cost = train_model(minibatch_index)
      iter = (epoch-1) * n_train_batches + minibatch_index
      if (iter+1)%validation_frequency == 0:
        validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
        this_validation_loss = numpy.mean(validation_losses)
        print('epoch %i, minibatch %i/%i, validation error %f %%' %
             (
               epoch,
               minibatch_index + 1,
               n_train_batches,
               this_validation_loss * 100.
             )
        )
        if this_validation_loss < best_validation_loss:
          if(this_validation_loss < best_validation_loss * improvement_threshold):
            patience = max(patience, iter * patience_increase)
            best_validation_loss = this_validation_loss
            best_iter = iter
            # test it on the test set:
            test_losses = [test_model(i) for i in xrange(n_test_batches)]
            test_score = numpy.mean(test_losses)
            print(('     epoch %i, minibatch %i/%i, test error of'
                   ' best model %f %%') %
                  (epoch, minibatch_index + 1, n_train_batches, test_score*100.))
      if patience <= iter:
        done_looping = True
        break

  end_time = time.clock()
  print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
  print >> sys.stderr, ('The code for file ' + os.path.split(__file__)[1] + ' ran for %.2fm' % ((end_time - start_time) / 60.))
  
  #########################################
  # SAUVEGARDE DES POIDS ET BIAIS OBTENUS #
  #########################################

  ParamsFile = file('MLPparams.save', 'wb')
  cPickle.dump(classifier.params, ParamsFile, protocol=cPickle.HIGHEST_PROTOCOL)
  ParamsFile.close()

  return 1  # sera utilise par la boucle principale pour tester la reussite de l'apprentissage



###################################################################################################

def use(parameters, datasets):

  def softmax(M):
    e = numpy.exp(numpy.array(M))
    dist = e/numpy.sum(e)
    return dist

  menu_answer = -1
  print
  print
  print ("          * * *  (4)  USING THE NETWORK   * * *")
  print ("Type 'test' to select an item from the test set of your already loaded datasets.")
  print ("Type 'ext' to select an item from an exterior file or enter it manually.")
  print ("Type 'back' to go back to MAIN MENU.")
  print
  while menu_answer == -1:
    string_input = raw_input(" > ")
    if string_input == 'test':
      menu_answer = 0
    elif string_input == 'ext':
      menu_answer = 1
    elif string_input == 'back':
      return 0


  # CHECKING if saved parameters suit the data and the chosen parameters
  print
  print ("Checking if the network's parameters saved in MLPparams.save suit the data dimensions and the options you have previously chosen:")
  if not os.path.isfile('MLPparams.save'):
    print("ERROR: File MLPparams.save not found.")
    print("       You can copy it manually or launch machine learning again from MAIN MENU.")
    return -1
  else:
    print("  - existence of MLPparams.save OK")
  MLPparamsfile = file('MLPparams.save')
  WBlist = cPickle.load(MLPparamsfile)
  MLPparamsfile.close()
  if len(WBlist) != (parameters[10]+1)*2:
    print("ERROR: Wrong number of weights and bias.")
    print("       network has "+str(parameters[10])+" hidden layers. Expected "+str((parameters[10]+1)*2)+" items. Found "+str(len(WBlist))+".")
    return -1
  else:
    print("  - network has "+str(parameters[10])+" hidden layers. "+str(len(WBlist))+" items needed in file: OK")
  # Checking shapes of Weight matrix and Bias vectors
  for i in xrange(0, parameters[10]):
    if i == 0:
      if WBlist[0].get_value().shape != (len(datasets[2][0][0]),parameters[11]):
        print("ERROR: Weights Matrix number "+str(1)+" has shape "+str(WBlist[0].get_value().shape)+". Must be "+str((len(datasets[2][0][0]), parameters[11]))+".")
        return -1
      else:
        print("  - shape of Weights Matrix number "+str(1)+": OK")
    else:
      if WBlist[2*i].get_value().shape != (parameters[10+i],parameters[11+i]):
        print("ERROR: Weights Matrix number "+str(i+1)+" has shape "+str(WBlist[2*i].get_value().shape)+". Must be "+str((parameters[10+i], parameters[11+i]))+".")
        return -1
      else:
        print("  - shape of Weights Matrix number "+str(i+1)+": OK")
    if WBlist[2*i+1].get_value().shape != (parameters[11+i],):
      print("ERROR: Bias Vector number "+str(i+1)+" has shape "+str(WBlist[2*i+1].get_value().shape)+". Must be "+str((parameters[11+i],))+".")
      return -1
    else:
      print("  - shape of Bias Vector number "+str(i+1)+": OK")
  if WBlist[-2].get_value().shape != (parameters[10+parameters[10]],parameters[25]):
    print("ERROR: Weigths Matrix number "+str(parameters[10]+1)+" has shape "+str(WBlist[-2].get_value().shape)+". Must be "+str((parameters[10+parameters[10]],parameters[25]))+".")
    return -1
  else:
    print("  - shape of Weigths Matrix number"+str(parameters[10]+1)+": OK")
  if WBlist[-1].get_value().shape != (parameters[25],):
    print("ERROR: Bias Vector number "+str(parameters[10]+1)+" has shape "+str(WBlist[-2].get_value().shape)+". Must be "+str((parameters[25],))+".")
    return -1
  else:
    print("  - shape of Bias Vector number "+str(parameters[10]+1)+": OK")
  # Fin de verification de la taille des matrices de poids et des vecteurs de biais    

  if menu_answer == 0:
    print
    print
    go_on = 1
    while go_on:
      wrong_input = 1
      while wrong_input:
        string_input = raw_input("Please select the item's index in the test set. > ")
        if not string_input.isdigit():
          print("! Please enter an integer.")
        else:
          index = int(string_input)
        if index >=0 and index < len(datasets[2][0]):
          wrong_input = 0
        else:
          print("! Value must be in [ 0 ; " + str(len(datasets[2][0])) + " ].")

      if not parameters[21]:
        activation=numpy.tanh
      else:
        print ("Need to implement sigmoid with numpy") #activation=numpy.sigmoid

      x = datasets[2][0][index]
      for i in xrange(0,parameters[10]):
        x = activation(numpy.dot(x, WBlist[2*i].get_value())+WBlist[2*i+1].get_value())
      y = softmax(numpy.dot(x, WBlist[-2].get_value()) + WBlist[-1].get_value())

      output = 0
      for i in range(1,10):
        if (y[i] > y[output]):
          output = i

      print
      print("**************************************************************")
      print("*                                                            *")
      print("* RESULT: Input was labelled as "+str(datasets[2][1][index])+", has been classified as "+str(output)+". *")
      print("*                                                            *")
      print("**************************************************************")
      print

      print("Individual results are:")
      for i in xrange(0,parameters[25]):
        if i == output:
          print("  > P(y="+str(i)+"|x) = " + "%.7f" % y[i]+" <")
        else:
          print("    P(y="+str(i)+"|x) = " + "%.7f" % y[i])
      print
    

  elif menu_answer == 1:
    print "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -"
    print "Type 'manual' to enter manually each component of your input item."
    print "Type 'file' to import an array of data from a Pickled file."
    print "Type 'picture' to turn a picture file to an array of data. Color will be ignored."
    print "Type 'back' to go back to MAIN MENU."
    print "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -"
    choice = 0
    while not choice:
      string_input = raw_input(" > ")
      if string_input == 'manual':
        choice = 1
      elif string_input == 'file':
        choice = 2
      elif string_input == 'picture':
        choice = 3
      elif string_input == 'back':
        return -1

    if choice == 1:
      input_array = numpy.zeros(len(datasets[0][0][0]))
      print("Usually, data format is float in [0;1].")
      for i in xrange(0,len(datasets[0][0][0])):
        string_input =raw_input("data[ "+str(i)+" / "+str(len(datasets[0][0][0]))+" ] > ")
        while string_input == '':
          string_input = raw_input("                 > ")
        input_array[i] = float(string_input)
      x = input_array

    elif choice ==2 :
      print "TO DO"

    elif choice == 3:
      print("- - - - - - - - - - - - - - - - - - - - -")
      print("Please enter the image's path, or 'quit'.")
      wrong_input = 1
      while wrong_input:
        string_input = raw_input(" > ")
        if string_input == 'quit':
          return 0
        elif os.path.isfile(string_input):
          wrong_input = 0
          im = Image.open(string_input,'r')
          x = numpy.ravel(numpy.array(im).astype(numpy.float32)/255)
          if x.shape != (len(datasets[0][0][0]),):
            print("Picture must have "+str(len(datasets[0][0][0]))+" pixels.")
            wrong_input = 1
        elif string_input != '':
          print("File '"+string_input+"' not found.")

    if not parameters[21]:
        activation=numpy.tanh
    else:
      print ("Need to implement sigmoid with numpy") #activation=numpy.sigmoid

    for i in xrange(0,parameters[10]):
      x = activation(numpy.dot(x, WBlist[2*i].get_value())+WBlist[2*i+1].get_value())
    y = softmax(numpy.dot(x, WBlist[-2].get_value()) + WBlist[-1].get_value())

    output = 0
    for i in range(1,10):
      if (y[i] > y[output]):
        output = i

    print
    print("*******************************************")
    print("*                                         *")
    print("* RESULT: Input has been classified as "+str(output)+". *")
    print("*                                         *")
    print("*******************************************")
    print

    print("Individual results are:")
    for i in xrange(0,parameters[25]):
      if i == output:
        print("  > P(y="+str(i)+"|x) = " + "%.7f" % y[i]+" <")
      else:
        print("    P(y="+str(i)+"|x) = " + "%.7f" % y[i])
    print




    # TO DO : ask for entering data 1 by 1 OR selecting a file (e.g. a picture)





###################################################################################################


def main_menu():
  run = 1
  wrong_input = 1
  print
  print
  print
  print ("          * * *    MAIN MENU    * * *")
  print ("- Type 'data' to choose a database for your neural network to learn from.")
  print ("- Type 'param' to set or update the parameters of your neural network.")
  print ("- Type 'learn' to launch the machine learning. This will take a while!")
  print ("- Type 'use' to classify a particular item with your neural network.")
  print
  print ("- Type 'quit' to leave.")
  print
  while wrong_input == 1:
    input = raw_input("  > ")
    if input == 'data':
      return 1
    elif input == 'param':
      return 2
    elif input == 'learn':
      return 3
    elif input == 'use':
      return 4
    elif input == 'quit':
      return 0
    elif input != '':
       print("/!\ wrong input")

###################################################################################################

#       M A I N   L O O P

run = 1
param_array = numpy.zeros(26).astype(numpy.int32)
datasets = -1 # not initialized with load_data()
param_array[0] = -1 # not initialized with param()
learning_done = 0 # learning not done yet
while run == 1:
  choice = main_menu()
  if choice == 1:
    datasets = load_data()
  elif choice == 2:
    param_array = param()
  elif choice == 3:
    if datasets == -1:
      print
      print("ERROR: You haven't chosen datasets yet.")
      print
    elif param_array[0] == -1:
      print         
      print("ERROR:  Please first set parameters.")
      print
    else:
      learning_done = learning(param_array, datasets)
  elif choice == 4:
    use(param_array, datasets)
  elif choice == 0:
    run = 0









#### NOTES
# 'back propagation error' on Google ou sur deeplearning.net


# rendu ecrit : pas de maths, qu'estce ue python, comment installer, comment utiliser le prgm,, a quoi il sert, donner les fonctions du code
# cuda ... toutes les difficultes rencontrees, etc...

# 5 ~ 6 pages, l'essentiel! Le truc qu'on aurait aime avoir au debut du projet en gros




# pre learning, dupliquer le rbm... entree RBM2 = sortie HiddenLayer1...  pas de pre training pour la couche de sortie. Pour le back propag final, propager l'erreur
