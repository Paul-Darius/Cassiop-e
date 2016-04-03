# v 1.0 by Paul Guelorget
#
#

# a coder :
#	Choix de la base
#	RBM oui ou non
#	Si oui, parametres du RBM (nombre d'epochs, etc.)
#	MLP : nombre de couches, nombre de neurones dans chaque couche
#	MLP : fonction(s) d'activation
#	MLP : taille des batches
#	MLP : pas de la descente de gradient
#	general : choix CPU/GPU

import os
import sys
import time
import cPickle
import numpy
import theano
import theano.tensor as T
from logistic_sgd import *
from mlp import *

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


def load_data():
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

###################################################################################################

# REMINDER - RESULT ARRAY :
# [0] : RBM?
# [1] : RBM epochs
# [10] : MLP hidden layers
# [11:20] : Size of each hidden layer
# [21] : Activation function (0=tanh / 1=sigmoid)
# [22] : Number of epochs
# [23] : Size of each minibatch
# [24] : Step of gradient descent
# [25] : CPU/GPU for learning (0/1)


def param():
  result = numpy.zeros(27).astype(numpy.int32)
  string_input = ''
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
    elif string_input == 'N' or string_input == 'n':
      result[0] = 0
      wrong_input = 0
    else:
      print("! wrong input")

# RBM epochs
  if result[0] == 1:
    wrong_input = 1
    while wrong_input == 1:
      result[1] = input("RBM's number of epochs? [?] > ")
      if result[1] > 0:
        wrong_input = 0
      else:
        print("Must be > 0")

# MLP hidden layers
  wrong_input = 1
  while wrong_input == 1:
    result[10] = input("Number of hidden layers in the neural network? [1]: ")
    if result[10] >= 0 or result[10] < 11:
      wrong_input = 0
    else:
      print("Must be in [0;10]")

# Size of hidden layers
  if result[10] > 0:
    for i in range(0,int(result[10])):
      wrong_input = 1
      while wrong_input == 1:
        result[11+i] = input("Size of hidden layer number " + str(i) + " [?] > ")
        if result[11+i] > 0:
          wrong_input = 0
        else:
          print("Must be > 0")

# ACTIVATION FUNCTION
  wrong_input = 1
  while wrong_input == 1:
    string_input = raw_input("Activation function? [tanh]/sigmoid > ")
    if string_input == 'tanh':
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
    result[22] = input("Number of epochs in MultiLayer Perceptron? [?] > ")
    if result[22] > 0:
      wrong_input = 0
    else:
      print("Must be > 0")


# SIZE OF MINI-BACTHES
  wrong_input = 1
  while wrong_input == 1:
    result[23] = input("Size of MultiLayer Perceptron mini-batches? [?] > ")
    if result[23] > 0:
      wrong_input = 0
    else:
      print("Must be > 0")

# STEP OF GRADIENT DESCENT
  wrong_input = 1
  while wrong_input == 1:
    result[24] = input("Step of gradient descent [?] > ")
    if result[24] > 0:
      wrong_input = 0
    else:
      print("Must be > 0")

# CPU/GPU
  wrong_input = 1
  while wrong_input == 1:
    print("Use CPU or GPU for computing?")
    string_input = raw_input("If you choose to use a nVidia GPU, make sure all CUDA drivers are installed and python launched as root. [cpu]/gpu > ")
    if string_input == 'cpu':
      result[25] = 0
      wrong_input = 0
    elif string_input == 'gpu':
      result[25] = 1
      wrong_input = 0
    else:
      print("! wrong input.")

# output size
  wrong_input = 1
  while wrong_input == 1:
    result[26] = input("How many classes / What ouput size? > ")
    if result[26] > 1:
      wrong_input = 0
    else:
      print("! wrong input.")

  return result


###################################################################################################

def learning(parameters, datasets):
  print
  print ("          * * *  (3)  MACHINE LEARNING   * * *")
  print ("Parameters sum-up:")
  if parameters[0] == 1:
    print ("  - RBM for pre-training : YES")
    print ("  - RBM epochs: " + parameters[1])
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
  print("  - Step of gradient descent: " + str(parameters[24]) + ".")
  if parameters[25] == 0:
    print("  - Computing using CPU.")
  else:
    print("  - Computing using GPU.")

  train_set_x, train_set_y = shared_dataset(datasets[0])
  valid_set_x, valid_set_y = shared_dataset(datasets[1])
  test_set_x, test_set_y = shared_dataset(datasets[2])
  shared_datasets = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
  batch_size = parameters[23]


###################################################################################################


def main_menu():
  run = 1
  wrong_input = 1
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
param_array = numpy.zeros(27).astype(numpy.int32)
datasets = -1 # not initialized with load_data()
param_array[0] = -1 # not initialized with param()
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
      learning(param_array, datasets)
  elif choice == 4:
    print("TODO")
  elif choice == 0:
    run = 0
