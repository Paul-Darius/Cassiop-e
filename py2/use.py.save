# In case of failure, launch as root.
#
#
# Paul Guelorget
#


import cPickle, gzip, numpy, theano
import theano.tensor as T

print
print("Bienvenue dans l'interface de test.")
print
print("    Chargement des donnees...")
mnist_file= gzip.open('../../mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(mnist_file)
mnist_file.close()

# ALL OF THE BELOW USELESS IN THIS SCRIPT! NUMPY USED INSTED OF THEANO
#
#def shared_dataset(data_xy):
#    data_x, data_y = data_xy
#    shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX))
#    shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX))
#    return shared_x, T.cast(shared_y, 'int32')
#test_set_x,  test_set_y  = shared_dataset(test_set)
#valid_set_x, valid_set_y = shared_dataset(valid_set)
#train_set_x, train_set_y = shared_dataset(train_set)

print("    Fait.")
print

W1_file = file('W1.save', 'rb')
b1_file = file('b1.save', 'rb')
W2_file = file('W2.save', 'rb')
b2_file = file('b2.save', 'rb')

W1 = cPickle.load(W1_file)
b1 = cPickle.load(b1_file)
W2 = cPickle.load(W2_file)
b2 = cPickle.load(b2_file)

W1_file.close()
b1_file.close()
W2_file.close()
b2_file.close()


def softmax(M):
        e = numpy.exp(numpy.array(M))
        dist = e/numpy.sum(e)
        return dist

while True:

    indice = input("Veuillez saisir l'indice de l'image a tester dans test_set : ")
    while not isinstance(indice,int):
        print("Erreur : l'indice doit etre un entier.")
        indice = input("Veuillez saisir l'indice de l'image a tester dans test_set : ")    


    x = test_set[0][indice]
    h = numpy.tanh(numpy.dot(x, W1.get_value())+b1.get_value())
    y = softmax(numpy.dot(h,W2.get_value())+b2.get_value())

    output = 0
    for i in range(1,10):
        if (y[i] > y[output]) :
            output = i

    print
    print("**************************************************************")
    print("*                                                            *")
    print("* OUTPUT : L'image etait un "+str(test_set[1][indice]) + ", elle est reconnue comme un " + str(output) + ". *")
    print("*                                                            *")
    print("**************************************************************")
    print

    for i in range(0,10):
        print("P(y="+str(i)+"|x) = " + "%.7f" % y[i])

    print
