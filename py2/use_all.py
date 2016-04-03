# Paul Guelorget

import cPickle, gzip, numpy, theano
import theano.tensor as T

print
print("Bienvenue dans use_all.py.")
print("Ce script va tester toutes les images de la base tes_set en utilisant les poids et biais contenus dans ce repertoire dans les fichiers W1.save, b1.save, W2.save, b2.save.")
print
print("    Chargement des donnees...")
mnist_file = gzip.open('../../mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(mnist_file)
mnist_file.close()

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

print("Test de toutes les images de test_set avec W1, b1, W2, b2...")

error_Matrix = numpy.zeros([10,10])
for i in range(0,10000):
    x = test_set[0][i]
    h = numpy.tanh(numpy.dot(x, W1.get_value())+b1.get_value())
    y = softmax(numpy.dot(h,W2.get_value())+b2.get_value())
    output = 0
    for j in range(1,10):
        if (y[j] > y[output]):
            output = j
    if test_set[1][i] != output :
        print("Erreur a l'indice "+str(i)+".")
        error_Matrix[test_set[1][i]][output] += 1

print
print("Voici les erreurs apparues au moins 3 fois:")
for i in range(0,10):
    for j in range(0,10):
        if error_Matrix[i][j] > 2 :
            print("    "+str(error_Matrix[i][j].astype(int)) + " chiffres " + str(i) + " ont ete vus comme des " + str(j) + ".")

print
print("TOTAL : "+str(error_Matrix.sum().astype(int)) + " erreurs sur 10 000 images testees, soit " + str (100.*error_Matrix.sum()/10000) + " %.")
print
