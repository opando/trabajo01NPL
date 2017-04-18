#!/usr/local/bin/python3

import numpy as np
import nltk
import os
import sys

from sklearn import svm, grid_search
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report, accuracy_score


##leer data
temp =open('/Users/opando/sourcegit/MachineLearnig/NPL/proyecto/trabajo01NPL/train3.txt','r',encoding='utf8').read().split('\n')
#train = [(post.substring(0, post.lenght - 2),post.substring(post.lenght - 2)];

#limpieza de datos
##limpieza de los post

#eliminar caracteres extraños
def quitarExtr(value):
    return ''.join([i if ord(i) < 128 else ' ' for i in value])

dataLimpia = [ quitarExtr(value) for value in temp ]
print(dataLimpia)

# variable post : contiene los post que seran analisados
post=[tuple(reversed(value.split('\t'))) for value in dataLimpia ]
del post[-1]

print(post)
#
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer

tokenregexp = RegexpTokenizer(r'\w+')

#cojunto de todas las palabras que formaran parte de la matriz
all_words = set(word.lower() for passage in post for word in tokenregexp.tokenize(passage[0]))
print(all_words)


t = [({word: (word in tokenregexp.tokenize(x[0])) for word in all_words}, x[1]) for x in post]
print('Matriz final de los post')
print(t)


#aplicacion de NaiveBayes
classifier = nltk.NaiveBayesClassifier.train(t)
classifier.show_most_informative_features()


##leear data test
test1 =open('/Users/opando/sourcegit/MachineLearnig/NPL/proyecto/trabajo01NPL/test3.txt','r',encoding='utf8').read().split('\n')



#evaluar si existe la palabra , para formar el array
def toArrayEval(value, all_words):
    return {word.lower(): (word in word_tokenize(word.lower())) for word in all_words}


array_test = [tuple(reversed( quitarExtr(value).split('\t'))) for value in test1]
del array_test[-1]


print('array_test')
print (array_test)


respuesta = [ classifier.classify(toArrayEval(value[0],all_words)) for value in array_test ]
respuesta_v = [ value[1] for value in array_test ]


print(respuesta)
print(respuesta_v)


print("[Test set] Accuracy: %.4f" % accuracy_score(respuesta,respuesta_v))
