#!/usr/bin/python
__author__ = 'as1986'

import theano.tensor as T
import theano
import numpy as np
import deeplearning.layer
import deeplearning.learning_rule
import deeplearning.utils
from io import open

def load_bitext(fname, using_vocab_l = None, using_vocab_r = None):
    '''

    :param fname:
    :param using_vocab:
    :return:
    '''
    with open(fname, mode='r', encoding='utf-8') as fh:
        sentences_l = []
        sentences_r = []
        for l in fh:
            splitted = l.split(u' ||| ')
            # print splitted
            sentences_l.append(splitted[0].strip())
            sentences_r.append(splitted[1].strip())
    dict_l, sens_l = load_sentences(sentences_l, using_vocab_l)
    dict_r, sens_r = load_sentences(sentences_r, using_vocab_r)
    return dict_l, dict_r, sens_l, sens_r


def load_sentences(sentences, using_vocab = None, embedding=None, read_only=False):
    ''' loads a space separated file, creates the vocab dict, and return arrays of sentences
    :param fname: file name
    :return: (vocab, sentences)
    '''
    if using_vocab is None:
        dict_to_return = dict()
    else:
        dict_to_return = using_vocab
    list_to_return = list()

    for l in sentences:
        # print l.encode(u'utf-8')
        this_line = []
        for w in l.strip().split(u' '):
            if len(w) == 0:
                continue
            if w not in dict_to_return:
                if not read_only:
                    dict_to_return[w] = len(dict_to_return) + 1
                else:
                    continue
            if embedding is None:
                this_line.append(dict_to_return[w])
            else:
                this_line.append(np.asarray(embedding[w.lower()], dtype=np.float32))
        list_to_return.append(this_line)
    return dict_to_return, list_to_return

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('textfile')
    args = parser.parse_args()
    (vocab_l, _, contents_l,_) = load_bitext(args.textfile)
    print 'vocab size: {}'.format(len(vocab_l))

    n_word = len(vocab_l)
    e_dim = 100
    x = T.imatrix('x')
    t = T.imatrix('t')
    layers = [
        deeplearning.layer.tProjection(n_word, e_dim),
        deeplearning.layer.LSTM(e_dim, 100, minibatch=True),
        deeplearning.layer.LSTM(100, 100, minibatch=True)
    ]

    for i, layer in enumerate(layers):
        if i == 0:
            layer_out = layer.fprop(x)
        else:
            layer_out = layer.fprop(layer_out)
    params = []
    for layer in layers:
        params += layer.params
    y = layers[-1].h[-1]
    y_func = theano.function([x], y)
    eval_y = y_func([[1, 2, 3, 4, 5, 6]])
    # eval_y_2 = y_func([[1, 2, 3, 4, 5]])
    cost = ((eval_y - y) ** 2 ).sum()
    cost_func = theano.function([x], cost)
    print cost_func([[1,2,3,4,5]])
    updates = learning_rule(cost, params, eps=1e-6, rho=0.65, clip=1., method="adadelta")


    return

if __name__ == '__main__':
    main()
