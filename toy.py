#!/usr/bin/python
__author__ = 'as1986'

import theano as T
import numpy as np
from deeplearning import *
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


def load_sentences(sentences, using_vocab = None):
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
            if w not in dict_to_return:
                dict_to_return[w] = len(dict_to_return) + 1
            this_line.append(dict_to_return[w])
        list_to_return.append(this_line)
    return dict_to_return, list_to_return

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('textfile')
    args = parser.parse_args()
    (vocab_l, _, contents_l,_) = load_bitext(args.textfile)
    print 'vocab size: {}'.format(len(vocab_l))
    for content in contents_l:
        print u' '.join([unicode(x) for x in content])
    return

if __name__ == '__main__':
    main()