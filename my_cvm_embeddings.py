#!/usr/bin/python

import learn
from toy import load_sentences
import numpy as np
from theano import tensor as T
import theano
import deeplearning.layer
import deeplearning.learning_rule
from my_cvm import *

def load_params(fname):
    with open(fname, mode='rb') as fh:
        import cPickle as pickle
        return pickle.load(fh)

def write_embeddings(fname, to_write, func, vocab):
    lines = []
    from io import open
    with open(fname, mode='r', encoding='utf-8') as fh:
        for l in fh:
            lines.append(l.strip())
    single_line = u' '.join(lines)
    _, sen = load_sentences([single_line], vocab, read_only=True)
    vec = get_np_sentence(sen)
    c = func(vec)[0]
    with open(to_write, mode='w', encoding='utf-8') as w_fh:
        w_fh.write(u'{}\n'.format(u' '.join([unicode(x) for x in c])))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('text')
    parser.add_argument('params')
    parser.add_argument('--right', action='store_true')
    args = parser.parse_args()
    params = load_params(args.params)
    if args.right:
        embeddings = params[1].get_value()
    else:
        embeddings = params[0].get_value()
    vocab = dict()
    sens = learn.mono_sentences(args.text)
    for sen in sens:
        vocab, _ = load_sentences([sen], vocab)


    import sys
    import codecs
    sys.stdout = codecs.getwriter('utf8')(sys.stdout)
    sys.stderr.write('all loaded now.\n')
    for k,v in vocab.iteritems():
        emb_str = u' '.join([unicode(x) for x in embeddings[v]])
        print u'{} {}'.format(k, emb_str)

    
    return

if __name__ == '__main__':
    main()
