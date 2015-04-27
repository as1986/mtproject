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
    parser.add_argument('params')
    parser.add_argument('--root', type=str, default=None)
    parser.add_argument('--lang', default='en')
    args = parser.parse_args()
    params = load_params(args.params)
    vocab = params[0][args.lang]

    sen_tensor = T.ivector('sen')
    proj = deeplearning.layer.Projection(embedding = params[1][args.lang][0].get_value())
    summed = get_sum(sen_tensor,proj)
    out_func = theano.function([sen_tensor], [summed], on_unused_input='warn')
    
    import sys
    sys.stderr.write('all loaded now.\n')
    if args.root is None:
        for l in sys.stdin:
            _, loaded = load_sentences([l], vocab, read_only=True)
            np_sen = get_np_sentence(loaded)
            summ = out_func(np_sen)
            print summ
    else:
        import glob2
        files = glob2.glob('{}/**/*.txt'.format(args.root))
        for f in files:
            write_embeddings(f, f + '.embeddings', out_func, vocab)
    
    return

if __name__ == '__main__':
    main()
