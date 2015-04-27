#!/usr/bin/python

import learn
from toy import load_sentences
import numpy as np
from theano import tensor as T
import theano
import deeplearning.layer
import deeplearning.learning_rule
from my_cvm import *


def load_language_pair(left, right, vocab_l, vocab_r):
    sens_l = learn.mono_sentences(left)
    sens_r = learn.mono_sentences(right)

    pairs = []
    for (sen_l, sen_r) in zip(sens_l, sens_r):
        vocab_l, loaded_l = load_sentences([sen_l], vocab_l)
        vocab_r, loaded_r = load_sentences([sen_r], vocab_r)
        pairs.append((loaded_l, loaded_r))
    return pairs


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--store', default='cvm_params_multilingual')
    parser.add_argument('--txtembeddings', action='store_true')
    parser.add_argument('--resume', default=None, type=str)
    langs = {'en', 'de'}
    vocab = dict()
    for l in langs:
        vocab[l] = dict()
    training = {('de', 'en'):('/gg/is13/europarl-v7.de-en.de.tokenized.noempty.lower','/gg/is13/europarl-v7.de-en.en.tokenized.noempty.lower'),}
    embeddings = {'en':'/gg/is13/cca-embeddings/glove.europarl-en.100d.txt', 'de':'/gg/is13/cca-embeddings/glove.europarl-de.100d.txt',}
    args = parser.parse_args()
    if args.resume is not None:
        import cPickle as pickle
        fh = open(args.resume, mode='rb')
        vocab, params = pickle.load(fh)
        fh.close()
    loaded_emb = dict()
    for l,fname in embeddings.iteritems():
        if args.txtembeddings:
            loaded_emb[l] = learn.load_txt_embeddings(fname)
        else:
            loaded_emb[l] = learn.load_embeddings(fname)
    loaded_training = dict()
    for ls, fnames in training.iteritems():
        loaded_training[ls] = load_language_pair(fnames[0], fnames[1], vocab[ls[0]], vocab[ls[1]])
    dim = 100
    shared_embeddings = dict()
    for l, emb in loaded_emb.iteritems():
        if args.resume is not None:
            shared_embeddings[l] = params[l][0].get_value()
            del params[l][0]
            del params[l]
        else:
            shared_embeddings[l] = learn.prepare_shared_embeddings(vocab[l], model=emb, dim=dim)
        del emb
    del loaded_emb

    good_tensors = dict()
    bad_tensors = dict()
    projections = dict()
    good_sums = dict()
    bad_sums = dict()
    params = dict()
    for l in langs:
        good_tensors[l] = T.ivector('good_{}'.format(l))
        bad_tensors[l] = T.ivector('bad_{}'.format(l))
        projections[l] = deeplearning.layer.Projection(embedding=shared_embeddings[l],update=True, hid_dim=dim)
        good_sums[l] = get_sum(good_tensors[l],projections[l])
        bad_sums[l] = get_sum(bad_tensors[l],projections[l])
        params[l] = projections[l].params
    # dist_equiv = rbf_dist(summed_l, summed_r)
    # dist_left_sanity = rbf_dist(summed_l, summed_bad_r)
    # dist_right_sanity = rbf_dist(summed_r, summed_bad_l)
    # cost = logloss(dist_equiv, dist_left_sanity) + logloss(dist_equiv, dist_right_sanity)
    costs = dict()
    updates = dict()
    cost_funcs = dict()
    for lang_pair in loaded_training.iterkeys():
        left = lang_pair[0]
        right = lang_pair[1]
        costs[lang_pair] = logloss_rbf_r(good_sums[left], good_sums[right], bad_sums[left]) + logloss_rbf_l(good_sums[left], good_sums[right], bad_sums[right]) + 1e-5 * (T.sum(good_sums[left] ** 2) + T.sum(good_sums[right] ** 2) + T.sum(bad_sums[left] ** 2) + T.sum(bad_sums[right] ** 2))
        updates[lang_pair] = deeplearning.learning_rule.learning_rule(costs[lang_pair], projections[left].params + projections[right].params, eps=1e-6, rho=0.65, method='adadelta')
        cost_funcs[lang_pair] = theano.function([good_tensors[left], good_tensors[right], bad_tensors[left], bad_tensors[right]], [costs[lang_pair]], on_unused_input='warn', updates=updates[lang_pair])
    for epoch in range(args.epochs):
        print 'epoch: {}'.format(epoch)
        from random import shuffle, choice

        positions = dict()
        for langs, pairs in loaded_training.iteritems():
            shuffle(pairs)
            positions[langs] = 0;
        total_loss = 0
        all_count = 0
        recent_loss = np.zeros((100,))
        while True:
            chosen_pair = choice(list(positions.keys()))
            idx = positions[chosen_pair]
            pairs = loaded_training[chosen_pair]
            p = pairs[idx]
            from random import randrange
            chosen = randrange(len(pairs))
            while chosen == idx:
                chosen = randrange(len(pairs))
            chosen_rand = pairs[chosen]
            np_senl = get_np_sentence(p[0])
            np_senr = get_np_sentence(p[1])
            np_badl = get_np_sentence(chosen_rand[0])
            np_badr = get_np_sentence(chosen_rand[1])
            c = cost_funcs[chosen_pair](np_senl, np_senr, np_badl, np_badr)
            recent_loss[all_count%100] = c[0]
            total_loss += c[0]
            if all_count % 100 == 0:
                print 'idx:{} recent 100 = {}'.format(all_count,recent_loss.sum())
            if all_count % 10000 == 0:
                store([vocab,params], args.store)
            all_count += 1
            positions[chosen_pair] += 1
            if len(pairs) == positions[chosen_pair]:
                positions.pop(chosen_pair, None)

        print 'total loss: {}'.format(total_loss)
        store([vocab, params], args.store)
    
    
    return

if __name__ == '__main__':
    main()
