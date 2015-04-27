#!/usr/bin/python

import learn
from toy import load_sentences
import numpy as np
from theano import tensor as T
import theano
import deeplearning.layer
import deeplearning.learning_rule

def get_embedding_chunk(sentence, embeddings_dict, dim=100):
    to_return = np.empty(shape=(len(sentence),dim), dtype='float32')
    for idx, x in enumerate(sentence):
        to_return[idx:] = embeddings_dict[x]
    return to_return
        
def get_np_sentence(sentence):
    return np.array(sentence, dtype='int32').reshape((-1,))

def get_sum(sentence, proj):
    proj.fprop(sentence)
    summed = T.sum(proj.h, axis=0)
    return summed

def cos_dist(a, b):
    eps = 1e-10
    return 0.5 - (T.dot(a, b) / ((a.norm(2)+eps) * (b.norm(2)+eps)))/2

def rbf_dist(a, b):
    return T.exp(- 1e-7 * ((a-b) ** 2).sum())

def logloss(t_cost, f_cost):
    eps = 1e-10
    return - (T.log(t_cost + eps) + T.log(1- f_cost + eps))

def logloss_rbf(l, r, bad_r):
    euc_good = ((l-r)**2).sum()
    euc_bad = ((l-bad_r)**2).sum()
    t = euc_good / (euc_good + euc_bad)
    return  t
    
def logloss_rbf_r(l, r, bad_l):
    eps = 1e-5
    euc_good = ((l-r)**2).sum()
    euc_bad = ((r-bad_l)**2).sum()
    t = T.log(euc_good+eps) - T.log( (euc_good + euc_bad) + eps)
    return t
    
def logloss_rbf_l(l, r, bad_r):
    eps = 1e-5
    euc_good = ((l-r)**2).sum()
    euc_bad = ((l-bad_r)**2).sum()
    t = T.log(euc_good+eps) - T.log( (euc_good + euc_bad) + eps)
    return t
    
def hingeloss_rbf_r(l, r, bad_l):
    euc_good = ((l-r)**2).sum()
    euc_bad = ((r-bad_l)**2).sum()
    return  T.max([0, 50 + euc_good - euc_bad])
    
def hingeloss_rbf_l(l, r, bad_r):
    euc_good = ((l-r)**2).sum()
    euc_bad = ((l-bad_r)**2).sum()
    return  T.max([0, 50 + euc_good - euc_bad])
    
def store(params, fname):
    import cPickle as pickle
    with open(fname, mode='wb') as fh:
        pickle.dump(params, fh, pickle.HIGHEST_PROTOCOL)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('left')
    parser.add_argument('right')
    parser.add_argument('embleft')
    parser.add_argument('embright')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--store', default='cvm_params_corrected_rbf_lower')
    parser.add_argument('--txtembeddings', action='store_true')
    args = parser.parse_args()
    sens_l = learn.mono_sentences(args.left)
    sens_r = learn.mono_sentences(args.right)
    vocab_l, vocab_r = dict(), dict()
    pairs = []
    for (sen_l, sen_r) in zip(sens_l, sens_r):
        vocab_l, loaded_l = load_sentences([sen_l], vocab_l)
        vocab_r, loaded_r = load_sentences([sen_r], vocab_r)
        pairs.append((loaded_l, loaded_r))
    if args.txtembeddings:
        embeddings_l = learn.load_txt_embeddings(args.embleft)
        embeddings_r = learn.load_txt_embeddings(args.embright)
        dim = 100
    else:
        embeddings_l = learn.load_embeddings(args.embleft)
        embeddings_r = learn.load_embeddings(args.embright)
        dim = 100
    shared_l = learn.prepare_shared_embeddings(vocab_l, model=embeddings_l, dim=dim)
    shared_r = learn.prepare_shared_embeddings(vocab_r, model=embeddings_r, dim=dim)

    senl_tensor = T.ivector('sen_l')
    senr_tensor = T.ivector('sen_r')
    badl_tensor = T.ivector('bad_l')
    badr_tensor = T.ivector('bad_r')
    proj_l = deeplearning.layer.Projection(embedding=shared_l,update=True, hid_dim=dim)
    proj_r = deeplearning.layer.Projection(embedding=shared_r,update=True, hid_dim=dim)
    summed_l = get_sum(senl_tensor,proj_l)
    summed_r = get_sum(senr_tensor,proj_r)
    summed_bad_l = get_sum(badl_tensor,proj_l)
    summed_bad_r = get_sum(badr_tensor,proj_r)
    # dist_equiv = rbf_dist(summed_l, summed_r)
    # dist_left_sanity = rbf_dist(summed_l, summed_bad_r)
    # dist_right_sanity = rbf_dist(summed_r, summed_bad_l)
    # cost = logloss(dist_equiv, dist_left_sanity) + logloss(dist_equiv, dist_right_sanity)
    cost = logloss_rbf(summed_l, summed_r, summed_bad_r) + logloss_rbf(summed_l, summed_r, summed_bad_l)
    params = proj_l.params + proj_r.params
    updates = deeplearning.learning_rule.learning_rule(cost, params, eps=1e-6, rho=0.65, method='adadelta')
    cost_func = theano.function([senl_tensor, senr_tensor, badl_tensor, badr_tensor], [cost], on_unused_input='warn', updates=updates)
    for epoch in range(args.epochs):
        print 'epoch: {}'.format(epoch)
        from random import shuffle
        shuffle(pairs)
        total_loss = 0
        for idx, p in enumerate(pairs):
            from random import randrange
            chosen = randrange(len(pairs))
            # chosen = idx + 1
            while chosen == idx:
                chosen = randrange(len(pairs))
            chosen = pairs[chosen]
            np_senl = get_np_sentence(p[0])
            np_senr = get_np_sentence(p[1])
            np_badl = get_np_sentence(chosen[0])
            np_badr = get_np_sentence(chosen[1])
            c = cost_func(np_senl, np_senr, np_badl, np_badr)
            total_loss += c[0]
            if idx % 100 == 0:
                print 'idx:{} {}'.format(idx,c)
            if idx % 10000 == 0:
                store(params, args.store)

        print 'total loss: {}'.format(total_loss)
        store(params, args.store)
        # for pair in pairs[:100]:
        #     print get_embedding_chunk(pair[0][0], shared_l)
    
    
    return

if __name__ == '__main__':
    main()
