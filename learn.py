import argparse
from itertools import islice  # slicing for iterators
from io import open
from random import choice, shuffle

from toy import load_sentences
import deeplearning.layer
from deeplearning.learning_rule import *
import deeplearning.utils

def prepare_shared_embeddings(vocab, model,dim=100):
    from numpy.random import rand
    assert isinstance(vocab, dict)
    to_return = np.zeros((len(vocab) + 1, dim), dtype=np.float32)
    for k, v in vocab.iteritems():
        if k in model:
            # to_return[v] = model[k.lower()]
            to_return[v] = model[k]
        else:
            # raise Exception('key {} not found'.format(k.encode('utf-8')))
            to_return[v] = rand(1,dim) - 0.5
    return to_return

def load_embeddings(fname):
    import gensim

    to_return = gensim.models.Word2Vec.load(fname)
    return to_return


def load_txt_embeddings(fname):
    import numpy as np
    to_return = dict()
    with open(fname, mode='r', encoding='utf-8') as fh:
        for l in fh:
            ss = l.strip().split(u' ', 1)
            if len(ss) != 2:
                raise Exception('ss: {} len: {} sth wrong {}'.format(ss, len(ss), l.strip()))
            to_return[ss[0]] = np.fromstring(ss[1], sep=' ', dtype='float32')
    return to_return

def mono_sentences(fname):
    with open(fname, encoding='utf-8', mode='r') as f:
        for l in f:
            yield l


def main():
    parser = argparse.ArgumentParser(description='Train LSTM for parallel corpora.')
    # PEP8: use ' and not " for strings
    parser.add_argument('-l', '--left', default='left_file',
                        help='left corpus')
    parser.add_argument('-r', '--right', default='right_file',
                        help='right corpus')
    parser.add_argument('-n', '--num_sentences', default=2000, type=int, )
    parser.add_argument('--test-file-left', default=None, type=str)
    parser.add_argument('--test-file-right', default=None, type=str)
    parser.add_argument('--save-every', default=1, type=int)
    parser.add_argument('--load-model', default=None, type=str)
    parser.add_argument('--predict', default=None, type=str)
    parser.add_argument('--embeddings-left', default='data/w2v_model_en', type=str)
    parser.add_argument('--embeddings-right', default='data/w2v_model_de', type=str)
    # note that if x == [2, 3, 3], then x[:None] == x[:] == x (copy); no need for sys.maxint
    opts = parser.parse_args()

    # we create a generator and avoid loading all sentences into a list
    def sentences(fname, infinite=False):
        with open(fname, encoding='utf-8', mode='r') as f:
            for pair in f:
                # yield [sentence.strip().split() for sentence in pair.split(u' ||| ')]
                yield pair.split(u' ||| ')

    vocab_l, vocab_r = dict(), dict()

    test_sentences = []
    if opts.test_file_left is not None:
        print 'loading test files {} {}'.format(opts.test_file_left, opts.test_file_right)
        for (sen_l, sen_r) in zip(mono_sentences(opts.test_file_left), mono_sentences(opts.test_file_right)):
            vocab_l, loaded_l = load_sentences([sen_l], vocab_l)
            vocab_r, loaded_r = load_sentences([sen_r], vocab_r)
            test_sentences.append((loaded_l, loaded_r))

    pairs = []
    idx = 0
    for (sen_l, sen_r) in islice(zip(mono_sentences(opts.left), mono_sentences(opts.right)), opts.num_sentences):
        print 'idx: {}'.format(idx)
        idx += 1
        vocab_l, loaded_l = load_sentences([sen_l], vocab_l)
        vocab_r, loaded_r = load_sentences([sen_r], vocab_r)
        s1 = loaded_l
        s2 = loaded_r
        pairs.append((np.asarray([s1], dtype=np.int32), np.asarray([s2], dtype=np.int32)))

    def save_model(model, fname):
        import cPickle as pickle

        f = file(fname, 'wb')
        print 'saving to file {}'.format(fname)
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

    def load_model(fname):
        import cPickle as pickle

        f = file(fname, 'rb')
        to_return = pickle.load(f)
        f.close()
        return to_return

    def prepare_nn(fname=None, predict_fname=None, shared_embeddings_left=None, shared_embeddings_right=None):
        print 'preparing nn...'
        if shared_embeddings_left is not None and shared_embeddings_right is not None:
            print 'left embeddings: {} {}'.format(len(shared_embeddings_left), len(shared_embeddings_left[0]))
            print 'right embeddings: {} {}'.format(len(shared_embeddings_right), len(shared_embeddings_right[0]))

        n_words_l = len(vocab_l)
        n_words_r = len(vocab_r)
        e_dim = 100
        lstm_dim = 50
        final_dim = 100
        left_trans = T.imatrix('left_trans')
        right_trans = T.imatrix('right_trans')
        left_non_trans = T.imatrix('left_non_trans')
        right_non_trans = T.imatrix('right_non_trans')

        layers_left = [
            deeplearning.layer.tProjection(n_words_l,
                                           e_dim) if shared_embeddings_left is None else deeplearning.layer.tProjection(
                n_words_l, e_dim, embedding=shared_embeddings_left),
            deeplearning.layer.LSTM(e_dim, lstm_dim, minibatch=True),
            deeplearning.layer.LSTM(lstm_dim, lstm_dim, minibatch=True),
            deeplearning.layer.LSTM(lstm_dim, final_dim, minibatch=True),
        ]

        layers_right = [
            deeplearning.layer.tProjection(n_words_r,
                                           e_dim) if shared_embeddings_right is None else deeplearning.layer.tProjection(
                n_words_r, e_dim, embedding=shared_embeddings_right),
            deeplearning.layer.LSTM(e_dim, lstm_dim, minibatch=True),
            deeplearning.layer.LSTM(lstm_dim, lstm_dim, minibatch=True),
            deeplearning.layer.LSTM(lstm_dim, final_dim, minibatch=True),
        ]

        if fname is not None:
            print 'loading model...'
            (old_layers_left, old_layers_right) = load_model(fname)
            layers_left[0].copy_constructor(orig=old_layers_left[0])
            # TODO hacky
            layers_left[0].set_embeddings(shared_embeddings_left)
            layers_left[1].copy_constructor(orig=old_layers_left[1])
            layers_left[2].copy_constructor(orig=old_layers_left[2])
            layers_left[3].copy_constructor(orig=old_layers_left[3])

            layers_right[0].copy_constructor(orig=old_layers_right[0])
            layers_right[0].set_embeddings(shared_embeddings_right)
            layers_right[1].copy_constructor(orig=old_layers_right[1])
            layers_right[2].copy_constructor(orig=old_layers_right[2])
            layers_right[3].copy_constructor(orig=old_layers_right[3])

        layers_left_bad = [
            deeplearning.layer.tProjection(orig=layers_left[0]),
            deeplearning.layer.LSTM(orig=layers_left[1]),
            deeplearning.layer.LSTM(orig=layers_left[2]),
            deeplearning.layer.LSTM(orig=layers_left[3]),
        ]

        layers_right_bad = [
            deeplearning.layer.tProjection(orig=layers_right[0]),
            deeplearning.layer.LSTM(orig=layers_right[1]),
            deeplearning.layer.LSTM(orig=layers_right[2]),
            deeplearning.layer.LSTM(orig=layers_right[3]),
        ]

        params = []

        for layer in layers_left:
            params += layer.params

        for layer in layers_right:
            params += layer.params

        for idx, (layer_left, layer_left_bad, layer_right, layer_right_bad) in enumerate(
                zip(layers_left, layers_left_bad, layers_right, layers_right_bad)):
            if idx == 0:
                layer_l_out = layer_left.fprop(left_trans)
                layer_l_out_bad = layer_left_bad.fprop(left_non_trans)
                layer_r_out = layer_right.fprop(right_trans)
                layer_r_out_bad = layer_right_bad.fprop(right_non_trans)
            else:
                layer_l_out = layer_left.fprop(layer_l_out)
                layer_l_out_bad = layer_left_bad.fprop(layer_l_out_bad)
                layer_r_out = layer_right.fprop(layer_r_out)
                layer_r_out_bad = layer_right_bad.fprop(layer_r_out_bad)

        y_l = layers_left[-1].h[-1]
        y_l_bad = layers_left_bad[-1].h[-1]
        y_r = layers_right[-1].h[-1]
        y_r_bad = layers_right_bad[-1].h[-1]

        def cos_dist(a, b):
            flat_a = a.flatten()
            flat_b = b.flatten()
            div = flat_a.norm(2) * flat_b.norm(2)
            return - flat_a.dot(flat_b) / div 

        cost_good = cos_dist(y_l, y_r)
        cost_l_bad = cos_dist(y_l_bad, y_l)

        cost_r_bad = cos_dist(y_r_bad, y_r)

        cost_trans_equiv = theano.tensor.max([0, 1 + cost_good])

        cost_l_sanity = theano.tensor.max([0, 1 - cost_l_bad])
        cost_r_sanity = theano.tensor.max([0, 1 - cost_r_bad])

        cost = cost_trans_equiv + cost_l_sanity + cost_r_sanity

        # L2
        for p in params:
            cost += 1e-4 * (p ** 2).sum()

        updates = learning_rule(cost, params, eps=1e-6, rho=0.65, method='adadelta')

        # cost_func = theano.function([left_trans, right_trans], [cost_trans_equiv], givens=[(left_non_trans, left_trans), (right_non_trans, right_trans)], on_unused_input='ignore')
        train = theano.function([left_trans, left_non_trans, right_trans, right_non_trans], [cost, y_l, y_r],
                                updates=updates)
        predictor_l = theano.function([left_trans,], y_l)
        predictor_r = theano.function([right_trans,], y_r)
        for r in xrange(2000):
            print 'round: {}'.format(r)
            shuffle(pairs)
            for idx, pair in enumerate(pairs):
                print 'idx: {}'.format(idx)
                random_idx = choice(range(len(pairs)))
                while random_idx == idx:
                    random_idx = choice(range(len(pairs)))
                random_pair = pairs[random_idx]
                # print 'debug print:'
                # theano.printing.debugprint(pair[0])
                this_cost, this_y_l, this_y_r = train(pair[0][0], random_pair[0][0], pair[1][0], random_pair[1][0])
                print 'this cost: {}'.format(this_cost)
                if idx % 50 == 0:
                    print 'this cost: {}'.format(this_cost)

            save_model((layers_left, layers_right), 'layers_round')
            if predict_fname is not None and r % 10 == 0:
                print 'predicting...'
                to_output = []
                for idx, (sent_l, sent_r) in enumerate(test_sentences):
                    print 'predicting #{}'.format(idx)
                    emb_left = predictor_l([sent_l[0],])
                    emb_right = predictor_r([sent_r[0],])
                    to_output.append((emb_left, emb_right))
                save_model(to_output, predict_fname)

    embeddings_l = load_embeddings(opts.embeddings_left)
    embeddings_r = load_embeddings(opts.embeddings_right)
    shared_l = prepare_shared_embeddings(vocab_l, model=embeddings_l)
    shared_r = prepare_shared_embeddings(vocab_r, model=embeddings_r)
    prepare_nn(fname=opts.load_model, predict_fname=opts.predict, shared_embeddings_left=shared_l,
               shared_embeddings_right=shared_r)


# convention to allow import of this file as a module
if __name__ == '__main__':
    main()
