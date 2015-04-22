#!/usr/bin/python

import cPickle as pickle

def score(e, f):
    from scipy.spatial.distance import cosine
    return cosine(e,f)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('evec')
    parser.add_argument('fvec')
    args = parser.parse_args()
    with open(args.evec, 'rb') as e_fh, open(args.fvec, 'rb') as f_fh:
        list_e = pickle.load(e_fh)
        list_f = pickle.load(f_fh)
        assert isinstance(list_e, list) and isinstance(list_f, list) and len(list_e) == len(list_f)
        for v_e, v_f in zip(list_e, list_f):
            print score(v_e, v_f)

    return

if __name__ == '__main__':
    main()
