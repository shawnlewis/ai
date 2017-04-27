#!/bin/python

import numpy as np
import sys

import os

def load(fname):
    return np.load(fname)

def predictions(logits, pos_thresh):
    return logits > pos_thresh

def read_labels(dataset_dir):
    f = open(os.path.join(dataset_dir, 'tensorflow', 'labels.txt'))
    result = {}
    for line in f:
        k, v = line.strip().split(':')
        result[int(k)] = v
    return result

def read_imagenames(dataset_dir):
    f = open(os.path.join(dataset_dir, 'tensorflow', 'image_names-test.txt'))
    return [line.strip() for line in f]

def main(argv):
    dataset_dir = argv[1]
    eval_fname = argv[2]
    threshold = float(argv[3])
    print 'Using threshold: %s' % threshold

    labels = read_labels(dataset_dir)
    print labels

    #image_names = read_imagenames(dataset_dir)

    data = load(argv[2])
    logits = data['logits']
    #assert len(logits) > len(image_names), 'wrong lens: %s %s' % (len(logits), len(image_names))
    #logits = logits[:len(image_names)]
    preds = predictions(data['logits'], threshold)

    result = open('submission.csv', 'w')
    result.write('image_name,tags\n')
    for image_name, pred in zip(data['names'], preds):
        tags = []
        for i, p in enumerate(pred):
            if p:
                tags.append(labels[i])
        result.write('%s,%s\n' % (image_name, ' '.join(tags)))
    result.close()

if __name__ == '__main__':
    main(sys.argv)
