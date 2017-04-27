#!/bin/python

import numpy as np
import sys

WEATHER_INDEXES = [5, 6, 10, 11]

def predictions_smarter(logits, pos_thresh):
    result = logits > pos_thresh
    # can only have one weather label, so take the highest prediction
    result[:, WEATHER_INDEXES] = False
    weather_logits = logits[:, WEATHER_INDEXES]
    weather_preds = np.zeros(weather_logits.shape, dtype=bool)
    weather_preds[
            np.arange(weather_preds.shape[0]),
            np.argmax(weather_logits, axis=1)] = True
    result[:, WEATHER_INDEXES] = weather_preds
    return result

def predictions_naive(logits, pos_thresh):
    return logits > pos_thresh

def load(fname):
    return np.load(fname)

def stats(preds, labels):
    tp = np.sum(np.logical_and(preds, labels), axis=1)
    fp = np.sum(np.logical_and(preds, np.logical_not(labels)), axis=1)
    fn = np.sum(np.logical_and(np.logical_not(preds), labels), axis=1)
    return tp, fp, fn

def precision(tp, fp):
    return tp.astype('float') / (tp + fp)

def recall(tp, fn):
    return tp.astype('float') / (tp + fn)

def F2(p, r):
    return 5 * p * r / (4 * p + r)

def f2_data(data, threshold):
    preds = predictions_naive(data['logits'], threshold)
    tp, fp, fn = stats(preds, data['labels'])
    prec = precision(tp, fp)
    rec = recall(tp, fn)
    f2 = F2(prec, rec)
    return np.nanmean(f2), np.nanmean(prec), np.nanmean(rec)

def main(argv):
    data = load(argv[1])
    for i in xrange(-50, 10, 1):
        thresh = float(i) / 10
        f2, prec, rec = f2_data(data, thresh)
        print('%.4f %.4f %.4f %.4f' % (thresh, f2, prec, rec))

if __name__ == '__main__':
    main(sys.argv)
