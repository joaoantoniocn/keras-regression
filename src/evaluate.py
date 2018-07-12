"""Evaluate results.

Usage:
    evaluate_nodeps.py <pred.csv> <truth.csv>
"""
import csv
import numpy as np

from docopt import docopt


def main(args):
    pred_fn = args['<pred.csv>']
    truth_fn = args['<truth.csv>']

    pred = csv.DictReader(open(pred_fn))
    truth = csv.DictReader(open(truth_fn))

    if pred.fieldnames != truth.fieldnames:
        raise ValueError("Files have mismatched headers")
    if set(pred.fieldnames) != {'id', 'slope', 'intercept'}:
        raise ValueError("Files have invalid headers")

    slope_ses = []
    intercept_aes = []

    for p, t in zip(pred, truth):
        if p['id'] != t['id']:
            raise ValueError("Files are out-of-order, please sort by ascending id")

        slope_ses.append((float(p['slope']) - float(t['slope'])) ** 2)
        intercept_aes.append(abs(float(p['intercept']) - float(t['intercept'])))

    print("Slope mse: %s" % np.mean(slope_ses))
    print("Intercept mae: %s" % np.mean(intercept_aes))


if __name__ == "__main__":
    main(docopt(__doc__))
