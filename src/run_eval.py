import sys
import argparse
from collections import defaultdict
from core.evaluation.eval import QGEvalCap


def evaluate_predictions(target_src, decoded_text):
    assert len(target_src) == len(decoded_text)
    eval_targets = {}
    eval_predictions = {}
    for idx in range(len(target_src)):
        eval_targets[idx] = [target_src[idx]]
        eval_predictions[idx] = [decoded_text[idx]]

    QGEval = QGEvalCap(eval_targets, eval_predictions)
    scores = QGEval.evaluate(include_meteor=True)
    return scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-gold', '--gold', required=True, type=str, help='path to the gold input file')
    parser.add_argument('-pred', '--pred', required=True, type=str, help='path to the pred input file')
    args = vars(parser.parse_args())

    pred = []
    with open(args['pred'], 'r') as f1:
        for line in f1:
            pred.append(line.strip())

    gold = []
    with open(args['gold'], 'r') as f2:
        for line in f2:
            gold.append(line.strip())

    metrics = evaluate_predictions(gold, pred)
    print(metrics)
