import os
import time
import numpy as np
import argparse
import pickle
import json
from copy import deepcopy

import torch
from torch.autograd import Variable

from data import build_vocab, get_batch, get_sentences, get_lengths
import dataset.Constant as Constant

_author_ = 'Linkai Luo'

parser = argparse.ArgumentParser()
# paths
parser.add_argument("--model_dir", type=str, default='model-train-big', help="model directory")
parser.add_argument("--output_dir", type=str, default='acc', help="output directory")
parser.add_argument("--model_name", type=str, default='model_19.pickle')
parser.add_argument("--use_cuda", action="store_true", default=False)
parser.add_argument("--batch_size", type=int, default=16)

params = parser.parse_args()

if not os.path.exists(params.output_dir):
    os.mkdir(params.output_dir)

file_path = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(file_path, 'dataset/data/data4')
GLOVE_PATH = os.path.join(file_path, 'dataset/GloVe/glove.840B.300d.txt')
# word vector
all_data = [line.rstrip() for line in open(os.path.join(DATA_PATH, 'data-all.en'), 'r')]
word_vec = build_vocab(all_data, GLOVE_PATH)
embed_size = 300

# load model
def load_model(model_dir, model_name, use_cuda=False):
    with open(os.path.join(model_dir, "params.json"), 'r') as f:
        model_config = json.load(f)
    model_dir = os.path.join(model_dir, model_name)
    if use_cuda:
        model = torch.load(model_dir)
    else:
        model = torch.load(model_dir, map_location=lambda storage, loc: storage)
    return model, model_config


def load_data(data_dir=DATA_PATH, model_config=None, action="dev"):
    fr_filename = 'friends_{}'.format(action)
    ep_filename = 'emotionpush_{}'.format(action)
    dev_fr = get_sentences(data_dir, fr_filename, num_classes=model_config["num_classes"])
    dev_ep = get_sentences(data_dir, ep_filename, num_classes=model_config["num_classes"])

    # get lengths for each dialogue
    dev_fr_lens = get_lengths(os.path.join(file_path, 'dataset/data/Friends'), fr_filename)
    dev_ep_lens = get_lengths(os.path.join(file_path, 'dataset/data/EmotionPush'), ep_filename)

    # add <s> and </s> to each of the sentences
    for data_type in ['dev_fr', 'dev_ep']:
        eval(data_type)['sent'] = np.array(
            [['<s>'] + [word for word in sent.split() if word in word_vec] + ['</s>'] for sent in
             eval(data_type)['sent']])

    return (dev_fr, dev_fr_lens), (dev_ep, dev_ep_lens)

"""
SET UP EVALUATION
"""
labels = Constant.EMOTION8
reverse_labels = {}

for key, val in zip(labels.keys(), labels.values()):
    reverse_labels[val] = key

labels_count = deepcopy(labels)
for key in labels.keys():
    labels_count[key] = 0

correct_count = deepcopy(labels_count)

def compute_acc(pred, label, correct_count, labels_count):
    for p, l in zip(pred, label):
        labels_count[reverse_labels[l]] += 1
        if p == l:
            correct_count[reverse_labels[l]] += 1
    return correct_count, labels_count

def evaluate(model, data_fr_tuple, data_ep_tuple, eval_type="valid"):
    model.eval()
    print('\nVALIDATION')

    # for counting
    labels = Constant.EMOTION8
    reverse_labels = {}
    for key, val in zip(labels.keys(), labels.values()):
        reverse_labels[val] = key
    labels_count = deepcopy(labels)
    for key in labels.keys():
        labels_count[key] = 0
    correct_count = deepcopy(labels_count)

    # data
    dev_fr, dev_fr_lens = data_fr_tuple
    dev_ep, dev_ep_lens = data_ep_tuple
    sent1 = dev_fr['sent'] if eval_type == 'valid' else None
    sent2 = dev_ep['sent'] if eval_type == 'valid' else None
    target1 = dev_fr['label'] if eval_type == 'valid' else None
    target2 = dev_ep['label'] if eval_type == 'valid' else None

    wa, uwa, acc_all = [], [], []

    for sent, target, diag_lens in zip([sent1, sent2], [target1, target2], [dev_fr_lens, dev_ep_lens]):
        stidx = 0
        for batch_size in diag_lens:
            sent_batch, len_batch = get_batch(sent[stidx:stidx + batch_size], word_vec, embed_size=embed_size)

            sent_batch = Variable(sent_batch.cuda()) if params.use_cuda else Variable(sent_batch.cpu())
            label_batch = Variable(torch.LongTensor(target[stidx:stidx + batch_size])).cuda() if params.use_cuda else \
                Variable(torch.LongTensor(target[stidx:stidx + batch_size])).cpu()

            stidx += batch_size

            output = model((sent_batch, len_batch))

            pred = output.data.max(1)[1]

            # counting
            correct_count, labels_count = compute_acc(pred=pred.long(), label=label_batch.data.long(),
                                                      correct_count=correct_count, labels_count=labels_count)

            # correct += pred.long().eq(label_batch.data.long()).cuda().sum() if params.use_cuda else pred.long().eq(
            #     label_batch.data.long()).cpu().sum()

        correct = list(correct_count.values())
        total = list(labels_count.values())
        correct = np.array(correct)
        total = np.array(total)
        acc = np.round(100 * correct / total, 1).tolist()

        eval_wa = round(100 * sum(correct[:4]) / sum(total[:4]), 1)
        eval_uwa = round(sum(acc[:4]) / 4, 1)

        wa.append(eval_wa)
        uwa.append(eval_uwa)
        acc_all.append(acc[:4])
        print("accuracy for each category\n{}".format(acc))
        print("wa: {}".format(eval_wa))
        print("uwa: {}".format(eval_uwa))

    result = {'wa': wa, 'uwa': uwa, 'acc': acc_all}
    return result

def predict(model, data_fr_tuple, data_ep_tuple, eval_type="valid"):
    model.eval()
    print('\nPREDICTING')
    # for counting
    labels = Constant.EMOTION8
    reverse_labels = {}
    for key, val in zip(labels.keys(), labels.values()):
        reverse_labels[val] = key
    labels_count = deepcopy(labels)
    for key in labels.keys():
        labels_count[key] = 0
    correct_count = deepcopy(labels_count)

    # data
    dev_fr, dev_fr_lens = data_fr_tuple
    dev_ep, dev_ep_lens = data_ep_tuple
    sent1 = dev_fr['sent'] if eval_type == 'valid' else None
    sent2 = dev_ep['sent'] if eval_type == 'valid' else None
    target1 = dev_fr['label'] if eval_type == 'valid' else None
    target2 = dev_ep['label'] if eval_type == 'valid' else None

    pred_lists = {}

    for sent, target, diag_lens, names in zip([sent1, sent2], [target1, target2], [dev_fr_lens, dev_ep_lens], ["friends", "emotionpush"]):
        stidx = 0
        pred_list = np.array([])
        for batch_size in diag_lens:
            sent_batch, len_batch = get_batch(sent[stidx:stidx + batch_size], word_vec, embed_size=embed_size)

            sent_batch = Variable(sent_batch.cuda()) if params.use_cuda else Variable(sent_batch.cpu())
            label_batch = Variable(torch.LongTensor(target[stidx:stidx + batch_size])).cuda() if params.use_cuda else \
                Variable(torch.LongTensor(target[stidx:stidx + batch_size])).cpu()
            stidx += batch_size

            output = model((sent_batch, len_batch))
            pred = output.data.max(1)[1]

            # number of sentences in this dialogue
            this_pred_list = np.array([reverse_labels[p] for p in pred.tolist()])
            pred_list = np.append(pred_list, this_pred_list)

        pred_lists[names] = pred_list
    return pred_lists

# load model
model, model_config = load_model(params.model_dir, params.model_name, params.use_cuda)
# load dev data
data_fr_tuple, data_ep_tuple = load_data(DATA_PATH, model_config)
# load dev data
test_fr_tuple, test_ep_tuple = load_data(DATA_PATH, model_config, action="test")


action = "test"

if action == "valid":
    result = evaluate(model, data_fr_tuple, data_ep_tuple)
elif action == "dev":

    pred_lists = predict(model, data_fr_tuple, data_ep_tuple)

    def compare_preds(pred_lists, output_dir, model_dir, true_label_path=None):
        if true_label_path is None:
            true_label_path = "dataset/data/data8/dev.label"

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        all_pred_list = np.append(pred_lists["friends"], pred_lists["emotionpush"])
        true_labels = np.array([p.rstrip("\n") for p in open(true_label_path)])

        with open(os.path.join(output_dir, "{}_compare.label".format(model_dir)), "w") as f:
            f.write("{}\t{}\n".format("true", "predict"))
            for pred, true in zip(all_pred_list, true_labels):
                f.write("{}\t{}\n".format(true, pred))

    print("making compare files")
    compare_preds(pred_lists, "results", params.model_dir)

    def put_back_labels(pred_lists, output_dir, filepath=None):
        if filepath is None:
            filepath = {}
            filepath["friends"] = "dataset/data/Friends/friends_dev.json"
            filepath["emotionpush"] = "dataset/data/EmotionPush/emotionpush_dev.json"

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        for data_type in ["friends", "emotionpush"]:
            idx = 0
            with open(filepath[data_type]) as f:
                data_json = json.load(f)
            for d in range(len(data_json)): # dialogue
                for u in range(len(data_json[d])): # urreance
                    data_json[d][u]["emotion"] = pred_lists[data_type][idx]
                    idx += 1
            name_data_json = [{"name": "Linkai Luo", "email": "llk1896@gmail.com"}, data_json]
            with open(os.path.join(output_dir, "{}_{}_pred.json".format(params.model_dir, data_type)), "w") as f:
                json.dump(name_data_json, f, ensure_ascii=False, indent=4)


    print("creating predicted files")
    put_back_labels(pred_lists, "results")

else:
    pred_lists = predict(model, test_fr_tuple, test_ep_tuple)

    def compare_preds(pred_lists, output_dir, model_dir, true_label_path=None):
        if true_label_path is None:
            true_label_path = "dataset/data/data4/test.label"

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        all_pred_list = np.append(pred_lists["friends"], pred_lists["emotionpush"])
        true_labels = np.array([p.rstrip("\n") for p in open(true_label_path)])

        with open(os.path.join(output_dir, "{}_compare-test.label".format(model_dir)), "w") as f:
            f.write("{}\t{}\n".format("true", "predict"))
            for pred, true in zip(all_pred_list, true_labels):
                f.write("{}\t{}\n".format(true, pred))

    print("making compare files")
    compare_preds(pred_lists, "results", params.model_dir)

    def put_back_labels(pred_lists, output_dir, filepath=None):
        if filepath is None:
            filepath = {}
            filepath["friends"] = "dataset/data/Friends/friends_test.json"
            filepath["emotionpush"] = "dataset/data/EmotionPush/emotionpush_test.json"

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        for data_type in ["friends", "emotionpush"]:
            idx = 0
            with open(filepath[data_type]) as f:
                data_json = json.load(f)
            for d in range(len(data_json)): # dialogue
                for u in range(len(data_json[d])): # urreance
                    data_json[d][u]["emotion"] = pred_lists[data_type][idx]
                    idx += 1
            name_data_json = [{"name": "Linkai Luo", "email": "llk1896@gmail.com"}, data_json]
            with open(os.path.join(output_dir, "{}_{}_test.json".format(params.model_dir, data_type)), "w") as f:
                json.dump(name_data_json, f, ensure_ascii=True, indent=4)


    print("creating predicted files")
    put_back_labels(pred_lists, "results")
