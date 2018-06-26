import os
import time
import numpy as np
import argparse
from copy import deepcopy
import json

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim

from data import build_vocab, get_batch, get_sentences, get_lengths
from model import BLSTMAttnNet, BLSTMNet
import dataset.Constant as Constant

_author_ = 'Linkai Luo'

parser = argparse.ArgumentParser()
# paths
parser.add_argument("--output_dir", type=str, default='blstm-attn-net-checkpoint', help="save checkpoint")
parser.add_argument("--model_name", type=str, default='model.pickle')

# training
parser.add_argument("--n_epochs", type=int, default=30)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--embed_size", type=int, default=300)
parser.add_argument("--lstm_dropout", type=float, default=0.1, help="lstm encoder dropout")
parser.add_argument("--attn_dropout", type=float, default=0.1, help="attention dropout")
parser.add_argument("--optimizer", type=str, default="adam", help="adam or sgd")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--lrshrink", type=float, default=5, help="shrink factor for sgd")
parser.add_argument("--decay", type=float, default=0.99, help="lr decay")
parser.add_argument("--minlr", type=float, default=1e-5, help="minimum lr")
parser.add_argument("--max_norm", type=float, default=5., help="max norm (grad clipping)")
parser.add_argument("--seed", type=int, default=12345, help="seed")
parser.add_argument("--save_at_epoch", type=int, default=0, help="only save this epoch")
parser.add_argument("--weight", action="store_true", default=False, help="weight cross entropy")

# model
parser.add_argument("--lstm_dim", type=int, default=256, help="encoder nhid dimension")
parser.add_argument("--n_enc_layers", type=int, default=1, help="encoder num layers")
parser.add_argument("--fc_dim", type=int, default=128, help="nhid of fc layers")
parser.add_argument("--num_classes", type=int, default=8, help="4 classes for the time being")
parser.add_argument("--max_sent_len", type=int, default=24, help="maximum sentence lengths")

parser.add_argument("--display_freq", type=int, default=100, help="how many steps to display message")
parser.add_argument("--use_cuda", action="store_true", default=False, help="use gpu or not")
parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")

params = parser.parse_args()
print(params)

if not os.path.exists(params.output_dir):
    os.makedirs(params.output_dir)

"""
Write model parameters to json files
"""
print('writing parameter to file..')
params_dict = vars(params)
with open(os.path.join(params.output_dir, 'params.json'), 'w') as f:
    json.dump(params_dict, f, ensure_ascii=False, indent=2)

"""
SEED
"""
np.random.seed(params.seed)
torch.manual_seed(params.seed)
if params.use_cuda:
    torch.cuda.manual_seed(params.seed)

"""
PATH
"""
assert params.embed_size in [50, 300]
glove_name = 'glove.840B.300d.txt' if params.embed_size == 300 else 'glove.twitter.27B.50d.txt'
file_path = os.path.dirname(os.path.realpath(__file__))
GLOVE_PATH = os.path.join(file_path, 'dataset/GloVe/' + glove_name)
DATA_PATH = os.path.join(file_path, 'dataset/data/data4')

"""
DATA
"""
train = get_sentences(DATA_PATH, 'train', num_classes=params.num_classes)
dev_fr = get_sentences(DATA_PATH, 'friends_dev', num_classes=params.num_classes)
dev_ep = get_sentences(DATA_PATH, 'emotionpush_dev', num_classes=params.num_classes)

# get lengths for each dialogue
train_fr_lens = get_lengths(os.path.join(file_path, 'dataset/data/Friends'), 'friends_train')
train_ep_lens = get_lengths(os.path.join(file_path, 'dataset/data/EmotionPush'), 'emotionpush_train')
train_lens = np.append(train_fr_lens, train_ep_lens)

dev_fr_lens = get_lengths(os.path.join(file_path, 'dataset/data/Friends'), 'friends_dev')
dev_ep_lens = get_lengths(os.path.join(file_path, 'dataset/data/EmotionPush'), 'emotionpush_dev')

all_data = [line.rstrip() for line in open(os.path.join(DATA_PATH, 'data-all.en'), 'r')]
word_vec = build_vocab(all_data, GLOVE_PATH)
# add <s> and </s> to each of the sentences
for data_type in ['train', 'dev_fr', 'dev_ep']:
        eval(data_type)['sent'] = np.array([['<s>'] + [word for word in sent.split() if word in word_vec] + ['</s>'] for sent in eval(data_type)['sent']])


"""
MODEL
"""
model = BLSTMAttnNet(embed_size=params.embed_size, lstm_dim=params.lstm_dim, fc_dim=params.fc_dim,
                         num_classes=params.num_classes, max_sent_len=params.max_sent_len,
                         attn_dropout=params.attn_dropout, lstm_dropout=params.lstm_dropout)
if params.use_cuda:
    # set gpu device
    torch.cuda.set_device(params.gpu_id)


"""
LOSS
"""
weight = torch.FloatTensor(params.num_classes).fill_(1)
if params.num_classes == 8:
    # only focus on 4 emotions
    weight[4:] = 0.0
    if params.weight:
        weight[0] = 0.9
        weight[1] = 0.95
        weight[2] = 1.0
        weight[2] = 1.0


loss_fn = nn.CrossEntropyLoss(weight=weight)
loss_fn.size_average = False

# do not train positional encoding
ignored_params = list(map(id, model.position_enc.parameters()))
trainable_params = filter(lambda p: id(p) not in ignored_params, model.parameters())

# optimizer (adam optimizer)
optim_fn = optim.Adam if params.optimizer.lower() == 'adam' else optim.SGD
# optimizer = optim_fn(model.parameters(), lr=params.lr)
optimizer = optim_fn(trainable_params, lr=params.lr)

# cuda
if params.use_cuda:
    model.cuda()
    loss_fn.cuda()

# train one epoch
def train_epoch(epoch):
    print('\n Traing: Epoch ', str(epoch))
    model.train()
    all_costs, logs = [], []
    words_count = 0

    last_time = time.time()
    correct = 0

    # shuffle the data
    permutation = np.random.permutation(len(train_lens))
    #
    perm_len_list = []
    group_list = []
    stidx = 0

    for l in train_lens:
        c_group = np.arange(stidx, stidx + l, 1)
        group_list.append(c_group)
        stidx += l
    for i in permutation.tolist():
        perm_len_list.extend(group_list[i])

    train_lens_ = train_lens[permutation]
    sent = train['sent'][perm_len_list]
    target = train['label'][perm_len_list]

    optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * params.decay if epoch > 1 and 'sgd' in params.optimizer.lower() else \
    optimizer.param_groups[0]['lr']

    stidx = 0
    for batch_size in train_lens_:
        sent_batch, len_batch = get_batch(sent[stidx:stidx + batch_size], word_vec, embed_size=params.embed_size)

        sent_batch = Variable(sent_batch.cuda()) if params.use_cuda else Variable(sent_batch.cpu())
        label_batch = Variable(torch.LongTensor(target[stidx:stidx + batch_size])).cuda() if params.use_cuda else \
            Variable(torch.LongTensor(target[stidx:stidx + batch_size])).cpu()

        stidx += batch_size
        k = sent_batch.size(0) # actual batch size

        output = model((sent_batch, len_batch))
        pred = output.data.max(1)[1]
        # pred = output.data.transpose(0, 1)

        correct += pred.long().eq(label_batch.data.long()).cuda().sum() if params.use_cuda else pred.long().eq(label_batch.data.long()).cpu().sum()
        # assert len(pred) == len(sent[stidx:stidx + batch_size])

        #loss
        loss = loss_fn(output, label_batch)
        all_costs.append(loss.data[0])
        words_count += sent_batch.nelement() / params.embed_size

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient clipping
        shrink_factor = 1
        total_norm = 0

        for p in model.parameters():
            if p.requires_grad:
                p.grad.data.div_(k)
                total_norm += p.grad.data.norm() ** 2
        total_norm = np.sqrt(total_norm)

        if total_norm > params.max_norm:
            shrink_factor = params.max_norm / total_norm
        current_lr = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = current_lr * shrink_factor

        # optimizer step
        optimizer.step()
        optimizer.param_groups[0]['lr'] = current_lr

        if len(all_costs) == params.display_freq:
            logs.append('{0} ; loss {1} ; sentence/s {2} ; words/s {3} ; accuracy train : {4}'.format(
                stidx, round(np.mean(all_costs), 2),
                int(len(all_costs) * params.batch_size / (time.time() - last_time)),
                int(words_count * 1.0 / (time.time() - last_time)),
                round(100. * correct / (stidx + k), 2)))
            print(logs[-1])
            last_time = time.time()
            words_count = 0
            all_costs = []
    train_acc = round(100 * correct/len(sent), 2)
    print('result: epoch {}; mean train accuracy: {}'.format(epoch, train_acc))

    # save checkpoint every epoch
    if params.save_at_epoch == 0:
        torch.save(model, os.path.join(params.output_dir, "model_{0}.pickle".format(epoch)))
    else:
        if params.save_at_epoch == epoch:
            torch.save(model, os.path.join(params.output_dir, "{}_{}.pickle".format(params.model_name, epoch)))
    return train_acc

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

def evaluate(epoch, eval_type='valid', correct_count=correct_count, labels_count=labels_count):
    model.eval()
    if eval_type == 'valid':
        print('\nVALIDATION : Epoch {0}'.format(epoch))

    sent1 = dev_fr['sent'] if eval_type == 'valid' else None
    sent2 = dev_ep['sent'] if eval_type == 'valid' else None
    target1 = dev_fr['label'] if eval_type == 'valid' else None
    target2 = dev_ep['label'] if eval_type == 'valid' else None

    eval_acc = []

    for sent, target, diag_lens in zip([sent1, sent2], [target1, target2], [dev_fr_lens, dev_ep_lens]):
        correct = 0
        stidx = 0
        for batch_size in diag_lens:
            sent_batch, len_batch = get_batch(sent[stidx:stidx + batch_size], word_vec, embed_size=params.embed_size)

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
        acc = np.round(100 * correct / total, 2)

        eval_wa = round(100 * sum(correct[:4]) / sum(total[:4]), 1)
        eval_uwa = round(sum(acc[:4]) / 4, 1)
        eval_acc.append([eval_wa, eval_uwa])
        print("accuracy for each category\n{}".format(acc))
        print("wa: {}".format(eval_wa))
        print("uwa: {}".format(eval_uwa))
    return eval_acc
#
if __name__ == '__main__':
    # train_acc = train_epoch(1)
    eval_acc_all = []
    #
    for i in range(params.n_epochs):
        train_acc = train_epoch(i)
        eval_acc = evaluate(i)
        eval_acc_all.append(eval_acc)
    print(eval_acc_all)
