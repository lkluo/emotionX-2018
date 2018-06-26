import torch
import os
import numpy as np
import json

def get_batch(batch, word_vec, embed_size=300):
    # sent in batch in decreasing order of lengths (bsize, max_len, word_dim)
    lengths = np.array([len(x) for x in batch])
    max_len = np.max(lengths)
    embed = np.zeros((max_len, len(batch), embed_size))

    for i in range(len(batch)):
        for j in range(len(batch[i])):
            try:
                embed[j, i, :] = word_vec[batch[i][j]]
            except:
                embed[j, i, :] = word_vec['<p>']

    return torch.from_numpy(embed).float(), lengths


def get_word_dict(sentences):
    word_dict = {}
    for s in sentences:
        for w in s.split():
            if w not in word_dict:
                word_dict[w] = ''
    word_dict['<s>'] = ''
    word_dict['</s>'] = ''
    word_dict['<p>'] = ''
    return word_dict


def get_vector(word_dict, glove_path):
    word_vec = {}
    with open(glove_path) as f:
        for line in f:
            word, vec = line.split(' ', 1)
            if word in word_dict:
                word_vec[word] = np.array(list(map(float, vec.split())))
    print('Found {0}/{1} words with glove vectors'.format(len(word_vec), len(word_dict)))
    return word_vec

def build_vocab(sentences, glove_path):
    word_dict = get_word_dict(sentences)
    word_vec = get_vector(word_dict, glove_path)
    print('Vocab size: {0}'.format(len(word_vec)))
    return word_vec



def get_sentences(data_dir, file_name, num_classes=8):
    assert num_classes in [4, 5, 8]

    if num_classes == 4:
        labels = {'neutral': 0, 'joy': 1, 'sadness': 2, 'anger': 3, 'fear': 0, 'surprise': 0, 'disgust': 0,
                  'non-neutral': 0}
    elif num_classes == 5:
        labels = {'neutral': 0, 'joy': 1, 'sadness': 2, 'anger': 3, 'fear': 4, 'surprise': 4, 'disgust': 4,
                    'non-neutral': 4}
    else:
        labels = {'neutral': 0, 'joy': 1, 'sadness': 2, 'anger': 3, 'fear': 4, 'surprise': 5, 'disgust': 6, 'non-neutral': 7}

    sent_path = os.path.join(data_dir, '{}.en'.format(file_name))
    target_path = os.path.join(data_dir, '{}.label'.format(file_name))

    sent = [line.rstrip() for line in open(sent_path, 'r')]
    target = np.array([labels[line.rstrip('\n')] for line in open(target_path, 'r')])
    data = {'sent': sent, 'label': target}

    return data

def get_lengths(data_dir, file_name):
    data_dir = os.path.join(data_dir, "{}.json".format(file_name))
    with open(data_dir) as f:
        diag = json.load(f)
    lengths = np.array([len(x) for x in diag])
    return lengths

if __name__ == '__main__':
    file_path = os.path.dirname(os.path.realpath(__file__))
    DATA_PATH = os.path.join(file_path, 'data')

    MAX_SEQ_LEN = 50
    MAX_SENT_NUM = 26
    N_CLASSES = 4
    train_path = os.path.join(DATA_PATH, 'train.json')
    dev_path = os.path.join(DATA_PATH, 'dev.json')

    train_fr_lengths = get_lengths("dataset/data/Friends/", "friends_train")
    train_ep_lengths = get_lengths("dataset/data/EmotionPush/", "emotionpush_train")
    train_lens = np.append(train_fr_lengths, train_ep_lengths)

    dev_fr_lengths = get_lengths("dataset/data/Friends/", "friends_dev")
    dev_ep_lengths = get_lengths("dataset/data/EmotionPush/", "emotionpush_dev")
    dev_lens = np.append(dev_fr_lengths, dev_ep_lengths)

    test_fr_lengths = get_lengths("dataset/data/Friends/", "friends_test")
    test_ep_lengths = get_lengths("dataset/data/EmotionPush/", "emotionpush_test")
    test_lens = np.append(test_fr_lengths, test_ep_lengths)

    train_lens = np.append(train_lens, dev_lens)

    print(len(train_lens))

    # train_lens.extend(train_fr_lengths)
    # train_lens.extend(train_ep_lengths)
    #
    # print(train_lens)
    # print(type(train_lens))
    # print(train_fr_lengths)
    #
    #
    # print(len(train_lens))

    # permutation = np.random.permutation(len(train_lens))
    # train_lens_ = train_lens[permutation]
    # print(train_lens_)

    # print("max train len: {}".format(max(train_lens)))
    # print("max dev len: {}".format(max(dev_lens)))
    # print("max test len: {}".format(max(test_lens)))

