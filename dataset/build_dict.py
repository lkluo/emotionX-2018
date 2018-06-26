import sys
import json
import os

from dataset.data_utils import get_word_dict, get_vector, build_vocab

_author_ = 'Linkai Luo'


file_path = os.path.dirname(os.path.realpath(__file__))

GLOVE_PATH = os.path.join(file_path, 'GloVe/glove.840B.300d.txt')
DATA_PATH = os.path.join(file_path, 'data')

def get_all_data(data_path=DATA_PATH, data_name='data.en'):
    all_data_path = os.path.join(data_path, data_name)
    # if not os.path.exists(all_data_path):
    for data_type in ['Friends', 'EmotionPush']:
        path1 = os.path.join(data_path, data_type)
        os.system('cat %s %s > %s' % (os.path.join(path1,  data_type.lower() + '_train.en'), os.path.join(path1,  data_type.lower() + '_dev.en'),
                                      os.path.join(path1, data_type.lower() + '.en')) )
    os.system('cat %s %s > %s' % (
    os.path.join(data_path, 'Friends/friends.en'), os.path.join(data_path, 'EmotionPush/emotionpush.en'),
    os.path.join(data_path, data_name)))
    print('merge all data to {}'.format(data_name))

    with open(os.path.join(data_path, data_name), 'r') as f:
        data = f.read().split('\n')[:-1]
    print('get {} lines of utterance'.format(len(data)))
    return data



if __name__ == '__main__':
    with open(os.path.join(DATA_PATH, 'data.en'), 'r') as f:
        sentences = f.read().split('\n')[:-1]

    word_dict = get_word_dict(sentences)
    print('total vocabulary {}'.format(len(word_dict)))

    word_vec = build_vocab(sentences, GLOVE_PATH)
    # print('total vocabulary found in GloVe {}'.format(len(word_vec)))


    unknown_dict = [key for key in word_dict if key not in word_vec.keys()]
    print("number of unknown words {}".format(len(unknown_dict)))
    print(unknown_dict)