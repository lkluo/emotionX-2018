import json
import os
from dataset.preproccess import tokenize


data_types = ['EmotionPush', 'Friends']

file_path = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(file_path, 'data')

def write_to_file(data_path, data_type, action, output_dir):

    assert action in ['train', 'dev', 'test']
    # if action == 'test':
    #     raise NotImplementedError()
    data_type = data_type.lower()

    # create output directory
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # load original data
    file_name = "{}_{}.json".format(data_type, action)
    with open(os.path.join(data_path, file_name), 'r') as f:
        source = json.load(f)

    utterance_lengths = []
    sentence_lengths = []

    all_utterance, all_emotion = [], []
    for n, diag in enumerate(source):
        utterance_lengths.append(len(diag))
        this_utterance, this_emotion = [], []
        for item in diag:
            # utterance = clean_utterance(item['utterance'])
            utterance = tokenize(item['utterance'])
            emotion = item['emotion']
            this_utterance.append(utterance)
            this_emotion.append(emotion)
            sentence_lengths.append(len(utterance))

        all_utterance.extend(this_utterance)
        all_emotion.extend(this_emotion)

    # save utterance as .en
    with open(os.path.join(output_dir, "{}_{}.en".format(data_type, action)), 'w') as f:
        for line in all_utterance:
            f.write(line + '\n')
    # save emotion as .label
    with open(os.path.join(output_dir, "{}_{}.label".format(data_type, action)), 'w') as f:
        for line in all_emotion:
            f.write(line + '\n')

    print('Write {}.{} successfully ({} dialogues.)'.format(data_type, action, len(utterance_lengths)))

def merge_files(output_dir):
    for action in ['train', 'dev', 'test']:
        for kind in ["en", "label"]:
            file1 = os.path.join(output_dir, "{}_{}.{}".format("friends", action, kind))
            file2 = os.path.join(output_dir, "{}_{}.{}".format("emotionpush", action, kind))
            file_final = os.path.join(output_dir, "{}.{}".format(action, kind))
            # file_temp = os.path.join(output_dir, "{}.{}".format(action, "tmp"))
            os.system('cat %s %s > %s' % (file1, file2, file_final))
        print("concat {} data".format(action))

    print("combine train and dev data")
    os.system('cat %s %s > %s' % (os.path.join(output_dir, "train.en"), os.path.join(output_dir, "dev.en"),
                                  os.path.join(output_dir, "data.en")))

def merge_files_with_tests(output_dir):
    # merge train and dev data to form a big train data
    os.system('cat %s %s > %s' % (os.path.join(output_dir,"train.en"), os.path.join(output_dir,"dev.en"), os.path.join(output_dir,"train-dev.en")))
    os.system('cat %s %s > %s' % (os.path.join(output_dir,"train.label"), os.path.join(output_dir,"dev.label"), os.path.join(output_dir,"train-dev.label")))
    print("concat train and dev data")
    # merge train/dev/test sentences to create dictionary
    print("combine train/dev/test sentence")
    os.system('cat %s %s %s > %s' % (os.path.join(output_dir, "train.en"), os.path.join(output_dir, "dev.en"), os.path.join(output_dir, "test.en"),
                                  os.path.join(output_dir, "data-all.en")))

if __name__ == '__main__':

    friend_data_path = os.path.join(DATA_PATH, 'Friends')
    emotionpush_data_path = os.path.join(DATA_PATH, 'EmotionPush')
    # output_dir = os.path.join(DATA_PATH, "data8")
    output_dir = os.path.join(DATA_PATH, "data4")

    # friends dialogue
    write_to_file(friend_data_path, 'Friends', 'train', output_dir)
    write_to_file(friend_data_path, 'Friends', 'dev', output_dir)
    write_to_file(friend_data_path, 'Friends', 'test', output_dir)
    # emotionpush dialogue
    write_to_file(emotionpush_data_path, 'EmotionPush', 'train', output_dir)
    write_to_file(emotionpush_data_path, 'EmotionPush', 'dev', output_dir)
    write_to_file(emotionpush_data_path, 'EmotionPush', 'test', output_dir)

    # get line-by-line data for dictionary
    merge_files(output_dir)
    merge_files_with_tests(output_dir)