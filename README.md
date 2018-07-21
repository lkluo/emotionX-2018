# emotionX-2018

This is source code for our paper [EmotionX-DLC: Self-Attentive BiLSTM for Detecting Sequential Emotions in Dialogues](http://www.aclweb.org/anthology/W18-3506)

**Note** We accidently used dropout duirng testing. Setting dropout rate 0 could improve performance as reported in our original paper.

## Dataset
Please go to the share task webpage: http://doraemon.iis.sinica.edu.tw/emotionlines/challenge.html to download or request data
needed for training and testing. You need to create a directory **EmotionPush** to store EmotionPush data (which you need to make
a request to the orginizer). The saved files should be renamed as emotionpush_{train, dev, test}. For either testset for **Friends**
and **EmotionPush**, you need to make a formal request to the orginizer.

## Glove
Please download pre-trained Glove data with 300 dimension size **glove.840B.300d.txt**.

## Model training
**python3 train.py** or **./train.sh**

## Model testing
python3 test.py
