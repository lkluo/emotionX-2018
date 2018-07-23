# emotionX-2018

This is source code for our paper [EmotionX-DLC: Self-Attentive BiLSTM for Detecting Sequential Emotions in Dialogues](http://www.aclweb.org/anthology/W18-3506)

**Note**: We accidently used dropout duirng testing. Setting dropout rate 0 could improve performance that is reported in our original paper.

## Dataset
Please go to the share task webpage: http://doraemon.iis.sinica.edu.tw/emotionlines/challenge.html to download or request data
needed for training and testing. You need to create a directory **EmotionPush** to store EmotionPush data (which you need to make
a request to the organizer). The saved files should be renamed as emotionpush_{train, dev, test}. For testset either of **Friends**
or **EmotionPush**, you need to make a formal request to the organizer.

## Glove
Please download pre-trained Glove data with 300d size (**glove.840B.300d.txt**) or 50d size (**glove.twitter.27B.50d**.

## Model training
**python3 train.py** or **./train.sh**

## Model testing
**python3 test.py**
