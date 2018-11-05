# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''

from nsml import DATASET_PATH
import os

class Hyperparams: # giga summary
    ## log path; frequency
    logdir = 'logdir' # log directory
    tb_dir = 'tbdir'
    checkpoint_steps = 1000
    eval_record_threshold = 5000
    eval_record_steps = 1000  # should be larger than checkpoint_steps? otherwise would duplicate
    train_record_steps = 200
    
    
    ## data source
    source_train = os.path.join(DATASET_PATH, 'train', 'train', 'train.article.txt')
    target_train = os.path.join(DATASET_PATH, 'train', 'train', 'train.title.txt')
    source_valid = os.path.join(DATASET_PATH, 'train', 'train', 'valid.article.filter.txt')
    target_valid = os.path.join(DATASET_PATH, 'train', 'train', 'valid.title.filter.txt')
    
    source_test = os.path.join(DATASET_PATH, 'train', 'test', 'test.giga.txt')
    sum_dict = os.path.join(DATASET_PATH, 'train', 'train', 'full_dict.txt')
    doc_dict = sum_dict

    ## data parameter
    min_cnt = 20  # words whose occurred less than min_cnt are encoded as <UNK>.
    article_minlen = 20
    article_maxlen = 40  # Maximum number of words in a sentence. alias = T.
    summary_minlen = 5
    summary_maxlen = 12  # Maximum number of words in a sentence. alias = T.
    
    
    ## training parameter
    sinusoid = False # If True, use sinusoid. If false, positional embedding.
    
    hidden_units = 512  # alias = C # orig: 512
    num_blocks = 3  # number of encoder/decoder blocks
    num_heads = 8
    
    batch_size = 80  # orig：32
    num_epochs = 10
    
    lr = 0.0001
    dropout_rate = 0.1


class NAVER_Hyperparams: # for NAVER data
    ## log path; frequency
    logdir = 'logdir' # log directory
    tb_dir = 'tbdir'
    checkpoint_steps = 1000
    eval_record_threshold = 1000
    eval_record_steps = 1000  # should be larger than checkpoint_steps? otherwise would duplicate
    train_record_steps = 50
    
    ## data source
    source_train = os.path.join(DATASET_PATH, 'train', 'train_content.txt')
    target_train = os.path.join(DATASET_PATH, 'train', 'train_title.txt')
    
    source_valid = os.path.join(DATASET_PATH, 'train', 'test_content.txt') # change
    target_valid = os.path.join(DATASET_PATH, 'train', 'test_title.txt') # change
    
    source_test = os.path.join(DATASET_PATH, 'train', 'test_content.txt')
    
    sum_dict = os.path.join(DATASET_PATH, 'train', 'full_dict.txt')  # full
    # sum_dict = os.path.join(DATASET_PATH, 'train', 'content_dict.txt') # tiny
    doc_dict = sum_dict
    
    ## data parameter
    ''' char: '''
    min_cnt = 20  # words whose occurred less than min_cnt are encoded as <UNK>.
    article_minlen = 100
    article_maxlen = 800  # Maximum number of words in a sentence. alias = T.
    summary_minlen = 5
    summary_maxlen = 25  # Maximum number of words in a sentence. alias = T.
    
    ''' word:
    min_cnt = 20  # words whose occurred less than min_cnt are encoded as <UNK>.
    article_minlen = 50
    article_maxlen = 700  # Maximum number of words in a sentence. alias = T.
    summary_minlen = 4
    summary_maxlen = 15  # Maximum number of words in a sentence. alias = T.
    '''
    
    ## training parameter
    sinusoid = False # If True, use sinusoid. If false, positional embedding.
    
    hidden_units = 512  # alias = C # orig: 512
    num_blocks = 3  # number of encoder/decoder blocks
    num_heads = 8
    
    batch_size = 32  # orig：32
    num_epochs = 10
    
    lr = 0.0001
    dropout_rate = 0.1

# sess 10: num_blocks = 3
# sess 13: num_blocks = 6

class CNN_Hyperparams: # for CNN data
    ## log path; frequency
    logdir = 'logdir' # log directory
    tb_dir = 'tbdir'
    checkpoint_steps = 1000
    eval_record_threshold = 1000
    eval_record_steps = 1000  # should be larger than checkpoint_steps? otherwise would duplicate
    train_record_steps = 50
    
    ## data source
    source_train = os.path.join(DATASET_PATH, 'train', 'train_content.txt')
    target_train = os.path.join(DATASET_PATH, 'train', 'train_summary.txt')
    
    source_valid = os.path.join(DATASET_PATH, 'train', 'val_content.txt') # change
    target_valid = os.path.join(DATASET_PATH, 'train', 'val_summary.txt') # change
    
    source_test = os.path.join(DATASET_PATH, 'train', 'test_content.txt')
    
    sum_dict = os.path.join(DATASET_PATH, 'train', 'dict.txt')
    doc_dict = sum_dict
    
    
    ## data parameter
    min_cnt = 20  # words whose occurred less than min_cnt are encoded as <UNK>.
    article_minlen = 100
    article_maxlen = 700  # Maximum number of words in a sentence. alias = T.
    summary_minlen = 20
    summary_maxlen = 50  # Maximum number of words in a sentence. alias = T.
    
    ## training parameter
    sinusoid = False # If True, use sinusoid. If false, positional embedding.
    
    hidden_units = 512  # alias = C # orig: 512
    num_blocks = 6  # number of encoder/decoder blocks
    num_heads = 8
    
    batch_size = 32  # orig：32
    num_epochs = 10
    
    lr = 0.0001
    dropout_rate = 0.1

# sess 10: num_blocks = 3
# sess 13: num_blocks = 6


