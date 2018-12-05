# -*- coding: utf-8 -*-
#/usr/bin/python3
# TRANSFORMER + RL (nsml version)

from __future__ import print_function
import os, codecs
import logging
import random
from tqdm import tqdm

#import nsml
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import corpus_bleu

from modules import *
from graph import Graph
from hyperparams import Hyperparams as hp
from data_load import load_doc_vocab, load_sum_vocab, load_data
from rouge_tensor import rouge_l_sentence_level

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def visualize():
    g = Graph(is_training=False)
    print("visualizes Graph loaded")
    

    # Load data
    # X, Sources, Targets = load_data(type=type)
    source = '本 文 总 结 了 十 个 可 穿 戴 产 品 的 设 计 原 则 ， 而 这 些 原 则 ， 同 样 也 是 笔 者 认 为 是 这 个 行 业 最 吸 引 人 的 地 方 ： 1 . 为 人 们 解 决 重 复 性 问 题 ； 2 . 从 人 开 始 ， 而 不 是 从 机 器 开 始 ； 3 . 要 引 起 注 意 ， 但 不 要 刻 意 ； 4 . 提 升 用 户 能 力 ， 而 不 是 取 代 人'.split(' ')
    
    de2idx, idx2de = load_doc_vocab()
    
    source_idx = [de2idx.get(word, 1) for word in (source + [u"</S>"])]
    if len(source_idx) < hp.article_maxlen:
        source_idx = np.lib.pad(source_idx, [0, hp.article_maxlen - len(source_idx)], 'constant', constant_values=(0, 0))

    with g.graph.as_default():
        sv = tf.train.Supervisor()
        with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            sv.saver.restore(sess, tf.train.latest_checkpoint(hp.pretrain_logdir))
            print("Restored!")
            
            x = [source_idx]
            sources = [source]
            
            ### Autoregressive inference
            preds = np.zeros((1, hp.summary_maxlen), np.int32) # only one output is generated
            # atten = sess.run(g.atten_node.enc_selfatten, {g.x: x, g.y: preds})
            
            for j in range(hp.summary_maxlen):
                print("the {} step: ".format(j))
                if j == hp.summary_maxlen - 1:
                    outp_lists = [g.atten_node.enc_selfatten, g.atten_node.dec_selfatten,
                                  g.atten_node.vanilla_atten, g.preds]
                    enc_atten, dec_atten, vanilla_atten, _preds = sess.run(outp_lists,
                                                                        {g.x: x, g.y: preds})
                else:
                    _preds = sess.run(g.preds, {g.x: x, g.y: preds})
                preds[:, j] = _preds[:, j]
            
            got = " ".join(idx2de[idx] for idx in preds[0]).split("</S>")[0].strip()
            got = got.split()

        # self_atten_visualize(source, enc_atten, 'enc_atten')
        self_atten_visualize(got, dec_atten, 'dec_atten')
        # vanilla_atten_visualize(source, got, vanilla_atten)
        print("vanilla_atten: ", vanilla_atten.shape)

        # rouge = rouge_l_sentence_level(hypotheses, list_of_refs)
        # (eval_sentences, ref_sentences):


def self_atten_visualize(source, atten, title):
    print(atten.shape)
    margin = 0.002

    import matplotlib.font_manager as mfm
    # font_path = '/Users/user/Downloads/cwtex-q-fonts-TTFs-0.4/ttf/cwTeXQFangsongZH-Medium.ttf'
    font_path = '/Users/user/Library/Fonts/msyh.ttf'
    prop = mfm.FontProperties(fname=font_path)

    # cur_len = min(len(source), hp.article_maxlen)
    if title == 'enc_atten':
        cur_len = min(len(source), hp.article_maxlen)
    elif title == 'dec_atten':
        cur_len = min(len(source), hp.summary_maxlen)

    
    left_i = random.randint(0, cur_len-1) # randomly choose an word in input to visualize

    plt.figure(figsize=(10, 5))

    colors = ['r', 'g', 'b', 'k']
    for i, c in enumerate(colors):
        # word
        x_s = margin
        y_s = 1.0 - i * 0.23
        step = (1.0 - margin * 2) / cur_len
        left_word_coord = []
        right_word_coord = []
        
        for word in source:
            left_word_coord.append((x_s, y_s))
            right_word_coord.append((x_s, y_s - 0.1))
            
            plt.text(x_s, y_s, word, fontproperties=prop)
            plt.text(x_s, y_s - 0.1, word, fontproperties=prop)
            x_s += step

        
        weights = atten[i, left_i, :cur_len]
        left_coord = left_word_coord[left_i]

        for right_i in range(cur_len):
            right_coord = right_word_coord[right_i]
            plt.plot((left_coord[0] + 0.005, right_coord[0] + 0.005),
                     (left_coord[1], right_coord[1] + 0.03),
                     c, alpha=weights[right_i])
    
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    plt.suptitle(title)
    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.8])

    plt.savefig(title + '.png')
    print("finished", title)



def vanilla_atten_visualize(source, pred, atten):
    print(atten.shape)
    margin = 0.002
    
    import matplotlib.font_manager as mfm
    # font_path = '/Users/user/Downloads/cwtex-q-fonts-TTFs-0.4/ttf/cwTeXQFangsongZH-Medium.ttf'
    font_path = '/Users/user/Library/Fonts/msyh.ttf'
    prop = mfm.FontProperties(fname=font_path)
    
    src_cur_len = min(len(source), hp.article_maxlen)
    prd_cur_len = min(len(pred), hp.summary_maxlen)

    left_i = random.randint(0, prd_cur_len-1) # randomly choose an word in input to visualize
    
    plt.figure(figsize=(20, 5))

    colors = ['r', 'g', 'b', 'k']
    for i, c in enumerate(colors):

        x_s = margin
        y_s = 1.0 - i * 0.23
        
        x_l = x_s
        x_r = x_s
        
        src_step = (1.0 - margin * 2) / src_cur_len
        prd_step = (1.0 - margin * 2) / prd_cur_len
        
        left_word_coord = []
        right_word_coord = []
        
        for word_i in range(src_cur_len):
            left_word_coord.append((x_l, y_s))
            right_word_coord.append((x_r, y_s - 0.1))
            
            if word_i < prd_cur_len:
                plt.text(x_l, y_s, pred[word_i], fontproperties=prop)
            plt.text(x_r, y_s - 0.1, source[word_i], fontproperties=prop)
            
            x_l += prd_step # left: query (which is the decoder input)
            x_r += src_step # right: key (which is the encoder input)
        
        weights = atten[i, left_i, :src_cur_len]
        left_coord = left_word_coord[left_i]
        
        for right_i in range(src_cur_len):
            right_coord = right_word_coord[right_i]
            plt.plot((left_coord[0] + 0.005, right_coord[0] + 0.005),
                     (left_coord[1], right_coord[1] + 0.03),
                     c, alpha=weights[right_i])

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    plt.suptitle('vanilla_atten')
    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.8])

    plt.savefig('vanilla_atten.png')
    print("finished vanila attention")

if __name__ == '__main__':
    logging.info("START")
    logging.info("tensorflow version: ", tf.__version__)
    
    from tensorflow.python.client import device_lib as dl
    logging.info("local device: ", dl.list_local_devices())

    visualize()
    # train()
    # eval()
    
    '''
    import matplotlib.font_manager as mfm
    # font_path = '/Users/user/Downloads/cwtex-q-fonts-TTFs-0.4/ttf/cwTeXQFangsongZH-Medium.ttf'
    font_path = '/Users/user/Library/Fonts/msyh.ttf'
    prop = mfm.FontProperties(fname=font_path)

    plt.text(0.3, 0.3, "这里", fontproperties=prop)
    plt.plot((0.3, 0.3), (0.5, 0.3))
    plt.plot((0.3 + 0.015, 0.5), (0.3 + 0.015, 0.3))

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.show()
    '''
