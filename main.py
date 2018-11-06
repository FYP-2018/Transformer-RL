# -*- coding: utf-8 -*-
#/usr/bin/python3

from __future__ import print_function
import os, codecs
import logging
from tqdm import tqdm

import nsml
import numpy as np
import tensorflow as tf
from nltk.translate.bleu_score import corpus_bleu

from modules import *
from graph import Graph
from hyperparams import Hyperparams as hp
from data_load import load_doc_vocab, load_sum_vocab, load_data

def train():
    try:
        if not os.path.exists(hp.logdir):
            tf.logging.info('making logdir')
            os.mkdir(hp.logdir)
    except:
        tf.logging.info('making logdir failed')
        pass
    
    # Load vocabulary
    de2idx, idx2de = load_doc_vocab()
    en2idx, idx2en = load_sum_vocab()
    
    print("Constructing graph...")
    train_g = Graph("train")

    print("Start training...")
    with train_g.graph.as_default():
        sv = tf.train.Supervisor(logdir=hp.logdir)
        # with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
        with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            
            print("Training with {} epoches".format(hp.num_epochs))
            for epoch in range(1, hp.num_epochs+1):
                print("Starting {}-th epoch".format(epoch))
                
                if sv.should_stop():
                    break
                    
                # for step in tqdm(range(train_g.num_batch), total=train_g.num_batch, ncols=70, leave=False, unit='b'):
                for step in range(train_g.num_batch):
                    # sess.run(train_g.train_op)
                    true_step = step + (epoch - 1) * train_g.num_batch
                    
                    if step % hp.train_record_steps == 0:
                        loss, acc, _, summary = sess.run([train_g.loss, train_g.acc, train_g.train_op, train_g.merged])
                        print("at step {}: loss = {}, rough-1 = {}".format(step, loss, acc))
                        
                        # print("REWARD DIFF: ", rd.shape)
                        # print('ml loss: {}, rl loss: {}'.format(mlls, rlls)) # normal & nan
                        # print('reward_diff:', rd): TODO: sometimes zeros
                        # print('sample_preds: ', sl) # ok (reasonable outcome) & hv correct length (80 * 12)
                        # print('length of train_g.current_logtis_testing: ', len(train_g.current_logtis_testing))
                        nsml.report(step=true_step, train_loss=float(loss), train_accuracy=float(acc))
                        
                        train_g.filewriter.add_summary(summary, true_step)

                    else:
                        sess.run(train_g.train_op)
                    
                    if true_step % hp.checkpoint_steps == 0:
                        # gs = sess.run(train_g.global_step)
                        # true_step = step + epoch * train_g.num_batch
                        sv.saver.save(sess, hp.logdir + '/model_epoch_%02d_step_%d' % (epoch, true_step))

                    if true_step > hp.eval_record_threshold and step % hp.eval_record_steps == 0:
                        blue_score = eval()
                        # eval(type='eval_tmp') # record result on training set
                        # true_step = step + epoch * train_g.num_batch
                        # nsml.report(step=true_step, blue_score=float(blue_score))

                gs = sess.run(train_g.global_step)
                sv.saver.save(sess, hp.logdir + '/model_epoch_%02d_gs_%d' % (epoch, gs))
                
                # test(num_epoch=epoch)  # test once per epoch
                
    print("Done")

def test(num_epoch=10):
    # Load graph
    g = Graph(is_training=False)
    print("Test Graph loaded")
    
    # Load data
    X, Sources = load_data(type='test')
    word2idx, idx2word = load_sum_vocab()
    
    # Start session         
    with g.graph.as_default():
        sv = tf.train.Supervisor(logdir=hp.logdir)
        # with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
        with sv.managed_session(
            start_standard_services=False,
            config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        ) as sess:
            
            ## Restore parameters
            sv.saver.restore(sess, tf.train.latest_checkpoint(hp.logdir))
            print("Restored!")
            
            ## Get model name
            mname = open(hp.logdir + '/checkpoint', 'r').read().split('"')[1] # model name
            
            ## Inference
            if not os.path.exists('results'): 
                os.mkdir('results')
            
            with codecs.open("results/test-" + mname, "w", "utf-8") as fout:
                list_of_refs, hypotheses = [], []
                for i in range(len(X) // hp.batch_size):
                    ### Get mini-batches
                    x = X[i*hp.batch_size: (i+1)*hp.batch_size]
                    sources = Sources[i*hp.batch_size: (i+1)*hp.batch_size]
                    
                    ### Autoregressive inference
                    # preds = np.zeros((hp.batch_size, hp.summary_maxlen), np.int32)
                    preds = np.zeros((x.shape[0], hp.summary_maxlen), np.int32)
                    for j in range(hp.summary_maxlen):
                        # _preds = sess.run(g.preds, {g.x: x, g.y: preds})
                        _preds, _acc = sess.run([g.preds, g.acc], {g.x: x, g.y: preds})
                        preds[:, j] = _preds[:, j]
                    
                    print("at step {}: rough-1 = {}".format(num_epoch, _acc))
                    # nsml.report(step=num_epoch, rough1=float(_acc))
                    
                    for source, pred in zip(sources, preds): # sentence-wise
                        got = " ".join(idx2word[idx] for idx in pred).split("</S>")[0].strip()
                        sentence_to_write = "- source: " + source + "\n- got: " + got + "\n\n"
                        
                        print(sentence_to_write)
                        fout.write(sentence_to_write)
                        fout.flush()


def eval(type='eval'):
    # Load graph
    g = Graph(is_training=False)
    print("Eval Graph loaded")
    
    # Load data
    # X, Sources, Targets = load_data(type='eval')
    X, Sources, Targets = load_data(type=type)
    
    de2idx, idx2de = load_doc_vocab()
    word2idx, idx2word = load_sum_vocab()
    
    # X, Sources, Targets = X[:65], Sources[:65], Targets[:65]
    
    # Start session
    with g.graph.as_default():
        sv = tf.train.Supervisor()
        with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            sv.saver.restore(sess, tf.train.latest_checkpoint(hp.logdir))
            print("Restored!")
            
            mname = open(hp.logdir + '/checkpoint', 'r').read().split('"')[1] # model name
            
            ## Inference
            if not os.path.exists('results'): os.mkdir('results')
            
            # for these 2 file, one version gonna be enough 
            fout2 = codecs.open("results/eval-pred", "w", "utf-8")
            fout3 = codecs.open("results/eval-title", "w", "utf-8")
            with codecs.open("results/eval-" + type + "_" + mname, "w", "utf-8") as fout:
                list_of_refs, hypotheses = [], []
                for i in range(len(X) // hp.batch_size):
                    print('The {}-th batch'.format(i))

                    ### Get mini-batches
                    x = X[i*hp.batch_size: (i+1)*hp.batch_size]
                    sources = Sources[i*hp.batch_size: (i+1)*hp.batch_size]
                    targets = Targets[i*hp.batch_size: (i+1)*hp.batch_size]
                    print('prepared input data')

                    ### Autoregressive inference
                    preds = np.zeros((hp.batch_size, hp.summary_maxlen), np.int32)
                    for j in range(hp.summary_maxlen):
                        _preds = sess.run(g.preds, {g.x: x, g.y: preds})
                        preds[:, j] = _preds[:, j]
                    
                    ### Write to file
                    for source, target, pred in zip(sources, targets, preds): # sentence-wise
                        got = " ".join(idx2word[idx] for idx in pred).split("</S>")[0].strip()
                        sentence_to_write = "-source: {}\n-expected: {}\n-got: {}\n\n".format(source, target, got)
                            
                        print(sentence_to_write)
                        fout.write(sentence_to_write)
                        fout2.write(got.strip() + '\n')
                        fout3.write(target.strip() + '\n')

                        fout.flush()
                        fout2.flush()
                        fout3.flush()

                        # bleu score
                        ref = target.split()
                        hypothesis = got.split()
                        if len(ref) > 3 and len(hypothesis) > 3:
                            list_of_refs.append([ref])
                            hypotheses.append(hypothesis)
            
                ## Calculate bleu score
                score = corpus_bleu(list_of_refs, hypotheses)
                print("Bleu Score = " + str(100*score))
                fout.write("Bleu Score = " + str(100*score))
    return score


if __name__ == '__main__':
    logging.info("START")
    logging.info("tensorflow version: ", tf.__version__)
    
    from tensorflow.python.client import device_lib as dl
    logging.info("local device: ", dl.list_local_devices())
    
    train()
    # eval()

