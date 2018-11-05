from collections import namedtuple
import tensorflow as tf

from hyperparams import Hyperparams as hp
from data_load import get_batch_data, load_doc_vocab, load_sum_vocab
from rouge_tensor import rouge_l_fscore
from modules import *

io_pairs = namedtuple(typename='io_pairs', field_names='input output')

class Graph():
    def __init__(self, is_training=True):
        self.graph = tf.Graph()
        # de2idx, idx2de = load_doc_vocab()
        # self.vocab_size = len(de2idx)
        self.vocab_size = len(load_doc_vocab()[0])
        
        with self.graph.as_default():
            if is_training:
                print('Getting batch data...')
                self.x, self.y, self.num_batch = get_batch_data()  # (N, T)
            
            else:  # inference
                self.x = tf.placeholder(tf.int32, shape=(None, hp.article_maxlen))
                self.y = tf.placeholder(tf.int32, shape=(None, hp.summary_maxlen))

            # define decoder inputs
            print('Loaded data...')

            self.decoder_inputs = tf.concat((tf.ones_like(self.y[:, :1]) * 2, self.y[:, :-1]), -1)  # 2:<S>
            

            self._add_encoder(is_training=is_training)
            # add decoder

            # eta = tf.get_variable(tf.zeros(shape=(1)))
            eta = tf.convert_to_tensor(0.4)
            self.ml_loss = self._add_ml_loss(is_training=is_training)
            self.rl_loss = self._add_rl_loss()
            
            # self.rl_loss = tf.Print(input_=self.rl_loss, data=[self.rl_loss, self.ml_loss, self.sl, self.reward_diff], message='LS / ML / self.sl / reward_diff')
            
            self.loss =  eta * self.rl_loss + (1 - eta) * self.ml_loss

            # Training Scheme
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
            # self.train_op = self.optimizer.minimize(self.mean_loss, global_step=self.global_step)
            
            self.grads_and_vars = self.optimizer.compute_gradients(loss=self.loss) # TODO: change mean_loss to mix_loss
            self.train_op = self.optimizer.apply_gradients(grads_and_vars=self.grads_and_vars, global_step=self.global_step)
            
            # record gradient
            '''
            for index, grad in enumerate(self.grads_and_vars):
                if index % 15 != 0:
                    continue
                if isinstance(grad[0], tf.IndexedSlices):
                    tf.summary.histogram(name="{}-grad".format(grad[1].name), values=grad[0].values)
                else:
                    tf.summary.histogram(name="{}-grad".format(grad[1].name), values=grad[0])
            '''
            # Summary
            tf.summary.histogram(name="embedding", values=self.get_embedding_table())
            tf.contrib.summary.scalar('rl_loss', self.rl_loss)
            tf.contrib.summary.scalar('ml_loss', self.ml_loss)
            tf.contrib.summary.scalar('loss', self.loss)
            self.merged = tf.summary.merge_all()
            
        self.filewriter = tf.summary.FileWriter(hp.tb_dir + '/train', self.graph)


    def _add_encoder(self, is_training):
        with self.graph.as_default():
            # de2idx, idx2de = load_doc_vocab()
            # en2idx, idx2en = load_sum_vocab()

            print('Constructing Encoder...')
            self.enc = embedding(self.x,
                                 # vocab_size=len(de2idx),
                                 vocab_size=self.vocab_size,
                                 num_units=hp.hidden_units,
                                 scale=True,
                                 scope="encoder_embed")
            
            self.batch_inp_emb = self.enc
            with tf.variable_scope("encoder"):
                if hp.sinusoid:
                    self.enc += positional_encoding(self.x,
                                                    num_units=hp.hidden_units,
                                                    zero_pad=False,
                                                    scale=False,
                                                    scope="enc_pe")
                else:
                    self.enc += embedding(
                        tf.tile(tf.expand_dims(tf.range(tf.shape(self.x)[1]), 0), [tf.shape(self.x)[0], 1]),
                        vocab_size=hp.article_maxlen,
                        num_units=hp.hidden_units,
                        zero_pad=False,
                        scale=False,
                        scope="enc_pe")

                ## Dropout ***
                self.enc = tf.layers.dropout(self.enc,
                                             rate=hp.dropout_rate,
                                             training=tf.convert_to_tensor(is_training))

                ## Blocks
                for i in range(hp.num_blocks):
                    with tf.variable_scope("num_blocks_{}".format(i)):
                        self.enc = multihead_attention(queries=self.enc,
                                                       keys=self.enc,
                                                       num_units=hp.hidden_units,
                                                       num_heads=hp.num_heads,
                                                       dropout_rate=hp.dropout_rate,
                                                       is_training=is_training,
                                                       causality=False)
                        
                        self.enc = feedforward(self.enc, num_units=[4 * hp.hidden_units, hp.hidden_units])
                        ## ATTENTION: the hard-coded >> 4 * hp.hidden_units <<
                        tf.summary.histogram(name="ffw-output/{}".format(i), values=self.enc)


    def _add_decoder(self, is_training, decoder_inputs, inside_loop=False, reuse=None):
        with self.graph.as_default():
            # Decoder
            self.dec = embedding(decoder_inputs,
                                 # vocab_size=len(en2idx),
                                 vocab_size=self.vocab_size,
                                 num_units=hp.hidden_units,
                                 lookup_table=self.get_embedding_table(concated=True),
                                 scale=True,
                                 scope="decoder_embed",
                                 reuse=reuse)
                                 
            self.batch_outp_emb = self.dec
            
            with tf.variable_scope("decoder"):
                ## Positional Encoding
                if hp.sinusoid:
                    self.dec += positional_encoding(decoder_inputs,
                                                    vocab_size=hp.summary_maxlen,
                                                    num_units=hp.hidden_units,
                                                    zero_pad=False,
                                                    scale=False,
                                                    scope="dec_pe",
                                                    reuse=reuse)
                else:
                    self.dec += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(decoder_inputs)[1]), 0),
                                                  [tf.shape(decoder_inputs)[0], 1]),
                                          vocab_size=hp.summary_maxlen,
                                          num_units=hp.hidden_units,
                                          zero_pad=False,
                                          scale=False,
                                          scope="dec_pe",
                                          reuse=reuse)

                ## Dropout
                self.dec = tf.layers.dropout(self.dec,
                                             rate=hp.dropout_rate,
                                             training=tf.convert_to_tensor(is_training))

                ## Blocks
                for i in range(hp.num_blocks):
                    with tf.variable_scope("num_blocks_{}".format(i)):
                        self.dec = multihead_attention(queries=self.dec,
                                                       keys=self.dec,
                                                       num_units=hp.hidden_units,
                                                       num_heads=hp.num_heads,
                                                       dropout_rate=hp.dropout_rate,
                                                       is_training=is_training,
                                                       causality=True,
                                                       scope="self_attention",
                                                       inside_loop=inside_loop,
                                                       reuse=reuse)

                        self.dec = multihead_attention(queries=self.dec,
                                                       keys=self.enc,
                                                       num_units=hp.hidden_units,
                                                       num_heads=hp.num_heads,
                                                       dropout_rate=hp.dropout_rate,
                                                       is_training=is_training,
                                                       causality=False,
                                                       scope="vanilla_attention",
                                                       inside_loop=inside_loop,
                                                       reuse=reuse)

                        self.dec = feedforward(self.dec,
                                               num_units=[4 * hp.hidden_units, hp.hidden_units],
                                               inside_loop=inside_loop,
                                               reuse=reuse)
                        # tf.summary.histogram(name="ffw-output/{}".format(i), values=self.dec)
        
            # Final linear projection
            # self.logits = tf.layers.dense(self.dec, len(en2idx), name='final_output_dense')
            self.logits = tf.layers.dense(self.dec, self.vocab_size, name='final_output_dense', reuse=reuse)
            return self.logits
    
    
    def _add_ml_loss(self, is_training):
        logits = self._add_decoder(is_training=is_training, decoder_inputs=self.decoder_inputs)

        with self.graph.as_default():
            self.preds = tf.to_int32(tf.argmax(logits, axis=-1))
            # IMPORTANT: here we havent pass the logits through softmax!! (yet since the index
            # of max-value is same so gonna be fine if we just want to use to predict
            # but for the rl part, or if u want to do beam-search, need to pass it to softmax
            # to obtain the probability
            self.istarget = tf.to_float(tf.not_equal(self.y, 0))
            self.acc = tf.reduce_sum(tf.to_float(tf.equal(self.preds, self.y)) * self.istarget) / (
                tf.reduce_sum(self.istarget))
            '''
            tf.summary.histogram(name="logits", values=self.logits)
            tf.summary.scalar('acc', self.acc)
            '''
            if is_training:
                # Loss
                # self.y_smoothed = label_smoothing(tf.one_hot(self.y, depth=len(en2idx)))
                self.y_smoothed = label_smoothing(tf.one_hot(self.y, depth=self.vocab_size))
                loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.y_smoothed)
                ml_loss = tf.reduce_sum(loss * self.istarget, name='fake_ml_loss') / (tf.reduce_sum(self.istarget))
                
        return ml_loss

    
    def _rl_step(self, last_logits, last_preds, cur_timestep, greedy, scope="rl_step", reuse=None):
        '''
            WARNING: this function is REALLY memory-inefficient, please improve in the future if possible
            
            rl actually share the same decoder with ml, yet since they have different input mechanism, we seperate them out
            so this function can be use as the loop-fn of tf.while_loop
            
            this function implement an auto-regressive greedy inference inside tf graph
            
            @params:
                last_logits: the result of previous timestep (prob), also the input of current timestep
                last_preds: predict outcome (the index of predicted word)
                cur_timestep: the current timestep, i.e. the timestep position where we are going to update 'logits' and 'pred'
                greedy: boolean, in greedy mode or sample mode
        '''
        # logits = self._add_decoder(is_training=is_training, decoder_input=last_preds, reuse=True)
        # padd last preds to max_length
        # tf.assert_equal(last_preds.get_shape()[-1], cur_timestep, message='dismatch timestep in rl')
        
        decoder_inputs = tf.pad(last_preds, [[0, 0], [0, hp.summary_maxlen - cur_timestep]])
        full_logits = self._add_decoder(is_training=True,
                                   decoder_inputs=decoder_inputs,
                                   inside_loop=True,
                                   reuse=True)
                                   # shape: (batch_size, time_steps, num_hidden_units)
                                   
        with self.graph.as_default():
            # cur_preds: an integer tensor
            '''
            # since "Using a `tf.Tensor` as a Python `bool` is not allowed"
            # -> switch to 'tf.cond'
            if greedy:
                cur_preds = tf.to_int32(tf.argmax(full_logits[:, cur_timestep, :], axis=-1)) # choose best one at current timestep
            else:
                cur_preds = tf.random_uniform(shape=(full_logits()[0], 1), minval=0, maxval=self.vocab_size, dtype=int32)
            '''
            with tf.variable_scope(scope, reuse=reuse):
                cur_preds = tf.cond(pred=greedy,
                                    true_fn=lambda: tf.to_int32(tf.argmax(full_logits[:, cur_timestep, :], axis=-1)), # for current timestep: choose the words with largest prob
                                    false_fn=lambda: tf.random_uniform(shape=(tf.shape(full_logits)[0], ), minval=0, maxval=self.vocab_size, dtype=tf.int32)) # sample
                                    # shape of cur_preds: (batch_size, ): one-dimensional array
                
                # WRONG since we r not allowed to direclty slice with tensor: cur_logit = full_logits[:, cur_timestep, cur_preds]
                cur_idx = tf.stack([tf.range(start=0, limit=tf.shape(full_logits)[0]), cur_preds], axis=-1)
                cur_logit = tf.gather_nd(params=full_logits[:, cur_timestep, :], indices=cur_idx) # select the wanted probabilities with current index from full_logits
                
                cur_logit = tf.nn.softmax(cur_logit, axis=-1) # get (batch_size, vocab_size), with each (, vocab_size) been softmax-ed
                # cur_logit = tf.Print(input_=cur_logit_real, data=[cur_timestep, cur_logit_real], message='current logits at timestep xx ')
                
                last_logits += tf.log(cur_logit) # shape: (num_batch, ) -- we only need the sum of log over each timestep
                last_preds = tf.concat(values=[last_preds, tf.reshape(cur_preds, shape=(hp.batch_size, 1))],
                                       axis=-1) # cannot do assignment on tensor & keep inputs hv same rank
                cur_timestep += 1
    
        return last_logits, last_preds, cur_timestep, greedy
    
    def _rl_autoinfer(self, greedy, scope='_rl_autoinfer', name=None):
        ''' greedy: type tf.bool '''
        
        def while_exit_cond(logits, preds, cur_timestep, greedy):
            ''' return True if not finished, False otherwise '''
            return cur_timestep < hp.summary_maxlen # stop when we reach the end
        
        with self.graph.as_default():
            logits = tf.zeros((hp.batch_size, ), name='logits') # should be float type & keep it to be 1d
            preds = tf.zeros((hp.batch_size, 0), dtype=tf.int32, name='preds')
            cur_timestep = tf.zeros(shape=(), dtype=tf.int32, name='cur_timestep')
            # here the shape should be None other than shape=(1), otherwise it will be treated like an array -> the return value of while_exit_cond will be an array other than bool;
            # also the dtype of cur_timpstep should be set as int32 explicitly, otherwise u will encounter error when using it as slice index & comparing it with a scalar
            
            # since the shape for preds change at each iteration
            shape_inv = [logits.get_shape(),
                         tf.TensorShape([hp.batch_size, None]),
                         cur_timestep.get_shape(),
                         greedy.get_shape()]
            logits, preds, cur_timestep, greedy = tf.while_loop(
                cond=while_exit_cond,
                body=self._rl_step,
                loop_vars=[logits, preds, cur_timestep, greedy],
                shape_invariants=shape_inv,
                back_prop=True, # ori : False,
                parallel_iterations=1,
                name=name)
        
        # the returned logits has been passed through tf.log()
        return logits, preds

    
    def _add_rl_loss(self):
        sample_logits, sample_preds = self._rl_autoinfer(greedy=tf.constant(value=False, dtype=tf.bool), name='sample_loop')
        greedy_logits, greedy_preds = self._rl_autoinfer(greedy=tf.constant(value=True, dtype=tf.bool), name='greedy_loop')
        
        # print("========= TYPE OF RETURNED SCORE: ", type(rouge_l_fscore(sample_preds, self.y)))
        self.reward_diff = rouge_l_fscore(greedy_preds, self.y) - rouge_l_fscore(sample_preds, self.y)
        # print("========= reward_diff ", rouge_l_fscore(sample_preds, self.y).get_shape()) # unknown
        self.sl = sample_logits
        
        self.reward_diff = tf.Print(input_=self.reward_diff, data=[tf.shape(self.reward_diff), self.reward_diff], message='shape of reward_diff')
        sample_logits = tf.Print(input_=sample_logits, data=[tf.shape(sample_logits), sample_logits], message='shape of sample_logits')
        
        self.istarget = tf.to_float(tf.not_equal(self.y, 0)) # only consider the target positions & average the loss
        rl_loss = tf.reduce_sum(tf.multiply(self.reward_diff, sample_logits), name='rl_loss') / (tf.reduce_sum(self.istarget))

        return rl_loss
    
    
    def get_filewriter(self):
        return self.filewriter

    def get_input_output(self, is_training=False):
        if is_training:
            return io_pairs(input=[self.x, self.y], output=[self.logits, self.loss])
        else:
            return io_pairs(input=[self.x, self.y], output=[self.logits])

    def get_embedding_table(self, scope='', concated=False):
        if not concated:
            with tf.variable_scope(scope, reuse=True):  # get lookup table
                lookup_table = tf.get_variable('lookup_table')
        else:
            lookup_table = self.graph.get_tensor_by_name("concated_lookup_table:0")
        
        return lookup_table

    def get_batch_embedding(self):
        """ only return the embedding of current batch"""
        return [self.batch_inp_emb, self.batch_outp_emb]
    
    def get_gradients(self):
        return self.grads_and_vars
