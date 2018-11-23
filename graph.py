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
                self.x, self.y, self.num_batch = get_batch_data()  # (N, T)
            else:  # inference
                self.x = tf.placeholder(tf.int32, shape=(None, hp.article_maxlen))
                self.y = tf.placeholder(tf.int32, shape=(None, hp.summary_maxlen))

            self.decoder_inputs = tf.concat((tf.ones_like(self.y[:, :1]) * 2, self.y[:, :-1]), -1)  # 2:<S> # define decoder inputs
            
            self._add_encoder(is_training=is_training)
            self.ml_loss = self._add_ml_loss(is_training=is_training)
            
            if is_training:
                self.eta = tf.Variable(initial_value=hp.eta_init, dtype=tf.float32,
                                       trainable=False, name='eta')
                                       # disallow eta to be updated by loss
                self.update_eta = tf.assign(self.eta, self.eta + 0.1)

                self.rl_loss = self._add_rl_loss()
                self.loss =  self.eta  * self.rl_loss + (1 - self.eta) * self.ml_loss

                # Training Scheme
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
                
                grads_and_vars_mix = self.optimizer.compute_gradients(loss=self.loss)
                grads_and_vars_ml = self.optimizer.compute_gradients(loss=self.ml_loss)
                
                grad_mix, vars_mix = zip(*grads_and_vars_mix) # parse grad and var
                grad_ml, vars_ml = zip(*grads_and_vars_ml) # parse grad and var
                
                # add gradient clipping
                clipped_grad_mix, globle_norm_mix = tf.clip_by_global_norm(grad_mix, hp.maxgradient)
                clipped_grad_ml, globle_norm_ml = tf.clip_by_global_norm(grad_ml, hp.maxgradient)
                self.globle_norm_ml = globle_norm_ml
                self.train_op_mix = self.optimizer.apply_gradients(grads_and_vars=zip(clipped_grad_mix, vars_mix),
                                                                   global_step=self.global_step)
                self.train_op_ml  = self.optimizer.apply_gradients(grads_and_vars=zip(clipped_grad_ml, vars_ml),
                                                                  global_step=self.global_step)
                '''
                # below: training wihtout gradient clipping
                self.train_op_mix = self.optimizer.apply_gradients(grads_and_vars=grads_and_vars_mix,
                                                                   global_step=self.global_step)
                self.train_op_ml  = self.optimizer.apply_gradients(grads_and_vars=grads_and_vars_ml,
                                                                   global_step=self.global_step)
                '''
                
                # Summary
                tf.summary.scalar('globle_norm_ml', globle_norm_ml)
                tf.summary.histogram(name='reward_diff', values=self.reward_dif)
                tf.summary.scalar('rl_loss', self.rl_loss)
                tf.summary.scalar('ml_loss', self.ml_loss)
                tf.summary.scalar('loss', self.loss)
                self.merged = tf.summary.merge_all()
            
                # prepare the Saver that restore all variables other than eta
                all_var = tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES)
                all_var.remove(self.eta)
                self.subset_saver = tf.train.Saver(var_list=all_var)

        self.filewriter = tf.summary.FileWriter(hp.tb_dir + '/train', self.graph)


    def _add_encoder(self, is_training):
        with self.graph.as_default():
            
            print('Constructing Encoder...')
            self.enc = embedding(self.x,
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
                        
                        self.enc = feedforward(self.enc, num_units=[hp.ffw_unit, hp.hidden_units])
                        ## ATTENTION: the hard-coded >> 4 * hp.hidden_units <<
                        tf.summary.histogram(name="ffw-output/{}".format(i), values=self.enc)


    def _add_decoder(self, is_training, decoder_inputs, inside_loop=False, reuse=None):
        with self.graph.as_default():
            # Decoder
            self.dec = embedding(decoder_inputs,
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
                                               num_units=[hp.ffw_unit, hp.hidden_units],
                                               inside_loop=inside_loop,
                                               reuse=reuse)
        
            # Final linear projection
            self.logits = tf.layers.dense(self.dec, self.vocab_size, name='final_output_dense', reuse=reuse)
            return self.logits

    
    def _add_ml_loss(self, is_training):
        logits = self._add_decoder(is_training=is_training, decoder_inputs=self.decoder_inputs)
        
        with self.graph.as_default():
            self.preds = tf.to_int32(tf.argmax(logits, axis=-1)) # shape: (batch_size, max_timestep)
            self.istarget = tf.to_float(tf.not_equal(self.y, 0)) # shape: (batch_size, max_timestep)
            self.acc = tf.reduce_sum(tf.to_float(tf.equal(self.preds, self.y)) * self.istarget) / (
                tf.reduce_sum(self.istarget))
            
            self.rouge = tf.reduce_sum(rouge_l_fscore(self.preds, self.y)) / float(hp.batch_size)
            tf.summary.scalar('acc', self.acc)

            ml_loss = -100
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
            
            @params:
                last_logits: the result of previous timestep (prob), also the input of current timestep
                last_preds: predict outcome (the index of predicted word)
                cur_timestep: the current timestep, i.e. the timestep position where we are going to update 'logits' and 'pred'
                greedy: tf.bool , in greedy mode or sample mode
        '''
        decoder_inputs = tf.pad(last_preds, [[0, 0], [0, hp.summary_maxlen - cur_timestep]])
        full_logits = self._add_decoder(is_training=True,
                                   decoder_inputs=decoder_inputs,
                                   inside_loop=True,
                                   reuse=True)
                                   # shape: (batch_size, time_steps, num_hidden_units)
                                   
        with self.graph.as_default():
            with tf.variable_scope(scope, reuse=reuse):
                cur_logit = full_logits[:, cur_timestep, :] # shape: (N, W)
                
                # add mask on <PAD>
                
                masks = tf.sign(cur_logit) # shape: (N, W); 0 for 0 entries, 1 or -1 for others
                # 这里应该改成用pred作为mask的依据吧？？logit怎么都不会为0的。
                # 而且应该是..不知道。又感觉不用mask。。先去掉好惹
                paddings = tf.ones_like(cur_logit)*(-2**32+1) # -inf (to reduce its influence in softmax)
                cur_logit = tf.where(tf.equal(masks, 0), paddings, cur_logit) # (N, W)
                
                cur_logit = tf.nn.softmax(cur_logit, axis=-1) # shape: (N, M), each (, M) is softmaxed
                
                
                # if greedy: apply greedy inference at each step,
                # otherwise sample according to current word distribution
                cur_preds = tf.cond(pred=greedy,
                                    true_fn=lambda: tf.argmax(cur_logit, axis=-1), # (N, )
                                    false_fn=lambda: tf.multinomial(logits=tf.log(cur_logit), num_samples=1)) # (N, 1)
                cur_preds = tf.to_int32(tf.reshape(cur_preds, shape=(hp.batch_size, )))  # (N, )
                cur_idx = tf.stack([tf.range(start=0, limit=tf.shape(full_logits)[0]), cur_preds],
                                   axis=-1) # (N, 2)
                                   
                cur_logit = tf.gather_nd(params=cur_logit, indices=cur_idx) # (N, ) # the logit of cur_idx
                
                last_logits += tf.log(cur_logit) # shape: (N, ) sum up the log prob at each timestep
                last_preds = tf.concat(values=[last_preds, tf.reshape(cur_preds, shape=(hp.batch_size, 1))],
                                       axis=-1)
                cur_timestep += 1
    
        return last_logits, last_preds, cur_timestep, greedy
    
    def _rl_autoinfer(self, greedy, scope='_rl_autoinfer', name=None):
        ''' greedy: type tf.bool '''
        
        def while_exit_cond(logits, preds, cur_timestep, greedy):
            return cur_timestep < hp.summary_maxlen # stop when we reach the end
        
        with self.graph.as_default():
            logits = tf.zeros((hp.batch_size, ), name='logits') # should be float type & keep it to be 1d
            preds = tf.zeros((hp.batch_size, 0), dtype=tf.int32, name='preds')
            cur_timestep = tf.zeros(shape=(), dtype=tf.int32, name='cur_timestep') # a scalar, used as index later & comparison
            
            shape_inv = [logits.get_shape(),
                         tf.TensorShape([hp.batch_size, None]), # shape for preds change at each iteration
                         cur_timestep.get_shape(),
                         greedy.get_shape()]
                         
            log_logits, preds, cur_timestep, greedy = tf.while_loop(
                cond=while_exit_cond,
                body=self._rl_step,
                loop_vars=[logits, preds, cur_timestep, greedy],
                shape_invariants=shape_inv,
                back_prop=True,
                parallel_iterations=1,
                name=name)
        
        return log_logits, preds

    
    def _add_rl_loss(self):
        sample_logits, sample_preds = self._rl_autoinfer(greedy=tf.constant(value=False, dtype=tf.bool), name='sample_loop')
        greedy_logits, greedy_preds = self._rl_autoinfer(greedy=tf.constant(value=True, dtype=tf.bool), name='greedy_loop')
        self.sl = sample_logits
        
        sample_logits = tf.Print(input_=sample_logits, data=[sample_logits], message='sample_logits: ')
        greedy_logits = tf.Print(input_=greedy_logits, data=[greedy_logits], message='greedy_logits: ')
        
        
        self.reward_diff = tf.zeros(shape=())
        for sent_i, ref in enumerate(tf.unstack(self.y)):
            real_y = ref[:tf.reduce_sum(tf.to_int32(tf.not_equal(self.y, 0)))] # remove the <PAD> in the end
            self.reward_diff += rouge_l_fscore([greedy_preds[sent_i]], [real_y]) - rouge_l_fscore([sample_preds[sent_i]], [real_y])
        
        self.reward_diff = tf.Print(input_=self.reward_diff, data=[self.reward_diff], message='reward_diff: ')

        rl_loss = tf.reduce_sum(self.reward_diff * sample_logits) / (hp.batch_size * hp.summary_maxlen)

        rl_loss = tf.Print(input_=rl_loss, data=[rl_loss], message='rl_loss: ')
        return rl_loss # masked
        
    
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
