# =========================================================================================
# Implementation of "Show, Attend and Tell: Neural Caption Generator With Visual Attention".
# There are some notations.
# N is batch size.
# L is spacial size of feature vector (196).
# D is dimension of image feature vector (512).
# T is the number of time step which is equal to caption's length-1 (16).
# V is vocabulary size (about 10000).
# M is dimension of word vector which is embedding size (default is 512).
# H is dimension of hidden state (default is 1024).
# =========================================================================================

from __future__ import division

import tensorflow as tf
import numpy as np
from bleu import evaluate
from tensorflow.contrib.layers import flatten
SEQUENCE_MAXLEN = 1000
from utils import *


def get_scope_variables(scope):
    
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)


class CaptionGenerator(object):
    def __init__(self, sess, word_to_idx, dim_feature=[49, 2048], dim_embed=2048, dim_hidden=1024, n_time_step=16, 
                  prev2out=True, ctx2out=True, alpha_c=0.0, selector=True, dropout=True):
        """
        Args:
            word_to_idx: word-to-index mapping dictionary.
            dim_feature: (optional) Dimension of vggnet19 conv5_3 feature vectors.
            dim_embed: (optional) Dimension of word embedding.
            dim_hidden: (optional) Dimension of all hidden state.
            n_time_step: (optional) Time step size of LSTM. 
            prev2out: (optional) previously generated word to hidden state. (see Eq (7) for explanation)
            ctx2out: (optional) context to hidden state (see Eq (7) for explanation)
            alpha_c: (optional) Doubly stochastic regularization coefficient. (see Section (4.2.1) for explanation)
            selector: (optional) gating scalar for context vector. (see Section (4.2.1) for explanation)
            dropout: (optional) If true then dropout layer is added.
        """
        self._sess = sess
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.iteritems()}
        self.prev2out = prev2out
        self.ctx2out = ctx2out
        self.alpha_c = alpha_c
        self.selector = selector
        self.dropout = dropout
        self.V = len(word_to_idx)
        self.L = dim_feature[0]
        self.D = dim_feature[1]
        self.M = dim_embed
        self.H = dim_hidden
        self.T = n_time_step
        self.num = 10
        self.detect = 4096
        self._start = word_to_idx['<START>']
        self._null = word_to_idx['<NULL>']
        self._weights = []
        self._learn_phase = 100
        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer(0.0)
        self.emb_initializer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)
        self._time = tf.Variable(0, name='time')
        # Place holder for features and captions
        self.features = tf.placeholder(tf.float32, [None, self.L, self.D])
        #self.features_fc = tf.placeholder(tf.float32, [None, self.D])
        self.features_detect = tf.placeholder(tf.float32, [None, self.num, self.detect])
        self.captions = tf.placeholder(tf.int32, [None, self.T + 1])
        #self.reference = tf.placeholder(tf.int32, [None, self.T + 1])
        self.reward_score = tf.placeholder(tf.float32, [])
        #self.batch_size = tf.placeholder(tf.int32, [])
        self.logdir = 'log/'
        self.log_every = 50
    @property
    def time(self):
        return self._time


    def get_weight(self, name, shape,
                   init='glorot',
                   device='gpu',
                   weight_val=None,
                   trainable=True):
        """Creates a new weight.
        Args:
            name: str, the name of the variable.
            shape: tuple of ints, the shape of the variable.
            init: str, the type of initialize to use.
            device: str, 'cpu' or 'gpu'.
            weight_val: Numpy array to use as the initial weights.
            trainable: bool, whether or not this weight is trainable.
        Returns:
            a trainable TF variable with shape `shape`.
        """

        if weight_val is None:
            init = init.lower()
            if init == 'normal':
                initializer = (lambda shape, dtype, partition_info:
                               tf.random_normal(shape, stddev=0.05))
            elif init == 'uniform':
                initializer = (lambda shape, dtype, partition_info:
                               tf.random_uniform(shape, stddev=0.05))
            elif init == 'glorot':
                initializer = (lambda shape, dtype, partition_info:
                               tf.random_normal(
                                   shape, stddev=np.sqrt(6. / sum(shape))))
            elif init == 'eye':
                assert all(i == shape[0] for i in shape)
                initializer = (lambda shape, dtype, partition_info:
                               tf.eye(shape[0]))
            elif init == 'zero':
                initializer = (lambda shape, dtype, partition_info:
                               tf.zeros(shape))
            else:
                raise ValueError('Invalid init: "%s"' % init)
        else:
            weight_val = weight_val.astype('float32')

        device = device.lower()
        if device == 'gpu':
            on_gpu = True
        elif device == 'cpu':
            on_gpu = False
        else:
            raise ValueError('Invalid device: "%s"' % device)

        with tf.device('/gpu:0'):
            weight = tf.get_variable(name=name,
                                     shape=shape,
                                     initializer=initializer,
                                     trainable=trainable)
        self._weights.append(weight)

        return weight

    def current_time(self):
        return self._sess.run(self.time)

    def _get_initial_lstm(self, features, reuse = False):
        with tf.variable_scope('initial_lstm', reuse= reuse):
            features_mean = tf.reduce_mean(features, 1)

            w_h = tf.get_variable('w_h', [self.D, self.H], initializer=self.weight_initializer)
            b_h = tf.get_variable('b_h', [self.H], initializer=self.const_initializer)
            h = tf.nn.tanh(tf.matmul(features_mean, w_h) + b_h)

            w_c = tf.get_variable('w_c', [self.D, self.H], initializer=self.weight_initializer)
            b_c = tf.get_variable('b_c', [self.H], initializer=self.const_initializer)
            c = tf.nn.tanh(tf.matmul(features_mean, w_c) + b_c)
            return c, h

    def _get_initial_lstm_detect(self, features, reuse = False):
        with tf.variable_scope('initial_lstm_detect', reuse= reuse):
            features_mean = tf.reduce_mean(features, 1)

            w_h = tf.get_variable('w_h', [self.detect, self.H*2], initializer=self.weight_initializer)
            b_h = tf.get_variable('b_h', [self.H*2], initializer=self.const_initializer)
            h = tf.nn.tanh(tf.matmul(features_mean, w_h) + b_h)

            w_c = tf.get_variable('w_c', [self.detect, self.H*2], initializer=self.weight_initializer)
            b_c = tf.get_variable('b_c', [self.H*2], initializer=self.const_initializer)
            c = tf.nn.tanh(tf.matmul(features_mean, w_c) + b_c)
            return c, h

    
    def _word_embedding(self, inputs, reuse=False):
        with tf.variable_scope('word_embedding', reuse=reuse):
            w_e = tf.get_variable('w_e', [self.V, self.M], initializer=self.emb_initializer)
            x = tf.nn.embedding_lookup(w_e, inputs, name='word_vector')  # (N, T, M) or (N, M)
            return x
    
    def _project_features_detect(self, features, reuse = False):
        with tf.variable_scope('project_detect', reuse=reuse):
            w_f = tf.get_variable('w_f', [self.detect, self.detect], initializer=self.weight_initializer)
            features_flat = tf.reshape(features, [-1, self.detect])
            features_proj = tf.matmul(features_flat, w_f)  
            features_proj = tf.reshape(features_proj, [-1, self.num, self.detect])
            return features_proj


    def _project_features(self, features, reuse = False):
        with tf.variable_scope('project', reuse=reuse):
            w_f = tf.get_variable('w_f', [self.D, self.D], initializer=self.weight_initializer)
            features_flat = tf.reshape(features, [-1, self.D])
            features_proj = tf.matmul(features_flat, w_f)  
            features_proj = tf.reshape(features_proj, [-1, self.L, self.D])
            return features_proj





    def _attention_layer(self, features, features_proj, h, temperature, reuse=False):
        with tf.variable_scope('attention', reuse=reuse):
            w_a = tf.get_variable('w_a', [self.H, self.D], initializer=self.weight_initializer)
            b_a = tf.get_variable('b_a', [self.D], initializer=self.const_initializer)
            w_att = tf.get_variable('w_att', [self.D, 1], initializer=self.weight_initializer)
            h_att = tf.nn.relu(features_proj + tf.expand_dims(tf.matmul(h, w_a), 1) + b_a)    # (N, L, D)
            out_att = tf.reshape(tf.matmul(tf.reshape(h_att, [-1, self.D]), w_att), [-1, self.L])   # (N, L)
            alpha = tf.nn.softmax(out_att)  
            context = tf.reduce_sum(features * tf.expand_dims(alpha, 2), 1, name='context')   #(N, D)
            return context, alpha
  
    def _attention_layer_detect(self, features, features_proj, h, temperature, reuse=False):
        with tf.variable_scope('attention_detect', reuse=reuse):
            w_a = tf.get_variable('w_a', [self.H*2, self.detect], initializer=self.weight_initializer)
            b_a = tf.get_variable('b_a', [self.detect], initializer=self.const_initializer)
            w_att = tf.get_variable('w_att', [self.detect, 1], initializer=self.weight_initializer)
            h_att = tf.nn.relu(features_proj + tf.expand_dims(tf.matmul(h, w_a), 1) + b_a)    # (N, L, D)
            out_att = tf.reshape(tf.matmul(tf.reshape(h_att, [-1, self.detect]), w_att), [-1, self.num])   # (N, L)
            alpha = tf.nn.softmax(out_att)  
            context = tf.reduce_sum(features * tf.expand_dims(alpha, 2), 1, name='context')   #(N, D)
            return context, alpha


    def _selector(self, context, h, reuse=False):
        with tf.variable_scope('selector', reuse=reuse):
            w_s = tf.get_variable('w_s', [self.H, 1], initializer=self.weight_initializer)
            b_s = tf.get_variable('b_s', [1], initializer=self.const_initializer)
            beta = tf.nn.sigmoid(tf.matmul(h, w_s) + b_s, 'beta')    # (N, 1)
            context = tf.multiply(beta, context, name='selected_context') 
            return context, beta
  
    def _selector_detect(self, context, h, reuse=False):
        with tf.variable_scope('selector_detect', reuse=reuse):
            w_s = tf.get_variable('w_s', [self.H*2, 1], initializer=self.weight_initializer)
            b_s = tf.get_variable('b_s', [1], initializer=self.const_initializer)
            beta = tf.nn.sigmoid(tf.matmul(h, w_s) + b_s, 'beta')    # (N, 1)
            context = tf.multiply(beta, context, name='selected_context') 
            return context, beta

    def _temp_detect(self, h, reuse=False):
        with tf.variable_scope('temp_detect', reuse=reuse):
            w_t = tf.get_variable('w_t', [self.H*2, 1], initializer=self.weight_initializer)
            b_t = tf.get_variable('b_t', [1], initializer=self.const_initializer)
            temperature = 1/(tf.nn.softplus(tf.matmul(h, w_t)+b_t)+1)    # (N, 1)
            return temperature


    def _temp(self, h, reuse=False):
        with tf.variable_scope('temp', reuse=reuse):
            w_t = tf.get_variable('w_t', [self.H, 1], initializer=self.weight_initializer)
            b_t = tf.get_variable('b_t', [1], initializer=self.const_initializer)
            temperature = 1/(tf.nn.softplus(tf.matmul(h, w_t)+b_t)+1)    # (N, 1)
            return temperature

    def _decode_lstm(self, x, h, context, dropout=False, reuse=False):
        with tf.variable_scope('logits', reuse=reuse):
            w_dh = tf.get_variable('w_dh', [self.H*3, self.M], initializer=self.weight_initializer)
            b_dh = tf.get_variable('b_dh', [self.M], initializer=self.const_initializer)
            w_out = tf.get_variable('w_out', [self.M, self.V], initializer=self.weight_initializer)
            b_out = tf.get_variable('b_out', [self.V], initializer=self.const_initializer)

            if dropout:
                h = tf.nn.dropout(h, 0.5)
            h_logits = tf.matmul(h, w_dh) + b_dh

            if self.ctx2out:
                w_ctx2out = tf.get_variable('w_ctx2out', [2*self.D, self.M], initializer=self.weight_initializer)
                h_logits += tf.matmul(context, w_ctx2out)

            if self.prev2out:
                h_logits += x
            h_logits = tf.nn.tanh(h_logits)

            if dropout:
                h_logits = tf.nn.dropout(h_logits, 0.5)
            out_logits = tf.matmul(h_logits, w_out) + b_out
            return out_logits
        
    
    def build_generator(self):
        with tf.variable_scope('generator'):
            
            features = self.features
            features_detect = self.features_detect
            captions = self.captions
            batch_size = tf.shape(features)[0]

            captions_in = captions[:, :self.T]      
            captions_out = captions[:, 1:]  
            mask = tf.to_float(tf.not_equal(captions_out, self._null))
        
            sampled_word_list = []
            class_scores_list = []
            # batch normalize feature vectors
            
           
            c, h = self._get_initial_lstm(features = features)
            c_detect, h_detect = self._get_initial_lstm_detect(features = features_detect)

            x = self._word_embedding(inputs = captions_in)
            features_proj = self._project_features(features = features)
            features_proj_detect = self._project_features_detect(features = features_detect)
            pretrain_loss = 0.0
            alpha_list = []
            alpha_list_detect = []

            
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.H)
            lstm_cell_detect = tf.contrib.rnn.BasicLSTMCell(num_units=self.H*2)
            for t in range(self.T):     
                temperature = self._temp(h,reuse=(t!=0))
                context, alpha = self._attention_layer(features, features_proj, h, temperature, reuse=(t!=0))
                alpha_list.append(alpha)
                if self.selector:
                    context, beta = self._selector(context, h, reuse=(t!=0)) 

                with tf.variable_scope('lstm', reuse=(t!=0)):
                    _, (c, h) = lstm_cell(inputs=tf.concat([x[:,t,:], context], 1), state=[c, h])



                temperature_detect = self._temp_detect(h_detect,reuse=(t!=0))
                context_detect, alpha_detect = self._attention_layer_detect(features_detect, features_proj_detect, h_detect, temperature_detect, reuse=(t!=0))
                alpha_list_detect.append(alpha_detect)
                if self.selector:
                    context_detect, beta_detect = self._selector_detect(context_detect, h_detect, reuse=(t!=0)) 

                with tf.variable_scope('lstm_detect', reuse=(t!=0)):
                    _, (c_detect, h_detect) = lstm_cell_detect(inputs=tf.concat([x[:,t,:], context_detect, h], 1), state=[c_detect, h_detect])



                logits = self._decode_lstm(x[:,t,:], tf.concat([h, h_detect], 1), context_detect, dropout=self.dropout, reuse=(t!=0))

                class_scores = tf.nn.softmax(logits)

                class_scores_list.append(class_scores)

                sampled_word = tf.argmax(logits, 1)  

                sampled_word_list.append(sampled_word) 
                         
                pretrain_loss += tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = captions_out[:, t]) * mask[:, t])
           
            if self.alpha_c > 0:
                alphas = tf.transpose(tf.stack(alpha_list), (1, 0, 2))     # (N, T, L)
                alphas_all = tf.reduce_sum(alphas, 1)      # (N, L)
                alpha_reg = self.alpha_c * tf.reduce_sum((16./196 - alphas_all) ** 2)     
                pretrain_loss += alpha_reg
            
            pretrain_loss = pretrain_loss/tf.to_float(batch_size)
 
            sampled_captions = tf.transpose(tf.stack(sampled_word_list), (1, 0))  
            
            scores = tf.transpose(tf.stack(class_scores_list), (1, 0, 2))
 
            return features, scores, captions_in, sampled_captions, pretrain_loss, batch_size


    def _batch_norm(self, x, mode='train', name=None):
        return tf.contrib.layers.batch_norm(inputs=x, 
                                            decay=0.95,
                                            center=True,
                                            scale=True,
                                            is_training=(mode=='train'),
                                            updates_collections=None,
                                            scope=(name+'batch_norm'))


    def build_sampler(self, max_len=30):
        with tf.variable_scope('generator', reuse=True):
            
            features = self.features
            features_detect = self.features_detect
            
            c, h = self._get_initial_lstm( features = features)
            c_detect, h_detect = self._get_initial_lstm_detect(features = features_detect)
            features_proj = self._project_features(features = features)
            features_proj_detect = self._project_features_detect(features = features_detect)
            sampled_word_list = []
            alpha_list = []
            alpha_list_detect = []
            beta_list = []
            beta_list_detect = []
        
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.H)
        lstm_cell_detect = tf.contrib.rnn.BasicLSTMCell(num_units=self.H*2)
        with tf.variable_scope('generator', reuse=True):
            for t in range(max_len):
                if t == 0:
                    x = self._word_embedding(inputs=tf.fill([tf.shape(features)[0]], self._start))
                else:
                    x = self._word_embedding(inputs=sampled_word, reuse=True)  
                temperature = self._temp(h,reuse=(t!=0))
                context, alpha = self._attention_layer(features, features_proj, h, temperature, reuse=(t!=0))
                alpha_list.append(alpha)
                
                if self.selector:
                    context, beta = self._selector(context, h, reuse=(t!=0)) 
                    beta_list.append(beta)
               
                with tf.variable_scope('lstm', reuse=(t!=0)):
                    _, (c, h) = lstm_cell(inputs=tf.concat([x, context], 1), state=[c, h])


                temperature_detect = self._temp_detect(h_detect,reuse=(t!=0))
                context_detect, alpha_detect = self._attention_layer_detect(features_detect, features_proj_detect, h_detect, temperature_detect, reuse=(t!=0))
                alpha_list_detect.append(alpha_detect)
                
                if self.selector:
                    context_detect, beta_detect = self._selector_detect(context_detect, h_detect, reuse=(t!=0)) 
                    beta_list_detect.append(beta_detect)

                with tf.variable_scope('lstm_detect', reuse=(t!=0)):
                    _, (c_detect, h_detect) = lstm_cell_detect(inputs=tf.concat([x, context_detect, h], 1), state=[c_detect, h_detect])


                logits = self._decode_lstm(x, tf.concat([h, h_detect], 1), context_detect, reuse=(t!=0))
                sampled_word = tf.argmax(logits, 1)       
                sampled_word_list.append(sampled_word)     

            alphas = tf.transpose(tf.stack(alpha_list), (1, 0, 2))     # (N, T, L)
            betas = tf.transpose(tf.squeeze(beta_list), (1, 0))    # (N, T)

            alphas_detect = tf.transpose(tf.stack(alpha_list_detect), (1, 0, 2))     # (N, T, L)
            betas_detect = tf.transpose(tf.squeeze(beta_list_detect), (1, 0))    # (N, T)

            sampled_captions = tf.transpose(tf.stack(sampled_word_list), (1, 0))     # (N, max_len)
            return alphas, betas, alphas_detect, betas_detect, sampled_captions

    def start_build_sampler(self):
         _, _, _, _, self.generated_captions = self.build_sampler(max_len=30)
   



    def build(self, reg_loss=1e-4):
        """Builds the model.
        Args:
            reg_loss: float, how much to weight regularization loss.
        """

        if hasattr(self, '_built') and self._built:
            raise RuntimeError('The model is already built.')

        features, g_classes, cap, g_seq, teach_loss, self.batch_size = self.build_generator()
        #cap = self._word_embedding(inputs = self.captions)
        #g_cap = self._word_embedding(inputs = g_seq)
        
        #reward_score = self.build_discriminator2(cap, g_seq)       

        #r_preds = self.build_discriminator(cap)
        #g_preds = self.build_discriminator(g_seq, reuse=True)
        

        #r_preds1 = self.build_discriminator1(cap, features)
        #g_preds1 = self.build_discriminator1(g_seq, features, reuse=True)


        g_weights = get_scope_variables('generator')

        #d_weights1 = get_scope_variables('discriminator')

        #d_weights2 = get_scope_variables('discriminator1')

        #d_weights = d_weights2
        #tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=vs.name)
        # Adds summaries of the real and fake predictions.
       # tf.summary.histogram('predictions/fake', g_preds)
        #tf.summary.histogram('predictions/real', r_preds)

        # Saves predictions for analysis later.
        #self.g_preds, self.r_preds = g_preds, r_preds
        #self.g_preds1, self.r_preds1 = g_preds1, r_preds1
        # Captures the generated sequence to use later.
        #self.generated_sequence = g_seq
        g_seq = tf.cast(g_seq, tf.int32)
        # Computes the weight updates for the discriminator and generator.
        #dis_op = self.get_discriminator_op(r_preds1, g_preds1, d_weights)
        #gen_op = self.get_generator_op(g_seq, g_preds1, g_classes, g_weights)

        # Adds the teacher forcing part, decaying at some rate.
        teach_lr = 10000. / (10000. + tf.cast(self.time, 'float32'))
        teach_lr *= 1e-3 
        teach_opt = tf.train.AdamOptimizer(teach_lr)
        teach_op = teach_opt.minimize(teach_loss, var_list=g_weights)
        self.teach_op = teach_op
        #gen_op = tf.group(gen_op, teach_op)
        tf.summary.scalar('teacher_lr', teach_lr)

        # Creates op to update time.
        step_op = self.time.assign(self.time + 1)

        # Allows the user to specify sequential vs. simultaneous updates.
        #if self._learn_phase is None:
        #    gan_train_op = gen_op
        #else:
        #    gan_train_op = tf.cond(
        #        tf.equal(tf.mod(self.time, self._learn_phase), 0),
        #        lambda: gen_op,
        #        lambda: dis_op)

        # Updates time every step.
        #self.train_op = tf.group(gan_train_op, step_op)
        self.alphas, self.betas, self.alphas_detect, self.betas_detect, self.generated_captions = self.build_sampler(max_len=30)
        # Creates the log directory and saving objects.
        if self.logdir is None:
            self.logdir = tempfile.mkdtemp()
            sys.stdout.write('logdir: "%s"\n' % self.logdir)
        self.summary_writer = tf.summary.FileWriter(
            self.logdir, self._sess.graph)
        self.summary_op = tf.summary.merge_all()

        self._saver = tf.train.Saver()
        self._sess.run(tf.global_variables_initializer())
        self._built = True
        

    def save(self):
        """Saves the model to the logdir."""

        self._saver.save(self._sess, self.logdir + 'model.ckpt')


    def pre_train_batch(self,features, features_detect, batch):
        """Trains on a single batch of data.
        Args:
            batch: numpy array with shape (batch_size, num_timesteps), where
                values are encoded tokens.
        """

       # self.batch_size, seq_len = batch.shape
        
        feed_dict = {
            self.features: features,
            self.features_detect: features_detect,
            self.captions: batch
           
            #self.text_len_pl: seq_len,
            #self.sample_pl: False,
        }

     
        self._sess.run(self.teach_op, feed_dict=feed_dict)
        


    def generate(self, features, features_detect):
        """Generates a sample from the model.
        Args:
            sample_len: int, length of the sample to generate.
        """
        alphas, betas, alphas_detect, betas_detect, sequence= self._sess.run( [self.alphas, self.betas, self.alphas_detect, self.betas_detect, self.generated_captions], feed_dict = {self.features: features, self.features_detect: features_detect})
        return alphas, betas, alphas_detect, betas_detect, sequence

        self.summary_writer.add_summary(summary, t)
