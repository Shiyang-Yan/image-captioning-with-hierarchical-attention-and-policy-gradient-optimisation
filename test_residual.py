import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import skimage.transform
import numpy as np
import time
import os 
import cPickle as pickle
from scipy import ndimage
from core.utils import *
from core.bleu import evaluate    
from core.model_cascade_2layer_detect import CaptionGenerator
def test(data, model, sess, test_model, batch_size, idx_to_word,split='test', attention_visualization=True, save_sampled_captions=True):
        '''
        Args:
            - data: dictionary with the following keys:
            - features: Feature vectors of shape (5000, 196, 512)
            - file_names: Image file names of shape (5000, )
            - captions: Captions of shape (24210, 17) 
            - image_idxs: Indices for mapping caption to image of shape (24210, ) 
            - features_to_captions: Mapping feature to captions (5000, 4~5)
            - split: 'train', 'val' or 'test'
            - attention_visualization: If True, visualize attention weights with images for each sampled word. (ipthon notebook)
            - save_sampled_captions: If True, save sampled captions to pkl file for computing BLEU scores.
        '''

        features = data['features']

        # build a graph to sample captions
        #alphas, betas, sampled_captions = model.generate(max_len=20)    # (N, max_len, L), (N, max_len)
        
       # config = tf.ConfigProto(allow_soft_placement=True)
       # config.gpu_options.allow_growth = True
        #with tf.Session(config=config) as sess:
        if 1:
            model.build()
            saver = tf.train.Saver(max_to_keep=40)
            saver.restore(sess, test_model)
            #features_batch, test_features_batch, image_files = sample_coco_minibatch(data, batch_size)
            #feed_dict = features_batch
            val_features = data['features']
            val_batch_size = batch_size
            n_iters_val = int(np.ceil(float(val_features.shape[0])/val_batch_size))
            all_gen_cap = np.ndarray((val_features.shape[0], 30))
            for i in range(n_iters_val):
                features_batch = val_features[i*val_batch_size:(i+1)*val_batch_size]
                val_detect_batch = np.empty((len(features_batch), 10, 4096))
                m = 0
                for j in range(i*val_batch_size, (i+1)*val_batch_size):
                    val_detect_single = hickle.load('./data_residue_detect/test/'+ 'test_' + str(j) + '.features.hkl')
                    val_detect_single = val_detect_single[-10:, :]
                    val_detect_batch[m,:] = val_detect_single 
                    m = m + 1
                _, _, _, _, gen_cap = model.generate(features_batch, val_detect_batch)
                all_gen_cap[i*val_batch_size:(i+1)*val_batch_size] = gen_cap
            all_decoded = decode_captions(all_gen_cap, idx_to_word)
            save_pickle(all_decoded, "./data/test/test.candidate.captions.pkl")
            scores = evaluate(data_path='./data', split='test', get_scores=True)
            

batch_size = 1013
pretrained_model = 'model_residue_cascade_attention_detect_10/model-9'
with open('data/train/word_to_idx.pkl', 'rb') as f:
    word_to_idx  = pickle.load(f)   
idx_to_word = {i: w for w, i in word_to_idx.iteritems()}
sess = tf.Session()
model = CaptionGenerator(sess, word_to_idx , dim_feature=[49, 2048], dim_embed=512,
                                       dim_hidden=512, n_time_step = 21, prev2out = True, 
                                                 ctx2out=True, alpha_c=1.0, selector=True, dropout=True)

val_data = load_coco_data(data_path='./data', split='test')
test(val_data, model, sess,'model_residue_cascade_attention_detect_10/model-9', batch_size, idx_to_word)
