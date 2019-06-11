from core.model_cascade_2layer_detect import CaptionGenerator
from core.utils import load_coco_data
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from core.utils import *
from core.bleu import evaluate, score, evaluate_part
import cPickle as pickle
import sys
import string
import hickle
def main():
    batch_size = 32
    val_batch_size = 12
    save_every = 1 
    #pretrained_model = None
    with open('./data/train/word_to_idx.pkl', 'rb') as f:
        word_to_idx  = pickle.load(f)   
    model_path = 'model_residue_cascade_attention_detect_10/'
    # load val dataset to print out bleu scores every epoch
    #
    #word_to_idx =1
    sess = tf.Session()
    model = CaptionGenerator(sess, word_to_idx , dim_feature=[49, 2048], dim_embed=512,
                                       dim_hidden=512, n_time_step = 21, prev2out = True, 
                                                 ctx2out=True, alpha_c=1.0, selector=True, dropout=True)
    n_examples = 117208
    val_data = load_coco_data(data_path='./data', split='val')
    n_iters_per_epoch = int(np.ceil(float(n_examples)/batch_size))
    with open('./data/train/train.captions.pkl', 'rb') as f:
        captions  = pickle.load(f) 
    with open('./data/train/train.image.idxs.pkl', 'rb') as f:
        image_idxs  = pickle.load(f)    
    print image_idxs
    val_features = val_data['features']
    print val_features.shape[0]
    n_iters_val = int(np.ceil(float(val_features.shape[0])/val_batch_size))
    model.build()    
    saver = tf.train.Saver()
    #variables = slim.get_variables_to_restore()
    #variables_to_restore = [v for v in variables if string.find(v.name, 'discriminator') == -1]
    #saver = tf.train.Saver(variables_to_restore)
    #if pretrained_model is not None:
        #saver = tf.train.import_meta_graph('./model_residue/model-10.meta')
     #   saver.restore(sess, pretrained_model)
    print 'start pre-traininig'
    for epoch in xrange(1, 10 + 1):
        rand_idxs = np.random.permutation(n_examples)
        captions = captions[rand_idxs]
        image_idxs = image_idxs[rand_idxs]
        for step in xrange(1, n_iters_per_epoch + 1):
            captions_batch = captions[step*batch_size:(step+1)*batch_size]
            image_idxs_batch = image_idxs[step*batch_size:(step+1)*batch_size]
            features_batch = np.empty((batch_size, 49, 2048))
            j = 0       
            for i in image_idxs_batch:
                features_single = hickle.load('./data_residue_single/train/'+ 'train_' + str(i) + '.features.hkl')
                features_batch[j,:] = features_single
                j = j + 1
            
            features_detect_batch = np.empty((batch_size, 10, 4096))
            j = 0       
            for i in image_idxs_batch:
                features_detect_single = hickle.load('./data_residue_detect/train/'+ 'train_' + str(i) + '.features.hkl')
                features_detect_single = features_detect_single[-10:, :]
                features_detect_batch[j,:] = features_detect_single
                j = j + 1



            if captions_batch.shape[0] == batch_size:
                model.pre_train_batch(features_batch, features_detect_batch, captions_batch)
            if step % 10 == 0:
                print 'epoch', epoch
                print 'step', step
            if step % 512 == 0:
                all_gen_cap = np.ndarray((val_features.shape[0], 30))
                for i in range(n_iters_val):
                    features_batch = val_features[i*val_batch_size:(i+1)*val_batch_size]
                    val_detect_batch = np.empty((len(features_batch), 10, 4096))
                    m = 0
                    for j in range(i*val_batch_size, (i+1)*val_batch_size):
                        val_detect_single = hickle.load('./data_residue_detect/val/'+ 'val_' + str(j) + '.features.hkl')
                        val_detect_single = val_detect_single[-10:, :]
                        val_detect_batch[m,:] = val_detect_single 
                        m = m + 1
                    _, _, _, _, gen_cap = model.generate(features_batch, val_detect_batch)
                    all_gen_cap[i*val_batch_size:(i+1)*val_batch_size] = gen_cap
                all_decoded = decode_captions(all_gen_cap, model.idx_to_word)
                save_pickle(all_decoded, "./data/val/val.candidate.captions.pkl")
                scores = evaluate(data_path='./data', split='val', get_scores=True)
                write_bleu(scores=scores, path=model_path, epoch=epoch)
                print "generative captions:%s\n"%all_decoded[0]
               
        if epoch % save_every == 0:
                saver.save(sess, os.path.join(model_path, 'model'), global_step=epoch)
                print "model-%s saved." %(epoch)
    print 'start reinforcement learning!'
    
    for epoch in xrange(1, 0 + 1):
        rand_idxs = np.random.permutation(n_examples)
        captions = captions[rand_idxs]
        image_idxs = image_idxs[rand_idxs]
        for step in xrange(1, n_iters_per_epoch + 1):
            captions_batch = captions[step*batch_size:(step+1)*batch_size]
            image_idxs_batch = image_idxs[step*batch_size:(step+1)*batch_size]
            features_batch = features[image_idxs_batch]
            if captions_batch.shape[0] == batch_size:
                #gen_cap = model.generate(features_batch)
                #decoded_cap = decode_captions(gen_cap, model.idx_to_word)
                #decoded_reference = decode_captions(captions_batch, model.idx_to_word)
                #scores = evaluate_part(candidate = decoded_cap, split = 'train', idx = image_idxs_batch, get_scores=True)
                #reward = (0.5*scores['Bleu_1']  + 0.5*scores['Bleu_2'] + scores['Bleu_3'] + scores['Bleu_4'])/3
                #print reward
                #reward = 1
                t = model.train_batch(features_batch, captions_batch)
            if step % 10 == 0:
                print 'epoch', epoch
                print 'step', step
                print 'time', t
            if step % 1024 ==0:
                ground_truths = captions[image_idxs == image_idxs_batch[0]]
                decoded = decode_captions(ground_truths, model.idx_to_word)
                for j, gt in enumerate(decoded):
                     print "Ground truth %d: %s" %(j+1, gt)                    
                gen_caps = model.generate(features_batch)
                decoded = decode_captions(gen_caps, model.idx_to_word)
                print "Generated caption: %s\n" %decoded[0]

            if step % 1024 ==0:
                all_gen_cap = np.ndarray((val_features.shape[0], 30))
                for i in range(n_iters_val):
                    features_batch = val_features[i*batch_size:(i+1)*batch_size]
                    feed_dict = features_batch
                    gen_cap = model.generate(feed_dict)
                    all_gen_cap[i*batch_size:(i+1)*batch_size] = gen_cap                    
                all_decoded = decode_captions(all_gen_cap, model.idx_to_word)
                save_pickle(all_decoded, "./data/val/val.candidate.captions.pkl")
                scores = evaluate(data_path='./data', split='val', get_scores=True)
                write_bleu(scores=scores, path=model_path, epoch=epoch)
                #print "generative captions:%s\n"%all_decoded[0]
        if epoch % save_every == 0:
                saver.save(sess, os.path.join(model_path, 'reinforcemodel'), global_step=epoch)
                print "model-%s saved." %(epoch)
if __name__ == "__main__":
    main()
