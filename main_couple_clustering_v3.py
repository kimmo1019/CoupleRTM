from __future__ import division
import os,sys
import time
import dateutil.tz
import datetime
import argparse
import importlib
import tensorflow as tf
tf.set_random_seed(0)
import numpy as np
import random
import copy
import math
import util
import metric
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score

import pandas as pd 
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
'''
It contains two Dx
two Roundtrip model using the same latent variable (co-embedding)
Instructions: Roundtrip model for clustering
    x,y - data drawn from base density and observation data (target density)
    y_  - learned distribution by G(.), namely y_=G(x)
    x_  - learned distribution by H(.), namely x_=H(y)
    y__ - reconstructed distribution, y__ = G(H(y))
    x__ - reconstructed distribution, x__ = H(G(y))
    G(.)  - generator network for mapping x space to y space
    H(.)  - generator network for mapping y space to x space
    Dx(.) - discriminator network in x space (latent space)
    Dy(.) - discriminator network in y space (observation space)
'''
class CoupleRTM(object):
    def __init__(self, g_net1, g_net2, h_net1, h_net2, dx_net1, dx_net2, dy_net1, dy_net2, x_sampler, y_sampler, nb_classes, data, pool, batch_size, alpha, beta, gamma, is_train):
        self.data = data
        self.g_net1 = g_net1
        self.g_net2 = g_net2
        self.h_net1 = h_net1
        self.h_net2 = h_net2
        self.dx_net1 = dx_net1
        self.dx_net2 = dx_net2
        self.dy_net1 = dy_net1
        self.dy_net2 = dy_net2
        self.x_sampler = x_sampler
        self.y_sampler = y_sampler
        self.A = self.y_sampler.A
        self.nb_classes = nb_classes
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.pool = pool
        self.x_dim = self.dx_net1.input_dim
        self.y_dim1 = self.dy_net1.input_dim
        self.y_dim2 = self.dy_net2.input_dim
        tf.reset_default_graph()


        self.x = tf.placeholder(tf.float32, [None, self.x_dim], name='x')
        self.x_onehot = tf.placeholder(tf.float32, [None, self.nb_classes], name='x_onehot')
        self.x_combine = tf.concat([self.x,self.x_onehot],axis=1,name='x_combine')

        self.y1 = tf.placeholder(tf.float32, [None, self.y_dim1], name='y1')
        self.y2 = tf.placeholder(tf.float32, [None, self.y_dim2], name='y2')

        self.y1_ = self.g_net1(self.x_combine,reuse=False)
        self.y2_ = self.g_net2(self.x_combine,reuse=False)

        self.x1_, self.x_onehot1_, self.x_logits1_ = self.h_net1(self.y1,reuse=False)#continuous + softmax + before_softmax
        self.x2_, self.x_onehot2_, self.x_logits2_ = self.h_net2(self.y2,reuse=False)
        
        self.x1__, self.x_onehot1__, self.x_logits1__ = self.h_net1(self.y1_)
        self.x2__, self.x_onehot2__, self.x_logits2__ = self.h_net2(self.y2_)

        self.x_combine1_ = tf.concat([self.x1_, self.x_onehot1_],axis=1)
        self.x_combine2_ = tf.concat([self.x2_, self.x_onehot2_],axis=1)
        self.y1__ = self.g_net1(self.x_combine1_)
        self.y2__ = self.g_net2(self.x_combine2_)

        self.dy1_ = self.dy_net1(self.y1_, reuse=False)
        self.dy2_ = self.dy_net2(self.y2_, reuse=False)

        #check this later, use one Dx or two Dx?
        self.dx1_ = self.dx_net1(self.x1_, reuse=False)
        self.dx2_ = self.dx_net2(self.x2_, reuse=False)

        self.l2_loss_x = (tf.reduce_mean((self.x - self.x1__)**2)+\
            tf.reduce_mean((self.x - self.x2__)**2))/2.0
        self.l2_loss_y = (tf.reduce_mean((self.y1 - self.y1__)**2)+\
            tf.reduce_mean((self.y2 - self.y2__)**2))/2.0

        #self.CE_loss_x = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.x_onehot, logits=self.x_logits__))
        self.CE_loss_x = (tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.x_logits1__,labels=self.x_onehot))+\
            tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.x_logits2__,labels=self.x_onehot)))/2.0
        
        #Couple loss
        self.couple_loss = -(tf.linalg.trace(tf.matmul(tf.matmul(self.y1_, self.A),tf.transpose(self.y2_))))*1.0/self.batch_size

        #-log(D(x))
        self.g_loss_adv = (tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dy1_, labels=tf.ones_like(self.dy1_)))+\
            tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dy2_, labels=tf.ones_like(self.dy2_))))/2.0
        
        self.h_loss_adv = (tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dx1_, labels=tf.ones_like(self.dx1_)))+\
            tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dx2_, labels=tf.ones_like(self.dx2_))))/2.0

        self.g_loss = self.g_loss_adv + self.alpha*self.l2_loss_x + self.beta*self.l2_loss_y
        self.h_loss = self.h_loss_adv + self.alpha*self.l2_loss_x + self.beta*self.l2_loss_y
        self.g_h_loss = self.g_loss_adv + self.h_loss_adv + self.alpha*(self.l2_loss_x + self.l2_loss_y) +\
             self.beta*self.CE_loss_x + self.gamma*self.couple_loss


        self.fake_x1 = tf.placeholder(tf.float32, [None, self.x_dim], name='fake_x1')
        self.fake_x_onehot1 = tf.placeholder(tf.float32, [None, self.nb_classes], name='fake_x_onehot1')
        self.fake_x_combine1 = tf.concat([self.fake_x1,self.fake_x_onehot1],axis=1,name='fake_x_combine1')
        self.fake_x2 = tf.placeholder(tf.float32, [None, self.x_dim], name='fake_x2')
        self.fake_x_onehot2 = tf.placeholder(tf.float32, [None, self.nb_classes], name='fake_x_onehot2')
        self.fake_x_combine2 = tf.concat([self.fake_x2,self.fake_x_onehot2],axis=1,name='fake_x_combine2')

        self.fake_y1 = tf.placeholder(tf.float32, [None, self.y_dim1], name='fake_y1')
        self.fake_y2 = tf.placeholder(tf.float32, [None, self.y_dim2], name='fake_y2')
        
        self.dx1 = self.dx_net1(self.x)
        self.dx2 = self.dx_net2(self.x)
        self.dy1 = self.dy_net1(self.y1)
        self.dy2 = self.dy_net2(self.y2)

        self.d_fake_x1 = self.dx_net1(self.fake_x1)
        self.d_fake_x2 = self.dx_net2(self.fake_x2)
        self.d_fake_y1 = self.dy_net1(self.fake_y1)
        self.d_fake_y2 = self.dy_net2(self.fake_y2)

        #-log(D(x))
        self.dx_loss = (tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dx1, labels=tf.ones_like(self.dx1))) \
            +tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dx2, labels=tf.ones_like(self.dx2)))
            +tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_fake_x1, labels=tf.zeros_like(self.d_fake_x1))) \
            +tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_fake_x2, labels=tf.zeros_like(self.d_fake_x2))))/4.0

        self.dy_loss = (tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dy1, labels=tf.ones_like(self.dy1))) \
            +tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_fake_y1, labels=tf.zeros_like(self.d_fake_y1))) \
            +tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dy2, labels=tf.ones_like(self.dy2))) \
            +tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_fake_y2, labels=tf.zeros_like(self.d_fake_y2))))/4.0

        self.d_loss = self.dx_loss + self.dy_loss
 
        #weight clipping
        self.clip_dx1 = [var.assign(tf.clip_by_value(var, -0.01, 0.01)) for var in self.dx_net1.vars]
        self.clip_dx2 = [var.assign(tf.clip_by_value(var, -0.01, 0.01)) for var in self.dx_net2.vars]
        self.clip_dy1 = [var.assign(tf.clip_by_value(var, -0.01, 0.01)) for var in self.dy_net1.vars]
        self.clip_dy2 = [var.assign(tf.clip_by_value(var, -0.01, 0.01)) for var in self.dy_net2.vars]

        self.lr = tf.placeholder(tf.float32, None, name='learning_rate')
        self.g_h_optim = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5, beta2=0.9) \
                .minimize(self.g_h_loss, var_list=self.g_net1.vars+self.h_net1.vars+self.g_net2.vars+self.h_net2.vars)
        self.d_optim = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5, beta2=0.9) \
                .minimize(self.d_loss, var_list=self.dx_net1.vars+self.dx_net2.vars+self.dy_net1.vars+self.dy_net2.vars)

        now = datetime.datetime.now(dateutil.tz.tzlocal())
        self.timestamp = now.strftime('%Y%m%d_%H%M%S')

        self.g_loss_adv_summary = tf.summary.scalar('g_loss_adv',self.g_loss_adv)
        self.h_loss_adv_summary = tf.summary.scalar('h_loss_adv',self.h_loss_adv)
        self.l2_loss_x_summary = tf.summary.scalar('l2_loss_x',self.l2_loss_x)
        self.l2_loss_y_summary = tf.summary.scalar('l2_loss_y',self.l2_loss_y)
        self.dx_loss_summary = tf.summary.scalar('dx_loss',self.dx_loss)
        self.dy_loss_summary = tf.summary.scalar('dy_loss',self.dy_loss)
        self.CE_loss_summary = tf.summary.scalar('CE_loss',self.CE_loss_x)
        self.couple_loss_summary = tf.summary.scalar('couple_loss',self.couple_loss)
        self.g_merged_summary = tf.summary.merge([self.g_loss_adv_summary, self.h_loss_adv_summary,\
            self.l2_loss_x_summary, self.l2_loss_y_summary, self.CE_loss_summary,self.couple_loss_summary])
        self.d_merged_summary = tf.summary.merge([self.dx_loss_summary,self.dy_loss_summary])

        #graph path for tensorboard visualization
        self.graph_dir = 'graph/cluster_{}_{}_x_dim={}_y_dim1={}_y_dim2={}_alpha={}_beta={}_gamma={}'.format(self.timestamp,self.data,self.x_dim, self.y_dim1, self.y_dim2, self.alpha, self.beta, self.gamma)
        if not os.path.exists(self.graph_dir) and is_train:
            os.makedirs(self.graph_dir)
        
        #save path for saving predicted data
        self.save_dir = 'data/cluster_{}_{}_x_dim={}_y_dim1={}_y_dim2={}_alpha={}_beta={}_gamma={}'.format(self.timestamp,self.data,self.x_dim, self.y_dim1, self.y_dim2, self.alpha, self.beta, self.gamma)
        if not os.path.exists(self.save_dir) and is_train:
            os.makedirs(self.save_dir)

        self.saver = tf.train.Saver(max_to_keep=500)

        run_config = tf.ConfigProto()
        run_config.gpu_options.per_process_gpu_memory_fraction = 1.0
        run_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=run_config)


    def train(self, epochs, patience):
        data_y1_train = self.y_sampler.load_all()[0]
        data_y2_train = self.y_sampler.load_all()[2]
        counter = 1
        self.sess.run(tf.global_variables_initializer())
        self.summary_writer=tf.summary.FileWriter(self.graph_dir,graph=tf.get_default_graph())
        start_time = time.time()
        for epoch in range(epochs):
            lr = 1e-4 #if epoch < epochs/2 else 1e-4 #*float(epochs-epoch)/float(epochs-epochs/2)
            batch_idxs = max(len(data_y1_train),len(data_y2_train)) // self.batch_size
            for idx in range(batch_idxs):
                bx, bx_onehot = self.x_sampler.train(batch_size)
                by1 = random.sample(data_y1_train,self.batch_size)
                by2 = random.sample(data_y2_train,self.batch_size)
                #update G and get generated fake data
                fake_bx1,fake_bx2,fake_by1,fake_by2,g_summary, _ = self.sess.run([self.x1_,self.x2_, self.y1_,self.y2_, self.g_merged_summary ,self.g_h_optim], feed_dict={self.x: bx, self.x_onehot: bx_onehot, self.y1: by1, self.y2: by2, self.lr:lr})
                self.summary_writer.add_summary(g_summary,counter)
                #random choose one batch from the previous 50 batches
                #[fake_bx1,fake_bx2,fake_by1,fake_by2] = self.pool([fake_bx1,fake_bx2,fake_by1,fake_by2])
                #update D
                d_summary,_ = self.sess.run([self.d_merged_summary, self.d_optim], feed_dict={self.x: bx, self.y1: by1, self.y2: by2,self.fake_x1: fake_bx1, self.fake_x2: fake_bx2,self.fake_y1: fake_by1,self.fake_y2: fake_by2,self.lr:lr})
                self.summary_writer.add_summary(d_summary,counter)
                #quick test on a batch data
                if counter % 100 == 0:
                    g_loss_adv, h_loss_adv, CE_loss,couple_loss,l2_loss_x, l2_loss_y, g_loss, \
                        h_loss, g_h_loss, fake_bx1, fake_bx2, fake_by1, fake_by2 = self.sess.run(
                        [self.g_loss_adv, self.h_loss_adv, self.CE_loss_x,self.couple_loss, self.l2_loss_x, self.l2_loss_y, \
                        self.g_loss, self.h_loss, self.g_h_loss, self.x1_,self.x2_, self.y1_, self.y2_],
                        feed_dict={self.x: bx, self.x_onehot: bx_onehot, self.y1: by1, self.y2: by2}
                    )
                    dx_loss, dy_loss, d_loss = self.sess.run([self.dx_loss, self.dy_loss, self.d_loss], \
                        feed_dict={self.x: bx, self.x_onehot: bx_onehot, self.y1: by1,self.y2:by2, self.fake_x1: fake_bx1,self.fake_x2: fake_bx2, self.fake_y1: fake_by1,self.fake_y2: fake_by2})

                    print('Epoch [%d] Iter [%d] Time [%.4f] g_loss_adv [%.4f] h_loss_adv [%.4f] CE_loss [%.4f] couple_loss [%.4f] l2_loss_x [%.4f] \
                        l2_loss_y [%.4f] g_loss [%.4f] h_loss [%.4f] g_h_loss [%.4f] dx_loss [%.4f] \
                        dy_loss [%.4f] d_loss [%.4f]' %
                        (epoch, counter, time.time() - start_time, g_loss_adv, h_loss_adv, CE_loss, couple_loss, l2_loss_x, l2_loss_y, \
                        g_loss, h_loss, g_h_loss, dx_loss, dy_loss, d_loss))                 
                counter+=1

            if (epoch+1) % 10 == 0:
                if epoch+1 == epochs:
                    self.evaluate(timestamp,epoch,True)
                else:
                    self.evaluate(timestamp,epoch)

    def evaluate(self,timestamp,epoch,run_kmeans=False):
        data_y1, label_y1, data_y2, label_y2= self.y_sampler.load_all()
        data_x1_, data_x_onehot1_ = self.predict_x1(data_y1)
        data_x2_, data_x_onehot2_ = self.predict_x2(data_y2)
        np.savez('{}/data_at_{}.npz'.format(self.save_dir, epoch+1),data_x1_,data_x_onehot1_,label_y1,data_x2_,data_x_onehot2_,label_y2)
        #scRNA-seq
        label_infer1 = np.argmax(data_x_onehot1_, axis=1)
        purity1 = metric.compute_purity(label_infer1, label_y1)
        nmi1 = normalized_mutual_info_score(label_y1, label_infer1)
        ari1 = adjusted_rand_score(label_y1, label_infer1)
        self.cluster_heatmap(epoch, label_infer1, label_y1,'scRNA')
        print('CoupleRTM scRNA-seq: NMI = {}, ARI = {}, Purity = {}'.format(nmi1,ari1,purity1))
        #scATAC-seq
        label_infer2 = np.argmax(data_x_onehot2_, axis=1)
        purity2 = metric.compute_purity(label_infer2, label_y2)
        nmi2 = normalized_mutual_info_score(label_y2, label_infer2)
        ari2 = adjusted_rand_score(label_y2, label_infer2)
        self.cluster_heatmap(epoch, label_infer2, label_y2,'scATAC')
        print('CoupleRTM scATAC-seq: NMI = {}, ARI = {}, Purity = {}'.format(nmi2,ari2,purity2))


        f = open('%s/log.txt'%self.save_dir,'a+')
        f.write('%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%d\n'%(nmi1,ari1,purity1,nmi2,ari2,purity2,epoch))
        f.close()
        #k-means
        if run_kmeans:
            #scRNA-seq
            km = KMeans(n_clusters=nb_classes, random_state=0).fit(data_y1)
            label_kmeans = km.labels_
            purity = metric.compute_purity(label_kmeans, label_y1)
            nmi = normalized_mutual_info_score(label_y1, label_kmeans)
            ari = adjusted_rand_score(label_y1, label_kmeans)
            print('K-means scRNA-seq: NMI = {}, ARI = {}, Purity = {}'.format(nmi,ari,purity))
            f = open('%s/log.txt'%self.save_dir,'a+')
            f.write('K-means scRNA-seq: NMI = {}, ARI = {}, Purity = {}'.format(nmi,ari,purity))
            #scATAC-seq
            km = KMeans(n_clusters=nb_classes, random_state=0).fit(data_y2)
            label_kmeans = km.labels_
            purity = metric.compute_purity(label_kmeans, label_y2)
            nmi = normalized_mutual_info_score(label_y2, label_kmeans)
            ari = adjusted_rand_score(label_y2, label_kmeans)
            print('K-means scATAC-seq: NMI = {}, ARI = {}, Purity = {}\n'.format(nmi,ari,purity))
            f = open('%s/log.txt'%self.save_dir,'a+')
            f.write('K-means scATAC-seq: NMI = {}, ARI = {}, Purity = {}'.format(nmi,ari,purity))
            f.close() 
    
    def cluster_heatmap(self,epoch,label_pre,label_true,suffix=''):
        assert len(label_pre)==len(label_true)
        confusion_mat = np.zeros((self.nb_classes,self.nb_classes))
        for i in range(len(label_true)):
            confusion_mat[label_pre[i]][label_true[i]] += 1
        plt.figure()
        df = pd.DataFrame(confusion_mat)
        sns.heatmap(df,annot=True, cmap="Blues")
        plt.savefig('%s/heatmap_%d_%s.png'%(self.save_dir,epoch,suffix),dpi=200)
        plt.close()


    #predict with y1_=G1(x)
    def predict_y1(self, x, x_onehot, bs=256):
        assert x.shape[-1] == self.x_dim
        N = x.shape[0]
        y_pred = np.zeros(shape=(N, self.y_dim1)) 
        for b in range(int(np.ceil(N*1.0 / bs))):
            if (b+1)*bs > N:
               ind = np.arange(b*bs, N)
            else:
               ind = np.arange(b*bs, (b+1)*bs)
            batch_x = x[ind, :]
            batch_x_onehot = x_onehot[ind, :]
            batch_y_ = self.sess.run(self.y1_, feed_dict={self.x:batch_x, self.x_onehot:batch_x_onehot})
            y_pred[ind, :] = batch_y_
        return y_pred

    #predict with y2_=G2(x)
    def predict_y1(self, x, x_onehot, bs=256):
        assert x.shape[-1] == self.x_dim
        N = x.shape[0]
        y_pred = np.zeros(shape=(N, self.y_dim2)) 
        for b in range(int(np.ceil(N*1.0 / bs))):
            if (b+1)*bs > N:
               ind = np.arange(b*bs, N)
            else:
               ind = np.arange(b*bs, (b+1)*bs)
            batch_x = x[ind, :]
            batch_x_onehot = x_onehot[ind, :]
            batch_y_ = self.sess.run(self.y2_, feed_dict={self.x:batch_x, self.x_onehot:batch_x_onehot})
            y_pred[ind, :] = batch_y_
        return y_pred
    
    #predict with x1_=H1(y)
    def predict_x1(self,y,bs=256):
        assert y.shape[-1] == self.y_dim1
        N = y.shape[0]
        x_pred = np.zeros(shape=(N, self.x_dim)) 
        x_onehot = np.zeros(shape=(N, self.nb_classes)) 
        for b in range(int(np.ceil(N*1.0 / bs))):
            if (b+1)*bs > N:
               ind = np.arange(b*bs, N)
            else:
               ind = np.arange(b*bs, (b+1)*bs)
            batch_y = y[ind, :]
            batch_x_,batch_x_onehot_ = self.sess.run([self.x1_, self.x_onehot1_], feed_dict={self.y1:batch_y})
            x_pred[ind, :] = batch_x_
            x_onehot[ind, :] = batch_x_onehot_
        return x_pred, x_onehot

    #predict with x2_=H2(y)
    def predict_x2(self,y,bs=256):
        assert y.shape[-1] == self.y_dim2
        N = y.shape[0]
        x_pred = np.zeros(shape=(N, self.x_dim)) 
        x_onehot = np.zeros(shape=(N, self.nb_classes)) 
        for b in range(int(np.ceil(N*1.0 / bs))):
            if (b+1)*bs > N:
               ind = np.arange(b*bs, N)
            else:
               ind = np.arange(b*bs, (b+1)*bs)
            batch_y = y[ind, :]
            batch_x_,batch_x_onehot_ = self.sess.run([self.x2_, self.x_onehot2_], feed_dict={self.y2:batch_y})
            x_pred[ind, :] = batch_x_
            x_onehot[ind, :] = batch_x_onehot_
        return x_pred, x_onehot

    def save(self,epoch):

        checkpoint_dir = 'checkpoint/cluster_{}_{}_x_dim={}_y_dim={}_alpha={}_beta={}'.format(self.timestamp,self.data,self.x_dim, self.y_dim, self.alpha, self.beta)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, 'model.ckpt'),global_step=epoch)

    def load(self, pre_trained = False, timestamp='',epoch=999):

        if pre_trained == True:
            print('Loading Pre-trained Model...')
            checkpoint_dir = 'pre_trained_models/{}/{}_{}_{}_{}}'.format(self.data, self.x_dim,self.y_dim, self.alpha, self.beta)
        else:
            if timestamp == '':
                print('Best Timestamp not provided.')
                checkpoint_dir = ''
            else:
                checkpoint_dir = 'checkpoint/cluster_{}_{}_x_dim={}_y_dim={}_alpha={}_beta={}'.format(self.timestamp,self.data,self.x_dim, self.y_dim, self.alpha, self.beta)
                self.saver.restore(self.sess, os.path.join(checkpoint_dir, 'model.ckpt-%d'%epoch))
                print('Restored model weights.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--data', type=str, default='cluster')
    parser.add_argument('--model', type=str, default='model')
    parser.add_argument('--K', type=int, default=11)
    parser.add_argument('--dx', type=int, default=10)
    parser.add_argument('--dy1', type=int, default=10)
    parser.add_argument('--dy2', type=int, default=10)
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--alpha', type=float, default=10.0)
    parser.add_argument('--beta', type=float, default=10.0)
    parser.add_argument('--gamma', type=float, default=0.001)
    parser.add_argument('--timestamp', type=str, default='')
    parser.add_argument('--train', type=bool, default=False)
    args = parser.parse_args()
    data = args.data
    model = importlib.import_module(args.model)
    nb_classes = args.K
    x_dim = args.dx
    y_dim1 = args.dy1
    y_dim2 = args.dy2
    batch_size = args.bs
    epochs = args.epochs
    patience = args.patience
    alpha = args.alpha
    beta = args.beta
    gamma = args.gamma
    timestamp = args.timestamp
    is_train = args.train
    #g_net = model.Generator(input_dim=x_dim,output_dim = y_dim,name='g_net',nb_layers=10,nb_units=512)
    #h_net = model.Encoder(input_dim=y_dim,output_dim = x_dim+nb_classes,feat_dim=x_dim,name='h_net',nb_layers=10,nb_units=256)
    #dx_net = model.Discriminator(input_dim=x_dim,name='dx_net',nb_layers=2,nb_units=128)
    #dy_net = model.Discriminator(input_dim=y_dim,name='dy_net',nb_layers=4,nb_units=256)
    g_net1 = model.Generator(input_dim=x_dim,output_dim = y_dim1,name='g_net1',nb_layers=10,nb_units=256,concat_every_fcl=False)
    g_net2 = model.Generator(input_dim=x_dim,output_dim = y_dim2,name='g_net2',nb_layers=10,nb_units=256,concat_every_fcl=False)
    #the last layer of G is linear without activation func, maybe add a relu
    h_net1 = model.Encoder(input_dim=y_dim1,output_dim = x_dim+nb_classes,feat_dim=x_dim,name='h_net1',nb_layers=10,nb_units=256)
    h_net2 = model.Encoder(input_dim=y_dim2,output_dim = x_dim+nb_classes,feat_dim=x_dim,name='h_net2',nb_layers=10,nb_units=256)

    dx_net1 = model.Discriminator(input_dim=x_dim,name='dx_net1',nb_layers=4,nb_units=256)
    dx_net2 = model.Discriminator(input_dim=x_dim,name='dx_net2',nb_layers=4,nb_units=256)
    
    dy_net1 = model.Discriminator(input_dim=y_dim1,name='dy_net1',nb_layers=4,nb_units=256)
    dy_net2 = model.Discriminator(input_dim=y_dim2,name='dy_net2',nb_layers=4,nb_units=256)
    pool = util.DataPool()

    #xs = util.Mixture_sampler_v2(nb_classes=nb_classes,N=10000,dim=x_dim,sd=1)
    xs = util.Mixture_sampler(nb_classes=nb_classes,N=10000,dim=x_dim,sd=1)
    #ys = util.DataSampler() #scRNA-seq data
    #ys = util.RA4_Sampler('scrna')
    ys = util.RA4CoupleSampler()
    #ys = util.scATAC_Sampler()
    #ys = util.GMM_sampler(N=10000,n_components=nb_classes,dim=y_dim,sd=8)


    CRTM = CoupleRTM(g_net1, g_net2, h_net1, h_net2, dx_net1, dx_net2, dy_net1, dy_net2, xs, ys, nb_classes, data, pool, batch_size, alpha, beta, gamma, is_train)

    if args.train:
        CRTM.train(epochs=epochs, patience=patience)
    else:
        print('Attempting to Restore Model ...')
        if timestamp == '':
            CRTM.load(pre_trained=True)
            timestamp = 'pre-trained'
        else:
            CRTM.load(pre_trained=False, timestamp = timestamp, epoch = epochs-1)
            
