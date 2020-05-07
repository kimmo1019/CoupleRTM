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
coupling clustering with two labels
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
class RoundtripModel(object):
    def __init__(self, g_net, h_net, dx_net, dy_net, x_sampler, y_sampler, nb_classes, data, pool, batch_size, alpha, beta, is_train):
        self.data = data
        self.g_net = g_net
        self.h_net = h_net
        self.dx_net = dx_net
        self.dy_net = dy_net
        self.x_sampler = x_sampler
        self.y_sampler = y_sampler
        self.A = self.y_sampler.A
        self.nb_classes = nb_classes
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.pool = pool
        self.x_dim = self.dx_net.input_dim
        self.y_dim1 = self.h_net.input_dim1
        self.y_dim2 = self.h_net.input_dim2
        tf.reset_default_graph()


        self.x = tf.placeholder(tf.float32, [None, self.x_dim], name='x')
        self.x_onehot1 = tf.placeholder(tf.float32, [None, self.nb_classes], name='x_onehot1')
        self.x_onehot2 = tf.placeholder(tf.float32, [None, self.nb_classes], name='x_onehot2')
        self.x_combine = tf.concat([self.x,self.x_onehot1,self.x_onehot2],axis=1,name='x_combine')

        self.y1 = tf.placeholder(tf.float32, [None, self.y_dim1], name='y1')
        self.y2 = tf.placeholder(tf.float32, [None, self.y_dim2], name='y2')
        self.y_combine = tf.concat([self.y1,self.y2],axis=1,name='y_combine')

        self.y1_, self.y2_ = self.g_net(self.x_combine,reuse=False)
        self.y_combine_ = tf.concat([self.y1_,self.y2_],axis=1,name='y_combine_')

        self.x_, self.x_onehot1_, self.x_logits1_, self.x_onehot2_, self.x_logits2_ = self.h_net(self.y_combine,reuse=False)#continuous + softmax + before_softmax
        
        self.x__, self.x_onehot1__, self.x_logits1__, self.x_onehot2__, self.x_logits2__  = self.h_net(self.y_combine_)

        self.x_combine_ = tf.concat([self.x_, self.x_onehot1_,self.x_onehot2_],axis=1)
        self.y1__, self.y2__ = self.g_net(self.x_combine_)
        self.y_combine__ = tf.concat([self.y1__,self.y2__],axis=1,name='y_combine__')

        self.dy_ = self.dy_net(self.y_combine_, reuse=False)
        self.dx_ = self.dx_net(self.x_, reuse=False)

        self.l2_loss_x = tf.reduce_mean((self.x - self.x__)**2)
        self.l2_loss_y = tf.reduce_mean((self.y_combine - self.y_combine__)**2)

        #self.CE_loss_x = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.x_onehot, logits=self.x_logits__))
        self.CE_loss_x = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.x_logits1__,labels=self.x_onehot1)) +\
            tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.x_logits2__,labels=self.x_onehot2))
        
        #-log(D(x))
        self.g_loss_adv = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dy_, labels=tf.ones_like(self.dy_)))
        self.h_loss_adv = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dx_, labels=tf.ones_like(self.dx_)))
        #coupling loss
        self.diag = tf.linalg.tensor_diag_part(tf.matmul(tf.matmul(self.y1, self.A),tf.transpose(self.y2)))
        self.ind = tf.reduce_mean(tf.multiply(self.x_onehot1,self.x_onehot2),axis=1)
        self.couple_loss = -tf.reduce_mean(tf.multiply(self.diag,self.ind))

        self.g_loss = self.g_loss_adv + self.alpha*self.l2_loss_x + self.beta*self.l2_loss_y
        self.h_loss = self.h_loss_adv + self.alpha*self.l2_loss_x + self.beta*self.l2_loss_y
        self.g_h_loss = self.g_loss_adv + self.h_loss_adv + self.alpha*(self.l2_loss_x + self.l2_loss_y) + self.beta*self.CE_loss_x+0.001*self.couple_loss


        self.fake_x = tf.placeholder(tf.float32, [None, self.x_dim], name='fake_x')
        self.fake_x_onehot1 = tf.placeholder(tf.float32, [None, self.nb_classes], name='fake_x_onehot1')
        self.fake_x_onehot2 = tf.placeholder(tf.float32, [None, self.nb_classes], name='fake_x_onehot2')
        self.fake_x_combine = tf.concat([self.fake_x,self.fake_x_onehot1,self.fake_x_onehot2],axis=1,name='fake_x_combine')

        self.fake_y = tf.placeholder(tf.float32, [None, self.y_dim1+self.y_dim2], name='fake_y')
        
        self.dx = self.dx_net(self.x)
        self.dy = self.dy_net(self.y_combine)

        self.d_fake_x = self.dx_net(self.fake_x)
        self.d_fake_y = self.dy_net(self.fake_y)

        #-log(D(x))
        self.dx_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dx, labels=tf.ones_like(self.dx))) \
            +tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_fake_x, labels=tf.zeros_like(self.d_fake_x)))
        self.dy_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dy, labels=tf.ones_like(self.dy))) \
            +tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_fake_y, labels=tf.zeros_like(self.d_fake_y)))

        self.d_loss = self.dx_loss + self.dy_loss
 
        #weight clipping
        self.clip_dx = [var.assign(tf.clip_by_value(var, -0.01, 0.01)) for var in self.dx_net.vars]
        self.clip_dy = [var.assign(tf.clip_by_value(var, -0.01, 0.01)) for var in self.dy_net.vars]

        self.lr = tf.placeholder(tf.float32, None, name='learning_rate')
        self.g_h_optim = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5, beta2=0.9) \
                .minimize(self.g_h_loss, var_list=self.g_net.vars+self.h_net.vars)
        self.d_optim = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5, beta2=0.9) \
                .minimize(self.d_loss, var_list=self.dx_net.vars+self.dy_net.vars)

        now = datetime.datetime.now(dateutil.tz.tzlocal())
        self.timestamp = now.strftime('%Y%m%d_%H%M%S')

        self.g_loss_adv_summary = tf.summary.scalar('g_loss_adv',self.g_loss_adv)
        self.h_loss_adv_summary = tf.summary.scalar('h_loss_adv',self.h_loss_adv)
        self.l2_loss_x_summary = tf.summary.scalar('l2_loss_x',self.l2_loss_x)
        self.l2_loss_y_summary = tf.summary.scalar('l2_loss_y',self.l2_loss_y)
        self.dx_loss_summary = tf.summary.scalar('dx_loss',self.dx_loss)
        self.dy_loss_summary = tf.summary.scalar('dy_loss',self.dy_loss)
        self.g_merged_summary = tf.summary.merge([self.g_loss_adv_summary, self.h_loss_adv_summary,\
            self.l2_loss_x_summary,self.l2_loss_y_summary])
        self.d_merged_summary = tf.summary.merge([self.dx_loss_summary,self.dy_loss_summary])

        #graph path for tensorboard visualization
        self.graph_dir = 'graph/cluster_{}_{}_x_dim={}_y_dim={}_alpha={}_beta={}'.format(self.timestamp,self.data,self.x_dim, self.y_dim1, self.alpha, self.beta)
        if not os.path.exists(self.graph_dir) and is_train:
            os.makedirs(self.graph_dir)
        
        #save path for saving predicted data
        self.save_dir = 'data/cluster_{}_{}_x_dim={}_y_dim={}_alpha={}_beta={}'.format(self.timestamp,self.data,self.x_dim, self.y_dim1, self.alpha, self.beta)
        if not os.path.exists(self.save_dir) and is_train:
            os.makedirs(self.save_dir)

        self.saver = tf.train.Saver(max_to_keep=500)

        run_config = tf.ConfigProto()
        run_config.gpu_options.per_process_gpu_memory_fraction = 1.0
        run_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=run_config)


    def train(self, epochs, patience):
        #data_y, label_y = self.y_sampler.load_all()
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
                bx, bx_onehot1, bx_onehot2 = self.x_sampler.train(batch_size,True)
                by1 = random.sample(data_y1_train,self.batch_size)
                by2 = random.sample(data_y2_train,self.batch_size)
                #update G and get generated fake data
                fake_bx, fake_by, g_summary, _ = self.sess.run([self.x_,self.y_combine_,self.g_merged_summary ,self.g_h_optim], feed_dict={self.x: bx, self.x_onehot1: bx_onehot1, self.x_onehot2: bx_onehot2, self.y1: by1, self.y2: by2, self.lr:lr})
                self.summary_writer.add_summary(g_summary,counter)
                #random choose one batch from the previous 50 batches
                #[fake_bx,fake_by] = self.pool([fake_bx,fake_by])
                #update D
                d_summary,_ = self.sess.run([self.d_merged_summary, self.d_optim], feed_dict={self.x: bx, self.y1: by1, self.y2: by2, self.fake_x: fake_bx, self.fake_y: fake_by,self.lr:lr})
                self.summary_writer.add_summary(d_summary,counter)
                #quick test on a random batch data
                if counter % 100 == 0:
                    # bx, bx_onehot = self.x_sampler.train(batch_size)
                    # by = self.y_sampler.train(batch_size)
                    g_loss_adv, h_loss_adv, CE_loss, couple_loss, l2_loss_x, l2_loss_y, g_loss, \
                        h_loss, g_h_loss, fake_bx, fake_by = self.sess.run(
                        [self.g_loss_adv, self.h_loss_adv, self.CE_loss_x, self.couple_loss, self.l2_loss_x, self.l2_loss_y, \
                        self.g_loss, self.h_loss, self.g_h_loss, self.x_, self.y_combine_],
                        feed_dict={self.x: bx, self.x_onehot1: bx_onehot1, self.x_onehot2: bx_onehot2, self.y1: by1,self.y2: by2}
                    )
                    dx_loss, dy_loss, d_loss = self.sess.run([self.dx_loss, self.dy_loss, self.d_loss], \
                        feed_dict={self.x: bx, self.x_onehot1: bx_onehot1,self.x_onehot2: bx_onehot2, self.y1: by1, self.y2: by2, self.fake_x: fake_bx, self.fake_y: fake_by})

                    print('Epoch [%d] Iter [%d] Time [%.4f] g_loss_adv [%.4f] h_loss_adv [%.4f] CE_loss [%.4f] l2_loss_x [%.4f] \
                        l2_loss_y [%.4f] g_loss [%.4f] h_loss [%.4f] g_h_loss [%.4f] dx_loss [%.4f] \
                        dy_loss [%.4f] d_loss [%.4f] couple_loss[%.4f]' %
                        (epoch, counter, time.time() - start_time, g_loss_adv, h_loss_adv, CE_loss, l2_loss_x, l2_loss_y, \
                        g_loss, h_loss, g_h_loss, dx_loss, dy_loss, d_loss, couple_loss))      
                counter+=1

            if (epoch+1) % 20 == 0:
                if epoch+1 == epochs:
                    self.evaluate(timestamp,epoch,True)
                else:
                    self.evaluate(timestamp,epoch)

    def evaluate(self,timestamp,epoch,run_kmeans=False):
        data_y1, label_y1, data_y2, label_y2= self.y_sampler.load_all()
        N = data_y1.shape[0]
        data_x_, data_x_onehot1_, data_x_onehot2_= self.predict_x(data_y1, data_y2)
        np.savez('{}/data_at_{}.npz'.format(self.save_dir, epoch+1),data_x_,data_x_onehot1_,data_x_onehot2_,label_y1,label_y2)
        label_infer1 = np.argmax(data_x_onehot1_, axis=1)
        label_infer2 = np.argmax(data_x_onehot2_, axis=1)

        purity1 = metric.compute_purity(label_infer1, label_y1)
        nmi1 = normalized_mutual_info_score(label_y1, label_infer1)
        ari1 = adjusted_rand_score(label_y1, label_infer1)

        purity2 = metric.compute_purity(label_infer2, label_y2)
        nmi2 = normalized_mutual_info_score(label_y2, label_infer2)
        ari2 = adjusted_rand_score(label_y2, label_infer2)

        self.cluster_heatmap(epoch, label_infer1, label_y1, label_infer2, label_y2)
        print('RTM scRNA-seq: Purity = {}, NMI = {}, ARI = {}'.format(purity1,nmi1,ari1))
        print('RTM scATAC-seq: Purity = {}, NMI = {}, ARI = {}'.format(purity2,nmi2,ari2))
        f = open('%s/log.txt'%self.save_dir,'a+')
        f.write('%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%d\n'%(purity1,nmi1,ari1,purity2,nmi2,ari2,epoch))
        f.close()
        #k-means
        if run_kmeans:
            #scRNA-seq
            km = KMeans(n_clusters=nb_classes, random_state=0).fit(data_y1)
            label_kmeans = km.labels_
            purity = metric.compute_purity(label_kmeans, label_y1)
            nmi = normalized_mutual_info_score(label_y1, label_kmeans)
            ari = adjusted_rand_score(label_y1, label_kmeans)
            print('K-means scRNA-seq: Purity = {}, NMI = {}, ARI = {}'.format(purity,nmi,ari))
            f = open('%s/log.txt'%self.save_dir,'a+')
            f.write('%.4f\t%.4f\t%.4f\n'%(purity,nmi,ari))
            #scATAC-seq
            km = KMeans(n_clusters=nb_classes, random_state=0).fit(data_y2)
            label_kmeans = km.labels_
            purity = metric.compute_purity(label_kmeans, label_y2)
            nmi = normalized_mutual_info_score(label_y2, label_kmeans)
            ari = adjusted_rand_score(label_y2, label_kmeans)
            print('K-means scATAC-seq: Purity = {}, NMI = {}, ARI = {}'.format(purity,nmi,ari))
            f = open('%s/log.txt'%self.save_dir,'a+')
            f.write('%.4f\t%.4f\t%.4f\n'%(purity,nmi,ari))
            f.close() 
    
    def cluster_heatmap(self,epoch,label_pre1,label_true1,label_pre2,label_true2):
        assert len(label_pre1)==len(label_true1)
        assert len(label_pre2)==len(label_true2)
        #scRNA-seq
        confusion_mat = np.zeros((self.nb_classes,self.nb_classes))
        for i in range(len(label_true1)):
            confusion_mat[label_pre1[i]][label_true1[i]] += 1
        plt.figure()
        df = pd.DataFrame(confusion_mat)
        sns.heatmap(df,annot=True, cmap="Blues",annot_kws={"size": 10})
        plt.savefig('%s/heatmap_%d_scRNA.png'%(self.save_dir,epoch),dpi=200)
        plt.close()
        #scATAC-seq
        confusion_mat = np.zeros((self.nb_classes,self.nb_classes))
        for i in range(len(label_true2)):
            confusion_mat[label_pre2[i]][label_true2[i]] += 1
        plt.figure()
        df = pd.DataFrame(confusion_mat)
        sns.heatmap(df,annot=True, cmap="Blues",annot_kws={"size": 10})
        plt.savefig('%s/heatmap_%d_scATAC.png'%(self.save_dir,epoch),dpi=200)
        plt.close()


    #predict with y_=G(x)
    def predict_y(self, x, x_onehot, bs=256):
        assert x.shape[-1] == self.x_dim
        N = x.shape[0]
        y_pred = np.zeros(shape=(N, self.y_dim)) 
        for b in range(int(np.ceil(N*1.0 / bs))):
            if (b+1)*bs > N:
               ind = np.arange(b*bs, N)
            else:
               ind = np.arange(b*bs, (b+1)*bs)
            batch_x = x[ind, :]
            batch_x_onehot = x_onehot[ind, :]
            batch_y_ = self.sess.run(self.y_, feed_dict={self.x:batch_x, self.x_onehot:batch_x_onehot})
            y_pred[ind, :] = batch_y_
        return y_pred
    
    #predict with x_=H(y)
    def predict_x(self,y1,y2,bs=256):
        assert y1.shape[-1] == self.y_dim1
        assert y2.shape[-1] == self.y_dim2
        N1 = y1.shape[0]
        N2 = y2.shape[0]
        N = max(N1,N2)
        if N1 > N2:
            indx = np.random.choice(N2, size=N-N2, replace=True)
            y2 = np.concatenate([y2,y2[indx,:]],axis=0) 
        else:
            indx = np.random.choice(N1, size=N-N1, replace=True)
            y1 = np.concatenate([y1,y1[indx,:]],axis=0) 
            
        x_pred = np.zeros(shape=(N, self.x_dim)) 
        x_onehot1 = np.zeros(shape=(N, self.nb_classes)) 
        x_onehot2 = np.zeros(shape=(N, self.nb_classes)) 
        for b in range(int(np.ceil(N*1.0 / bs))):
            if (b+1)*bs > N:
               ind = np.arange(b*bs, N)
            else:
               ind = np.arange(b*bs, (b+1)*bs)
            batch_y1 = y1[ind, :]
            batch_y2 = y2[ind, :]
            batch_x_,batch_x_onehot1_,batch_x_onehot2_ = self.sess.run([self.x_, self.x_onehot1_,self.x_onehot2_], feed_dict={self.y1:batch_y1,self.y2:batch_y2})
            x_pred[ind, :] = batch_x_
            x_onehot1[ind, :] = batch_x_onehot1_
            x_onehot2[ind, :] = batch_x_onehot2_
        if N1 > N2:
            return x_pred, x_onehot1, x_onehot2[:N2,:]
        else:
            return x_pred, x_onehot1[:N1,:], x_onehot2


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
    timestamp = args.timestamp
    is_train = args.train
    #g_net = model.Generator(input_dim=x_dim,output_dim = y_dim,name='g_net',nb_layers=10,nb_units=512)
    #h_net = model.Encoder(input_dim=y_dim,output_dim = x_dim+nb_classes,feat_dim=x_dim,name='h_net',nb_layers=10,nb_units=256)
    #dx_net = model.Discriminator(input_dim=x_dim,name='dx_net',nb_layers=2,nb_units=128)
    #dy_net = model.Discriminator(input_dim=y_dim,name='dy_net',nb_layers=4,nb_units=256)
    g_net = model.GeneratorCouple(input_dim=x_dim+2*nb_classes,feat_dim=x_dim,output_dim1=y_dim1, output_dim2=y_dim2,name='g_net',nb_layers=10,nb_units=256,concat_every_fcl=False)
    h_net = model.EncoderCouple(input_dim1=y_dim1,input_dim2=y_dim2,output_dim = x_dim+2*nb_classes,feat_dim=x_dim,name='h_net',nb_layers=10,nb_units=256)
    dx_net = model.Discriminator(input_dim=x_dim,name='dx_net',nb_layers=2,nb_units=256)
    dy_net = model.Discriminator(input_dim=y_dim1+y_dim2,name='dy_net',nb_layers=2,nb_units=256)
    pool = util.DataPool()

    #xs = util.Mixture_sampler_v2(nb_classes=nb_classes,N=10000,dim=x_dim,sd=1)
    xs = util.Mixture_sampler(nb_classes=nb_classes,N=10000,dim=x_dim,sd=1)
    #ys = util.DataSampler() #scRNA-seq data
    ys = util.RA4CoupleSampler()
    #ys = util.scATAC_Sampler()
    #ys = util.GMM_sampler(N=10000,n_components=nb_classes,dim=y_dim,sd=8)


    RTM = RoundtripModel(g_net, h_net, dx_net, dy_net, xs, ys, nb_classes, data, pool, batch_size, alpha, beta, is_train)

    if args.train:
        RTM.train(epochs=epochs, patience=patience)
    else:
        print('Attempting to Restore Model ...')
        if timestamp == '':
            RTM.load(pre_trained=True)
            timestamp = 'pre-trained'
        else:
            RTM.load(pre_trained=False, timestamp = timestamp, epoch = epochs-1)
            
