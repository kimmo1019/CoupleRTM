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
Two Dx
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
    def __init__(self, g_net, h_net, dx_net, dx_cat_net, dy_net, x_sampler, y_sampler, nb_classes, data, pool, batch_size, alpha, beta, is_train):
        self.data = data
        self.g_net = g_net
        self.h_net = h_net
        self.dx_net = dx_net
        self.dx_cat_net = dx_cat_net
        self.dy_net = dy_net
        self.x_sampler = x_sampler
        self.y_sampler = y_sampler
        self.nb_classes = nb_classes
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.pool = pool
        self.x_dim = self.dx_net.input_dim
        self.y_dim = self.dy_net.input_dim
        tf.reset_default_graph()


        self.x = tf.placeholder(tf.float32, [None, self.x_dim], name='x')
        self.x_onehot = tf.placeholder(tf.float32, [None, self.nb_classes], name='x_onehot')
        
        self.W = tf.Variable(tf.random_normal([self.nb_classes,self.x_dim], stddev=0.35),name="centers")
        
        self.x_combine = tf.concat([self.x,self.x_onehot],axis=1,name='x_combine')
        #self.x_combine = self.x + tf.matmul(self.x_onehot,self.W)

        self.y = tf.placeholder(tf.float32, [None, self.y_dim], name='y')

        self.y_ = self.g_net(self.x_combine,reuse=False)

        self.x_, self.x_onehot_, self.x_logits_ = self.h_net(self.y,reuse=False)#continuous + softmax + before_softmax
        #self.x_onehot_ = 1-tf.sign(tf.reduce_max(self.x_softmax_,axis=1,keepdims=True)-self.x_softmax_)
        
        #tf.random_normal(shape=tf.shape(self.y), mean=0.0, stddev=0.1, dtype=tf.float32)

        self.x__, _ , self.x_logits__ = self.h_net(self.y_)

        self.x_combine_ = tf.concat([self.x_, self.x_onehot_],axis=1,name='x_combine_')
        #self.x_combine_ = self.x_ + tf.matmul(self.x_onehot_,self.W)

        self.y__ = self.g_net(self.x_combine_)

        self.dy_ = self.dy_net(self.y_, reuse=False)
        self.dx_ = self.dx_net(self.x_, reuse=False)

        self.dx_cat_ = self.dx_cat_net(self.x_onehot_, reuse=False)

        self.l2_loss_x = tf.reduce_mean((self.x - self.x__)**2)
        self.l2_loss_y = tf.reduce_mean((self.y - self.y__)**2)

        #distence between different clusters
        thred = 5*self.x_dim
        wij2 = tf.matmul(self.W,tf.transpose(self.W))
        c = tf.constant([1,self.nb_classes], tf.int32)
        wi2 = tf.tile(tf.reshape(tf.reduce_sum(tf.square(self.W),axis=1),[self.nb_classes,1]),c)
        c_t = tf.constant([self.nb_classes,1], tf.int32)
        wj2 = tf.tile(tf.reshape(tf.reduce_sum(tf.square(self.W),axis=1),[1,self.nb_classes]),c_t)
        diff_w  = wi2 - 2*wij2 + wj2#l2 between (mu_i, mu_j)
        diff_w = tf.nn.relu(thred-diff_w)
        self.loss_w = (tf.reduce_sum(diff_w)-tf.trace(diff_w))/(self.nb_classes*1.*(self.nb_classes-1))

        

        #self.CE_loss_x = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.x_onehot, logits=self.x_logits__))
        self.CE_loss_x = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.x_logits__,labels=self.x_onehot))
        
        #-log(sigmoid(D(x))),D(x) larger, loss smaller
        self.g_loss_adv = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dy_, labels=tf.ones_like(self.dy_)))
        self.h_loss_adv = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dx_, labels=tf.ones_like(self.dx_)))
        self.h_cat_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dx_cat_, labels=tf.ones_like(self.dx_cat_)))
        #(1-D(x))^2
        #self.g_loss_adv = tf.reduce_mean((tf.ones_like(self.dy_)  - self.dy_)**2)
        #self.h_loss_adv = tf.reduce_mean((tf.ones_like(self.dx_) - self.dx_)**2)
        #-D(x)
        #self.g_loss_adv = -tf.reduce_mean(self.dy_)
        #self.h_loss_adv = -tf.reduce_mean(self.dx_)

        self.g_loss = self.g_loss_adv + self.alpha*self.l2_loss_x + self.beta*self.l2_loss_y
        self.h_loss = self.h_loss_adv + self.alpha*self.l2_loss_x + self.beta*self.l2_loss_y
        #self.g_h_loss = self.g_loss_adv + self.h_loss_adv + self.alpha*(self.l2_loss_x + self.l2_loss_y) + self.beta*self.CE_loss_x
        self.g_h_loss = self.g_loss_adv + self.h_loss_adv + self.alpha*(self.l2_loss_x) + self.CE_loss_x
        
        self.dx = self.dx_net(self.x)
        self.dx_cat = self.dx_cat_net(self.x_onehot)
        self.dy = self.dy_net(self.y)
        #+tf.random_normal(shape=tf.shape(self.y), mean=0.0, stddev=0.05, dtype=tf.float32)


        #(1-D(x))^2
        # self.dx_loss = (tf.reduce_mean((tf.ones_like(self.dx) - self.dx)**2) \
        #         +tf.reduce_mean((self.d_fake_x)**2))/2.0
        # self.dy_loss = (tf.reduce_mean((tf.ones_like(self.dy) - self.dy)**2) \
        #         +tf.reduce_mean((self.d_fake_y)**2))/2.0
        #-log(sigmoid(D(x)))-log(1-sigmoid(D(G(z))))
        self.dx_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dx, labels=tf.ones_like(self.dx))) \
            +tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dx_, labels=tf.zeros_like(self.dx_)))
        self.dy_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dy, labels=tf.ones_like(self.dy))) \
            +tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dy_, labels=tf.zeros_like(self.dy_)))
        self.dx_cat_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dx_cat, labels=tf.ones_like(self.dx_cat))) \
            +tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dx_cat_, labels=tf.zeros_like(self.dx_cat_)))
        
        #D(x)
        #self.dx_loss = -tf.reduce_mean(self.dx) + tf.reduce_mean(self.d_fake_x)
        #self.dy_loss = -tf.reduce_mean(self.dy) + tf.reduce_mean(self.d_fake_y)

        #self.d_loss = self.dx_loss + self.dx_cat_loss + self.dy_loss 
        self.d_loss = self.dx_loss + self.dx_cat_loss + self.dy_loss 
        

        #weight clipping
        self.clip_dx = [var.assign(tf.clip_by_value(var, -0.01, 0.01)) for var in self.dx_net.vars]
        self.clip_dy = [var.assign(tf.clip_by_value(var, -0.01, 0.01)) for var in self.dy_net.vars]

        self.lr = tf.placeholder(tf.float32, None, name='learning_rate')
        # self.g_h_optim = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5, beta2=0.9) \
        #         .minimize(self.g_h_loss, var_list=self.g_net.vars+self.h_net.vars)
        # self.d_optim = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5, beta2=0.9) \
        #         .minimize(self.d_loss, var_list=self.dx_net.vars+self.dy_net.vars)
        self.recon_optim = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5, beta2=0.9) \
                .minimize(self.l2_loss_y, var_list=self.g_net.vars+self.h_net.vars)
        self.dx_optim = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5, beta2=0.9) \
                .minimize(self.dx_loss, var_list=self.dx_net.vars)
        self.h_optim = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5, beta2=0.9) \
                .minimize(self.h_loss_adv, var_list=self.h_net.vars)
        self.dx_cat_optim = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5, beta2=0.9) \
                .minimize(self.dx_cat_loss, var_list=self.dx_cat_net.vars)
        self.h_cat_optim = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5, beta2=0.9) \
                .minimize(self.h_cat_loss, var_list=self.h_net.vars)
                
        # self.dy_optim = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5, beta2=0.9) \
        #         .minimize(self.dy_loss, var_list=self.dy_net.vars)
        # self.g_optim = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5, beta2=0.9) \
        #         .minimize(self.g_loss_adv, var_list=self.g_net.vars)

        # self.CE_optim = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5, beta2=0.9) \
        #         .minimize(self.CE_loss_x, var_list=self.g_net.vars+self.h_net.vars)
        
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
        self.graph_dir = 'graph/cluster_{}_{}_x_dim={}_y_dim={}_alpha={}_beta={}'.format(self.timestamp,self.data,self.x_dim, self.y_dim, self.alpha, self.beta)
        if not os.path.exists(self.graph_dir) and is_train:
            os.makedirs(self.graph_dir)
        
        #save path for saving predicted data
        self.save_dir = 'data/cluster_{}_{}_x_dim={}_y_dim={}_alpha={}_beta={}'.format(self.timestamp,self.data,self.x_dim, self.y_dim, self.alpha, self.beta)
        if not os.path.exists(self.save_dir) and is_train:
            os.makedirs(self.save_dir)

        self.saver = tf.train.Saver(max_to_keep=500)

        run_config = tf.ConfigProto()
        run_config.gpu_options.per_process_gpu_memory_fraction = 1.0
        run_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=run_config)


    def train(self, epochs, patience):
        #data_y, label_y = self.y_sampler.load_all()
        data_y_train = copy.copy(self.y_sampler.load_all()[0])
        counter = 1
        self.sess.run(tf.global_variables_initializer())
        self.summary_writer=tf.summary.FileWriter(self.graph_dir,graph=tf.get_default_graph())
        start_time = time.time()
        weights = np.ones(self.nb_classes, dtype=np.float64) / float(self.nb_classes)
        for epoch in range(epochs):
            np.random.shuffle(data_y_train)
            lr = 1e-4 #if epoch < epochs/2 else 1e-4 #*float(epochs-epoch)/float(epochs-epochs/2)
            batch_idxs = len(data_y_train) // self.batch_size
            for idx in range(batch_idxs):
                bx, bx_onehot = self.x_sampler.train(batch_size,weights)
                by = data_y_train[self.batch_size*idx:self.batch_size*(idx+1)]
                #by = random.sample(data_y_train,batch_size)
                _, l2_loss_y,CE_loss_x = self.sess.run([self.recon_optim,self.l2_loss_y,self.CE_loss_x], feed_dict={self.x: bx, self.x_onehot: bx_onehot, self.y: by, self.lr:lr})
                _, dx_loss = self.sess.run([self.dx_optim,self.dx_loss], feed_dict={self.x: bx, self.x_onehot: bx_onehot, self.y: by, self.lr:lr})
                _, h_loss_adv = self.sess.run([self.h_optim,self.h_loss_adv], feed_dict={self.x: bx, self.x_onehot: bx_onehot, self.y: by, self.lr:lr})
                #_,dx_cat_loss, h_cat_loss,dy_loss,g_loss_adv= self.sess.run([self.CE_optim,self.dx_cat_loss,self.h_cat_loss,self.dy_loss,self.g_loss_adv], feed_dict={self.x: bx, self.x_onehot: bx_onehot, self.y: by, self.lr:lr})
                _, dx_cat_loss = self.sess.run([self.dx_cat_optim,self.dx_cat_loss], feed_dict={self.x: bx, self.x_onehot: bx_onehot, self.y: by, self.lr:lr})
                _, h_cat_loss = self.sess.run([self.h_cat_optim,self.h_cat_loss], feed_dict={self.x: bx, self.x_onehot: bx_onehot, self.y: by, self.lr:lr})

                #_, dy_loss = self.sess.run([self.dy_optim,self.dy_loss], feed_dict={self.x: bx, self.x_onehot: bx_onehot, self.y: by, self.lr:lr})
                #_, g_loss_adv = self.sess.run([self.g_optim,self.g_loss_adv], feed_dict={self.x: bx, self.x_onehot: bx_onehot, self.y: by, self.lr:lr})


                #self.summary_writer.add_summary(g_summary,counter)
                #    _,_ = self.sess.run([self.clip_dx,self.clip_dy])
                #self.summary_writer.add_summary(d_summary,counter)
                #quick test on a random batch data
                if counter % 100 == 0:
                    print('Epoch [%d] Iter [%d] Time [%.4f] l2_loss_y [%.4f] CE_loss_x [%.4f] dx_loss [%.4f] h_loss_adv [%.4f] dx_cat_loss [%.4f] h_cat_loss [%.4f]' %
                        (epoch, counter, time.time() - start_time, l2_loss_y, CE_loss_x, dx_loss, h_loss_adv, dx_cat_loss, h_cat_loss))                 
                counter += 1

            if (epoch+1) % 1 == 0:
                if epoch+1 == epochs:
                    self.evaluate(timestamp,epoch,True)
                else:
                    self.evaluate(timestamp,epoch)
            weights = self.estimate_weights(use_kmeans=False)
    
    def estimate_weights(self,use_kmeans=False):
        data_y, label_y = self.y_sampler.load_all()
        data_x_, data_x_onehot_ = self.predict_x(data_y)
        if use_kmeans:
            km = KMeans(n_clusters=nb_classes, random_state=0).fit(data_x_)
            label_infer = km.labels_
        else:
            label_infer = np.argmax(data_x_onehot_, axis=1)
        weights = np.empty(self.nb_classes, dtype=np.float32)
        for i in range(self.nb_classes):
            weights[i] = list(label_infer).count(i)  
        return weights/float(np.sum(weights))      


    def evaluate(self,timestamp,epoch,run_kmeans=False):
        data_y, label_y = self.y_sampler.load_all()
        N = data_y.shape[0]
        data_x_, data_x_onehot_ = self.predict_x(data_y)
        np.savez('{}/data_at_{}.npz'.format(self.save_dir, epoch+1),data_x_,data_x_onehot_,label_y)
        label_infer = np.argmax(data_x_onehot_, axis=1)
        purity = metric.compute_purity(label_infer, label_y)
        nmi = normalized_mutual_info_score(label_y, label_infer)
        ari = adjusted_rand_score(label_y, label_infer)
        self.cluster_heatmap(epoch, label_infer, label_y)
        print('RTM: NMI = {}, ARI = {}, Purity = {}'.format(nmi,ari,purity))
        f = open('%s/log.txt'%self.save_dir,'a+')
        f.write('RTM\t%.4f\t%.4f\t%.4f\t%d\n'%(nmi,ari,purity,epoch))
        km = KMeans(n_clusters=nb_classes, random_state=0).fit(data_x_)
        label_kmeans = km.labels_
        purity = metric.compute_purity(label_kmeans, label_y)
        nmi = normalized_mutual_info_score(label_y, label_kmeans)
        ari = adjusted_rand_score(label_y, label_kmeans)
        print('Latent-kmeans: NMI = {}, ARI = {}, Purity = {}'.format(nmi,ari,purity))
        f.write('Latent-kmeans\t%.4f\t%.4f\t%.4f\t%d\n'%(nmi,ari,purity,epoch))
        #k-means
        if run_kmeans:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=10)
            pca.fit(data_y)
            data_pca_y = pca.fit_transform(data_y)

            km = KMeans(n_clusters=nb_classes, random_state=0).fit(data_pca_y)
            label_kmeans = km.labels_
            purity = metric.compute_purity(label_kmeans, label_y)
            nmi = normalized_mutual_info_score(label_y, label_kmeans)
            ari = adjusted_rand_score(label_y, label_kmeans)
            print('PCA + K-means: NMI = {}, ARI = {}, Purity = {}'.format(nmi,ari,purity))
            f.write('PCA+Kmeans%.4f\t%.4f\t%.4f\n'%(nmi,ari,purity))

            km = KMeans(n_clusters=nb_classes, random_state=0).fit(data_y)
            label_kmeans = km.labels_
            purity = metric.compute_purity(label_kmeans, label_y)
            nmi = normalized_mutual_info_score(label_y, label_kmeans)
            ari = adjusted_rand_score(label_y, label_kmeans)
            print('K-means: NMI = {}, ARI = {}, Purity = {}'.format(nmi,ari,purity))
            f.write('Kmeans%.4f\t%.4f\t%.4f\n'%(nmi,ari,purity))
            f.close() 
    
    def cluster_heatmap(self,epoch,label_pre,label_true):
        assert len(label_pre)==len(label_true)
        confusion_mat = np.zeros((self.nb_classes,self.nb_classes))
        for i in range(len(label_true)):
            confusion_mat[label_pre[i]][label_true[i]] += 1
        #columns=[item for item in range(1,11)]
        #index=[item for item in range(1,11)]
        #df = pd.DataFrame(confusion_mat,columns=columns,index=index)
        plt.figure()
        df = pd.DataFrame(confusion_mat)
        sns.heatmap(df,annot=True, cmap="Blues")
        plt.savefig('%s/heatmap_%d.png'%(self.save_dir,epoch),dpi=200)
        plt.close()


    #predict with y_=G(x)
    def predict_y(self, x, x_onehot, bs=128):
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
    def predict_x(self,y,bs=128):
        assert y.shape[-1] == self.y_dim
        N = y.shape[0]
        x_pred = np.zeros(shape=(N, self.x_dim)) 
        x_onehot = np.zeros(shape=(N, self.nb_classes)) 
        for b in range(int(np.ceil(N*1.0 / bs))):
            if (b+1)*bs > N:
               ind = np.arange(b*bs, N)
            else:
               ind = np.arange(b*bs, (b+1)*bs)
            batch_y = y[ind, :]
            batch_x_,batch_x_onehot_ = self.sess.run([self.x_, self.x_onehot_], feed_dict={self.y:batch_y})
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
    parser.add_argument('--dy', type=int, default=10)
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
    y_dim = args.dy
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
    #g_node_list = [100,200,360,360,360,360,360,360,360,360]
    g_net = model.Generator(input_dim=x_dim,output_dim = y_dim,name='g_net',nb_layers=2,nb_units=1000,concat_every_fcl=False)
    #the last layer of G is linear without activation func, maybe add a relu
    h_net = model.Encoder(input_dim=y_dim,output_dim = x_dim+nb_classes,feat_dim=x_dim,name='h_net',nb_layers=2,nb_units=1000)
    dx_net = model.Discriminator(input_dim=x_dim,name='dx_net',nb_layers=2,nb_units=256)
    dx_cat_net = model.Discriminator(input_dim=nb_classes,name='dx_cat_net',nb_layers=2,nb_units=256)
    dy_net = model.Discriminator(input_dim=y_dim,name='dy_net',nb_layers=2,nb_units=500)
    pool = util.DataPool(50)

    #xs = util.Mixture_sampler_v2(nb_classes=nb_classes,N=10000,dim=x_dim,sd=1)
    xs = util.Mixture_sampler(nb_classes=nb_classes,N=10000,dim=x_dim,sd=1)
    #ys = util.DataSampler() #scRNA-seq data
    #ys = util.RA4_Sampler('scatac')
    ys = util.scATAC_Sampler()
    #ys = util.GMM_sampler(N=10000,n_components=nb_classes,dim=y_dim,sd=8)


    RTM = RoundtripModel(g_net, h_net, dx_net, dx_cat_net, dy_net, xs, ys, nb_classes, data, pool, batch_size, alpha, beta, is_train)

    if args.train:
        RTM.train(epochs=epochs, patience=patience)
    else:
        print('Attempting to Restore Model ...')
        if timestamp == '':
            RTM.load(pre_trained=True)
            timestamp = 'pre-trained'
        else:
            RTM.load(pre_trained=False, timestamp = timestamp, epoch = epochs-1)
            
