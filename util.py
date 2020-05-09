from __future__ import division
import scipy.sparse
import scipy.io
import numpy as np
import copy
from scipy.special import polygamma
import scipy.special
from scipy.stats import t, uniform, norm, truncnorm, invgamma, gamma
from scipy import pi
from tqdm import tqdm
import sys
import pandas as pd
from os.path import join
from collections import Counter
import cPickle as pickle
import gzip
import hdf5storage

class RA4CoupleSampler(object):
    def __init__(self):
        #(464,21973)
        self.X_rna = np.loadtxt('datasets/RAd4/scRNA_seq_RAd4.txt',dtype='float32',delimiter='\t',usecols=range(1,465))
        self.X_rna = self.X_rna.T
        self.Y_rna = np.array([int(item)-1 for item in open('datasets/RAd4/Label_RNAseq').readline().strip().split('\t')])
        self.N1 = len(self.Y_rna)
        #(415,23180)
        self.X_atac = np.loadtxt('datasets/RAd4/scATAC_seq_RAd4.txt',dtype='float32',delimiter='\t',usecols=range(1,416))
        self.X_atac = self.X_atac.T
        self.Y_atac = np.array([int(item)-1 for item in open('datasets/RAd4/Label_ATACseq').readline().strip().split('\t')])
        self.N2 = len(self.Y_atac)
        #coupling matrix
        self.A = np.load('datasets/RAd4/couple_mat.npy').astype('float32')
        print np.max(self.A),np.min(self.A)

    def train(self, batch_size):
        indx1 = np.random.randint(low = 0, high = self.N1, size = batch_size)
        indx2 = np.random.randint(low = 0, high = self.N2, size = batch_size)
        return self.X_rna[indx1,:],self.X_atac[indx2,:]
    def load_all(self):
         return self.X_rna, self.Y_rna, self.X_atac, self.Y_atac

class RA4_Sampler(object):
    def __init__(self,dtype='scrna'):
        if dtype == 'scrna':
            #(464,21973)
            self.X = np.loadtxt('datasets/RAd4/scRNA_seq_RAd4.txt',dtype='float32',delimiter='\t',usecols=range(1,465))
            self.X = self.X.T
            self.Y = np.array([int(item)-1 for item in open('datasets/RAd4/Label_RNAseq').readline().strip().split('\t')])
            self.total_size = len(self.Y)
        elif dtype == 'scatac':
            #(415,23180)
            self.X = np.loadtxt('datasets/RAd4/scATAC_seq_RAd4.txt',dtype='float32',delimiter='\t',usecols=range(1,416))
            self.X = self.X.T
            self.Y = np.array([int(item)-1 for item in open('datasets/RAd4/Label_ATACseq').readline().strip().split('\t')])
            self.total_size = len(self.Y)
    def train(self, batch_size, label = False):
        indx = np.random.randint(low = 0, high = self.total_size, size = batch_size)

        if label:
            return self.X[indx, :], self.Y[indx].flatten()
        else:
            return self.X[indx, :]

    def load_all(self):
         return self.X, self.Y


#scATAC data
class scATAC_Sampler(object):
    def __init__(self,name='GMvsHL'):
        #GMvsHL,(105233,700),3 classes
        #GMvsHeK,(104260,748),3 classes
        #InSilico, (68069,1377),6 classes
        self.X = hdf5storage.loadmat('data/scATAC/%s.mat'%name)['count_mat']
        self.Y = np.load('datasets/scATAC/%s_label.npy'%name).astype('int64')
        self.X = self.X*1.0/np.max(self.X)
        self.X = self.X.T

        self.total_size = len(self.Y)
    # for data sampling given batch size
    def train(self, batch_size, label = False):
        indx = np.random.randint(low = 0, high = self.total_size, size = batch_size)

        if label:
            return self.X[indx, :], self.Y[indx].flatten()
        else:
            return self.X[indx, :]

    def load_all(self):
         return self.X, self.Y


#pbmc ~68k single cell RNA-seq data
class DataSampler(object):
    def __init__(self):
        self.total_size = 68260
        #self.X = self._load_gene_mtx()
        #self.Y = self._load_labels()
        #np.savez('data/pbmc68k/data_all.npz',self.X,self.Y)
        if True:
            data = np.load('datasets/pbmc68k/data_5types.npz')
            self.X,self.Y = data['arr_0'], data['arr_1']
            idx = [i for i,item in enumerate(self.Y) if item!=4]
            print len(idx)
            self.X = self.X[idx,:]
            self.Y = self.Y[idx]
            
            
        else:
            data = np.load('datasets/pbmc68k/data_all.npz')
            self.X,self.Y = data['arr_0'], data['arr_1']
            #only keep 1,3,7,8,9 where nb>5000
            # dic_nb = {item:0 for item in range(11)}
            # keep_ind=[]
            # print len(self.Y)
            # for i in range(len(self.Y)):
            #     dic_nb[self.Y[i]]+=1
            #     if dic_nb[self.Y[i]] <= 5000 and self.Y[i] in [1,3,7,8,9]:
            #         keep_ind.append(i)
            # self.X = self.X[keep_ind, :]
            # dic_label = {1:0,3:1,7:2,8:3,9:4}
            # self.Y = self.Y[keep_ind]
            # self.Y = np.array([dic_label[item] for item in self.Y])
            # self.total_size = 25000
            # np.savez('data/pbmc68k/data_5types.npz',self.X,self.Y)


    def _read_mtx(self, filename):
        buf = scipy.io.mmread(filename)
        return buf


    def _load_gene_mtx(self):
        data_path = 'datasets/pbmc68k/filtered_mat.txt'
        data = np.loadtxt(data_path,delimiter=' ',skiprows=1,usecols=range(1,68261))
        data = data.T
        scale = np.max(data)
        data = data / scale
        np.random.seed(0)
        indx = np.random.permutation(np.arange(self.total_size))
        data = data[indx, :]
        return data

    def _load_labels(self):
        data_path = 'datasets/pbmc68k/label_info.txt'
        labels = np.array([int(item.split('\t')[-1].strip()) for item in open(data_path).readlines()])
        np.random.seed(0)
        indx = np.random.permutation(np.arange(self.total_size))
        labels = labels[indx]
        return labels

    # for data sampling given batch size
    def train(self, batch_size, label = False):
        indx = np.random.randint(low = 0, high = self.total_size, size = batch_size)

        if label:
            return self.X[indx, :], self.Y[indx].flatten()
        else:
            return self.X[indx, :]

    def load_all(self):
         return self.X, self.Y

#outliner dataset (http://odds.cs.stonybrook.edu/)
class Outliner_sampler(object):
    def __init__(self,data_path='datasets/Outliner/Shuttle/data.npz'):
        data_dic = np.load(data_path)
        self.X_train, self.X_val,self.X_test,self.label_test = self.normalize(data_dic)
        self.Y=None
        self.nb_train = self.X_train.shape[0]
        self.mean = 0
        self.sd = 0
    def normalize(self,data_dic):
        data = data_dic['arr_0']
        label = data_dic['arr_1']
        N_test = int(0.1*data.shape[0])
        data_test = data[-N_test:]
        label_test = label[-N_test:]
        data = data[0:-N_test]
        N_validate = int(0.1*data.shape[0])
        data_validate = data[-N_validate:]
        data_train = data[0:-N_validate]
        data = np.vstack((data_train, data_validate))
        return data_train, data_validate, data_test, label_test
    def train(self, batch_size, label = False):
        indx = np.random.randint(low = 0, high = self.nb_train, size = batch_size)
        if label:
            return self.X_train[indx, :], self.Y[indx]
        else:
            return self.X_train[indx, :]
    def load_all(self):
        return self.X_train, None


#UCI dataset
class UCI_sampler(object):
    def __init__(self,data_path='datasets/AReM/data.npy'):
        data = np.load(data_path)
        self.X_train, self.X_val,self.X_test = self.normalize(data)
        self.Y=None
        self.nb_train = self.X_train.shape[0]
        self.mean = 0
        self.sd = 0
    def normalize(self,data):
        rng = np.random.RandomState(42)
        rng.shuffle(data)
        N_test = int(0.1*data.shape[0])
        data_test = data[-N_test:]
        data = data[0:-N_test]
        N_validate = int(0.1*data.shape[0])
        data_validate = data[-N_validate:]
        data_train = data[0:-N_validate]
        data = np.vstack((data_train, data_validate))
        return data_train, data_validate, data_test
    def train(self, batch_size, label = False):
        indx = np.random.randint(low = 0, high = self.nb_train, size = batch_size)
        if label:
            return self.X_train[indx, :], self.Y[indx]
        else:
            return self.X_train[indx, :]
    def load_all(self):
        return self.X_train, None

#miniboone dataset
class miniboone_sampler(object):
    def __init__(self,data_path='/home/liuqiao/software/maf/data/miniboone/data.npy'):
        data = np.load(data_path)
        self.X_train, self.X_val,self.X_test = self.normalize(data)
        self.Y=None
        self.nb_train = self.X_train.shape[0]
        self.mean = 0
        self.sd = 0
    def normalize(self,data):
        N_test = int(0.1*data.shape[0])
        data_test = data[-N_test:]
        data = data[0:-N_test]
        N_validate = int(0.1*data.shape[0])
        data_validate = data[-N_validate:]
        data_train = data[0:-N_validate]
        data = np.vstack((data_train, data_validate))
        mu = data.mean(axis=0)
        s = data.std(axis=0)
        data_train = (data_train - mu)/s
        data_validate = (data_validate - mu)/s
        data_test = (data_test - mu)/s
        return data_train, data_validate, data_test
    def train(self, batch_size, label = False):
        indx = np.random.randint(low = 0, high = self.nb_train, size = batch_size)
        if label:
            return self.X_train[indx, :], self.Y[indx]
        else:
            return self.X_train[indx, :]
    def load_all(self):
        return self.X_train, None
#power dataset
class power_sampler(object):
    def __init__(self,data_path='/home/liuqiao/software/maf/data/power/data.npy'):
        data = np.load(data_path)
        self.X_train, self.X_val,self.X_test = self.normalize(data)
        self.nb_train = self.X_train.shape[0]
        self.Y=None
        self.mean = 0
        self.sd = 0
    def normalize(self,data):
        rng = np.random.RandomState(42)
        rng.shuffle(data)
        N = data.shape[0]
        data = np.delete(data, 3, axis=1)
        data = np.delete(data, 1, axis=1)
        voltage_noise = 0.01*rng.rand(N, 1)
        gap_noise = 0.001*rng.rand(N, 1)
        sm_noise = rng.rand(N, 3)
        time_noise = np.zeros((N, 1))
        noise = np.hstack((gap_noise, voltage_noise, sm_noise, time_noise))
        data = data + noise
        N_test = int(0.1*data.shape[0])
        data_test = data[-N_test:]
        data = data[0:-N_test]
        N_validate = int(0.1*data.shape[0])
        data_validate = data[-N_validate:]
        data_train = data[0:-N_validate]
        data = np.vstack((data_train, data_validate))
        mu = data.mean(axis=0)
        s = data.std(axis=0)
        data_train = (data_train - mu)/s
        data_validate = (data_validate - mu)/s
        data_test = (data_test - mu)/s
        return data_train, data_validate, data_test
    def train(self, batch_size, label = False):
        indx = np.random.randint(low = 0, high = self.nb_train, size = batch_size)
        if label:
            return self.X_train[indx, :], self.Y[indx]
        else:
            return self.X_train[indx, :]
    def load_all(self):
        return self.X_train, None

#power dataset
class gas_sampler(object):
    def __init__(self,data_path='/home/liuqiao/software/maf/data/gas/ethylene_CO.pickle'):
        data = pd.read_pickle(data_path)
        self.X_train, self.X_val,self.X_test = self.normalize(data)
        self.nb_train = self.X_train.shape[0]
        self.Y=None
        self.mean = 0
        self.sd = 0
    def normalize(self,data):
        data.drop("Meth", axis=1, inplace=True)
        data.drop("Eth", axis=1, inplace=True)
        data.drop("Time", axis=1, inplace=True)
        C = data.corr()
        A = C > 0.98
        B = A.as_matrix().sum(axis=1)
        while np.any(B > 1):
            col_to_remove = np.where(B > 1)[0][0]
            col_name = data.columns[col_to_remove]
            data.drop(col_name, axis=1, inplace=True)
            C = data.corr()
            A = C > 0.98
            B = A.as_matrix().sum(axis=1)
        data = (data-data.mean())/data.std()
        data = data.as_matrix()
        N_test = int(0.1*data.shape[0])
        data_test = data[-N_test:]
        data_train = data[0:-N_test]
        N_validate = int(0.1*data_train.shape[0])
        data_validate = data_train[-N_validate:]
        data_train = data_train[0:-N_validate]
        return data_train, data_validate, data_test

    def train(self, batch_size, label = False):
        indx = np.random.randint(low = 0, high = self.nb_train, size = batch_size)
        if label:
            return self.X_train[indx, :], self.Y[indx]
        else:
            return self.X_train[indx, :]
    def load_all(self):
        return self.X_train, None

#HEPMASS dataset
class hepmass_sampler(object):
    def __init__(self,data_path='/home/liuqiao/software/maf/data/hepmass/'):
        self.X_train, self.X_val,self.X_test = self.normalize(data_path)
        self.Y=None
        self.nb_train = self.X_train.shape[0]
        self.mean = 0
        self.sd = 0
    def normalize(self,data_path):
        data_train = pd.read_csv(filepath_or_buffer=join(data_path, "1000_train.csv"), index_col=False)
        data_test = pd.read_csv(filepath_or_buffer=join(data_path, "1000_test.csv"), index_col=False)
        data_train = data_train[data_train[data_train.columns[0]] == 1]
        data_train = data_train.drop(data_train.columns[0], axis=1)
        data_test = data_test[data_test[data_test.columns[0]] == 1]
        data_test = data_test.drop(data_test.columns[0], axis=1)
        # Because the data set is messed up!
        data_test = data_test.drop(data_test.columns[-1], axis=1)
        mu = data_train.mean()
        s = data_train.std()
        data_train = (data_train - mu)/s
        data_test = (data_test - mu)/s
        data_train, data_test = data_train.as_matrix(), data_test.as_matrix()
        i = 0
        # Remove any features that have too many re-occurring real values.
        features_to_remove = []
        for feature in data_train.T:
            c = Counter(feature)
            max_count = np.array([v for k, v in sorted(c.iteritems())])[0]
            if max_count > 5:
                features_to_remove.append(i)
            i += 1
        data_train = data_train[:, np.array([i for i in range(data_train.shape[1]) if i not in features_to_remove])]
        data_test = data_test[:, np.array([i for i in range(data_test.shape[1]) if i not in features_to_remove])]

        N = data_train.shape[0]
        N_validate = int(N*0.1)
        data_validate = data_train[-N_validate:]
        data_train = data_train[0:-N_validate]
        return data_train, data_validate, data_test
    def train(self, batch_size, label = False):
        indx = np.random.randint(low = 0, high = self.nb_train, size = batch_size)
        if label:
            return self.X_train[indx, :], self.Y[indx]
        else:
            return self.X_train[indx, :]
    def load_all(self):
        return self.X_train, None

class mnist_sampler(object):
    def __init__(self,data_path='/home/liuqiao/software/maf/data/mnist/mnist.pkl.gz'):
        f = gzip.open(data_path, 'rb')
        trn, val, tst = pickle.load(f)
        self.trn_data = trn[0]
        self.trn_label = trn[1]
        self.trn_one_hot = np.eye(10)[self.trn_label]
        self.tst_data = tst[0]
        self.tst_label = tst[1]
        self.N = self.trn_data.shape[0]
        self.mean = 0
        self.sd = 0
    def train(self, batch_size, indx = None, label = False):
        if indx is None:
            indx = np.random.randint(low = 0, high = self.N, size = batch_size)
        if label:
            return self.trn_data[indx, :], self.trn_one_hot[indx]
        else:
            return self.trn_data[indx, :]
    def load_all(self):
        return self.trn_data, self.trn_label, self.trn_one_hot

class cifar10_sampler(object):
    def __init__(self,data_path='/home/liuqiao/software/maf/data/cifar10'):
        trn_data = []
        trn_label = []
        for i in xrange(1, 6):
            f = open(data_path + '/data_batch_' + str(i), 'rb')
            dict = pickle.load(f)
            trn_data.append(dict['data'])
            trn_label.append(dict['labels'])
            f.close()
        trn_data = np.concatenate(trn_data, axis=0)
        trn_data = trn_data.reshape(trn_data.shape[0],3,32,32)
        trn_data = trn_data.transpose(0, 2, 3, 1)
        trn_data = trn_data/256.0
        self.trn_data = trn_data.reshape(trn_data.shape[0],-1)
        self.trn_label = np.concatenate(trn_label, axis=0)
        self.trn_one_hot = np.eye(10)[self.trn_label]
        self.N = self.trn_data.shape[0]
        f = open(data_path + '/test_batch', 'rb')
        dict = pickle.load(f)
        tst_data = dict['data']
        tst_data = tst_data.reshape(tst_data.shape[0],3,32,32)
        tst_data = tst_data.transpose(0, 2, 3, 1)
        tst_data = tst_data/256.0
        self.tst_data = tst_data.reshape(tst_data.shape[0],-1)
        self.tst_label = np.array(dict['labels'])
        self.mean = 0
        self.sd = 0
    def train(self, batch_size, indx = None, label = False):
        if indx is None:
            indx = np.random.randint(low = 0, high = self.N, size = batch_size)
        if label:
            return self.trn_data[indx, :], self.trn_one_hot[indx]
        else:
            return self.trn_data[indx, :]
    def load_all(self):
        return self.trn_data, self.trn_label, self.trn_one_hot

# Gaussian mixture sampler by either given parameters or random component centers and fixed sd
class GMM_sampler(object):
    def __init__(self, N, mean=None, n_components=None, cov=None, sd=None, dim=None, weights=None):
        np.random.seed(1024)
        self.total_size = N
        self.n_components = n_components
        self.dim = dim
        self.sd = sd
        self.weights = weights
        if mean is None:
            assert n_components is not None and dim is not None and sd is not None
            #self.mean = np.random.uniform(-0.5,0.5,(self.n_components,self.dim))
            self.mean = np.random.uniform(-5,5,(self.n_components,self.dim))
        else:
            assert cov is not None    
            self.mean = mean
            self.n_components = self.mean.shape[0]
            self.dim = self.mean.shape[1]
            self.cov = cov
        if weights is None:
            self.weights = np.ones(self.n_components, dtype=np.float64) / float(self.n_components)
        self.Y = np.random.choice(self.n_components, size=N, replace=True, p=self.weights)
        if mean is None:
            self.X = np.array([np.random.normal(self.mean[i],scale=self.sd) for i in self.Y],dtype='float64')
        else:
            self.X = np.array([np.random.multivariate_normal(mean=self.mean[i],cov=self.cov[i]) for i in self.Y],dtype='float64')
        self.X_train, self.X_val,self.X_test = self.split(self.X)

    def split(self,data):
        #N_test = int(0.1*data.shape[0])
        N_test = 2000
        data_test = data[-N_test:]
        data = data[0:-N_test]
        #N_validate = int(0.1*data.shape[0])
        N_validate = 2000
        data_validate = data[-N_validate:]
        data_train = data[0:-N_validate]
        data = np.vstack((data_train, data_validate))
        return data_train, data_validate, data_test

    def train(self, batch_size, label = False):
        indx = np.random.randint(low = 0, high = len(self.X_train), size = batch_size)
        if label:
            return self.X_train[indx, :], self.Y[indx]
        else:
            return self.X_train[indx, :]

    def load_all(self):
        return self.X, self.Y

#Swiss roll (r*sin(scale*r),r*cos(scale*r)) + Gaussian noise
class Swiss_roll_sampler(object):
    def __init__(self, N, theta=2*np.pi, scale=2, sigma=0.4):
        np.random.seed(1024)
        self.total_size = N
        self.theta = theta
        self.scale = scale
        self.sigma = sigma
        params = np.linspace(0,self.theta,self.total_size)
        self.X_center = np.vstack((params*np.sin(scale*params),params*np.cos(scale*params)))
        self.X = self.X_center.T + np.random.normal(0,sigma,size=(self.total_size,2))
        np.random.shuffle(self.X)
        self.X_train, self.X_val,self.X_test = self.split(self.X)
        self.Y = None
        self.mean = 0
        self.sd = 0

    def split(self,data):
        N_test = int(0.1*data.shape[0])
        data_test = data[-N_test:]
        data = data[0:-N_test]
        N_validate = int(0.1*data.shape[0])
        data_validate = data[-N_validate:]
        data_train = data[0:-N_validate]
        data = np.vstack((data_train, data_validate))
        return data_train, data_validate, data_test
        
    def train(self, batch_size, label = False):
        indx = np.random.randint(low = 0, high = self.total_size, size = batch_size)
        return self.X[indx, :]

    def load_all(self):
        return self.X, self.Y

#Gaussian mixture + normal + uniform distribution
class GMM_Uni_sampler(object):
    def __init__(self, N, mean, cov, norm_dim=2,uni_dim=10,weights=None):
        self.total_size = N
        self.mean = mean
        self.n_components = self.mean.shape[0]
        self.norm_dim = norm_dim
        self.uni_dim = uni_dim
        self.cov = cov
        np.random.seed(1024)
        if weights is None:
            weights = np.ones(self.n_components, dtype=np.float64) / float(self.n_components)
        self.Y = np.random.choice(self.n_components, size=self.total_size, replace=True, p=weights)
        #self.X = np.array([np.random.normal(self.mean[i],scale=self.sd) for i in self.Y],dtype='float64')
        self.X_gmm = np.array([np.random.multivariate_normal(mean=self.mean[i],cov=self.cov[i]) for i in self.Y],dtype='float64')
        self.X_normal = np.random.normal(0.5, np.sqrt(0.1), (self.total_size,self.norm_dim))
        self.X_uni = np.random.uniform(-0.5,0.5,(self.total_size,self.uni_dim))
        self.X = np.concatenate([self.X_gmm,self.X_normal,self.X_uni],axis = 1)
        
    def train(self, batch_size, label = False):
        indx = np.random.randint(low = 0, high = self.total_size, size = batch_size)
        if label:
            return self.X[indx, :], self.Y[indx].flatten()
        else:
            return self.X[indx, :]

    def load_all(self):
        return self.X, self.Y

#each dim is a gmm
class GMM_indep_sampler(object):
    def __init__(self, N, sd, dim, n_components, weights=None, bound=1):
        np.random.seed(1024)
        self.total_size = N
        self.dim = dim
        self.sd = sd
        self.n_components = n_components
        self.centers = np.linspace(-bound, bound, n_components)
        self.X = np.vstack([self.generate_gmm() for _ in range(dim)]).T
        self.X_train, self.X_val,self.X_test = self.split(self.X)
        self.nb_train = self.X_train.shape[0]
        self.Y=None
        self.mean=0
    def generate_gmm(self,weights = None):
        if weights is None:
            weights = np.ones(self.n_components, dtype=np.float64) / float(self.n_components)
        Y = np.random.choice(self.n_components, size=self.total_size, replace=True, p=weights)
        return np.array([np.random.normal(self.centers[i],self.sd) for i in Y],dtype='float64')
    def split(self,data):
        #N_test = int(0.1*data.shape[0])
        N_test = 2000
        data_test = data[-N_test:]
        data = data[0:-N_test]
        #N_validate = int(0.1*data.shape[0])
        N_validate = 2000
        data_validate = data[-N_validate:]
        data_train = data[0:-N_validate]
        data = np.vstack((data_train, data_validate))
        return data_train, data_validate, data_test

    def train(self, batch_size):
        indx = np.random.randint(low = 0, high = self.nb_train, size = batch_size)
        return self.X_train[indx, :]

    def load_all(self):
        return self.X, self.Y


#Gaussian + uniform distribution
class Multi_dis_sampler(object):
    def __init__(self, N, dim):
        np.random.seed(1024)
        self.total_size = N
        self.dim = dim
        assert dim >= 5
        #first two dims are GMM
        self.mean = np.array([[0.2,0.3],[0.7,0.8]])
        self.cov = [0.1**2*np.eye(2),0.1**2*np.eye(2)]
        comp_idx = np.random.choice(2, size=self.total_size, replace=True)
        self.X_gmm = np.array([np.random.multivariate_normal(mean=self.mean[i],cov=self.cov[i]) for i in comp_idx],dtype='float64')
        #dim 3 is a normal
        self.X_gau = np.random.normal(0.5, 0.1, size=(self.total_size,1))
        #dim 4 is a uniform
        self.X_uni = np.random.uniform(0,1,size=(self.total_size,1))
        #dim >=5 is a GMM for each dim
        self.centers=np.array([0.2,0.6])
        self.sd = np.array([0.1,0.05])
        self.X_indep_gmm = np.vstack([self.generate_gmm(self.centers,self.sd) for _ in range(dim-4)]).T
        self.X = np.hstack((self.X_gmm,self.X_gau,self.X_uni,self.X_indep_gmm))
        self.X_train, self.X_val,self.X_test = self.split(self.X)
    def generate_gmm(self,centers,sd):
            Y = np.random.choice(2, size=self.total_size, replace=True)
            return np.array([np.random.normal(centers[i],sd[i]) for i in Y],dtype='float64')
        
    def split(self,data):
        N_test = int(0.1*data.shape[0])
        data_test = data[-N_test:]
        data = data[0:-N_test]
        N_validate = int(0.1*data.shape[0])
        data_validate = data[-N_validate:]
        data_train = data[0:-N_validate]
        data = np.vstack((data_train, data_validate))
        return data_train, data_validate, data_test
        
    def train(self, batch_size):
        indx = np.random.randint(low = 0, high = len(self.X_train), size = batch_size)
        return self.X[indx, :]
    def get_single_density(self,data):
        #gmm
        p1 = 1./(np.sqrt(2*np.pi)*0.1) * np.exp(-np.sum((self.mean[0]-data[:2])**2) / (2*0.1**2)) 
        p2 = 1./(np.sqrt(2*np.pi)*0.1) * np.exp(-np.sum((self.mean[1]-data[:2])**2) / (2*0.1**2)) 
        p_gmm = (p1+p2)/2.
        #Gaussian
        p_gau = 1./(np.sqrt(2*np.pi)*0.1)**2 * np.exp(-np.sum((0.5-data[2])**2) / (2*0.1**2)) 
        #Uniform
        p_uni = 1
        #indep gmm
        p_indep_gmm = 1
        for i in range(4,self.dim):
            p1 = 1./(np.sqrt(2*np.pi)*self.sd[0]) * np.exp(-np.sum((self.centers[0]-data[i])**2) / (2*self.sd[0]**2)) 
            p2 = 1./(np.sqrt(2*np.pi)*self.sd[1]) * np.exp(-np.sum((self.centers[1]-data[i])**2) / (2*self.sd[1]**2)) 
            p_indep_gmm *= (p1+p2)/2.
        return np.prod([p_gmm,p_gau,p_uni,p_indep_gmm])
    def get_all_density(self,batch_data):
        assert batch_data.shape[1]==self.dim
        p_all = map(self.get_single_density,batch_data)
        return np.array(p_all)


class Uniform_sampler(object):
    def __init__(self, N, dim, mean):
        self.total_size = N
        self.dim = dim 
        self.mean = mean
        np.random.seed(1024)
        self.centers = np.random.uniform(-0.5,0.5,(self.dim,))
        #print self.centers
        #self.X = np.random.uniform(self.centers-0.5,self.centers+0.5,size=(self.total_size,self.dim))
        self.Y = None
        self.X = np.random.uniform(self.mean-0.5,self.mean+0.5,(self.total_size,self.dim))

    def get_batch(self, batch_size):
        return np.random.uniform(self.mean-0.5,self.mean+0.5,(batch_size,self.dim))
    #for data sampling given batch size
    def train(self, batch_size, label = False):
        return np.random.uniform(self.mean-0.5,self.mean+0.5,(batch_size,self.dim))

    def load_all(self):
        return self.X, self.Y

class Gaussian_sampler(object):
    def __init__(self, N, mean, sd=1):
        self.total_size = N
        self.mean = mean
        self.sd = sd
        np.random.seed(1024)
        self.X = np.random.normal(self.mean, self.sd, (self.total_size,len(self.mean)))
        self.Y = None

    def train(self, batch_size, label = False):
        indx = np.random.randint(low = 0, high = self.total_size, size = batch_size)
        return self.X[indx, :]

    def get_batch(self,batch_size):
        return np.random.normal(self.mean, self.sd, (batch_size,len(self.mean)))

    def load_all(self):
        return self.X, self.Y

#sample continuous (Gaussian) and discrete (Catagory) latent variables together
class Mixture_sampler(object):
    def __init__(self, nb_classes, N, dim, sd, scale=1):
        self.nb_classes = nb_classes
        self.total_size = N
        self.dim = dim
        self.sd = sd 
        self.scale = scale
        np.random.seed(1024)
        self.X_c = self.scale*np.random.normal(0, self.sd**2, (self.total_size,self.dim))
        #self.X_c = self.scale*np.random.uniform(-1, 1, (self.total_size,self.dim))
        self.label_idx = np.random.randint(low = 0 , high = self.nb_classes, size = self.total_size)
        self.X_d = np.eye(self.nb_classes)[self.label_idx]
        self.X = np.hstack((self.X_c,self.X_d))

    def train(self, batch_size, two_label = False):
        X_c = np.random.normal(0, self.sd**2, (batch_size,self.dim))
        if two_label:
            indx1 = np.random.randint(low = 0, high = self.total_size, size = batch_size)
            indx2 = np.random.randint(low = 0, high = self.total_size, size = batch_size)
            return X_c, self.X_d[indx1, :], self.X_d[indx2, :]
        else:
            indx = np.random.randint(low = 0, high = self.total_size, size = batch_size)
            return self.X_c[indx, :],self.X_d[indx, :]
            #return X_c,self.X_d[indx, :]
    
    def get_batch(self,batch_size,weights=None):
        X_batch_c = self.scale*np.random.normal(0, 1, (batch_size,self.dim))
        #X_batch_c = self.scale*np.random.uniform(-1, 1, (batch_size,self.dim))
        if weights is None:
            weights = np.ones(self.nb_classes, dtype=np.float64) / float(self.nb_classes)
        label_batch_idx =  np.random.choice(self.nb_classes, size=batch_size, replace=True, p=weights)
        X_batch_d = np.eye(self.nb_classes)[label_batch_idx]
        return X_batch_c,X_batch_d,label_batch_idx

    def load_all(self):
        return self.X_c, self.X_d, self.label_idx

#sample continuous (Gaussian Mixture) and discrete (Catagory) latent variables together
class Mixture_sampler_v2(object):
    def __init__(self, nb_classes, N, dim, weights=None,sd=0.5):
        self.nb_classes = nb_classes
        self.total_size = N
        self.dim = dim
        np.random.seed(1024)
        if nb_classes<=dim:
            self.mean = np.random.uniform(-5,5,size =(nb_classes, dim))
            #self.mean = np.zeros((nb_classes,dim))
            #self.mean[:,:nb_classes] = np.eye(nb_classes)
        else:
            if dim==2:
                self.mean = np.array([(np.cos(2*np.pi*idx/float(self.nb_classes)),np.sin(2*np.pi*idx/float(self.nb_classes))) for idx in range(self.nb_classes)])
            else:
                self.mean = np.zeros((nb_classes,dim))
                self.mean[:,:2] = np.array([(np.cos(2*np.pi*idx/float(self.nb_classes)),np.sin(2*np.pi*idx/float(self.nb_classes))) for idx in range(self.nb_classes)])
        self.cov = [sd**2*np.eye(dim) for item in range(nb_classes)]
        if weights is None:
            weights = np.ones(self.nb_classes, dtype=np.float64) / float(self.nb_classes)
        self.Y = np.random.choice(self.nb_classes, size=N, replace=True, p=weights)
        self.X_c = np.array([np.random.multivariate_normal(mean=self.mean[i],cov=self.cov[i]) for i in self.Y],dtype='float64')
        self.X_d = np.eye(self.nb_classes)[self.Y]
        self.X = np.hstack((self.X_c,self.X_d))

    def train(self, batch_size, label = False):
        indx = np.random.randint(low = 0, high = self.total_size, size = batch_size)
        if label:
            return self.X_c[indx, :], self.X_d[indx, :], self.Y[indx, :]
        else:
            return self.X_c[indx, :], self.X_d[indx, :]

    def get_batch(self,batch_size,weights=None):
        if weights is None:
            weights = np.ones(self.nb_classes, dtype=np.float64) / float(self.nb_classes)
        label_batch_idx =  np.random.choice(self.nb_classes, size=batch_size, replace=True, p=weights)
        return self.X_c[label_batch_idx, :], self.X_d[label_batch_idx, :]
    def predict_onepoint(self,array):#return component index with max likelyhood
        from scipy.stats import multivariate_normal
        assert len(array) == self.dim
        return np.argmax([multivariate_normal.pdf(array,self.mean[idx],self.cov[idx]) for idx in range(self.nb_classes)])

    def predict_multipoints(self,arrays):
        assert arrays.shape[-1] == self.dim
        return map(self.predict_onepoint,arrays)
    def load_all(self):
        return self.X_c, self.X_d, self.label_idx


def sample_Z(batch, z_dim , sampler = 'one_hot', num_class = 10, n_cat = 1, label_index = None):
    if sampler == 'mul_cat':
        if label_index is None:
            label_index = np.random.randint(low = 0 , high = num_class, size = batch)
        return np.hstack((0.10 * np.random.randn(batch, z_dim-num_class*n_cat),
                          np.tile(np.eye(num_class)[label_index], (1, n_cat))))
    elif sampler == 'one_hot':
        if label_index is None:
            label_index = np.random.randint(low = 0 , high = num_class, size = batch)
        #return np.hstack((0.10 * np.random.randn(batch, z_dim-num_class), np.eye(num_class)[label_index]))
        return np.hstack((0.10 * np.random.normal(0,1,(batch, z_dim-num_class)), np.eye(num_class)[label_index]))
    elif sampler == 'uniform':
        return np.random.uniform(-1., 1., size=[batch, z_dim])
    elif sampler == 'normal':
        return 0.15*np.random.randn(batch, z_dim)
    elif sampler == 'mix_gauss':
        if label_index is None:
            label_index = np.random.randint(low = 0 , high = num_class, size = batch)
        return (0.1 * np.random.randn(batch, z_dim) + np.eye(num_class, z_dim)[label_index])

#get a batch of data from previous 50 batches, add stochastic
class DataPool(object):
    def __init__(self, maxsize=50):
        self.maxsize = maxsize
        self.nb_batch = 0
        self.pool = []

    def __call__(self, data):
        if self.nb_batch < self.maxsize:
            self.pool.append(data)
            self.nb_batch += 1
            return data
        if np.random.rand() > 0.5:
            results=[]
            for i in range(len(data)):
                idx = int(np.random.rand()*self.maxsize)
                results.append(copy.copy(self.pool[idx])[i])
                self.pool[idx][i] = data[i]
            return results
        else:
            return data

class Bayesian_sampler(object):
    def __init__(self, N, dim1=10, dim2=5+1):
        self.total_size = N
        self.dim1 = dim1#y
        self.dim2 = dim2#theta
        np.random.seed(1024)
        self.data = np.load('TS-data_block1.npy')
        
        self.X1 = self.data[:N,-self.dim1:]
        self.X2 = self.data[:N,:self.dim2]
        assert self.X2.shape[1]==self.dim2
        assert self.X1.shape[1]==self.dim1
        self.X = np.hstack((self.X1,self.X2))

    def train(self, batch_size, label = False):
        indx = np.random.randint(low = 0, high = self.total_size, size = batch_size)
        return self.X1[indx, :], self.X2[indx, :]
    
    def get_batch(self,batch_size,weights=None):
        indx = np.random.randint(low = 0, high = self.total_size, size = batch_size)
        return self.X1[indx, :], self.X2[indx, :]

    def load_all(self):
        return self.X1,self.X2


class SV_sampler(object):#stochastic volatility model
    def __init__(self, theta_init, sample_size, block_size=10,seed = 1):
        np.random.seed(seed)
        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt
        self.sample_size = sample_size
        self.theta_init = theta_init
        self.block_size = block_size
        #self.y_true, _, self.h_true = self.generate_data(sample_size=1,time_step=1000,use_prior=True)
        #print self.y_true[0,-5:]

    def generate_data(self,sample_size,time_step,theta=None,h_0=None,use_prior=False, seed = 1):
        np.random.seed(seed)
        assert len(self.theta_init)==5
        h_t = np.empty((sample_size, time_step), dtype=np.float64)#log-volatility
        y_t = np.empty((sample_size, time_step), dtype=np.float64)#observation data
        if use_prior:
            theta = np.empty((sample_size, len(self.theta_init)), dtype=np.float64)
            #theta[:,0] = self.generate_mu(sample_size)
            theta[:,0] = np.random.normal(loc=0.0314,scale=1.0,size=(sample_size,))
            theta[:,1] = self.generate_phi(sample_size)
            theta[:,2] = self.generate_sigma2(sample_size)
            #theta[:,3] = self.generate_nu(sample_size)
            #theta[:,4] = self.generate_lambda(sample_size)
            #theta[0,:] = self.theta_init
            theta[0,:] = [0.0314, 0.9967, 0.0107, 19.6797, -1.1528]
        mu = theta[:, 0]
        phi = theta[:, 1]
        sigma2 = theta[:, 2]
        sigma = sigma2 ** 0.5
        #nu = theta[:, 3]
        #lambda_ = theta[:, 4]

        if use_prior:
            #h_t[:, 0] = norm.rvs(size=sample_size) * (sigma2 / (1 - phi ** 2)) ** 0.5 + mu
            h_t[:, 0] = 0
        else:
            h_t[:, 0] = mu + phi * (h_0 - mu) + sigma * norm.rvs(size=sample_size)

        for t_ in range(1, time_step):
            h_t[:, t_] = mu + phi * (h_t[:, t_-1] - mu) + sigma * norm.rvs(size=sample_size)
           
        #generate y_t
        for i in range(sample_size):
            #zeta, omega = self.get_zeta_omega(lambda_=lambda_[i], nu=nu[i])
            #epsilon = self.generate_skew_student_t(sample_size=time_step, zeta=zeta, omega=omega, lambda_=lambda_[i], nu=nu[i])
            epsilon = self.generate_normal(time_step)
            y_t[i, :] = np.exp(h_t[i, :] / 2) * epsilon
        return y_t, theta[:,:3], h_t
   
    def generate_normal(self,sample_size,low=0.,high=1.):
        return np.random.normal(size=sample_size)

    def generate_mu(self,sample_size):
        return norm.rvs(scale=1, size=sample_size)
    
    def generate_phi(self,sample_size):
        my_mean = 0.95
        my_std = 10
        myclip_a = -1
        myclip_b = 1
        a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
        return truncnorm.rvs(a, b, loc=my_mean, scale=my_std, size=sample_size)

    def generate_sigma2(self,sample_size):
        return invgamma.rvs(a=2.5, scale=0.025, size=sample_size)

    def prior_of_nu(self, nu):
        first = nu / (nu + 3)
        first **= 0.5
        second = polygamma(1, nu / 2) - polygamma(1, (nu + 1) / 2) - 2 * (nu + 3) / nu / (nu + 1) ** 2
        second **= 0.5
        return first * second

    def generate_nu(self, sample_size, left = 10, right = 40):
        out = []
        temp = self.prior_of_nu(left)
        while len(out) < sample_size:
            nu = uniform.rvs() * (right - left) + left
            if uniform.rvs() < self.prior_of_nu(nu) / temp:
                out.append(nu)
        return np.array(out, dtype=np.float64)

    def generate_lambda(self, sample_size):
        # return t.rvs(df=0.5, loc=0.0, scale=pi ** 2 / 4, size=sample_size)
        return norm.rvs(loc=0.0, scale=pi ** 2 / 4, size=sample_size)

    def get_zeta_omega(self,lambda_, nu):
        k1 = (nu / 2) ** 0.5 * scipy.special.gamma(nu / 2 - 0.5) / scipy.special.gamma(nu / 2)
        k2 = nu / (nu - 2)
        delta = lambda_ / (1 + lambda_ ** 2) ** 0.5
        omega = (k2 - 2 / pi * k1 ** 2 * delta ** 2) ** (-0.5)
        zeta = - (2 / pi) ** 0.5 * k1 * omega * delta
        return zeta, omega

    def generate_skew_student_t(self, sample_size, zeta, omega, lambda_, nu):
        delta = lambda_ / (1 + lambda_ ** 2) ** 0.5
        w = truncnorm.rvs(a=0, b=float('inf'), size=sample_size)
        epsilon = norm.rvs(size=sample_size)
        u = gamma.rvs(a=nu / 2, scale=2 / nu, size=sample_size)
        return zeta + u ** (-0.5) * omega * (delta * w + (1 - delta ** 2) ** 0.5 * epsilon)

    def train(self, batch_size, label = False):
        indx = np.random.randint(low = 0, high = self.total_size, size = batch_size)
        return self.X1[indx, :], self.X2[indx, :]
    
    def get_batch(self,batch_size,weights=None):
        indx = np.random.randint(low = 0, high = self.total_size, size = batch_size)
        return self.X1[indx, :], self.X2[indx, :]

    def load_all(self):
        return self.X1,self.X2

# model: y_t = A cos ( 2 pi omega t + phi ) + sigma w_t, w_t ~ N (0, 1)
# T = 1/omega
# parameters: (omega, phi, logsigma, logA)
# prior:
#     logA ~ N (0, 1)
#     phi ~ Unif(0, pi)
#     omega ~ Unif(0, 0.1)
#     logsigma ~ N (0, 1)
class Cosine_sampler(object):
    def __init__(self, omega=1./10,sigma=0.3,iteration=10,block_size=10):
        self.omega = omega
        self.sigma = sigma
        self.block_size = block_size
        self.observation = np.zeros(iteration*block_size)
    def generate_data(self,sample_size,time_step,seed=0):#time series data t=1,2,3...
        np.random.seed(seed)
        theta = np.empty((sample_size, 4), dtype=np.float64)
        data = np.empty((sample_size, time_step), dtype=np.float64)
        theta[:, 0] = np.random.normal(size=sample_size)
        theta[:, 1] = np.random.uniform(low=0, high=np.pi, size=sample_size)
        theta[:, 2] = 1./80
        theta[:, 3] = -10
        #theta[:, 0] = np.random.uniform(low=0, high=0.1, size=sample_size)
        #theta[:, 1] = np.random.uniform(low=0, high=2 * np.pi, size=sample_size)
        #theta[:, 2] = np.random.normal(size=sample_size)
        #theta[:, 2] = np.random.uniform(low=-5, high=-2, size=sample_size)
        #theta[:, 3] = np.random.normal(size=sample_size)
        #theta[0, :] = self.theta_init
        theta[0,:2] = [np.log(2),np.pi / 4]
        #theta[0, :] = [1 / 80, np.pi / 4, 0, np.log(2)]

        logA, phi, omega, logsigma = theta.transpose()
        sigma = np.exp(logsigma)
        A = np.exp(logA)

        for t in range(time_step):
            data[:, t] = A * np.cos(2 * np.pi * omega * (t + 1) + phi) + sigma * np.random.normal(size=sample_size)
        return data, theta[:,:2]
    def generate_data2(self,sample_size,time_step=10,seed=0):#time series data t is continous from [0,1]
        np.random.seed(seed)
        theta = np.empty((sample_size, 4), dtype=np.float64)
        data = np.empty((sample_size, 2), dtype=np.float64)
        theta[:, 0] = np.random.normal(size=sample_size)
        theta[:, 1] = np.random.uniform(low=0, high=np.pi, size=sample_size)
        theta[:, 2] = 0.5
        theta[:, 3] = -10
        #theta[:, 0] = np.random.uniform(low=0, high=0.1, size=sample_size)
        #theta[:, 1] = np.random.uniform(low=0, high=2 * np.pi, size=sample_size)
        #theta[:, 2] = np.random.normal(size=sample_size)
        #theta[:, 2] = np.random.uniform(low=-5, high=-2, size=sample_size)
        #theta[:, 3] = np.random.normal(size=sample_size)
        #theta[0, :] = self.theta_init
        #theta[0,:2] = [np.log(2),np.pi / 4]
        #theta[0,:2] = [0.0, np.pi / 2]
        theta[0,:2] = [1, 3.*np.pi / 4]
        #theta[0, :] = [1 / 80, np.pi / 4, 0, np.log(2)]

        logA, phi, omega, logsigma = theta.transpose()
        sigma = np.exp(logsigma)
        A = np.exp(logA)
        for i in range(sample_size):
            if i<time_step:
                t = float(i) / time_step
                y_t = A[0] * np.cos(2 * np.pi * omega[0] * t + phi[0]) + sigma[0] * np.random.normal()
            else:
                t = np.random.uniform()
                y_t = A[i] * np.cos(2 * np.pi * omega[i] * t + phi[i]) + sigma[i] * np.random.normal()
            data[i,:] = [y_t,t]
        return data, theta[:,:2] 
        
    def generate_data3(self,sample_size,iteration,prior=None,seed=0):#minibatch=1 in the above generate_data2()
        #np.random.seed(seed)
        np.random.seed(iteration)
        params = np.empty((sample_size, 2), dtype=np.float64)
        data = np.empty((sample_size, self.block_size), dtype=np.float64)
        if prior is None:
            params[:, 0] = np.random.normal(size=sample_size)
            params[:, 1] = np.random.uniform(low=-2*np.pi, high=2*np.pi, size=sample_size)
        else:
            params[:,:2] = prior
        #params[0,:2] = [np.log(2),np.pi / 4]# (0.69, 0.78)
        params[0,:] = [0.0, np.pi / 2]  # (0, 1.57)
        #params[0,:2] = [1, 3.*np.pi / 4]  #(1, 2.35)
        #params[0, :] = [1 / 80, np.pi / 4, 0, np.log(2)] 
        logA, phi = params.transpose()
        A = np.exp(logA)
    
        #t = np.linspace(self.block_size*iteration,self.block_size*(iteration+1)-1,self.block_size)
        t = np.random.uniform(self.block_size*iteration,self.block_size*(iteration+1),size=self.block_size)
        #Note that the first row is the observation data
        for i in range(sample_size):
            data[i,:] = A[i] * np.cos(2 * np.pi * self.omega * t + phi[i]) + self.sigma * np.random.normal(size=self.block_size)
        return data, params, t

    #calculate likelihood
    def cal_bayesian_likelihood(self, data, t, params_list,use_log=True):
        if len(params_list.shape) == 1:
            logA, phi = params_list[0], params_list[1]
            A = np.exp(logA)
            log_likelyhood = -np.sum((data-A*np.cos(2*np.pi * self.omega *t+phi))**2) / (2*self.sigma**2)
        else:
            logA, phi = params_list
            A = np.exp(logA)
            log_params_prior=  -logA**2/2
            log_likelyhood = np.array([-np.sum((data-A_*np.cos(2*np.pi * self.omega *t+phi_))**2) / (2*self.sigma**2) \
                for A_, phi_ in zip(A, phi)])
        if use_log:
            return log_likelyhood
        else:
            return np.exp(log_likelyhood)

    #calculate posterior using bayesian formula
    def cal_bayesian_posterior(self, data, t, params_list,use_log=True):
        if len(params_list.shape) == 1:
            logA, phi = params_list[0], params_list[1]
            A = np.exp(logA)
            log_params_prior = -logA**2/2
            log_likelyhood = -np.sum((data-A*np.cos(2*np.pi * self.omega *t+phi))**2) / (2*self.sigma**2)
        else:
            logA, phi = params_list
            A = np.exp(logA)
            log_params_prior = -logA**2/2
            log_likelyhood = np.array([-np.sum((data-A_*np.cos(2*np.pi * self.omega *t+phi_))**2) / (2*self.sigma**2) \
                for A_, phi_ in zip(A, phi)])
        if use_log:
            return log_params_prior+log_likelyhood
        else:
            return np.exp(log_params_prior+log_likelyhood)

    #sample theta by Metroplis-Hasting algorithm, random Gaussian walk
    def sample_posterior(self, data, params, t, sample_size=1000, chain_len=500, sd=0.1, seed=0):
        np.random.seed(seed)
        if params is None:
            para_temp = np.zeros((2, sample_size), dtype=np.float64)
            #starting states of Markov Chain
            para_temp[0, :] = 0 # logA
            para_temp[1, :] = 0  #phi
        else:
            para_temp = copy.copy(params).T
            sample_size = params.shape[0]
        for _ in tqdm(range(chain_len)):
            para_propose = para_temp + np.random.normal(scale=sd, size=(2, sample_size))
            #para_propose[0, :] %= 0.1
            para_propose[1, :] += 2*np.pi
            para_propose[1, :] %= (4 * np.pi)
            para_propose[1, :] -= 2*np.pi
            #para_propose[1, :] = para_propose[1, :]%(2 * np.pi)-np.pi 

            mask = self.cal_bayesian_posterior(data, t, para_propose) > np.log(np.random.uniform(size=sample_size)) \
                + self.cal_bayesian_posterior(data, t, para_temp)
            para_temp[:, mask] = para_propose[:, mask]
        return para_temp.T

    #plot posterior of both gruth truth(2D) and sampled theta
    def get_posterior(self, data, t, res = 100):
        #get the truth params
        _,params,_ = self.generate_data3(1,0,self.block_size)
        logA, phi = params[0,:]
        logA_axis = np.linspace(logA-1,logA+1,res)
        phi_axis = np.linspace(phi-1,phi+1,res)
        X,Y = np.meshgrid(logA_axis,phi_axis)
        params_stacks = np.vstack([X.ravel(), Y.ravel()]) #shape (2, N*N)
        #log_posterior_list = map(cal_beyesian_posterior,theta_list)
        bayesian_posterior_stacks = self.cal_bayesian_posterior(data, t, params_stacks)
        bayesian_posterior_mat = np.reshape(bayesian_posterior_stacks,(len(phi_axis),len(logA_axis)))
        params_sampled = self.sample_posterior(data,None,t)
        return bayesian_posterior_mat, logA_axis, phi_axis, params_sampled
    
    #MCMC refinement
    def refine_posterior(self, data, params, t, chain_len=50, seed=0):
        np.random.seed(seed)
        #starting states of Markov Chain
        para_temp = params.T
        for _ in tqdm(range(chain_len)):
            para_propose = para_temp + np.random.normal(scale=0.1, size=para_temp.shape)
            #para_propose[0, :] %= 0.1
            para_propose[1, :] += 2*np.pi
            para_propose[1, :] %= (4 * np.pi)
            para_propose[1, :] -= 2*np.pi
            #para_propose[1, :] = para_propose[1, :]%(2 * np.pi)-np.pi 
            mask = self.cal_bayesian_posterior(data, t, para_propose) > np.log(np.random.uniform(size=para_temp.shape[1])) \
                + self.cal_bayesian_posterior(data, t, para_temp)
            para_temp[:, mask] = para_propose[:, mask]
        return para_temp.T
    #MCMC resampling with N chains and get the last data point
    def resampling(self, data, params, t, chain_len=100, seed=0):
        np.random.seed(seed)
        #t,data: time and data of a minibatch, e.g., (10,)
        #params: params set as proposals
        para_propose_set = copy.copy(params)
        para_temp = params.T
        for _ in tqdm(range(chain_len)):
            label_batch_idx =  np.random.choice(len(para_propose_set), size=len(para_propose_set), replace=True)
            para_propose = para_propose_set[label_batch_idx].T
            mask = self.cal_bayesian_likelihood(data, t, para_propose) > np.log(np.random.uniform(size=para_temp.shape[1])) \
                + self.cal_bayesian_likelihood(data, t, para_temp)
            para_temp[:, mask] = para_propose[:, mask]
        return para_temp.T

    #MCMC resampling with 1 chain and get the last N data points
    def resampling_v2(self, data, params, t, chain_len=10000, seed=0):
        np.random.seed(seed)
        #t,data: time and data of a minibatch, e.g., (10,)
        #params: params set as proposals
        a = copy.copy(params)
        para_propose_set = copy.copy(params)
        para_sampled = np.zeros(params.shape)
        para_temp = params[0,:]
        for i in tqdm(range(chain_len+len(para_propose_set)*10)):
            label_batch_idx =  np.random.choice(len(para_propose_set))
            para_propose = para_propose_set[label_batch_idx,:]
            #print(self.cal_bayesian_likelihood(data, t, para_propose))
            #print(self.cal_bayesian_likelihood(data, t, para_temp))
            if self.cal_bayesian_likelihood(data, t, para_propose) > np.log(np.random.uniform()) \
                + self.cal_bayesian_likelihood(data, t, para_temp):
                para_temp = para_propose
            if i >= chain_len and i%10==0:
                para_sampled[int((i-chain_len)/10),:] = para_temp
        return para_sampled

    #directly sample with multinomial distribution using likelihood as weights
    def resampling_v3(self, data, params, t, chain_len=10000, seed=0):
        np.random.seed(seed)
        #t,data: time and data of a minibatch, e.g., (10,)
        #params: params set as proposals
        para_propose_set = copy.copy(params)
        log_likelihood = self.cal_bayesian_likelihood(data, t, params.T)
        likelihood = np.exp(log_likelihood)
        likelihood /= np.sum(likelihood)
        sampled_idx = np.random.choice(len(params),size=len(params),replace=True,p=likelihood)
        return para_propose_set[sampled_idx]

    #adaptive mh 
    def adaptive_sampling(self, data, params, t, chain_len=1000, bound=0.2, seed=2):
        np.random.seed(seed)
        #t,data: time and data 
        #params: params set as proposals
        para_propose_set = copy.copy(params)
        sample_size,param_size = params.shape
        para_sampled = np.zeros(params.shape)
        params_sorted = np.zeros(params.shape)
        para_temp = para_propose_set[0,:]
        temp_idx = 0
        adjacent_list = [[] for _ in range(sample_size)]
        #params with shape (N,m)
        order_idx = [[] for _ in range(param_size)]
        dic_order_idx = [{} for _ in range(param_size)]
        for i in range(param_size):
            params_with_idx = np.vstack([params[:,i],np.arange(sample_size)]).T
            params_with_idx = np.array(sorted(params_with_idx,key=lambda a:a[0]))
            params_sorted[:,i] = params_with_idx[:,0]
            order_idx[i] = list(params_with_idx[:,1])
            dic_order_idx[i] = {item[0]:item[1] for item in zip(order_idx[i],np.arange(sample_size))}
        #calculate proposal point density 
        def cal_proposal_density(param,idx):
            neighbor_list=[]
            for i in range(param_size):
                param_ith_idx = dic_order_idx[i][idx]
                left_idx, right_idx = find_inserted_idx(params_sorted[:,i],param[i],param_ith_idx)
                neighbor_list.append(order_idx[i][left_idx:right_idx])
            return len(set(neighbor_list[0]).intersection(*neighbor_list[1:]))
        def find_inserted_idx(array,value,idx):
            left_idx,right_idx = 0,len(array)
            isleftbreak=0
            for i in range(idx,len(array)):
                if value+bound < array[i]:
                    right_idx = i
                    break
            for i in range(idx,-1,-1):
                if value-bound > array[i]:
                    left_idx = i
                    isleftbreak = 1
                    break
            if isleftbreak:
                left_idx+=1
            return left_idx,right_idx

        def cal_proposal_density_v2(param,idx):
            neighbor_list=[]
            for i in range(param_size):
                param_ith_idx = dic_order_idx[i][idx]
                left_idx, right_idx = find_inserted_idx_v2(params_sorted[:,i],param[i],param_ith_idx)
                neighbor_list.append(order_idx[i][left_idx:right_idx])
            return len(set(neighbor_list[0]).intersection(*neighbor_list[1:]))
        #find idx with binary division search
        def find_inserted_idx_v2(param_array,value,idx):
            def binary_search(value,array):
                if len(array)==0:
                    return 0
                if value>array[-1]:
                    return len(array)
                Min, Max = 0, len(array)-1
                while Min<=Max:
                    mid = int((Min+Max)/2)
                    if array[mid]<value:
                        Min = mid + 1
                    elif array[mid]>value:
                        Max = mid - 1
                    else:
                        return mid
                return Min

            right_idx = binary_search(value+bound,param_array[idx:])
            left_idx = binary_search(value-bound,param_array[:idx])
            return left_idx, idx+right_idx

        #neighboring points as the unnormalized density
        def mixture_square_density(points,point,idx):
            nb=0
            for i in adjacent_list[idx]:
                neighbor_point = points[i]
                if np.sum(abs(neighbor_point-point) <= bound) == points.shape[1]:
                    nb+=1
            return nb

        for i in tqdm(range(chain_len+len(para_propose_set)*5)):
            mixture_idx =  np.random.choice(len(para_propose_set))
            para_propose = para_propose_set[mixture_idx,:] + np.random.uniform(-bound,bound,size=(2,))

            if self.cal_bayesian_posterior(data, t, para_propose) + np.log(cal_proposal_density_v2(para_temp,temp_idx)) > \
                + self.cal_bayesian_posterior(data, t, para_temp) + np.log(cal_proposal_density_v2(para_propose,mixture_idx)) \
                + np.log(np.random.uniform()):
                para_temp = para_propose
                temp_idx = mixture_idx
            if i >= chain_len and i%5==0:
                para_sampled[int((i-chain_len)/5),:] = para_temp
        return para_sampled


if __name__=='__main__':
    from scipy import stats
    from sklearn.neighbors import KernelDensity
    from  sklearn.mixture import GaussianMixture
    from scipy.stats import gaussian_kde
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from sklearn.cluster import KMeans
    from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
    from sklearn.manifold import TSNE
    from sklearn.decomposition import NMF
    #import seaborn as sns
    a = RA4CoupleSampler()
    X = a.X_atac
    Y = a.Y_atac
    print X.shape, Y.shape,np.min(X),np.max(X),np.mean(X)
    sys.exit()
    nb_classes=3
    model = NMF(n_components=nb_classes, init='random', random_state=0, solver='cd', max_iter=1000)
    W10 = model.fit_transform(X) #(n_samples,K)
    H10 = model.components_ #(K x n_feats)
    S10 = np.argmax(W10, 1)
    print W10.shape, H10.shape,S10.shape
    nmi1 = normalized_mutual_info_score(Y , S10)
    ari1 = adjusted_rand_score(Y , S10)
    print nmi1,ari1
    sys.exit()

    
    tsne = TSNE(n_components=2, verbose=1, init='pca', random_state=0)
    tsne_enc = tsne.fit_transform(X)
    colors = cm.rainbow(np.linspace(0, 1, nb_classes))
    fig, ax = plt.subplots(figsize=(8,6))
    for iclass in range(0, nb_classes):
        idxs = Y==iclass
        ax.scatter(tsne_enc[idxs, 0],tsne_enc[idxs, 1],c=colors[iclass],s=3,edgecolor=None,label='%d'%iclass)
    plt.legend(title=r'Class', loc='best', numpoints=1, fontsize=8)
    ax.set_xlabel('tSNE-dim1', fontsize=18)
    ax.set_ylabel('tSNE-dim2', fontsize=18)
    plt.savefig('datasets/RAd4/tsn_scRNA.png')
    sys.exit()
    import random
    np.random.seed(2)
    genes = [item.split('\t')[0] for item in open('datasets/RAd4/scRNA_seq_RAd4.txt').readlines()]
    REs = [item.split('\t')[0] for item in open('datasets/RAd4/scATAC_seq_RAd4.txt').readlines()]
    relations = [item.strip().split('\t') for item in open('datasets/RAd4/RAd4_coupling_matrix.txt').readlines()]
    print len(genes),len(REs)
    A = np.zeros((len(genes),len(REs)))
    for each in relations:
        gene, RE = each[0],each[1]
        A[genes.index(gene)][REs.index(RE)] = float(each[2])
    np.save('datasets/RAd4/couple_mat.npy',A)
    sys.exit()
    a=RA4_Sampler('scrna')
    print a.X.shape,np.min(a.X),np.max(a.X),np.mean(a.X)

    sys.exit()
    name='InSilico'
    X = hdf5storage.loadmat('/home/chenshengquan/dataForQiaoLiu/%s.mat'%name)['count_mat']
    Y = hdf5storage.loadmat('/home/chenshengquan/dataForQiaoLiu/%s.mat'%name)['label_mat'][:,0]
    X = np.array(X).astype('float32')
    a=X/np.max(X)
    print X.shape,type(X),np.max(X)
    
    sys.exit()
    Y =[item[0] for item in Y]
    print Y[:10]
    uniq_types = list(set(Y))
    dic = {item:i for i,item in enumerate(uniq_types)}
    label = np.array([dic[item] for item in Y],dtype='float32')
    print len(label),set(label)
    np.save('data/scATAC/%s_label.npy'%name,label)
    sys.exit()
    
    ys = DataSampler()
    print ys.X.shape, ys.Y.shape
    sys.exit()
    
    # x = list(range(3,11))
    # y_kde = [0.989,	0.967,0.911,0.743,0.549,0.378,0.226,0.14]
    # y_made = [0.014,0.03,0.014,0.018,0.029,0.039,0.033,0.032]
    # y_nvp = [0.741,0.785,0.812,0.803,0.764,0.709,0.618,0.669]
    # y_maf = [0.879,0.847,0.796,0.744,0.69,0.663,0.654,0.595]
    # y_cf = [0.841,0.776,0.745,0.703,0.612,0.566,0.481,0.505]
    # y_is = [0.942,0.922,0.929,0.88,0.85,0.847,0.836,0.829]
    # lw=2
    # plt.figure()
    # plt.plot(x,y_kde,'*-',color=(135/255. ,135/255., 135/255.),label='KDE',linewidth=lw)
    # plt.plot(x,y_made,'o-',color=(253/255. ,198/255., 122/255.),label='MADE',linewidth=lw)
    # plt.plot(x,y_nvp,'s-',color=(87/255. ,104/255., 180/255.),label='Real NVP',linewidth=lw)
    # plt.plot(x,y_maf,'v-',color=(240/255. ,168/255., 171/255.),label='MAF',linewidth=lw)
    # plt.plot(x,y_cf,'D-',color=(215/255. ,220/255., 254/255.),label='Rountrip-CF',linewidth=lw)
    # plt.plot(x,y_is,'d-',color=(169/255. ,209/255., 142/255.),label='Rountrip-IS',linewidth=lw)
    # plt.legend(loc = "best")
    # plt.savefig('com.png',dpi=300)
    # sys.exit(0)
    ys = UCI_sampler('datasets/YearPredictionMSD/data.npy')
    #ys = hepmass_sampler()
    print ys.X_train.shape, ys.X_val.shape,ys.X_test.shape
    X = np.concatenate([ys.X_train,ys.X_val])
    gkde1 = gaussian_kde(X.T,'silverman')
    gkde2 = gaussian_kde(X.T,'scott')
    py_gau_kernel1=gkde1.logpdf(ys.X_test.T)
    py_gau_kernel2=gkde2.logpdf(ys.X_test.T)
    print np.mean(py_gau_kernel1),2.*np.std(py_gau_kernel1)/np.sqrt(len(py_gau_kernel1))
    print np.mean(py_gau_kernel2),2.*np.std(py_gau_kernel2)/np.sqrt(len(py_gau_kernel2))


    sys.exit()
    ys = Swiss_roll_sampler(N=20000)
    print ys.X_train.shape, ys.X_val.shape, ys.X_test.shape
    np.savez('data_swill_roll.npz',ys.X_train,ys.X_val,ys.X_test)
    sys.exit()
    
    # for dim in range(30,100,20):
    #     ys = Multi_dis_sampler(N=50000,dim=dim)
    #     print ys.X_train.shape, ys.X_val.shape, ys.X_test.shape
    #     np.savez('data_multi_v2_dim%d.npz'%dim,ys.X_train,ys.X_val,ys.X_test)
    # sys.exit()
    # a = ys.get_all_density(ys.X_test)
    # print a.shape
    # sys.exit()
    for dim in [5,10,30,50,100]:
        ys = GMM_indep_sampler(N=50000, sd=0.1, dim=dim, n_components=3, bound=1)
        #ys = GMM_sampler(N=10000,n_components=dim,dim=dim,sd=0.05)
        print ys.X_train.shape, ys.X_val.shape, ys.X_test.shape
        np.savez('data_indep_dim%d.npz'%dim,ys.X_train,ys.X_val,ys.X_test)

    sys.exit()
    # s=miniboone_sampler()
    # print s.X_train.shape,s.X_val.shape,s.X_test.shape,s.nb_train
    # s=gas_sampler()
    # print s.X_train.shape,s.X_val.shape,s.X_test.shape,s.nb_train
    # s=power_sampler()
    # print s.X_train.shape,s.X_val.shape,s.X_test.shape,s.nb_train
    # s=hepmass_sampler()
    # print s.X_train.shape,s.X_val.shape,s.X_test.shape,s.nb_train

    s=UCI_sampler('datasets/AReM/data.npy')
    print s.X_train.shape,s.X_val.shape,s.X_test.shape,s.nb_train
    print s.X_train[0,:5],s.X_val[0,:5]
    a = copy.copy(s.X_train)
    #a=s.X_train
    np.random.shuffle(a)
    print s.X_train[0,:5],s.X_val[0,:5]
    sys.exit()
    s=UCI_sampler('datasets/Protein/data.npy')
    print s.X_train.shape,s.X_val.shape,s.X_test.shape,s.nb_train
    s=UCI_sampler('datasets/Superconductivty/data.npy')
    print s.X_train.shape,s.X_val.shape,s.X_test.shape,s.nb_train
    s=UCI_sampler('datasets/YearPredictionMSD/data.npy')
    print s.X_train.shape,s.X_val.shape,s.X_test.shape,s.nb_train
    s=UCI_sampler('datasets/BankMarketing/data.npy')
    print s.X_train.shape,s.X_val.shape,s.X_test.shape,s.nb_train
    sys.exit()
    data_list,t_list=[],[]
    s = Cosine_sampler(block_size=10)
    for i in range(10):
        if i==0:
            data, theta, t = s.generate_data3(1000,i)
        else:
            data, theta, t = s.generate_data3(100,i,prior=theta)
        data_list.append(data[0,:])
        t_list.append(t)
        #log_prob,axis_x,axis_y,sampled_theta = s.get_posterior(np.concatenate(data_list,axis=0),np.concatenate(t_list,axis=0))
        #theta_refine = s.refine_posterior(np.concatenate(data_list,axis=0),theta,np.concatenate(t_list,axis=0))
        theta = np.random.uniform(0,0.2,size=theta.shape)
        theta_resample = s.adaptive_sampling(data[0,:],theta,t,chain_len=10000)

        print theta_resample.shape
        sys.exit()
        plt.figure(figsize=(5,15))
        fig, ax = plt.subplots(1, 2)
        ax[0].hist(sampled_theta[:,0], bins=30,alpha=0.75)
        ax[1].hist(sampled_theta[:,1], bins=30,alpha=0.75)
        plt.savefig('data/bayes_infer/posterior2/iter_%d_sampled_posterior.png'%i)
        plt.close('all')
        prob = np.exp(log_prob)
        z_min, z_max = -np.abs(prob).max(), np.abs(prob).max()
        plt.imshow(prob, cmap='RdBu', vmin=z_min, vmax=z_max,
                extent=[axis_x.min(), axis_x.max(), axis_y.min(), axis_y.max()],
                interpolation='nearest', origin='lower')
        plt.title('Beyesian posterior')
        plt.colorbar()
        plt.savefig('data/bayes_infer/posterior2/iter_%d_posterior_2d.png'%i)
        plt.close('all')
        
    sys.exit()
    data, theta, t = s.generate_data3(50000,0)
    print data[0],t
    data, theta, t = s.generate_data3(50000,1,prior=np.zeros(theta.shape))
    print data[0],t
    data, theta, t = s.generate_data3(50000,2,prior=np.zeros(theta.shape))
    print data[0],t
    sys.exit(0)
    log_prob,axis_x,axis_y,sampled_theta = s.get_posterior(data[0,:],t,0)
    fig, ax = plt.subplots(1, 2)
    ax[0].hist(sampled_theta[:,0], bins=100,alpha=0.75)
    ax[1].hist(sampled_theta[:,1], bins=100,alpha=0.75)
    plt.savefig('sampled_posterior.png')
    plt.close()
    print log_prob.shape
    print np.max(log_prob)
    print(np.where(log_prob==np.max(log_prob)))
    # sns.set(style='whitegrid', color_codes=True)
    # sns.heatmap(np.exp(prob))
    prob = np.exp(log_prob)
    z_min, z_max = -np.abs(prob).max(), np.abs(prob).max()
    plt.imshow(prob, cmap='RdBu', vmin=z_min, vmax=z_max,
            extent=[axis_x.min(), axis_x.max(), axis_y.min(), axis_y.max()],
            interpolation='nearest', origin='lower')
    plt.title('posterior')
    plt.colorbar()
    plt.savefig('posterior_map.png')
    sys.exit()
    print data.shape
    print theta[0]
    plt.plot(data[0,:50])
    plt.xlabel('t')
    plt.ylabel('y_t')
    plt.savefig('a.png')
    sys.exit()
    
    X, Y = np.mgrid[-2:2:100j, -2:2:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    print positions.shape

    kernel = stats.gaussian_kde(values)
    Z = kernel(positions)
    #Z = np.reshape(kernel(positions).T, X.shape)
    print Z.shape
    Z2 = kernel.pdf(positions)
    print Z[0:4],Z2[0:4]
    X = np.random.normal(size=(100,3))
    kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(X)
    log_density = kde.score_samples(X)
    log_density1 = kde.score(X)
    print len(log_density)
    print log_density[:3],sum(log_density)
    print log_density1
    a=np.log(2)
    print np.e**a
    sys.exit()
    a=np.ones((3,5))
    c=np.ones((2,5))
    b.append(a)
    b.append(c)
    print np.concatenate(b,axis=0)
    sys.exit()
    ys = Gaussian_sampler(N=10000,mean=np.zeros(5),sd=1.0)
    print ys.get_batch(3)
    print ys.get_batch(3)
    print ys.get_batch(3)
    a=np.array([0.0314, 0.9967, 0.0107, 19.6797, -1.1528])
    a=np.ones((4,3))
    import random
    print random.sample(a,2)
    sys.exit()
    ys = SV_sampler(np.array([0.0314, 0.9967, 0.0107, 19.6797, -1.1528]),1)


