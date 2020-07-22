import metric
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from sklearn.metrics.cluster import homogeneity_score, adjusted_mutual_info_score
import numpy as np
import random
import sys,os
from scipy.io import loadmat
from sklearn.metrics import confusion_matrix
import pandas as pd 
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid", {'axes.grid' : False})
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 
matplotlib.rcParams.update({'font.size': 22})

def sort_by_classes(X, y, y_infer, classes):
    #classes = np.unique(y)
    classes = np.unique(y_infer)
    index = []
    for c in classes:
        #ind = np.where(y==c)[0]
        ind = np.where(y_infer==c)[0]
        group_idx = np.argsort(y[ind])
        ind = ind[group_idx]
        index.append(ind)
    index = np.concatenate(index)
    #X = X.iloc[:, index]
    X = X[:, index]
    y = y[index]
    y_infer = y_infer[index]
    return X, y, y_infer,classes, index

def plot_heatmap(X, y, y_infer, classes=None, y_pred=None, row_labels=None, colormap=None, row_cluster=False,
                 cax_title='', xlabel='', ylabel='', yticklabels='', legend_font=10, 
                 show_legend=True, show_cax=True, tick_color='black', ncol=3,
                 bbox_to_anchor=(0.5, 1.3), position=(0.8, 0.78, .1, .04), return_grid=False,
                 save=None, **kw):
    """
    plot hidden code heatmap with labels

    Params:
        X: fxn array, n is sample number, f is feature
        y: a array of labels for n elements or a list of array
    """
    #y=['a','c','c','a',...]
    #y_infer = [0,1,1,0,...]
    import matplotlib.patches as mpatches  # add legend
    X, y, y_infer, classes, index = sort_by_classes(X, y, y_infer,classes)
    use_y_color=True
    if use_y_color:
        colormap = plt.cm.tab20
        colors = {c:colormap(i) for i, c in enumerate(classes)}
        col_colors = [colors[c] for c in y_infer]
    else:
        classes = np.unique(y)
        colormap = sns.color_palette('husl', n_colors=len(classes),desat=0.7)
        colors = {c:colormap[i] for i, c in enumerate(classes)}
        col_colors = [colors[c] for c in y]


        
    legend_TN = [mpatches.Patch(color=color, label=c) for c, color in colors.items()]

    if row_labels is not None:
        row_colors = [ colors[c] for c in row_labels ]
        kw.update({'row_colors':row_colors})

    kw.update({'col_colors':col_colors})

    cbar_kws={"orientation": "horizontal"}
    grid = sns.clustermap(X, yticklabels=True, 
            col_cluster=False,
            row_cluster=row_cluster,
            cbar_kws=cbar_kws, **kw)
    if show_cax:
        grid.cax.set_position(position)
        grid.cax.tick_params(length=1, labelsize=4, rotation=0)
        grid.cax.set_title(cax_title, fontsize=6, y=0.35)

    if show_legend:
        grid.ax_heatmap.legend(loc='upper center', 
                               bbox_to_anchor=bbox_to_anchor, 
                               handles=legend_TN, 
                               fontsize=legend_font, 
                               frameon=False, 
                               ncol=ncol)
        grid.ax_col_colors.tick_params(labelsize=6, length=0, labelcolor='orange')

    if (row_cluster==True) and (yticklabels is not ''):
        yticklabels = yticklabels[grid.dendrogram_row.reordered_ind]

    grid.ax_heatmap.set_xlabel(xlabel)
    grid.ax_heatmap.set_ylabel(ylabel, fontsize=8)
    grid.ax_heatmap.set_xticklabels('')
    grid.ax_heatmap.set_yticklabels(yticklabels, color=tick_color)
    grid.ax_heatmap.yaxis.set_label_position('left')
    grid.ax_heatmap.tick_params(axis='x', length=0)
    grid.ax_heatmap.tick_params(axis='y', labelsize=6, length=0, rotation=0, labelleft=True, labelright=False)
    grid.ax_row_dendrogram.set_visible(False)
    grid.cax.set_visible(show_cax)
    grid.row_color_labels = classes

    if save:
        plt.savefig(save, format='png', bbox_inches='tight',dpi=600)
    else:
        plt.show()
    if return_grid:
        return grid

def plot_embedding(X, labels, classes=None, method='tSNE', cmap='tab20', figsize=(8, 8), markersize=15, dpi=600,marker=None,
                   return_emb=False, save=False, save_emb=False, show_legend=True, show_axis_label=True, **legend_params):
    if marker is not None:
        X = np.concatenate([X, marker], axis=0)
    N = len(labels)
    if X.shape[1] != 2:
        if method == 'tSNE':
            from sklearn.manifold import TSNE
            #X = TSNE(n_components=2, random_state=124,metric='correlation').fit_transform(X)
            X = TSNE(n_components=2, random_state=124).fit_transform(X)
        if method == 'PCA':
            from sklearn.decomposition import PCA
            X = PCA(n_components=2, random_state=124).fit_transform(X)
        if method == 'UMAP':
            from umap import UMAP
            X = UMAP(n_neighbors=15, min_dist=0.1, metric='correlation').fit_transform(X)
    labels = np.array(labels)
    plt.figure(figsize=figsize)
    if classes is None:
        classes = np.unique(labels)
    #tab10, tab20, husl, hls
    if cmap is not None:
        cmap = cmap
    elif len(classes) <= 10:
        cmap = 'tab10'
    elif len(classes) <= 20:
        cmap = 'tab20'
    else:
        cmap = 'husl'
    #colors = sns.color_palette('husl', n_colors=len(classes),desat=0.7)
    colors = sns.husl_palette(len(classes), s=.8)
    #markersize = 80
    #colors = sns.husl_palette(7, s=.8)
    for i, c in enumerate(classes):
        dic = {0:'*',1:'^',2:'P'}
        idx1,idx2=[],[]
        for j in range(N):
            if labels[j]==c and donors[j]=='BM0828':
                idx1.append(j)
            if labels[j]==c and donors[j]=='BM1077':
                idx2.append(j)
        plt.scatter(X[:N][idx1, 0], X[:N][idx1, 1], s=50, color=colors[i], label=c)
        plt.scatter(X[:N][idx2, 0], X[:N][idx2, 1], s=80, color=colors[i], label=c,marker='^')
        
        
        #plt.scatter(X[:N][labels==c, 0], X[:N][labels==c, 1], s=markersize, color=colors[2*i-1], label=c,marker='P')
        #plt.scatter(X[:N][labels==c, 0], X[:N][labels==c, 1], s=markersize, color=colors[i], label=c)
    if marker is not None:
        plt.scatter(X[N:, 0], X[N:, 1], s=10*markersize, color='black', marker='*')
    
    legend_params_ = {'loc': 'center left',
                     'bbox_to_anchor':(1.0, 0.45),
                     'fontsize': 20,
                     'ncol': 1,
                     'frameon': False,
                     'markerscale': 1.5
                    }
    legend_params_.update(**legend_params)
    if show_legend:
        plt.legend(**legend_params_)
    sns.despine(offset=10, trim=True)
    if show_axis_label:
        plt.xlabel(method+' dim 1', fontsize=12)
        plt.ylabel(method+' dim 2', fontsize=12)

    if save:
        plt.savefig(save, format='png', bbox_inches='tight',dpi=dpi)

#visulizat at given gradient colors
def plot_embedding_v2(X, values, classes=None, method='tSNE', cmap='tab20', figsize=(8, 8), markersize=15, dpi=600,marker=None,
                   return_emb=False, save=False, save_emb=False, show_legend=False, show_axis_label=True, **legend_params):
    if marker is not None:
        X = np.concatenate([X, marker], axis=0)
    N = X.shape[0]
    if X.shape[1] != 2:
        if method == 'tSNE':
            from sklearn.manifold import TSNE
            #X = TSNE(n_components=2, random_state=124,metric='correlation').fit_transform(X)
            X = TSNE(n_components=2, random_state=124).fit_transform(X)
        if method == 'PCA':
            from sklearn.decomposition import PCA
            X = PCA(n_components=2, random_state=124).fit_transform(X)
        if method == 'UMAP':
            from umap import UMAP
            X = UMAP(n_neighbors=15, min_dist=0.1, metric='correlation').fit_transform(X)

    cmap = 'RdBu_r'
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    sm = ScalarMappable(norm=Normalize(vmin=-np.max(values), vmax=np.max(values)), cmap='RdBu_r')
    #colors = sns.color_palette('husl', n_colors=len(classes),desat=0.7)
    #colors = sns.husl_palette(len(classes), s=.8)
    for i in range(N):
        plt.scatter(X[i,0], X[i,1], s=markersize, color=sm.to_rgba(values[i]))

    legend_params_ = {'loc': 'center left',
                     'bbox_to_anchor':(1.0, 0.45),
                     'fontsize': 20,
                     'ncol': 1,
                     'frameon': False,
                     'markerscale': 1.5
                    }
    legend_params_.update(**legend_params)
    if show_legend:
        plt.legend(**legend_params_)
    sns.despine(offset=10, trim=True)
    if show_axis_label:
        plt.xlabel(method+' dim 1', fontsize=12)
        plt.ylabel(method+' dim 2', fontsize=12)

    if save:
        plt.savefig(save, format='png', bbox_inches='tight',dpi=dpi)

def analyze_forebrain():
    idx = 7600
    dic = {0:'IN1',1:'AC',2:'EX3',3:'MG',4:'EX1',5:'OC',6:'EX2',7:'IN2'}
    data = np.load('%s/data_at_%d.npz'%(path,idx))
    data_x_,data_x_onehot_,label_y = data['arr_0'],data['arr_1'],data['arr_2']
    label = np.array([dic[item] for item in label_y])
    plot_embedding(data_x_,label,save='%s/embedding_tsne_cont_%d.png'%(path,idx))
    plot_embedding(data_x_onehot_,label,save='%s/embedding_tsne_onehot_%d.png'%(path,idx))
    plot_embedding(np.concatenate([data_x_, data_x_onehot_],axis=1),label,save='%s/embedding_tsne_combine_%d.png'%(path,idx))


def analyze_splenocyte():
    #idx = 19899+1
    idx = 1900
    uniq_label = ['CD27+_Natural_Killer', 'CD27-_Natural_Killer', 'Dendritic_cell', 'Follicular_B',\
        'Granulocyte', 'Macrophage', 'Marginal_Zone_B', 'Memory_CD8_T', 'Naive_CD4_T', \
            'Naive_CD8_T','Regulatory_T', 'Transitional_B']
    data = np.load('%s/data_at_%d.npz'%(path,idx))
    data_x_,data_x_onehot_,label_y = data['arr_0'],data['arr_1'],data['arr_2']
    label = np.array([uniq_label[item] for item in label_y])
    plot_embedding(data_x_,label,save='%s/embedding_tsne_cont_%d.png'%(path,idx))
    plot_embedding(data_x_onehot_,label,save='%s/embedding_tsne_onehot_%d.png'%(path,idx))
    plot_embedding(np.concatenate([data_x_, data_x_onehot_],axis=1),label,save='%s/embedding_tsne_combine_%d.png'%(path,idx))

def analyze_mouse_atlas():
    #idx = 19899+1
    ratio=0.1
    idx = 14899+1
    labels = np.array([item.strip() for item in open('datasets/scATAC/Mouse_atlas/label_%s.txt'%str(ratio)).readlines()])
    data = np.load('%s/data_at_%d.npz'%(path,idx))
    data_x_,data_x_onehot_,label_y = data['arr_0'],data['arr_1'],data['arr_2']
    plot_embedding(data_x_,labels,save='%s/embedding_tsne_cont_%d.png'%(path,idx))
    plot_embedding(data_x_onehot_,labels,save='%s/embedding_tsne_onehot_%d.png'%(path,idx))
    plot_embedding(np.concatenate([data_x_, data_x_onehot_],axis=1),labels,save='%s/embedding_tsne_combine_%d.png'%(path,idx))


def analyze_SCALE():
    ratio=0.1
    data = np.loadtxt('SCALE/%s/feature.txt'%path,delimiter='\t',usecols=range(1,1+10))
    labels = [item.strip() for item in open('datasets/scATAC/Mouse_atlas/label_%s.txt'%str(ratio)).readlines()]
    uniq_labels = list(np.unique(labels))
    Y = np.array([uniq_labels.index(item) for item in labels])
    print(data.shape)
    assert len(Y)==data.shape[0]
    km = KMeans(n_clusters=len(uniq_labels), random_state=0).fit(data)
    label_kmeans = km.labels_
    purity = metric.compute_purity(label_kmeans, Y)
    nmi = normalized_mutual_info_score(Y, label_kmeans)
    ari = adjusted_rand_score(Y, label_kmeans)
    print(nmi,ari,purity)
    plot_embedding(data,np.array(labels),save='SCALE/embedding_tsne_cont_%s.png'%path)

def analyze_cistopic():
    ratio=0.7
    df = pd.read_csv('cisTopic/results/%s'%path,sep=' ',header=0,index_col=[0])
    data = df.values
    data = data*1.0 / np.sum(data,axis=0)
    data = data.T
    #mouse atlas
    labels = [item for item in open('datasets/scATAC/Mouse_atlas/label_%s.txt'%str(ratio)).readlines()]
    #forebrain
    #labels = loadmat('datasets/scATAC/scATAC-seq_data_for_liuqiao/REN/original/cell_labels.mat')['cell_labels']
    uniq_labels = list(np.unique(labels))
    Y = np.array([uniq_labels.index(item) for item in labels]) 
    #GMvsHL,
    #name='GMvsHL'
    #Y = np.load('datasets/scATAC/%s_label.npy'%name).astype('int64')
    print(data.shape)
    assert len(Y)==data.shape[0]
    km = KMeans(n_clusters=len(uniq_labels), random_state=0).fit(data)
    label_kmeans = km.labels_
    purity = metric.compute_purity(label_kmeans, Y)
    nmi = normalized_mutual_info_score(Y, label_kmeans)
    ari = adjusted_rand_score(Y, label_kmeans)
    print(nmi,ari,purity)
    
#generate input for baseline methods
def generate_benchmark_input():
    pd_data = pd.read_csv('%s/sc_mat.txt'%path,sep='\t',header=0,index_col=[0])
    #chrom_list = [str(i+1) for i in range(22)]
    #chrom_list += ['X','Y']
    labels = [item.strip() for item  in open('%s/label.txt'%path).readlines()]
    regions = [item for item in list(pd_data.index) if item.split(':')[0][3:]!='Un']
    pd_data = pd_data.loc[regions,:]
    regions = [item.split(':')[0]+'_'+item.strip().split(':')[1].split('-')[0]+'_'+item.strip().split(':')[1].split('-')[1] \
        for item in regions]
    pd_data.index = regions
    f_out=open('%s/metadata.tsv'%path,'w')
    f_out.write('\tlabel\n')
    for i in range(len(labels)):
        f_out.write('cell%d\t%s\n'%(i,labels[i]))
    f_out.close()
    pd_data.to_csv('%s/benchmark_count.csv'%path,sep='\t',float_format='%.0f')

def analyze_all_methods():
    #load_label
    labels = [item.strip() for item  in open('datasets/scATAC/%s/label.txt'%dataset).readlines()]
    uniq_labels = list(np.unique(labels))
    Y = np.array([uniq_labels.index(item) for item in labels])
    #load latent feature
    #SCALE
    data_scale = np.loadtxt('SCALE/%s_l=%d/feature.txt'%(dataset,latent_dim),delimiter='\t',usecols=range(1,1+latent_dim))
    scale_idx = [int(item.split('\t')[0][4:]) for item in open('SCALE/%s_l=%d/cluster_assignments.txt'%(dataset,latent_dim)).readlines()]
    Y_scale = Y[scale_idx]
    print('SCALE', data_scale.shape)
    #cluster_eval(Y_scale,data_scale)
    plot_embedding(data_scale,np.array(labels)[scale_idx],save='figs/%s/scale_tsne_%d.png'%(dataset,latent_dim))
    #cisTopic
    df = pd.read_csv('cisTopic/results/%s_%d_pre.csv'%(dataset,latent_dim),sep=' ',header=0,index_col=[0])
    data = df.values
    data = data*1.0 / np.sum(data,axis=0)
    data_cistopic = data.T
    print('cisTopic',data_cistopic.shape)
    #cluster_eval(Y,data_cistopic)
    plot_embedding(data_cistopic,labels,save='figs/%s/cistopic_tsne_%d.png'%(dataset,latent_dim),dpi=600)
    #cisTopic
    #Cusanovich2018
    df = pd.read_csv('scATAC_benchmarks/Cusanovich2018_%s_%d.csv'%(dataset,latent_dim),sep=' ',header=0,index_col=[0])
    data_cusanovich2018 = df.values.T
    print('cusanovich2018',data_cusanovich2018.shape)
    #cluster_eval(Y,data_cusanovich2018)
    plot_embedding(data_cusanovich2018,labels,save='figs/%s/cusanovich2018_tsne_%d.png'%(dataset,latent_dim),dpi=600)
    #Scasat
    df = pd.read_csv('scATAC_benchmarks/Scasat_%s_%d.csv'%(dataset,latent_dim),sep=' ',header=0,index_col=[0])
    data_scasat = df.values.T
    print('scasat',data_scasat.shape)
    #cluster_eval(Y,data_scasat)
    plot_embedding(data_scasat,labels,save='figs/%s/scasat_tsne_%d.png'%(dataset,latent_dim),dpi=600)
    #SnapATAC
    df = pd.read_csv('scATAC_benchmarks/SnapATAC_%s_%d.csv'%(dataset,latent_dim),sep=' ',header=0,index_col=[0])
    data_snapatac = df.values.T
    print('SnapATAC',data_snapatac.shape)
    #cluster_eval(Y,data_snapatac)
    plot_embedding(data_snapatac,labels,save='figs/%s/snapatac_tsne_%d.png'%(dataset,latent_dim),dpi=600)
    return None

def confusion_mat(labels_true,labels_pre):
    #donors
    #['CLP', 'LMPP', 'MPP'] --> 0,1,2 for lables_trues
    n_clusters = len(np.unique(labels_true))
    mat = np.zeros((n_clusters,2*n_clusters))
    dic={0:2,1:1,2:0}
    for i in range(len(donors)):
        if donors[i]=='BM0828':
            mat[dic[labels_pre[i]]][labels_true[i]]+=1
        elif donors[i]=='BM1077':
            mat[dic[labels_pre[i]]][labels_true[i]+3]+=1
        else:
            print('Wrong donor label!')
            sys.exit()
    fig, ax = plt.subplots(figsize=(10,5))
    sns.heatmap(mat, annot=True,ax=ax,square=True,cmap='Blues')
    plt.savefig('figs/BM0828_BM1077/confusion_mat_scRT.png',dpi=600)
    sys.exit()
    return None


#given true and latent feature
def cluster_eval(labels_true,latent_feat):
    assert len(labels_true)==latent_feat.shape[0]
    n_clusters = len(np.unique(labels_true))
    km = KMeans(n_clusters=n_clusters, n_init=20, random_state=0).fit(latent_feat)
    label_kmeans = km.labels_
    confusion_mat(labels_true,label_kmeans)
    purity = metric.compute_purity(label_kmeans, labels_true)
    nmi = normalized_mutual_info_score(labels_true, label_kmeans)
    ari = adjusted_rand_score(labels_true, label_kmeans)
    homogeneity = homogeneity_score(labels_true, label_kmeans)
    ami = adjusted_mutual_info_score(labels_true, label_kmeans)
    print('NMI = {}, ARI = {}, Purity = {},AMI = {}, Homogeneity = {}'.format(nmi,ari,purity,ami,homogeneity))

#directly given true and predicted label
def cluster_eval_v2(labels_true,labels_infer):
    purity = metric.compute_purity(labels_infer, labels_true)
    nmi = normalized_mutual_info_score(labels_true, labels_infer)
    ari = adjusted_rand_score(labels_true, labels_infer)
    homogeneity = homogeneity_score(labels_true, labels_infer)
    ami = adjusted_mutual_info_score(labels_true, labels_infer)
    print('NMI = {}, ARI = {}, Purity = {},AMI = {}, Homogeneity = {}'.format(nmi,ari,purity,ami,homogeneity))


#automatically scan all exps
def analyze_scRT():
    results = []
    for exp in os.listdir('results/%s'%dataset):
    #for exp in os.listdir('results_6.28/%s'%dataset):
        if os.path.exists('results/%s/%s/log.txt'%(dataset,exp)):
            model = 'v1' if 'v1' in exp else 'v0'
            ratio = float(exp[-3:])
            metrics = [[float(each) for each in item.strip().split('\t')] for item in open('results/%s/%s/log.txt'%(dataset,exp)).readlines()[:-1] if item.startswith('0')] 
            metrics_rank_nmi = sorted(metrics,key=lambda x:x[0],reverse=True)
            metrics_rank_ari = sorted(metrics,key=lambda x:x[1],reverse=True)
            #print(model, ratio, metrics_rank_nmi[0],metrics_rank_ari[0])
            #print len(metrics_rank_nmi),len(metrics_rank_ari),exp,model
            results.append([model, ratio, metrics_rank_nmi[0],metrics_rank_ari[0],exp])
    results.sort(key=lambda a:(a[1],a[0],-a[2][0]))
    for each in results:
        print(each)

#given a specific exp
def analyze_scRT_v2():
    #load_label
    labels = [item.strip() for item  in open('datasets/scATAC/%s/label.txt'%dataset).readlines()]
    uniq_labels = list(np.unique(labels))
    Y = np.array([uniq_labels.index(item) for item in labels])
    print(Y.shape)
    if dataset=='InSilico':
        data = np.load('results_6.28/InSilico/cluster_20200626_132038_x_dim=10_y_dim=20_alpha=10.0_beta=10.0_ratio=0.0/data_at_2700.npz')
        data_x_,data_x_onehot_,label_y = data['arr_0'],data['arr_1'],data['arr_2']
        feat_cont = data_x_[:,:latent_dim]
        feat_logits = data_x_[:,latent_dim:]
        label_infer = np.argmax(data_x_onehot_, axis=1)
        cluster_eval_v2(label_y,label_infer)
        print(data_x_.shape)
        plot_embedding(data_x_,labels,save='figs/%s/scRT_tsne_cont_logits_%d.png'%(dataset,latent_dim),dpi=600)
        plot_embedding(feat_logits,labels,save='figs/%s/scRT_tsne_logits_%d.png'%(dataset,latent_dim),dpi=600)
        plot_embedding(feat_cont,labels,save='figs/%s/scRT_tsne_cont_%d.png'%(dataset,latent_dim),dpi=600)
        plot_embedding(data_x_onehot_,labels,save='figs/%s/scRT_tsne_onehot_%d.png'%(dataset,latent_dim),dpi=600)
    if dataset=='Forebrain':
        data = np.load('data/cluster_20200607_115059_Forebrain_x_dim=20_y_dim=20_alpha=10.0_beta=10.0/data_at_11200.npz')
        data_x_,data_x_onehot_,label_y = data['arr_0'],data['arr_1'],data['arr_2']
        print(data_x_.shape)
        label_infer = np.argmax(data_x_onehot_, axis=1)
        cluster_eval_v2(label_y,label_infer)
        plot_embedding(data_x_,labels,save='figs/%s/scRT_tsne_cont_logits_%d.png'%(dataset,latent_dim),dpi=600)
        plot_embedding(data_x_onehot_,labels,save='figs/%s/scRT_tsne_onehot_%d.png'%(dataset,latent_dim),dpi=600)
    if dataset=="Splenocyte":
        data = np.load('results_6.28/Splenocyte/cluster_20200625_030854_x_dim=10_y_dim=20_alpha=10.0_beta=10.0_ratio=0.0/data_at_6300.npz')
        data_x_,data_x_onehot_,label_y = data['arr_0'],data['arr_1'],data['arr_2']
        feat_cont = data_x_[:,:latent_dim]
        feat_logits = data_x_[:,latent_dim:]
        label_infer = np.argmax(data_x_onehot_, axis=1)
        cluster_eval_v2(label_y,label_infer)
        print(data_x_.shape)
        assert data_x_.shape[0]==len(labels)
        plot_embedding(data_x_,labels,save='figs/%s/scRT_tsne_cont_logits_%d.png'%(dataset,latent_dim),dpi=600)
        plot_embedding(feat_logits,labels,save='figs/%s/scRT_tsne_logits_%d.png'%(dataset,latent_dim),dpi=600)
        plot_embedding(feat_cont,labels,save='figs/%s/scRT_tsne_cont_%d.png'%(dataset,latent_dim),dpi=600)
        plot_embedding(data_x_onehot_,labels,save='figs/%s/scRT_tsne_onehot_%d.png'%(dataset,latent_dim),dpi=600)
    if dataset=='All_blood':
        data = np.load('results_6.28/All_blood/cluster_20200626_162055_x_dim=10_y_dim=20_alpha=10.0_beta=10.0_ratio=0.7/data_at_10000.npz')
        data_x_,data_x_onehot_,label_y = data['arr_0'],data['arr_1'],data['arr_2']
        feat_cont = data_x_[:,:latent_dim]
        feat_logits = data_x_[:,latent_dim:]
        label_infer = np.argmax(data_x_onehot_, axis=1)
        cluster_eval_v2(label_y,label_infer)
        print(data_x_.shape)
        assert data_x_.shape[0]==len(labels)
        plot_embedding(data_x_,labels,save='figs/%s/scRT_tsne_cont_logits_%d.png'%(dataset,latent_dim),dpi=600)
        plot_embedding(feat_logits,labels,save='figs/%s/scRT_tsne_logits_%d.png'%(dataset,latent_dim),dpi=600)
        plot_embedding(feat_cont,labels,save='figs/%s/scRT_tsne_cont_%d.png'%(dataset,latent_dim),dpi=600)
        plot_embedding(data_x_onehot_,labels,save='figs/%s/scRT_tsne_onehot_%d.png'%(dataset,latent_dim),dpi=600)
    if dataset=="GMvsHL":
        data = np.load('data/cluster_20200525_054226_GMvsHL_x_dim=20_y_dim=20_alpha=10.0_beta=10.0/data_at_900.npz')
        data_x_,data_x_onehot_,label_y = data['arr_0'],data['arr_1'],data['arr_2']
        print(data_x_.shape)
        label_infer = np.argmax(data_x_onehot_, axis=1)
        cluster_eval_v2(label_y,label_infer)
        plot_embedding(data_x_,labels,save='figs/%s/scRT_tsne_cont_logits_%d.png'%(dataset,latent_dim),dpi=600)
        plot_embedding(data_x_onehot_,labels,save='figs/%s/scRT_tsne_onehot_%d.png'%(dataset,latent_dim),dpi=600)
    if dataset=="BM0828":
        data = np.load('results/BM0828/cluster_20200710_173928_x_dim=10_y_dim=20_alpha=10.0_beta=10.0_ratio=0.0/data_at_2400.npz')
        data_x_,data_x_onehot_,label_y = data['arr_0'],data['arr_1'],data['arr_2']
        feat_cont = data_x_[:,:latent_dim]
        feat_logits = data_x_[:,latent_dim:]
        label_infer = np.argmax(data_x_onehot_, axis=1)
        print(uniq_labels,Y.shape)
        # idx=[]
        # for i in range(len(uniq_labels)):
        #     idx+=[j for j,item in enumerate(label_y) if item==i]
        # print(len(idx))
        # label_y = label_y[idx]
        # label_infer= label_infer[idx]
        # data_x_onehot_ = data_x_onehot_[idx,:]
        # label_cmp = np.vstack([label_y,label_infer])
        # print(label_cmp.shape)
        # fig, ax = plt.subplots()
        # sns.heatmap(data_x_onehot_,ax = ax,cmap='rainbow')
        # plt.savefig('figs/%s/label_infered.png'%(dataset))       
        #fig, ax = plt.subplots(figsize=(10,2))
        #sns.heatmap(label_cmp*1./np.max(label_cmp),ax = ax,cmap='rainbow')
        #plt.savefig('figs/%s/label_cmp.png'%(dataset))

        cluster_eval_v2(label_y,label_infer)
        assert data_x_.shape[0]==len(labels)
        #umap
        # plot_embedding(data_x_,labels,method='UMAP',save='figs/%s/scRT_umap_cont_logits_%d.png'%(dataset,latent_dim),dpi=600)
        # plot_embedding(feat_logits,labels,method='UMAP',save='figs/%s/scRT_umap_logits_%d.png'%(dataset,latent_dim),dpi=600)
        # plot_embedding(feat_cont,labels,method='UMAP',save='figs/%s/scRT_umap_cont_%d.png'%(dataset,latent_dim),dpi=600)
        # plot_embedding(data_x_onehot_,labels,method='UMAP',save='figs/%s/scRT_umap_onehot_%d.png'%(dataset,latent_dim),dpi=600)
        #tsne
        plot_embedding(data_x_,labels,save='figs/%s/scRT_tsne_cont_logits_%d.png'%(dataset,latent_dim),dpi=600)
        plot_embedding(feat_logits,labels,save='figs/%s/scRT_tsne_logits_%d.png'%(dataset,latent_dim),dpi=600)
        plot_embedding(feat_cont,labels,save='figs/%s/scRT_tsne_cont_%d.png'%(dataset,latent_dim),dpi=600)
        plot_embedding(data_x_onehot_,labels,save='figs/%s/scRT_tsne_onehot_%d.png'%(dataset,latent_dim),dpi=600)
        #pca
        # plot_embedding(data_x_,labels,method='PCA',save='figs/%s/scRT_pca_cont_logits_%d.png'%(dataset,latent_dim),dpi=600)
        # plot_embedding(feat_logits,labels,method='PCA',save='figs/%s/scRT_pca_logits_%d.png'%(dataset,latent_dim),dpi=600)
        # plot_embedding(feat_cont,labels,method='PCA',save='figs/%s/scRT_pca_cont_%d.png'%(dataset,latent_dim),dpi=600)
        # plot_embedding(data_x_onehot_,labels,method='PCA',save='figs/%s/scRT_pca_onehot_%d.png'%(dataset,latent_dim),dpi=600)
    if dataset=="BM0828_BM1077":
        data = np.load('results/BM0828_BM1077/cluster_20200717_192338_x_dim=10_y_dim=20_alpha=10.0_beta=10.0_ratio=0.0/data_at_5500.npz')
        data_x_,data_x_onehot_,label_y = data['arr_0'],data['arr_1'],data['arr_2']
        feat_cont = data_x_[:,:latent_dim]
        feat_logits = data_x_[:,latent_dim:]
        label_infer = np.argmax(data_x_onehot_, axis=1)
        matplotlib.rc('xtick', labelsize=6) 
        matplotlib.rc('ytick', labelsize=6) 
        matplotlib.rcParams.update({'font.size': 6})
        if True:
            #Y, donors
            idx_all = []
            dic={'BM0828':0,'BM1077':1}
            label_donor = np.array([dic[item] for item in donors])
            for i in range(3):
                ind = np.where(Y==i)[0]
                group_idx = np.argsort(label_donor[ind])
                ind = ind[group_idx]
                idx_all.append(ind)
            idx_all = np.concatenate(idx_all)
            feat = data_x_[idx_all,:]
            #normalization
            #feat, cells * dims
            feat = (feat-np.mean(feat,axis=0))/np.std(feat,axis=0)
            feat = feat.T
            Y = Y[idx_all]
            label_donor = label_donor[idx_all]
            #colormap = sns.color_palette('husl', n_colors=6,desat=0.7)
            colormap = plt.cm.tab20
            col_colors = [colormap(2*item[0]+item[1]) for item in zip(Y,label_donor)]
            kw={}
            kw.update({'col_colors':col_colors})
            cbar_kws={"orientation": "horizontal"}
            grid = sns.clustermap(feat, yticklabels=True, figsize=(25, 10),cmap='RdBu_r',
                    col_cluster=False,
                    row_cluster=False,
                    cbar_kws=cbar_kws, **kw)
            plt.savefig('figs/%s/feat_heatmap.png'%dataset,dpi=600)
            sys.exit()

        cluster_eval_v2(label_y,label_infer)
        plot_embedding(feat_cont,labels,save='figs/%s/scRT_tsne_cont_%d_donor.png'%(dataset,latent_dim),dpi=600)
        plot_embedding(data_x_,labels,save='figs/%s/scRT_tsne_cont_logits_%d_donor.png'%(dataset,latent_dim),dpi=600)
        plot_embedding(data_x_onehot_,labels,save='figs/%s/scRT_tsne_onehot_%d_donor.png'%(dataset,latent_dim),dpi=600)
        confusion_mat(Y,label_infer)

def trajactory_infer(nb_classes = 3,org=False):
    if nb_classes == 3:
        data = np.load('results/BM0828_sub3/cluster_20200713_183137_x_dim=10_y_dim=20_alpha=10.0_beta=10.0_ratio=0.0/data_at_400.npz')
        data_x_,data_x_onehot_,label_y = data['arr_0'],data['arr_1'],data['arr_2']
        feat_cont = data_x_[:,:latent_dim]
        feat_logits = data_x_[:,latent_dim:]
        label_infer = np.argmax(data_x_onehot_, axis=1)
        cluster_eval_v2(label_y,label_infer)
        #tsne
        dic={0:'CLP',1:'LMPP',2:'MPP'}
        label_y = [dic[item] for item in label_y]
        plot_embedding(data_x_,label_y,cmap='tab10',save='figs/%s/scRT_tsne_cont_logits_markers_%d.png'%(dataset,latent_dim),dpi=600)
        #plot_embedding(feat_logits,label_y,save='figs/%s/scRT_tsne_logits_%d.png'%(dataset,latent_dim),dpi=600)
        #plot_embedding(feat_cont,label_y,save='figs/%s/scRT_tsne_cont_%d.png'%(dataset,latent_dim),dpi=600)
        #plot_embedding(data_x_onehot_,label_y,save='figs/%s/scRT_tsne_onehot_%d.png'%(dataset,latent_dim),dpi=600)
    if nb_classes ==2 :
        data = np.load('results/BM0828_sub2/cluster_20200714_062251_x_dim=10_y_dim=20_alpha=10.0_beta=10.0_ratio=0.0/data_at_400.npz')
        data_x_,data_x_onehot_,label_y = data['arr_0'],data['arr_1'],data['arr_2']
        feat_cont = data_x_[:,:latent_dim]
        feat_logits = data_x_[:,latent_dim:]
        label_infer = np.argmax(data_x_onehot_, axis=1)
        cluster_eval_v2(label_y,label_infer)
        #tsne
        dic={0:'CLP',1:'MPP'}
        label_y = [dic[item] for item in label_y]
        plot_embedding(data_x_,label_y,save='figs/%s/MPP_CLP/scRT_tsne_cont_logits_%d.png'%(dataset,latent_dim),dpi=600)
        plot_embedding(feat_logits,label_y,save='figs/%s/MPP_CLP/scRT_tsne_logits_%d.png'%(dataset,latent_dim),dpi=600)
        plot_embedding(feat_cont,label_y,save='figs/%s/MPP_CLP/scRT_tsne_cont_%d.png'%(dataset,latent_dim),dpi=600)
        plot_embedding(data_x_onehot_,label_y,save='figs/%s/MPP_CLP/scRT_tsne_onehot_%d.png'%(dataset,latent_dim),dpi=600)
       
    #org_pca
    if org:
        data = np.load('datasets/scATAC/BM0828/pca_sub3.npz')
        X, Y = data['arr_0'],data['arr_1']
        plot_embedding(X,label_y,save='figs/%s/scRT_pca_org.png'%dataset,dpi=600)

def multi_test(df,labels,anova=True):
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm
    from scipy.stats import mannwhitneyu
    uniq_labels = np.unique(labels)
    nb_classes = len(uniq_labels)
    p_values=[]
    for each in df.index:
        dic = {}
        dic['label'] = labels
        dic['value'] = df.loc[each].values
        df_motif = pd.DataFrame(dic)
        if anova:
            model = ols('value~C(label)',data=df_motif).fit()
            anova_table = anova_lm(model, typ = 2)
            p_values.append(anova_table.loc['C(label)'][-1])
        else:
            p_values.append(-np.std(dic['value']))
    return np.array(p_values)

def single_test(df,labels, thred=1e-5):
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm
    from scipy.stats import mannwhitneyu
    #patterns: 0; 0,5; 1,2,3; 4; 5; 6; 7
    nums = [10,   0,   8,   10, 10,10,5]
    nums = np.array(nums)
    data = df.values
    assert data.shape[1]==len(labels)
    all_idx = []
    #0 vs others,pattern 0
    motif_idx = []
    p_list = []
    for i in range(data.shape[0]):
        idx = labels==0
        idx_ = ~(labels==0) 
        p_value = mannwhitneyu(data[i,idx],data[i,idx_],alternative='greater')[1]
        if p_value < thred and np.max(data[i,idx])>0.8:
            motif_idx.append(i)
            p_list.append(p_value)
    motif_idx = np.array(motif_idx)[np.argsort(p_list)]
    all_idx.append(motif_idx[:nums[0]])
    #0,5 vs others, pattern 1
    motif_idx = []
    p_list = []
    for i in range(data.shape[0]):
        idx = [j for j,item in enumerate(labels) if item in [0,5]]
        idx_ = [j for j,item in enumerate(labels) if item not in [0,5]]
        p_value = mannwhitneyu(data[i,idx],data[i,idx_],alternative='greater')[1]
        if p_value < thred and np.max(data[i,idx])>0.8:
            motif_idx.append(i)
            p_list.append(p_value)
    motif_idx = np.array(motif_idx)[np.argsort(p_list)]
    all_idx.append(motif_idx[:nums[1]])

    #1,2,3 vs others,pattern 2
    motif_idx = []
    p_list = []
    for i in range(data.shape[0]):
        idx = [j for j,item in enumerate(labels) if item in [1,2,3]]
        idx_ = [j for j,item in enumerate(labels) if item not in [1,2,3]]
        p_value = mannwhitneyu(data[i,idx],data[i,idx_],alternative='greater')[1]
        if p_value < thred and np.max(data[i,idx])>0.5:
            motif_idx.append(i)
            p_list.append(p_value)
    motif_idx = np.array(motif_idx)[np.argsort(p_list)]
    all_idx.append(motif_idx[:nums[2]])
    #4 vs others, pattern 3
    motif_idx = []
    p_list = []
    for i in range(data.shape[0]):
        idx = labels==4
        idx_ = ~(labels==4) 
        p_value = mannwhitneyu(data[i,idx],data[i,idx_],alternative='greater')[1]
        if p_value < thred and np.max(data[i,idx])>0.3:
            motif_idx.append(i)
            p_list.append(p_value)
    motif_idx = np.array(motif_idx)[np.argsort(p_list)]
    all_idx.append(motif_idx[:nums[3]])
    #5 vs others,pattern 4
    motif_idx = []
    p_list = []
    for i in range(data.shape[0]):
        idx = labels==5
        idx_ = ~(labels==5) 
        p_value = mannwhitneyu(data[i,idx],data[i,idx_],alternative='greater')[1]
        if p_value < thred and np.max(data[i,idx])>0.8:
            motif_idx.append(i)
            p_list.append(p_value)
    motif_idx = np.array(motif_idx)[np.argsort(p_list)]
    all_idx.append(motif_idx[:nums[4]])
    #6 vs others, pattern 5
    motif_idx = []
    p_list = []
    for i in range(data.shape[0]):
        idx = labels==6
        idx_ = ~(labels==6) 
        p_value = mannwhitneyu(data[i,idx],data[i,idx_],alternative='greater')[1]
        if p_value < thred and np.max(data[i,idx])>0.8:
            motif_idx.append(i)
            p_list.append(p_value)
    motif_idx = np.array(motif_idx)[np.argsort(p_list)]
    all_idx.append(motif_idx[:nums[5]])
    #7 vs others, pattern 6
    motif_idx = []
    p_list = []
    for i in range(data.shape[0]):
        idx = labels==7
        idx_ = ~(labels==7) 
        p_value = mannwhitneyu(data[i,idx],data[i,idx_],alternative='greater')[1]
        if p_value < thred and np.max(data[i,idx])>0.5:
            motif_idx.append(i)
            p_list.append(p_value)
    motif_idx = np.array(motif_idx)[np.argsort(p_list)]
    all_idx.append(motif_idx[:nums[6]])
    print('number of each pattern',[len(item) for item in all_idx])
    all_idx = np.concatenate(all_idx)
    print len(all_idx),len(np.unique(all_idx))
    return all_idx

def motif_enrich(nb_top=None,filter_motif=True):
    data = np.load('data/cluster_20200607_115059_Forebrain_x_dim=20_y_dim=20_alpha=10.0_beta=10.0/data_at_11200.npz')
    data_x_onehot_ = data['arr_1']
    label_infer = np.argmax(data_x_onehot_, axis=1)
    df = pd.read_csv('motifenrich/chromVAR_Forebrain.csv',sep=' ',header=0,index_col=[0])
    labels = np.array([item.strip() for item  in open('datasets/scATAC/Forebrain/label.txt').readlines()])
    motifs = np.array([item.split('_')[1] for item in df.index])

    #re arrange labels
    dic1 = {'AC':0,'EX1':1,'EX2':2,'EX3':3,'IN1':4,'IN2':5,'MG':6,'OC':7}
    dic2 = {0:7,1:0,2:5,3:3,4:4,5:2,6:1,7:6}
    labels = np.array([dic1[item] for item in labels])
    label_infer = np.array([dic2[item] for item in label_infer])
    data = df.values
    #re arrange motifs
    motif_idx = single_test(df,labels)
    if filter_motif:
        data = data[motif_idx,:]
        motifs = motifs[motif_idx]

    p_values = multi_test(df,labels,False)
    
    if nb_top:
        #nb_top = 50
        ind = np.argpartition(-p_values, -nb_top)[-nb_top:]
        data = data[ind,:]
        motifs = motifs[ind]
    print(np.max(data),np.min(data),np.mean(data))
    # data from up to down, but y_ticks label from down to up, note this setting
    plot_heatmap(data, labels, label_infer, 
                #figsize=(6, 10), cmap='RdBu_r', vmax=20, vmin=-10, center=0,
                figsize=(6, 10), cmap='RdBu_r',center=0,vmax=np.max(data), vmin=-np.max(data),
                ylabel='%d motifs'%len(motifs), yticklabels=motifs[::-1], 
                cax_title='', legend_font=6, ncol=1,
                bbox_to_anchor=(1.1, 1.1), position=(0.92, 0.15, .08, .04),save='motifenrich/motif_heatmap_rerange.png')

def motif_enrich_visual():
    df = pd.read_csv('motifenrich/chromVAR_Forebrain.csv',sep=' ',header=0,index_col=[0])
    motif = 'MA0027.2_EN1' #AX
    motif = 'MA0668.1_NEUROD2' #EX1-3
    motif = 'MA0722.1_VAX1' #IN2
    motif = 'MA0775.1_MEIS3' #IN1
    motif = 'MA0080.4_SPI1' #MG
    #motif = 'MA0077.1_SOX9' #
    #motif = 'MA0687.1_SPIC' #  NeuroD1
    #motif = 'MA0027.2_EN1'
    #motif = 'MA0826.1_OLIG1'
    #motif = 'MA0722.1_VAX1'
    
    scores = df.loc[motif].values
    data = np.load('data/cluster_20200607_115059_Forebrain_x_dim=20_y_dim=20_alpha=10.0_beta=10.0/data_at_11200.npz')
    data_x_,data_x_onehot_,label_y = data['arr_0'],data['arr_1'],data['arr_2']
    print(data_x_.shape)
    label_infer = np.argmax(data_x_onehot_, axis=1)
    cluster_eval_v2(label_y,label_infer)
    plot_embedding_v2(data_x_,scores,save='motifenrich/%s.png'%motif,dpi=600)




if __name__=="__main__":
    dataset = sys.argv[1]
    latent_dim  = int(sys.argv[2])
    path = 'datasets/scATAC/BM0828_BM1077'
    #generate_benchmark_input()
    #trajactory_infer()
    #motif_enrich()
    #motif_enrich_visual()
    #analyze_scRT()
    donors = [item.strip() for item  in open('datasets/scATAC/%s/donor.txt'%dataset).readlines()]
    #analyze_scRT_v2()
    analyze_all_methods()




