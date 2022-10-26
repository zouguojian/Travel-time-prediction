# -- coding: utf-8 --
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
from model import tf_utils
import tensorflow as tf
import seaborn as sns

def FC(x, units, activations, bn, bn_decay, is_training, use_bias=True):
    if isinstance(units, int):
        units = [units]
        activations = [activations]
    elif isinstance(units, tuple):
        units = list(units)
        activations = list(activations)
    assert type(units) == list
    for num_unit, activation in zip(units, activations):
        x = tf_utils.conv2d(
            x, output_dims=num_unit, kernel_size=[1, 1], stride=[1, 1],
            padding='VALID', use_bias=use_bias, activation=activation,
            bn=bn, bn_decay=bn_decay, is_training=is_training)
    return x

def gatedFusion(HS, HT, D, bn, bn_decay, is_training):
    '''
    gated fusion
    HS:     [batch_size, num_step, N, D]
    HT:     [batch_size, num_step, N, D]
    D:      output dims
    return: [batch_size, num_step, N, D]
    '''
    XS = FC(
        HS, units=D, activations=None,
        bn=bn, bn_decay=bn_decay,
        is_training=is_training, use_bias=False)
    XT = FC(
        HT, units=D, activations=None,
        bn=bn, bn_decay=bn_decay,
        is_training=is_training, use_bias=True)
    z = tf.nn.sigmoid(tf.add(XS, XT))
    H = tf.add(tf.multiply(z, HS), tf.multiply(1 - z, HT))
    H = FC(
        H, units=[D, D], activations=[tf.nn.relu, None],
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    return H

def STEmbedding(SE, TE, T, D, bn, bn_decay, is_training):
    '''
    spatio-temporal embedding
    SE:     [N, D]
    TE:     [batch_size, P + Q, 2] (dayofweek, timeofday)
    T:      num of time steps in one day
    D:      output dims
    retrun: [batch_size, P + Q, N, D]
    '''
    # spatial embedding
    SE = FC(
        SE, units=[D, D], activations=[tf.nn.relu, None],
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    # temporal embedding
    TE = tf.add_n(TE)
    # TE = tf.concat((TE), axis=-1)
    TE = FC(
        TE, units=[D, D], activations=[tf.nn.relu, None],
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    return tf.add(SE, TE)
    # return tf.concat([SE, TE],axis=-1)

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    print(rowsum.shape)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    print(r_inv.shape)
    r_mat_inv = sp.diags(r_inv)
    print(r_mat_inv.shape)
    features = r_mat_inv.dot(features)
    print('features shape is : ',features.shape)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    '''
    :param adj: Symmetrically normalize adjacency matrix
    :return:
    '''
    adj = sp.coo_matrix(adj) # 转化为稀疏矩阵表示的形式
    rowsum = np.array(adj.sum(1)) # 原连接矩阵每一行的元素和
    d_inv_sqrt = np.power(rowsum, -0.5).flatten() #先根号，再求倒数，然后flatten返回一个折叠成一维的数组
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0. #
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    '''
    :param adj:  A=A+E, and then to normalize the the adj matrix,
    preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation.
    :return:
    '''
    # 邻接矩阵 加上 单位矩阵
    '''
    [[1,0,0],[0,1,0],[0,0,1]]
    '''
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    print('adj_normalized shape is : ', adj_normalized.shape)

    return sparse_to_tuple(adj_normalized)

def construct_feed_dict(x_s = None,
                        week = 1,
                        day = 1,
                        hour = 1,
                        minute = 1,
                        label_s = None,
                        x_tra = None,
                        element_index = [],
                        separate_trajectory_time = [0.1],
                        total_time = 0.1, trajectory_inds=[0], placeholders = None):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['position']: np.array([[i for i in range(108)]],dtype=np.int32)})
    feed_dict.update({placeholders['week']: week})
    feed_dict.update({placeholders['day']: day})
    feed_dict.update({placeholders['hour']: hour})
    feed_dict.update({placeholders['minute']: minute})
    feed_dict.update({placeholders['feature_s']: x_s})
    feed_dict.update({placeholders['label_s']: label_s})
    feed_dict.update({placeholders['feature_tra']: x_tra})
    feed_dict.update({placeholders['label_tra']: separate_trajectory_time})
    feed_dict.update({placeholders['label_tra_sum']: total_time})
    feed_dict.update({placeholders['feature_inds']: element_index})
    feed_dict.update({placeholders['trajectory_inds']: trajectory_inds[0]})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)

def one_hot_concatenation(features=[]):
    '''
    :param features:
    :return: [N, p]
    '''
    features = np.concatenate(features, axis=-1)
    return features


def seaborn(x =None):
    '''
    :param x:
    :return:
    '''
    """
    document: https://seaborn.pydata.org/generated/seaborn.heatmap.html#seaborn.heatmap
    根据data传入的值画出热力图，一般是二维矩阵
    vmin设置最小值, vmax设置最大值
    cmap换用不同的颜色
    center设置中心值
    annot 是否在方格上写上对应的数字
    fmt 写入热力图的数据类型，默认为科学计数，d表示整数，.1f表示保留一位小数
    linewidths 设置方格之间的间隔
    xticklabels，yticklabels填到横纵坐标的值。可以是bool，填或者不填。可以是int，以什么间隔填，可以是list
    color: Accent, Accent_r, Blues, Blues_r, BrBG, BrBG_r, BuGn, BuGn_r, BuPu, BuPu_r, CMRmap, CMRmap_r, 
    Dark2, Dark2_r, GnBu, GnBu_r, Greens, Greens_r, Greys, Greys_r, OrRd, OrRd_r, Oranges, Oranges_r, PRGn, 
    PRGn_r, Paired, Paired_r, Pastel1, Pastel1_r, Pastel2, Pastel2_r, PiYG, PiYG_r, PuBu, PuBuGn, PuBuGn_r, 
    PuBu_r, PuOr, PuOr_r, PuRd, PuRd_r, Purples, Purples_r, RdBu, RdBu_r, RdGy, RdGy_r, RdPu, RdPu_r, RdYlBu, 
    RdYlBu_r, RdYlGn, RdYlGn_r, Reds, Reds_r, Set1, Set1_r, Set2, Set2_r, Set3, Set3_r, Spectral, Spectral_r, Wistia, 
    Wistia_r, YlGn, YlGnBu, YlGnBu_r, YlGn_r, YlOrBr, YlOrBr_r, YlOrRd, YlOrRd_r, afmhot, afmhot_r, autumn, autumn_r, 
    binary, binary_r, bone, bone_r, brg, brg_r, bwr, bwr_r, cividis, cividis_r, cool, cool_r, coolwarm, coolwarm_r, 
    copper, copper_r, cubehelix, cubehelix_r, flag, flag_r, gist_earth, gist_earth_r, gist_gray, gist_gray_r, gist_heat, 
    gist_heat_r, gist_ncar, gist_ncar_r, gist_rainbow, gist_rainbow_r, gist_stern, gist_stern_r, gist_yarg, gist_yarg_r, 
    gnuplot, gnuplot2, gnuplot2_r, gnuplot_r, gray, gray_r, hot, hot_r, hsv, hsv_r, icefire, icefire_r, inferno, inferno_r, 
    jet, jet_r, magma, magma_r, mako, mako_r, nipy_spectral, nipy_spectral_r, ocean, ocean_r, pink, pink_r, plasma, plasma_r, 
    prism, prism_r, rainbow, rainbow_r, rocket, rocket_r, seismic, seismic_r, spring, spring_r, summer, summer_r, tab10, tab10_r, 
    tab20, tab20_r, tab20b, tab20b_r, tab20c, tab20c_r, terrain, terrain_r, twilight, twilight_r, twilight_shifted, twilight_shifted_r, 
    viridis, viridis_r, vlag, vlag_r, winter, winter_r

    """
    f, (ax1,ax2) = plt.subplots(nrows=2,ncols=2)
    #
    # sns.heatmap(x, annot=False, ax=ax1)
    # sns.heatmap(x1, annot=False, ax=ax2)

    sns.heatmap(x[:,0], annot=False,
                yticklabels=[i+1 for i in range(x.shape[0])],
                xticklabels=[i+1 for i in range(x.shape[2])],cbar=True, ax=ax1[0],cmap='Blues')
    sns.heatmap(x[:,1], annot=False,
                yticklabels=[i+1 for i in range(x.shape[0])],
                xticklabels=[i+1 for i in range(x.shape[2])],cbar=True, ax=ax1[1], cmap='Greens')
    sns.heatmap(x[:,2], annot=False,
                yticklabels=[i+1 for i in range(x.shape[0])],
                xticklabels=[i+1 for i in range(x.shape[2])],cbar=True, ax=ax2[0])
    sns.heatmap(x[:,3], annot=False,
                yticklabels=[i+1 for i in range(x.shape[0])],
                xticklabels=[i+1 for i in range(x.shape[2])],cbar=True, ax=ax2[1],cmap='Greys')


    plt.show()

import matplotlib.pyplot as plt
def describe(label, predict):
    '''
    :param label:
    :param predict:
    :param prediction_size:
    :return:
    '''
    plt.figure()
    # Label is observed value,Blue
    plt.plot(label[0:], 'b', label=u'actual value')
    # Predict is predicted value，Red
    plt.plot(predict[0:], 'r', label=u'predicted value')
    # use the legend
    plt.legend()
    # plt.xlabel("time(hours)", fontsize=17)
    # plt.ylabel("pm$_{2.5}$ (ug/m$^3$)", fontsize=17)
    # plt.title("the prediction of pm$_{2.5}", fontsize=17)
    plt.show()

def metric(pred, label):
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = np.not_equal(label, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(pred, label)).astype(np.float32)
        rmse = np.square(mae)
        mape = np.divide(mae, label)
        # mae = np.nan_to_num(mae * mask)
        # wape = np.divide(np.sum(mae), np.sum(label))
        mae = np.mean(mae)
        # rmse = np.nan_to_num(rmse * mask)
        rmse = np.sqrt(np.mean(rmse))
        mape = np.nan_to_num(mape * mask)
        mape = np.mean(mape)
        cor = np.mean(np.multiply((label - np.mean(label)),
                                  (pred - np.mean(pred)))) / (np.std(pred) * np.std(label))
        sse = np.sum((label - pred) ** 2)
        sst = np.sum((label - np.mean(label)) ** 2)
        r2 = 1 - sse / sst  # r2_score(y_actual, y_predicted, multioutput='raw_values')
        print('mae is : %.6f'%mae)
        print('rmse is : %.6f'%rmse)
        print('mape is : %.6f'%mape)
        print('r is : %.6f'%cor)
        print('r$^2$ is : %.6f'%r2)
    return mae, rmse, mape, cor, r2