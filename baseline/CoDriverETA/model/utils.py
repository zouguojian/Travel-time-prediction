# -- coding: utf-8 --
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import tensorflow as tf

def conv2d(x, output_dims, kernel_size, stride = [1, 1],
           padding = 'SAME', use_bias = True, activation = tf.nn.relu,
           bn = False, bn_decay = None, is_training = None):
    input_dims = x.get_shape()[-1].value
    kernel_shape = kernel_size + [input_dims, output_dims]
    kernel = tf.Variable(
        tf.glorot_uniform_initializer()(shape = kernel_shape),
        dtype = tf.float32, trainable = True, name = 'kernel')
    x = tf.nn.conv2d(x, kernel, [1] + stride + [1], padding = padding)
    if use_bias:
        bias = tf.Variable(
            tf.zeros_initializer()(shape = [output_dims]),
            dtype = tf.float32, trainable = True, name = 'bias')
        x = tf.nn.bias_add(x, bias)
    if activation is not None:
        if bn:
            x = batch_norm(x, is_training = is_training, bn_decay = bn_decay)
        x = activation(x)
    return x

def batch_norm(x, is_training, bn_decay):
    input_dims = x.get_shape()[-1].value
    moment_dims = list(range(len(x.get_shape()) - 1))
    beta = tf.Variable(
        tf.zeros_initializer()(shape = [input_dims]),
        dtype = tf.float32, trainable = True, name = 'beta')
    gamma = tf.Variable(
        tf.ones_initializer()(shape = [input_dims]),
        dtype = tf.float32, trainable = True, name = 'gamma')
    batch_mean, batch_var = tf.nn.moments(x, moment_dims, name='moments')

    decay = bn_decay if bn_decay is not None else 0.9
    ema = tf.train.ExponentialMovingAverage(decay = decay)
    # Operator that maintains moving averages of variables.
    ema_apply_op = tf.cond(
        is_training,
        lambda: ema.apply([batch_mean, batch_var]),
        lambda: tf.no_op())
    # Update moving average and return current batch's avg and var.
    def mean_var_with_update():
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)
    # ema.average returns the Variable holding the average of var.
    mean, var = tf.cond(
        is_training,
        mean_var_with_update,
        lambda: (ema.average(batch_mean), ema.average(batch_var)))
    x = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return x

def dropout(x, drop, is_training):
    x = tf.cond(
        is_training,
        lambda: tf.nn.dropout(x, rate = drop),
        lambda: x)
    return x

def FC(x, units, activations, bn, bn_decay, is_training, use_bias=True):
    if isinstance(units, int):
        units = [units]
        activations = [activations]
    elif isinstance(units, tuple):
        units = list(units)
        activations = list(activations)
    assert type(units) == list
    for num_unit, activation in zip(units, activations):
        x = conv2d(
            x, output_dims=num_unit, kernel_size=[1, 1], stride=[1, 1],
            padding='VALID', use_bias=use_bias, activation=activation,
            bn=bn, bn_decay=bn_decay, is_training=is_training)
    return x

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
                        total_time = 0.1, trajectory_inds=[0], placeholders = None, ave_s=None):
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
    feed_dict.update({placeholders['ave_s']: ave_s})
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