# -- coding: utf-8 --

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.svm import SVR

file='/Users/guojianzou/Travel-time-prediction/data/'

class svm_i():
    def __init__(self,
                 site_id=0,
                 is_training=True,
                 time_size=6,
                 prediction_size=6,
                 data_divide=0.8,
                 window_step=6,
                 normalize=False):
        '''
        :param is_training: while is_training is True,the model is training state
        :param field_len:
        :param time_size:
        :param prediction_size:
        :param target_site:
        '''
        self.site_id=site_id                   # ozone ID
        self.time_size=time_size               # time series length of input
        self.prediction_size=prediction_size   # the length of prediction
        self.is_training=is_training           # true or false
        self.data_divide=data_divide           # the divide between in training set and test set ratio
        self.window_step=window_step           # windows step
        self.data=self.get_source_data(file+'trajectory_1.csv')
        self.length=self.data.values.shape[0]  #data length

    def get_source_data(self,file_path):
        '''
        :return:
        '''
        data = pd.read_csv(file_path, encoding='utf-8')
        return data

    def describe(self, label, predict):
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

    def metric(self, pred, label):
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
            print('mae is : %.6f' % mae)
            print('rmse is : %.6f' % rmse)
            print('mape is : %.6f' % mape)
            print('r is : %.6f' % cor)
            print('r$^2$ is : %.6f' % r2)
        return mae, rmse, mape, cor, r2

    def train_data(self,data,input_length,predict_length):
        low,high=0,data.shape[0]
        x ,y=[],[]
        while low+predict_length+input_length<high:
            x.append(np.reshape(data[low:low+input_length],newshape=[-1]))
            y.append(np.reshape(data[low+input_length:low+input_length+predict_length],newshape=[-1]))
            low+=self.window_step
        return np.array(x), np.array(y)

    def model(self):
        toll_label = list()
        toll_predict = list()
        for time_step in range(6):
            self.dictionary_label = []
            self.dictionary_predict = []

            predict_index=time_step

            for site in range(108):
                data1=self.data[(self.data['node']==self.data.values[site][0])]
                x = data1.values[:, -1]

                x,y=self.train_data(data=x,input_length=6,predict_length=6)

                train_size = int(len(x) * 0.8)
                train_x, train_y, test_x, test_y = x[:train_size],y[:train_size,predict_index], x[train_size:],y[train_size:,predict_index]
                print(train_x.shape, train_y.shape, test_x.shape,test_y.shape)
                # print(data1.shape)
                # svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5,
                #                    param_grid={"C": [1e0, 1e1, 1e2, 1e3],
                #                                "gamma": np.logspace(-2, 2, 5)})

                svr = SVR(C=4, degree=2)

                # model=svm.NuSVR(nu=0.457,C=.8,degree=3)
                svr.fit(X=train_x, y=train_y)

                pre=svr.predict(X=test_x)
                self.dictionary_predict.append(pre)
                self.dictionary_label.append(test_y)
            self.metric(np.reshape(np.array(self.dictionary_predict), newshape=[-1]), np.reshape(np.array(self.dictionary_label), newshape=[-1]))
            toll_label.append(self.dictionary_label)
            toll_predict.append(self.dictionary_predict)
        self.metric(np.reshape(np.array(toll_predict), newshape=[-1]), np.reshape(np.array(toll_label), newshape=[-1]))

#
if __name__=='__main__':

    ha=svm_i(site_id=0,normalize=False)

    ha.model()
    # ha.metric(np.reshape(np.array(ha.dictionary_label),newshape=[-1]),np.reshape(np.array(ha.dictionary_predict),newshape=[-1]))
    # print(iter.data.loc[iter.data['ZoneID']==0])