# -- coding: utf-8 --

import argparse
class parameter(object):
    def __init__(self,parser):
        self.parser=parser

        self.parser.add_argument('--save_path', type=str, default='weights/CoDriverETA-4/', help='save path')
        self.parser.add_argument('--model_name', type=str, default='CoDriverETA', help='training or testing model name')

        self.parser.add_argument('--divide_ratio', type=float, default=0.8, help='data_divide')
        self.parser.add_argument('--is_training', type=bool, default=True, help='is training')
        self.parser.add_argument('--epoch', type=int, default=50, help='epoch')
        self.parser.add_argument('--step', type=int, default=1, help='step')
        self.parser.add_argument('--batch_size', type=int, default=64, help='batch size')
        self.parser.add_argument('--learning_rate', type=float, default=0.0005, help='learning rate')
        self.parser.add_argument('--dropout', type=float, default=0.0, help='drop out')
        self.parser.add_argument('--site_num', type=int, default=108, help='total number of road')
        self.parser.add_argument('--k', type=int, default=64, help='latent vector dimension')
        self.parser.add_argument('--field_cnt', type=int, default=17, help='the number of filed for trajectory features')
        self.parser.add_argument('--feature_tra', type=int, default=10284, help='number of the trajectory feature elements')
        self.parser.add_argument('--trajectory_length', type=int, default=5, help='length of trajectory')
        self.parser.add_argument('--num_heads', type=int, default=4, help='total number of head attentions')
        self.parser.add_argument('--num_blocks', type=int, default=1, help='total number of attention layers')

        #每个点表示a->b路线，目前8个收费站
        self.parser.add_argument('--emb_size', type=int, default=64, help='embedding size')
        self.parser.add_argument('--feature_s', type=int, default=1, help='number of the speed feature elements')
        self.parser.add_argument('--normalize', type=bool, default=True, help='normalize')
        self.parser.add_argument('--input_length', type=int, default=12, help='input length')
        self.parser.add_argument('--output_length', type=int, default=6, help='output length')

        self.parser.add_argument('--hidden_size', type=int, default=64, help='hidden size')
        self.parser.add_argument('--hidden_layer', type=int, default=1, help='hidden layer')

        self.parser.add_argument('--training_set_rate', type=float, default=0.7, help='training set rate')
        self.parser.add_argument('--validate_set_rate', type=float, default=0.15, help='validate set rate')
        self.parser.add_argument('--test_set_rate', type=float, default=0.15, help='test set rate')

        self.parser.add_argument('--file_train_s', type=str, default='/Users/guojianzou/Travel-time-prediction/data/train.csv', help='training_speed file address')
        self.parser.add_argument('--file_val', type=str, default='data/val.csv', help='validate set file address')
        self.parser.add_argument('--file_test', type=str, default='data/test.csv', help='test set file address')

        self.parser.add_argument('--file_train_t', type=str, default='/Users/guojianzou/Travel-time-prediction/data/trajectory_4.csv', help='trajectory file address')

        self.parser.add_argument('--file_adj', type=str,default='data/adjacent.csv', help='adj file address')

    def get_para(self):
        return self.parser.parse_args()

if __name__=='__main__':
    para=parameter(argparse.ArgumentParser())

    print(para.get_para().batch_size)