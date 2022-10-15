import os
import json
import time
import utils
from baseline.DeepTTE import models
from baseline.DeepTTE.models import TTE
import logger
import inspect
import datetime
import argparse
import data_loader

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.autograd import Variable

import numpy as np

parser = argparse.ArgumentParser()
# basic args
parser.add_argument('--task', type = str, default='test')
parser.add_argument('--batch_size', type = int, default = 10)
parser.add_argument('--epochs', type = int, default = 10)
# evaluation args
parser.add_argument('--weight_file', type = str, default = './saved_weights/DeepTTE')
parser.add_argument('--result_file', type = str, default = './result/deeptte.res')
# cnn args
parser.add_argument('--kernel_size', type = int, default=3)
# rnn args
parser.add_argument('--pooling_method', type = str, default= 'attention')
# multi-task args
parser.add_argument('--alpha', type = float, default=0.1)
# log file name
parser.add_argument('--log_file', type = str, default='run_log')
args = parser.parse_args()

config = json.load(open('./config.json', 'r'))

"""
distance gap length: 16
lngs length: 16
states length: 16
time gap length: 16

time_gap
dist
lats
driverID
weekID
states
timeID
dateID
time
lngs
dist_gap
"""

def train(model, elogger, train_set, eval_set):
    # record the experiment setting
    elogger.log(str(model))
    elogger.log(str(args._get_kwargs()))

    if torch.cuda.is_available():
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr = 1e-3)

    for epoch in range(args.epochs):
        model.train()
        print('Training on epoch {}'.format(epoch))
        for input_file in train_set:
            print('Train on file {}'.format(input_file))

            # data loader, return two dictionaries, attr and traj
            data_iter = data_loader.get_loader(input_file, args.batch_size)

            running_loss = 0.0
            for idx, (attr, traj) in enumerate(data_iter):
                # transform the input to pytorch variable
                attr, traj = utils.to_var(attr), utils.to_var(traj)

                _, loss = model.eval_on_batch(attr, traj, config)

                # update the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.data
                print('Progress {:.2f}%, average loss {}'.format((idx + 1) * 100.0 / len(data_iter), running_loss / (idx + 1.0)))
            elogger.log('Training Epoch {}, File {}, Loss {}'.format(epoch, input_file, running_loss / (idx + 1.0)))

        # evaluate the model after each epoch
        evaluate(model, elogger, eval_set, save_result = False)

        # save the weight file after each epoch
        # weight_name = '{}_{}'.format(args.log_file, str(datetime.datetime.now()))
        # elogger.log('Save weight file {}'.format(weight_name))
        torch.save(model.state_dict(), args.weight_file)

def write_result(fs, pred_dict, attr):
    pred = pred_dict['pred'].data.cpu().numpy()
    label = pred_dict['label'].data.cpu().numpy()

    for i in range(pred_dict['pred'].size()[0]):
        fs.write('%.6f %.6f\n' % (label[i][0], pred[i][0]))

        dateID = attr['dateID'].data[i]
        timeID = attr['timeID'].data[i]
        driverID = attr['driverID'].data[i]


def evaluate(model, elogger, files, save_result = False):
    model.eval()
    if save_result:
        fs = open('%s' % args.result_file, 'w')

    for input_file in files:
        running_loss = 0.0
        data_iter = data_loader.get_loader(input_file, args.batch_size)

        for idx, (attr, traj) in enumerate(data_iter):
            attr, traj = utils.to_var(attr), utils.to_var(traj)

            pred_dict, loss = model.eval_on_batch(attr, traj, config)

            if save_result: write_result(fs, pred_dict, attr)
            running_loss += loss.data

        print('Evaluate on file {}, loss {}'.format(input_file, running_loss / (idx + 1.0)))
        elogger.log('Evaluate File {}, Loss {}'.format(input_file, running_loss / (idx + 1.0)))

    if save_result: fs.close()

def get_kwargs(model_class):
    # model_args = inspect.getargspec(model_class.__init__).args
    model_args = inspect.getfullargspec(model_class.__init__).args
    shell_args = args._get_kwargs()

    kwargs = dict(shell_args)

    for arg, val in shell_args:
        if not arg in model_args:
            kwargs.pop(arg)

    return kwargs

def run():
    # model instance
    # get the model arguments
    kwargs = get_kwargs(TTE.Net)

    # model instance
    model = TTE.Net(**kwargs)
    # model = TTE.Net(kernel_size = args, num_filter = 32, pooling_method = 'attention', num_final_fcs = 3, final_fc_size = 128, alpha = 0.3)

    # experiment logger
    elogger = logger.Logger(args.log_file)

    if args.task == 'train':
        train(model, elogger, train_set = config['train_set'], eval_set = config['eval_set'])

    elif args.task == 'test':
        # load the saved weight file
        model.load_state_dict(torch.load(args.weight_file))
        if torch.cuda.is_available():
            model.cuda()
        evaluate(model, elogger, config['test_set'], save_result = True)

if __name__ == '__main__':
    run()
