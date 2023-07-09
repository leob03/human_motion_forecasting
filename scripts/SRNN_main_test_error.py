#!/usr/bin/env python
#only for the predictions error, without comparisons (ground truth) and no publication on RVIZ

import rospy
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray
import torch
import time

import sys
sys.path.append('/home/bartonlab-user/workspace/src/human_motion_forecasting/scripts')
import os

from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import h5py
import torch.optim as optim

from utils.h36motion3d import Datasets
from model import AttModel
from utils.opt import Options
from utils import util
from utils import log

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange # pylint: disable=redefined-builtin

from utils import data_utils_srnn
from utils import seq2seq_model
import argparse

# Learning
parser = argparse.ArgumentParser(description='Train RNN for human pose estimation')
parser.add_argument('--learning_rate', dest='learning_rate',
                  help='Learning rate',
                  default=0.005, type=float)
parser.add_argument('--learning_rate_decay_factor', dest='learning_rate_decay_factor',
                  help='Learning rate is multiplied by this much. 1 means no decay.',
                  default=0.95, type=float)
parser.add_argument('--learning_rate_step', dest='learning_rate_step',
                  help='Every this many steps, do decay.',
                  default=10000, type=int)
parser.add_argument('--batch_size', dest='batch_size',
                  help='Batch size to use during training.',
                  default=16, type=int)
parser.add_argument('--max_gradient_norm', dest='max_gradient_norm',
                  help='Clip gradients to this norm.',
                  default=5, type=float)
parser.add_argument('--iterations', dest='iterations',
                  help='Iterations to train for.',
                  default=1e5, type=int)
parser.add_argument('--test_every', dest='test_every',
                  help='',
                  default=200, type=int)
# Architecture
parser.add_argument('--architecture', dest='architecture',
                  help='Seq2seq architecture to use: [basic, tied].',
                  default='tied', type=str)
parser.add_argument('--loss_to_use', dest='loss_to_use',
                  help='The type of loss to use, supervised or sampling_based',
                  default='sampling_based', type=str)
parser.add_argument('--residual_velocities', dest='residual_velocities',
                  help='Add a residual connection that effectively models velocities',action='store_true',
                  default=False)
parser.add_argument('--size', dest='size',
                  help='Size of each model layer.',
                  default=1024, type=int)
parser.add_argument('--num_layers', dest='num_layers',
                  help='Number of layers in the model.',
                  default=1, type=int)
parser.add_argument('--seq_length_in', dest='seq_length_in',
                  help='Number of frames to feed into the encoder. 25 fp',
                  default=50, type=int)
parser.add_argument('--seq_length_out', dest='seq_length_out',
                  help='Number of frames that the decoder has to predict. 25fps',
                  default=10, type=int)
parser.add_argument('--omit_one_hot', dest='omit_one_hot',
                  help='', action='store_true',
                  default=False)
# Directories
parser.add_argument('--data_dir', dest='data_dir',
                  help='Data directory',
                  default=os.path.normpath("./data/h3.6m/dataset"), type=str)
parser.add_argument('--train_dir', dest='train_dir',
                  help='Training directory',
                  default=os.path.normpath("./checkpoint/SRNN"), type=str)
parser.add_argument('--action', dest='action',
                  help='The action to train on. all means all the actions, all_periodic means walking, eating and smoking',
                  default='all', type=str)
parser.add_argument('--use_cpu', dest='use_cpu',
                  help='', action='store_true',
                  default=False)
parser.add_argument('--load', dest='load',
                  help='Try to load a previous checkpoint.',
                  default=0, type=int)
parser.add_argument('--sample', dest='sample',
                  help='Set to True for sampling.', action='store_true',
                  default=False)

args = parser.parse_args()

batch_size = 1
num_frames = 60
num_joints = 32
num_new_frames = 10  # Number of new frames to collect in each iteration

train_dir = os.path.normpath(os.path.join( args.train_dir, args.action,
  'out_{0}'.format(args.seq_length_out),
  'iterations_{0}'.format(args.iterations),
  args.architecture,
  args.loss_to_use,
  'omit_one_hot' if args.omit_one_hot else 'one_hot',
  'depth_{0}'.format(args.num_layers),
  'size_{0}'.format(args.size),
  'lr_{0}'.format(args.learning_rate),
  'residual_vel' if args.residual_velocities else 'not_residual_vel'))

print(train_dir)
os.makedirs(train_dir, exist_ok=True)

def create_model(actions, sampling=False):
  """Create translation model and initialize or load parameters in session."""

  model = seq2seq_model.Seq2SeqModel(
      args.architecture,
      args.seq_length_in if not sampling else 50,
      args.seq_length_out if not sampling else 100,
      args.size, # hidden layer size
      args.num_layers,
      args.max_gradient_norm,
      args.batch_size,
      args.learning_rate,
      args.learning_rate_decay_factor,
      args.loss_to_use if not sampling else "sampling_based",
      len( actions ),
      not args.omit_one_hot,
      args.residual_velocities,
      dtype=torch.float32)

  if args.load <= 0:
    return model

  print("Loading model")
  model = torch.load(train_dir + '/model_' + str(args.load))
  if sampling:
    model.source_seq_len = 50
    model.target_seq_len = 100
  return model








opt = Options().parse()

lr_now = opt.lr_now
start_epoch = 1
# opt.is_eval = True
print('>>> create models')
in_features = 66
d_model = opt.d_model
kernel_size = opt.kernel_size
net_pred = AttModel.AttModel(in_features=in_features, kernel_size=kernel_size, d_model=d_model,
                            num_stage=opt.num_stage, dct_n=opt.dct_n)
net_pred.cuda()
# model_path_len = '{}/ckpt_best.pth.tar'.format(opt.ckpt)
model_path_len = os.path.join('/home/bartonlab-user/workspace/src/human_motion_forecasting/scripts/checkpoint/HRI/main_h36m_3d_in50_out10_ks10_dctn20/ckpt_best.pth.tar')
# model_path_len = os.path.join('/home/bartonlab-user/workspace/src/azure_bodytracking/scripts/checkpoint/pretrained/h36m_3d_in50_out10_dctn20/ckpt_best.pth.tar')
print(">>> loading ckpt len from '{}'".format(model_path_len))
ckpt = torch.load(model_path_len)
start_epoch = ckpt['epoch'] + 1
err_best = ckpt['err']
lr_now = ckpt['lr']
net_pred.load_state_dict(ckpt['state_dict'])
print(">>> ckpt len loaded (epoch: {} | err: {})".format(ckpt['epoch'], ckpt['err']))

processed_data = torch.zeros(batch_size, num_frames, num_joints*3)
past_frames = torch.zeros(batch_size, num_frames - num_new_frames, num_joints*3)  # Store past frames


frame_count = 0
start_timestamp = time.time()

def run_model(net_pred, optimizer=None, is_train=0, input_data=None, epo=1, opt=None):
    net_pred.eval()
    titles = np.array(range(opt.output_n)) + 1
    m_p3d_h36 = np.zeros([opt.output_n])
    n = 0
    in_n = opt.input_n
    out_n = opt.output_n
    dim_used = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25,
                         26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                         46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68,
                         75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92])
    seq_in = opt.kernel_size
    # joints at same loc
    joint_to_ignore = np.array([16, 20, 23, 24, 28, 31])
    index_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
    joint_equal = np.array([13, 19, 22, 13, 27, 30])
    index_to_equal = np.concatenate((joint_equal * 3, joint_equal * 3 + 1, joint_equal * 3 + 2))

    itera = 3
    
    batch_size, seq_n, _ = input_data.shape

    n += batch_size
    bt = time.time()
    p3d_h36 = input_data.float().cuda()
    # print(p3d_h36.shape)
    # print(p3d_h36[0,0,:])
    # p3d_sup = p3d_h36.clone()[:, :, dim_used][:, -out_n - seq_in:].reshape(
    #     [-1, seq_in + out_n, len(dim_used) // 3, 3])
    p3d_src = p3d_h36.clone()[:, :, dim_used]
    # p3d_src = p3d_src.permute(1, 0, 2)  # seq * n * dim
    # p3d_src = p3d_src[:in_n]
    p3d_out_all = net_pred(p3d_src*1000, input_n=in_n, output_n=10, itera=itera)
    p3d_out_all = p3d_out_all*0.001

    p3d_out_all = p3d_out_all[:, seq_in:].transpose(1, 2).reshape([batch_size, 10 * itera, -1])[:, :out_n]

    p3d_out = p3d_h36.clone()[:, in_n:in_n + out_n]
    p3d_out[:, :, dim_used] = p3d_out_all
    p3d_out[:, :, index_to_ignore] = p3d_out[:, :, index_to_equal]
    p3d_out = p3d_out.reshape([-1, out_n, 32, 3])
    # print(p3d_out.shape)

    p3d_h36 = p3d_h36.reshape([-1, in_n + out_n, 32, 3])
    
    mpjpe_p3d_h36 = torch.sum(torch.mean(torch.norm(p3d_h36[:, in_n:] - p3d_out, dim=3), dim=2), dim=0)
    # print(p3d_h36[:, -9])
    # print(p3d_out[:,-9])
    # print(mpjpe_p3d_h36.shape)
    # print(mpjpe_p3d_h36)
    m_p3d_h36 += mpjpe_p3d_h36.cpu().data.numpy()

    ###

    ret = {}
    m_p3d_h36 = m_p3d_h36 *256
    for j in range(out_n):
        ret["#{:d}".format(titles[j])] = m_p3d_h36[j]
    return ret

def body_tracking_callback(msg):
    
    global frame_count, start_timestamp, past_frames

    marker_array = msg.markers
    coordinates = [[marker.pose.position.x, marker.pose.position.y, marker.pose.position.z]
                   for marker in marker_array]
    marker_tensor = torch.tensor(coordinates, dtype = torch.float32)

    # print(marker_tensor.shape)
    # print(marker_tensor)

    processed_data[:,frame_count] = marker_tensor.view(-1)
    
    frame_count +=1

    if frame_count == num_frames:

        if frame_count > num_new_frames:
            input_data = torch.cat((past_frames, processed_data[:,num_frames-num_new_frames:]), dim=1)
        else:
            input_data = processed_data

        input_data = processed_data.view(batch_size, num_frames, num_joints*3)

        elapsed_time = time.time() - start_timestamp
        print("Elapsed time:", elapsed_time)

        # errs = np.zeros([len(acts) + 1, opt.output_n])

        ret_test = run_model(net_pred, is_train=3, input_data=input_data, opt=opt)
        print('testing error: {:.3f}'.format(ret_test['#10']))
        # print(ret_test)
        # ret_log = np.array([])
        # for k in ret_test.keys():
        #     ret_log = np.append(ret_log, [ret_test[k]])
        # errs[0] = ret_log

        # errs[-1] = np.mean(errs[:-1], axis=0)
        # acts = np.expand_dims(np.array(acts + ["average"]), axis=1)
        # value = np.concatenate([acts, errs.astype(np.str)], axis=1)
        # # log.save_csv_log(opt, head, value, is_create=True, file_name='test_pre_action')

        frame_count = num_frames - num_new_frames
        processed_data.zero_()

        if num_new_frames < num_frames:
            past_frames = input_data[:, num_new_frames:]
        
        start_timestamp = time.time()


def listener():
    rospy.init_node('subscriber_node', anonymous=True)
    rospy.Subscriber('body_tracking_data', MarkerArray, body_tracking_callback)
    rospy.spin()

if __name__ == '__main__':
    listener()