#!/usr/bin/env python
#only for the predictions error, without comparisons (ground truth) and no publication on RVIZ

from __future__ import print_function, absolute_import, division
import rospy
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray
import torch
import time

import sys
sys.path.append('/home/bartonlab-user/workspace/src/human_motion_forecasting/scripts')
import os

from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import h5py
import torch.optim as optim
from progress.bar import Bar
import pandas as pd

from utils import loss_funcs, utils as utils
import model.GCN as nnmodel
from utils.opt import Options
from utils import util
from utils import log
import utils.data_utils as data_utils

batch_size = 1
num_frames = 20
num_joints = 32
num_new_frames = 10  # Number of new frames to collect in each iteration



opt = Options().parse()

lr_now = opt.lr_now
start_epoch = 1
opt.is_eval = True
# save option in log
script_name = os.path.basename(__file__).split('.')[0]
script_name = script_name + '_3D_in{:d}_out{:d}_dct_n_{:d}'.format(opt.input_n, opt.output_n, opt.dct_n)

print('>>> create models')
input_n = opt.input_n
output_n = opt.output_n
dct_n = opt.dct_n
sample_rate = opt.sample_rate

model = nnmodel.GCN(input_feature=dct_n, hidden_feature=opt.linear_size, p_dropout=opt.dropout,
                    num_stage=opt.num_stage, node_n=66)
model.cuda()
# model_path_len = '{}/ckpt_best.pth.tar'.format(opt.ckpt)
model_path_len = os.path.join('/home/bartonlab-user/workspace/src/human_motion_forecasting/scripts/checkpoint/LTD/pretrained/h36m3D_in10_out25_dctn30.pth.tar')
print(">>> loading ckpt len from '{}'".format(model_path_len))
ckpt = torch.load(model_path_len)
start_epoch = ckpt['epoch'] + 1
err_best = ckpt['err']
lr_now = ckpt['lr']
model.load_state_dict(ckpt['state_dict'])
print(">>> ckpt len loaded (epoch: {} | err: {})".format(ckpt['epoch'], ckpt['err']))

processed_data = torch.zeros(batch_size, num_frames, num_joints*3)
past_frames = torch.zeros(batch_size, num_frames - num_new_frames, num_joints*3)  # Store past frames


frame_count = 0
start_timestamp = time.time()

def run_model(model, optimizer=None, is_train=0, input_data=None, epo=1, opt=None):
    model.eval()
    titles = np.array(range(opt.output_n)) + 1
    m_p3d_h36 = np.zeros([opt.output_n])
    n = 0
    in_n = opt.input_n
    out_n = opt.output_n
    dct_used = in_n + out_n
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
    
    batch_size, seq_n, _ = input_data.shape
    dim_used_len = len(dim_used)

    n += batch_size
    bt = time.time()
    p3d_h36 = input_data.float().cuda()

    p3d_src = p3d_h36.clone()[:, :, dim_used]

    #First I need to transform p3d_src in dct coeficients before I give it to the predictor
    dct_m_in, _ = data_utils.get_dct_matrix(in_n + out_n)
    pad_idx = np.repeat([input_n - 1], output_n)
    i_idx = np.append(np.arange(0, input_n), pad_idx)
    input_dct_seq = np.matmul(dct_m_in[0:dct_used, :], p3d_src[i_idx, :])
    input_dct_seq = input_dct_seq.transpose().reshape([-1, len(dim_used), dct_used])
    # input_dct_seq = input_dct_seq.reshape(-1, len(dim_used) * dct_used)

    #make sure dim_used in LTD is the same as in HRI

    p3d_out_all = model(input_dct_seq*1000)
    p3d_out_all = p3d_out_all*0.001

    _, idct_m = data_utils.get_dct_matrix(seq_n)
    idct_m = torch.from_numpy(idct_m).float().cuda()
    outputs_t = p3d_out_all.view(-1, dct_n).transpose(0, 1)
    outputs_3d = torch.matmul(idct_m[:, 0:dct_n], outputs_t).transpose(0, 1).contiguous().view(-1, dim_used_len,seq_n).transpose(1,2)
    
    pred_3d = p3d_h36.clone()
    pred_3d[:, :, dim_used] = outputs_3d
    pred_3d[:, :, index_to_ignore] = pred_3d[:, :, index_to_equal]
    pred_p3d = pred_3d.contiguous().view(n, seq_n, -1, 3)[:, in_n:, :, :]

    targ_p3d = p3d_h36.contiguous().view(n, seq_n, -1, 3)[:, in_n:, :, :]

    # p3d_out_all = p3d_out_all[:, seq_in:].transpose(1, 2).reshape([batch_size, 10 * itera, -1])[:, :out_n]

    # p3d_out = p3d_h36.clone()[:, in_n:in_n + out_n]
    # p3d_out[:, :, dim_used] = p3d_out_all
    # p3d_out[:, :, index_to_ignore] = p3d_out[:, :, index_to_equal]

    #method in HRI
    # mpjpe_p3d_h36 = torch.sum(torch.mean(torch.norm(pred_p3d - targ_p3d, dim=3), dim=2), dim=0)

    #method in STS_GCN

    mpjpe_p3d_h36 =torch.mean(torch.norm(pred_p3d-targ_p3d,2,1))

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

        ret_test = run_model(model, is_train=3, input_data=input_data, opt=opt)
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
