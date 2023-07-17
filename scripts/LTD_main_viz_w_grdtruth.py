#!/usr/bin/env python

#prediciton of the present using the past (with ground truth)

from __future__ import print_function, absolute_import, division
import rospy
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
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
script_name = script_name + '_3D_in{:d}_out{:d}_dct_n_{:d}'.format(10, opt.output_n, 20)

print('>>> create models')
input_n = 10
output_n = opt.output_n
dct_n = 20
sample_rate = opt.skip_rate

model = nnmodel.GCN(input_feature=dct_n, hidden_feature=opt.d_model, p_dropout=opt.dropout,
                    num_stage=opt.num_stage, node_n=66)
model.cuda()
# model_path_len = '{}/ckpt_best.pth.tar'.format(opt.ckpt)
model_path_len = os.path.join('/home/bartonlab-user/workspace/src/human_motion_forecasting/scripts/checkpoint/LTD/test/ckpt_main_3d_3D_in10_out10_dct_n_20_best.pth.tar')
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

        input_data = input_data.view(1, num_frames, num_joints*3)

        # errs = np.zeros([len(acts) + 1, opt.output_n])

        is_train=3
        epo=1

        #network pred

        model.eval()
        titles = np.array(range(opt.output_n)) + 1
        m_p3d_h36 = np.zeros([opt.output_n])
        n = 0
        in_n = 10
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
        # dct_m_in = torch.from_numpy(dct_m_in).float().cuda()

        pad_idx = np.repeat([input_n - 1], output_n)
        i_idx = np.append(np.arange(0, input_n), pad_idx)

        p3d_src = p3d_src.cpu()
        # print(dct_m_in[0:dct_used, :].shape, p3d_src.shape, i_idx.shape)
        input_dct_seq = np.matmul(dct_m_in[0:dct_used, :], p3d_src[0][i_idx, :])
        # input_dct_seq = input_dct_seq.transpose().reshape([-1, len(dim_used), dct_used])
        input_dct_seq = input_dct_seq.reshape([-1, len(dim_used), dct_used])

        # input_dct_seq = input_dct_seq.reshape(-1, len(dim_used) * dct_used)

        #make sure dim_used in LTD is the same as in HRI
        input_dct_seq = input_dct_seq.float().cuda()

        p3d_out_all = model(input_dct_seq*1000)
        p3d_out_all = p3d_out_all*0.001
        # print(p3d_out_all.shape)


        _, idct_m = data_utils.get_dct_matrix(seq_n)
        idct_m = torch.from_numpy(idct_m).float().cuda()

        outputs_t = p3d_out_all.view(-1, dct_n).transpose(0, 1)
        # print(outputs_t.shape)
        outputs_3d = torch.matmul(idct_m[:, 0:dct_n], outputs_t).transpose(0, 1).contiguous().view(-1, dim_used_len,seq_n).transpose(1,2)
        # print(outputs_3d.shape)

        pred_3d = p3d_h36.clone()
        pred_3d[:, :, dim_used] = outputs_3d
        pred_3d[:, :, index_to_ignore] = pred_3d[:, :, index_to_equal]
        pred_p3d = pred_3d.contiguous().view(n, seq_n, -1, 3)[:, in_n:, :, :]
        # print(pred_p3d.shape)

        targ_p3d = p3d_h36.contiguous().view(n, seq_n, -1, 3)[:, in_n:, :, :]
        # print(targ_p3d.shape)

        #end of network

        #publish the groundtruth for visualization 
        grnd = targ_p3d[:,-1]
        grnd_coordinates = grnd.view(num_joints, 3)

        marker1 = Marker()
        marker1.header.frame_id = "depth_camera_link"
        marker1.type = Marker.SPHERE_LIST
        # marker1.type = Marker.LINE_LIST
        # marker.type = 2
        marker1.action = Marker.ADD
        # marker1.scale.x = 0.01
        # marker1.scale.y = 0.01
        # marker1.scale.z = 0.01
        marker1.scale.x = 0.05
        marker1.scale.y = 0.05
        marker1.scale.z = 0.05
        marker1.color.a = 1.0
        marker1.color.r = 0.0
        marker1.color.g = 0.0
        marker1.color.b = 1.0     
        
        # for simple points at each body joint
        for coordinate in grnd_coordinates:
            point = Point()
            point.x = coordinate[0].item()
            point.y = coordinate[1].item()
            point.z = coordinate[2].item()
            marker1.points.append(point)

        #for edges that connects the body joints to give a skeleton representation        
        # connections = [(0, 2), (2, 3), (3, 5), (5, 6),(6, 8),(3,12),(12, 13), (13, 15), (18,22),(18, 19), (19,21), (22,23),(23, 25), (26, 3), (26,27)]

        # for start_idx, end_idx in connections:
        #     start_coordinate = grnd_coordinates[start_idx]
        #     end_coordinate = grnd_coordinates[end_idx]

        #     # Add start point
        #     start_point = Point()
        #     start_point.x = start_coordinate[0].item()
        #     start_point.y = start_coordinate[1].item()
        #     start_point.z = start_coordinate[2].item()
        #     marker1.points.append(start_point)

        #     # Add end point
        #     end_point = Point()
        #     end_point.x = end_coordinate[0].item()
        #     end_point.y = end_coordinate[1].item()
        #     end_point.z = end_coordinate[2].item()
        #     marker1.points.append(end_point)
        
        marker1.pose.orientation.x = 0.0
        marker1.pose.orientation.y = 0.0
        marker1.pose.orientation.z = 0.0
        marker1.pose.orientation.w = 1.0

        marker_publisher1.publish(marker1)

        #publish the prediction for visualization 
        prediction = pred_p3d[:,-1]
        prediction_coordinates = prediction.view(num_joints, 3)

        marker = Marker()
        marker.header.frame_id = "depth_camera_link"
        marker.type = Marker.SPHERE_LIST
        # marker1.type = Marker.LINE_LIST
        # marker.type = 2
        marker.action = Marker.ADD
        # marker.scale.x = 0.01
        # marker.scale.y = 0.01
        # marker.scale.z = 0.01
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.05
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0

        for coordinate in prediction_coordinates:
            point = Point()
            point.x = coordinate[0].item()
            point.y = coordinate[1].item()
            point.z = coordinate[2].item()
            marker.points.append(point)

        # for start_idx, end_idx in connections:
        #     start_coordinate = prediction_coordinates[start_idx]
        #     end_coordinate = prediction_coordinates[end_idx]

        #     # Add start point
        #     start_point = Point()
        #     start_point.x = start_coordinate[0].item()
        #     start_point.y = start_coordinate[1].item()
        #     start_point.z = start_coordinate[2].item()
        #     marker.points.append(start_point)

        #     # Add end point
        #     end_point = Point()
        #     end_point.x = end_coordinate[0].item()
        #     end_point.y = end_coordinate[1].item()
        #     end_point.z = end_coordinate[2].item()
        #     marker.points.append(end_point)
        
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        marker_publisher.publish(marker)

        #calculate the elapsed time
        elapsed_time = time.time() - start_timestamp
        print("Elapsed time:", elapsed_time)

        #update the counters
        frame_count = num_frames - num_new_frames
        processed_data.zero_()

        if num_new_frames < num_frames:
            past_frames = input_data[:, num_new_frames:]

        start_timestamp = time.time()


if __name__ == '__main__':
    rospy.init_node('motion_forecasting_node', anonymous=True)
    marker_publisher1 = rospy.Publisher("/visualization_marker1", Marker, queue_size=1)
    marker_publisher = rospy.Publisher("/visualization_marker", Marker, queue_size=1)
    rospy.Subscriber('body_tracking_data', MarkerArray, body_tracking_callback)
    rospy.spin()