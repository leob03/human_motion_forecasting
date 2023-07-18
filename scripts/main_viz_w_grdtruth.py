#!/usr/bin/env python

#prediciton of the present using the past (with ground truth)

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
import torch.nn as nn
import numpy as np
import h5py
import torch.optim as optim

from utils.h36motion3d import Datasets
from model import AttModel
from utils.opt import Options
from utils import util
from utils import log

batch_size = 1
num_frames = 60
num_joints = 32
num_new_frames = 10  # Number of new frames to collect in each iteration (for sliding windows)

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
# model_path_len = os.path.join('/home/bartonlab-user/workspace/src/azure_bodytracking/scripts/checkpoint/HRI/pretrained/h36m_3d_in50_out10_dctn20/ckpt_best.pth.tar')
print(">>> loading ckpt len from '{}'".format(model_path_len))
ckpt = torch.load(model_path_len)
start_epoch = ckpt['epoch'] + 1
err_best = ckpt['err']
lr_now = ckpt['lr']
net_pred.load_state_dict(ckpt['state_dict'])
print(">>> ckpt len loaded (epoch: {} | err: {})".format(ckpt['epoch'], ckpt['err']))

torch.cuda.empty_cache()

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

        net_pred.eval()
        titles = np.array(range(opt.output_n)) + 1
        m_p3d_h36 = np.zeros([opt.output_n])
        n = 0
        in_n = opt.input_n
        out_n = opt.output_n
        dim_used = np.array([0,1,2,6, 7, 8, 9, 10, 11, 15, 16, 17, 18, 19, 20, 24, 25,
                            26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 45,
                            46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68,
                            69, 70, 71, 75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92])
        seq_in = opt.kernel_size
        # joints at same loc
        joint_to_ignore = np.array([16, 20, 24, 24, 28, 31])
        index_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
        joint_equal = np.array([13, 19, 23, 13, 27, 30])
        index_to_equal = np.concatenate((joint_equal * 3, joint_equal * 3 + 1, joint_equal * 3 + 2))

        itera = 3
        
        batch_size, seq_n, _ = input_data.shape

        n += batch_size
        bt = time.time()
        p3d_h36 = input_data.float().cuda()
        # print(p3d_h36.shape)
        # print(p3d_h36[0,0,:])
        p3d_sup = p3d_h36.clone()[:, :, dim_used][:, -out_n - seq_in:].reshape(
            [-1, seq_in + out_n, len(dim_used) // 3, 3])
        p3d_src = p3d_h36.clone()[:, :, dim_used]
        # p3d_src = p3d_src.permute(1, 0, 2)  # seq * n * dim
        # p3d_src = p3d_src[:in_n]
        p3d_out_all = net_pred(p3d_src*1000, input_n=in_n, output_n=10, itera=itera)

        p3d_out_all = p3d_out_all[:, seq_in:].transpose(1, 2).reshape([batch_size, 10 * itera, -1])[:, :out_n]
        # print(p3d_out_all.shape)

        p3d_out = p3d_h36.clone()[:, in_n:in_n + out_n]
        p3d_out[:, :, dim_used] = p3d_out_all
        p3d_out[:, :, index_to_ignore] = p3d_out[:, :, index_to_equal]
        p3d_out = p3d_out.reshape([-1, out_n, 32, 3])*0.001
        # print(p3d_out.shape)
        
        #mpjpe_p3d_h36 = torch.sum(torch.mean(torch.norm(p3d_h36[:, in_n:] - p3d_out, dim=3), dim=2), dim=0)

        #end of network

        #publish the groundtruth for visualization 
        p3d_h36 = p3d_h36.reshape([-1, in_n + out_n, 32, 3])
        grnd_truth = p3d_h36[:, in_n:]
        grnd = grnd_truth[:,-1]
        grnd_coordinates = grnd.view(num_joints, 3)

        marker1 = Marker()
        marker1.header.frame_id = "depth_camera_link"
        # marker1.type = Marker.SPHERE_LIST
        marker1.type = Marker.LINE_LIST
        # marker.type = 2
        marker1.action = Marker.ADD
        marker1.scale.x = 0.01
        marker1.scale.y = 0.01
        marker1.scale.z = 0.01
        marker1.color.a = 1.0
        marker1.color.r = 0.0
        marker1.color.g = 0.0
        marker1.color.b = 1.0     
        
        #for simple points at each body joint
        # for coordinate in grnd_coordinates:
        #     point = Point()
        #     point.x = coordinate[0].item()
        #     point.y = coordinate[1].item()
        #     point.z = coordinate[2].item()
        #     marker1.points.append(point)

        #for edges that connects the body joints to give a skeleton representation
        # connections = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 6), (6, 7), (7, 8), (11, 12),(12, 13), (13, 14), (14, 15), (18,22),(18, 19), (19, 20), (20, 21), (22, 23), (23, 24), (24, 25), (26, 3), (26,27)]
        connections = [(0, 2), (2, 3), (3, 5), (5, 6),(6, 8),(3,12),(12, 13), (13, 15), (18,22),(18, 19), (19,21), (22,23),(23, 25), (26, 3), (26,27)]

        for start_idx, end_idx in connections:
            start_coordinate = grnd_coordinates[start_idx]
            end_coordinate = grnd_coordinates[end_idx]

            # Add start point
            start_point = Point()
            start_point.x = start_coordinate[0].item()
            start_point.y = start_coordinate[1].item()
            start_point.z = start_coordinate[2].item()
            marker1.points.append(start_point)

            # Add end point
            end_point = Point()
            end_point.x = end_coordinate[0].item()
            end_point.y = end_coordinate[1].item()
            end_point.z = end_coordinate[2].item()
            marker1.points.append(end_point)
        
        # marker1.pose.orientation.x = 0.0
        # marker1.pose.orientation.y = 0.0
        # marker1.pose.orientation.z = 0.0
        # marker1.pose.orientation.w = 1.0

        marker_publisher1.publish(marker1)

        #publish the prediction for visualization 
        prediction = p3d_out[:,-1]
        prediction_coordinates = prediction.view(num_joints, 3)

        marker = Marker()
        marker.header.frame_id = "depth_camera_link"
        marker.type = Marker.LINE_LIST
        # marker.type = 2
        marker.action = Marker.ADD
        marker.scale.x = 0.01
        marker.scale.y = 0.01
        marker.scale.z = 0.01
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0

        for start_idx, end_idx in connections:
            start_coordinate = prediction_coordinates[start_idx]
            end_coordinate = prediction_coordinates[end_idx]

            # Add start point
            start_point = Point()
            start_point.x = start_coordinate[0].item()
            start_point.y = start_coordinate[1].item()
            start_point.z = start_coordinate[2].item()
            marker.points.append(start_point)

            # Add end point
            end_point = Point()
            end_point.x = end_coordinate[0].item()
            end_point.y = end_coordinate[1].item()
            end_point.z = end_coordinate[2].item()
            marker.points.append(end_point)
        
        # marker.pose.orientation.x = 0.0
        # marker.pose.orientation.y = 0.0
        # marker.pose.orientation.z = 0.0
        # marker.pose.orientation.w = 1.0

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

        # allocated_memory = torch.cuda.memory_allocated(torch.device("cuda"))
        # allocated_memory = allocated_memory/(1024**3)
        # print("Allocated GPU Memory:", allocated_memory, "GB")


if __name__ == '__main__':
    rospy.init_node('motion_forecasting_node', anonymous=True)
    marker_publisher1 = rospy.Publisher("/visualization_marker1", Marker, queue_size=1)
    marker_publisher = rospy.Publisher("/visualization_marker", Marker, queue_size=1)
    rospy.Subscriber('body_tracking_data', MarkerArray, body_tracking_callback)
    rospy.spin()