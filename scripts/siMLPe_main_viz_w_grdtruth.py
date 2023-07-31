#!/usr/bin/env python

#prediciton of the present using the past (with ground truth)

import rospy
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point

import torch
import time
import matplotlib.pyplot as plt
import argparse
from scipy.spatial.transform import Rotation as R

from utils.config  import config
from model.siMLPe import siMLPe as Model
from utils.misc import rotmat2xyz_torch, rotmat2euler_torch

import sys
sys.path.append('/home/bartonlab-user/workspace/src/human_motion_forecasting/scripts')
import os

from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import h5py
import torch.optim as optim

batch_size = 1
num_frames = 75
num_joints = 32
num_new_frames = 10  # Number of new frames to collect in each iteration

errors = []
error_cumulate = 0
fig, ax = plt.subplots()
line, = ax.plot([], [])

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--model-pth', type=str, default=None, help='=encoder path')
args = parser.parse_args()

model = Model(config)

state_dict = torch.load(args.model_pth)
model.load_state_dict(state_dict, strict=True)
model.eval()
model.cuda()

config.motion.h36m_target_length = config.motion.h36m_target_length_eval

torch.cuda.empty_cache()

processed_data = torch.zeros(batch_size, num_frames, num_joints*3)
past_frames = torch.zeros(batch_size, num_frames - num_new_frames, num_joints*3)  # Store past frames


frame_count = 0
start_timestamp = time.time()
inference_time_mean = 0
actual_callbacks = 0

results_keys = ['#2', '#4', '#8', '#10', '#14', '#18', '#22', '#25']

def get_dct_matrix(N):
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    return dct_m, idct_m

dct_m,idct_m = get_dct_matrix(config.motion.h36m_input_length_dct)
dct_m = torch.tensor(dct_m).float().cuda().unsqueeze(0)
idct_m = torch.tensor(idct_m).float().cuda().unsqueeze(0)


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
        joint_used_xyz = np.array([2,3,4,5,7,8,9,10,12,13,14,15,17,18,19,21,22,25,26,27,29,30]).astype(np.int64)
        num_samples = 0
        joint_to_ignore = np.array([16, 20, 23, 24, 28, 31]).astype(np.int64)
        joint_equal = np.array([13, 19, 22, 13, 27, 30]).astype(np.int64)

        input_data = input_data.reshape(batch_size, num_frames, num_joints, 3)
        motion_input = input_data[:,:50].cuda()
        b,n,c,_ = motion_input.shape
        num_samples += b

        motion_input = motion_input.reshape(b, n, 32, 3)
        motion_input = motion_input[:, :, joint_used_xyz].reshape(b, n, -1)
        outputs = []
        step = config.motion.h36m_target_length_train
        if step == 25:
            num_step = 1
        else:
            num_step = 25 // step + 1
        for idx in range(num_step):
            with torch.no_grad():
                if config.deriv_input:
                    motion_input_ = motion_input.clone()
                    motion_input_ = torch.matmul(dct_m[:, :, :config.motion.h36m_input_length], motion_input_.cuda())
                else:
                    motion_input_ = motion_input.clone()
                output = model(motion_input_)
                output = torch.matmul(idct_m[:, :config.motion.h36m_input_length, :], output)[:, :step, :]
                if config.deriv_output:
                    output = output + motion_input[:, -1:, :].repeat(1,step,1)

            output = output.reshape(-1, 22*3)
            output = output.reshape(b,step,-1)
            outputs.append(output)
            motion_input = torch.cat([motion_input[:, step:], output], axis=1)
        motion_pred = torch.cat(outputs, axis=1)[:,:25]

        #define motion_target
        motion_target = input_data[:,-25:].detach()
        b,n,c,_ = motion_target.shape

        motion_gt = motion_target.clone()

        motion_pred = motion_pred.detach().cpu()
        pred_rot = motion_pred.clone().reshape(b,n,22,3)
        motion_pred = motion_target.clone().reshape(b,n,32,3)
        motion_pred[:, :, joint_used_xyz] = pred_rot

        tmp = motion_gt.clone()
        tmp[:, :, joint_used_xyz] = motion_pred[:, :, joint_used_xyz]
        motion_pred = tmp
        motion_pred[:, :, joint_to_ignore] = motion_pred[:, :, joint_equal]
        
        #mpjpe_p3d_h36 = torch.sum(torch.mean(torch.norm(p3d_h36[:, in_n:] - p3d_out, dim=3), dim=2), dim=0)

        #end of network

        #publish the groundtruth for visualization 
        p3d_h36 = input_data.reshape([-1, 50 + 25, 32, 3])
        grnd_truth = p3d_h36[:, -25:]
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
        prediction = motion_pred[:,-1]
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
    rospy.init_node('siMLPe_motion_forecasting_node', anonymous=True)
    marker_publisher1 = rospy.Publisher("/siMLPe_visualization_marker_gt", Marker, queue_size=1)
    marker_publisher = rospy.Publisher("/siMLPe_visualization_marker", Marker, queue_size=1)
    rospy.Subscriber('body_tracking_data', MarkerArray, body_tracking_callback)
    rospy.spin()