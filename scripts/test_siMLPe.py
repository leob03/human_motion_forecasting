#!/usr/bin/env python

import rospy
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point

import torch
import time
import matplotlib.pyplot as plt
import argparse
from scipy.spatial.transform import Rotation as R

from utils.misc import rotmat2xyz_torch, rotmat2euler_torch

import sys
sys.path.append('/home/bartonlab-user/workspace/src/human_motion_forecasting/scripts')
import os

from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import h5py
import torch.optim as optim

from utils.config  import config
from model.siMLPe import siMLPe as Model
from utils.h36m_eval import H36MEval

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

def regress_pred(model, pbar, num_samples, joint_used_xyz, m_p3d_h36, prediction_publisher, ground_truth_publisher):
    joint_to_ignore = np.array([16, 20, 23, 24, 28, 31]).astype(np.int64)
    joint_equal = np.array([13, 19, 22, 13, 27, 30]).astype(np.int64)
    connections = [(0, 2), (2, 3), (3, 5), (5, 6),(6, 8),(3,12),(12, 13), (13, 15), (18,22),(18, 19), (19,21), (22,23),(23, 25), (26, 3), (26,27)]

    for (motion_input, motion_target) in pbar:
        motion_input = motion_input.cuda()
        b,n,c,_ = motion_input.shape
        num_samples += b
        print(n)

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

        motion_target = motion_target.detach()
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
        publish_marker_array(ground_truth_publisher, motion_gt[0,10,:,:], connections, 0.0, 0.0, 1.0)
        publish_marker_array(prediction_publisher, motion_pred[0,10,:,:], connections, 0.0, 1.0, 0.0)

        mpjpe_p3d_h36 = torch.sum(torch.mean(torch.norm(motion_pred*1000 - motion_gt*1000, dim=3), dim=2), dim=0)
        m_p3d_h36 += mpjpe_p3d_h36.cpu().numpy()
        time.sleep(0.100)

    m_p3d_h36 = m_p3d_h36 / num_samples
    return m_p3d_h36

def test(config, model, dataloader, prediction_publisher, ground_truth_publisher) :

    m_p3d_h36 = np.zeros([config.motion.h36m_target_length])
    titles = np.array(range(config.motion.h36m_target_length)) + 1
    joint_used_xyz = np.array([2,3,4,5,7,8,9,10,12,13,14,15,17,18,19,21,22,25,26,27,29,30]).astype(np.int64)
    num_samples = 0

    pbar = dataloader
    m_p3d_h36 = regress_pred(model, pbar, num_samples, joint_used_xyz, m_p3d_h36, prediction_publisher, ground_truth_publisher)

    ret = {}
    for j in range(config.motion.h36m_target_length):
        ret["#{:d}".format(titles[j])] = [m_p3d_h36[j], m_p3d_h36[j]]
    return [round(ret[key][0], 1) for key in results_keys]

def publish_marker_array(publisher, joint_coordinates, connections, color_r, color_g, color_b):
    marker_array = MarkerArray()
    marker = Marker()
    marker.header.frame_id = "depth_camera_link"
    marker.type = Marker.SPHERE_LIST
    # marker.type = Marker.LINE_LIST
    marker.action = Marker.ADD
    marker.scale.x = 0.05
    marker.scale.y = 0.05
    marker.scale.z = 0.05
    marker.color.a = 1.0
    marker.color.r = color_r
    marker.color.g = color_g
    marker.color.b = color_b

    # for simple points at each body joint
    for coordinate in joint_coordinates:
        point = Point()
        point.x = coordinate[0].item()
        point.y = coordinate[1].item()
        point.z = coordinate[2].item()
        marker.points.append(point)

    # for start_idx, end_idx in connections:
    #     start_coordinate = joint_coordinates[start_idx]
    #     end_coordinate = joint_coordinates[end_idx]

    #     # Add start point
    #     start_point = Point()
    #     start_point.x = start_coordinate[0]
    #     start_point.y = start_coordinate[1]
    #     start_point.z = start_coordinate[2]
    #     marker.points.append(start_point)

    #     # Add end point
    #     end_point = Point()
    #     end_point.x = end_coordinate[0]
    #     end_point.y = end_coordinate[1]
    #     end_point.z = end_coordinate[2]
    #     marker.points.append(end_point)

    marker_array.markers.append(marker)
    publisher.publish(marker_array)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--model-pth', type=str, default='/home/bartonlab-user/workspace/src/human_motion_forecasting/scripts/checkpoint/siMLPe/model-iter-40000.pth', help='=encoder path')
    args = parser.parse_args()

    rospy.init_node('pose_estimation_visualizer', anonymous=True)
    prediction_publisher = rospy.Publisher('/pose_estimation/predictions', MarkerArray, queue_size=10)
    ground_truth_publisher = rospy.Publisher('/pose_estimation/ground_truth', MarkerArray, queue_size=10)


    model = Model(config)

    state_dict = torch.load(args.model_pth)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.cuda()

    config.motion.h36m_target_length = config.motion.h36m_target_length_eval
    dataset = H36MEval(config, 'test')

    shuffle = False
    sampler = None
    train_sampler = None
    dataloader = DataLoader(dataset, batch_size=128,
                            num_workers=1, drop_last=False,
                            sampler=sampler, shuffle=shuffle, pin_memory=True)
    
    results = test(config, model, dataloader, prediction_publisher, ground_truth_publisher)
    
    rospy.spin()

