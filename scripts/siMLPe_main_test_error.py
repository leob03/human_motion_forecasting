#!/usr/bin/env python
#only for the predictions error, without comparisons (ground truth) and no publication on RVIZ

import rospy
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray
import torch
import time
import matplotlib.pyplot as plt
import threading
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
num_frames = 60
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

m_p3d_h36 = np.zeros([config.motion.h36m_target_length])
titles = np.array(range(config.motion.h36m_target_length)) + 1
joint_used_xyz = np.array([2,3,4,5,7,8,9,10,12,13,14,15,17,18,19,21,22,25,26,27,29,30]).astype(np.int64)
num_samples = 0

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

def run_model(model, num_samples, joint_used_xyz, m_p3d_h36, input_data=None):
    joint_to_ignore = np.array([16, 20, 23, 24, 28, 31]).astype(np.int64)
    joint_equal = np.array([13, 19, 22, 13, 27, 30]).astype(np.int64)

    input_data = input_data.reshape(batch_size, num_frames, num_joints, 3)
    motion_input = input_data.cuda()
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

    #method in siMLPe
    mpjpe_p3d_h36 = torch.sum(torch.mean(torch.norm(motion_pred*1000 - motion_gt*1000, dim=3), dim=2), dim=0)
    m_p3d_h36 += mpjpe_p3d_h36.cpu().numpy()

    #method in HRI
    # p3d_out = p3d_out.reshape([-1, out_n, 32, 3])
    # p3d_h36 = p3d_h36.reshape([-1, in_n + out_n, 32, 3])
    # mpjpe_p3d_h36 = torch.sum(torch.mean(torch.norm(p3d_h36[:, in_n:] - p3d_out, dim=3), dim=2), dim=0)

    #method in STS_GCN
    batch_pred=p3d_out.view(-1,out_n,32,3).contiguous().view(-1,3)
    batch_gt= p3d_h36[:, in_n:].view(-1, out_n,32,3).contiguous().view(-1,3)
    mpjpe_p3d_h36 =torch.mean(torch.norm(batch_gt-batch_pred,2,1))

    m_p3d_h36 += mpjpe_p3d_h36.cpu().data.numpy()

    ###

    # ret = {}
    # m_p3d_h36 = m_p3d_h36
    # for j in range(out_n):
    #     ret["#{:d}".format(titles[j])] = m_p3d_h36[j]
    # return ret

    ret = {}
    for j in range(config.motion.h36m_target_length):
        ret["#{:d}".format(titles[j])] = [m_p3d_h36[j], m_p3d_h36[j]]
    return [round(ret[key][0], 1) for key in results_keys]

def body_tracking_callback(msg):
    
    global frame_count, start_timestamp, past_frames, error_cumulate, inference_time_mean, actual_callbacks

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
        print("Elapsed time between two callbacks:", elapsed_time)

        start_timestamp_inference = time.time()
        ret_test = run_model(net_pred, is_train=3, input_data=input_data, opt=opt)
        inference_time = time.time() - start_timestamp_inference
        print("Inference time:", inference_time)
        actual_callbacks += 1
        if actual_callbacks>1:
            inference_time_mean += inference_time
        print('testing error: {:.3f}'.format(ret_test['#10']))
        
        error_cumulate += ret_test['#10']
        errors.append(error_cumulate)
        update_plot()

        frame_count = num_frames - num_new_frames
        processed_data.zero_()

        if num_new_frames < num_frames:
            past_frames = input_data[:, num_new_frames:]
        
        start_timestamp = time.time()

        # allocated_memory = torch.cuda.memory_allocated(torch.device("cuda"))
        # print("Allocated GPU Memory:", allocated_memory)

def update_plot():
    line.set_data(range(len(errors)), errors)
    ax.relim()
    ax.autoscale_view() 
    fig.canvas.draw_idle()


def main():
    global inference_time_mean, actual_callbacks
    rospy.init_node('subscriber_node', anonymous=True)
    rospy.Subscriber('body_tracking_data', MarkerArray, body_tracking_callback)
    # plt.ion()
    # plot_thread = threading.Thread(target=update_plot)
    # plot_thread.start()
    rospy.spin()
    # plot_thread.join()  # Wait for the plot thread to finish
    # plt.ioff()
    inference_time_mean = inference_time_mean/(actual_callbacks-1)
    print("mean Inference time:", inference_time_mean)
    plt.show()


if __name__ == '__main__':
    main()