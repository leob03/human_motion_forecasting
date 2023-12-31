#!/usr/bin/env python
#only for the predictions error, without comparisons (ground truth) and no publication on RVIZ

import rospy
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray
import torch
import time
import matplotlib.pyplot as plt
import threading
# from matplotlib.animation import FuncAnimation
# import matplotlib
# matplotlib.use('TkAgg')

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
num_new_frames = 1  # Number of new frames to collect in each iteration

errors = []
error_cumulate = 0
fig, ax = plt.subplots()
line, = ax.plot([], [])

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

torch.cuda.empty_cache()

processed_data = torch.zeros(batch_size, num_frames, num_joints*3)
past_frames = torch.zeros(batch_size, num_frames - num_new_frames, num_joints*3)  # Store past frames


frame_count = 0
start_timestamp = time.time()
inference_time_mean = 0
actual_callbacks = 0

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

    ret = {}
    m_p3d_h36 = m_p3d_h36
    for j in range(out_n):
        ret["#{:d}".format(titles[j])] = m_p3d_h36[j]
    return ret

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
