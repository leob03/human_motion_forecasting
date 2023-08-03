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
      args.seq_length_out if not sampling else 10,
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

def define_actions( action ):
  """
  Define the list of actions we are using.

  Args
    action: String with the passed action. Could be "all"
  Returns
    actions: List of strings of actions
  Raises
    ValueError if the action is not included in H3.6M
  """

  actions = ["walking", "eating", "smoking", "discussion",  "directions",
              "greeting", "phoning", "posing", "purchases", "sitting",
              "sittingdown", "takingphoto", "waiting", "walkingdog",
              "walkingtogether"]

  if action in actions:
    return [action]

  if action == "all":
    return actions

  if action == "all_srnn":
    return ["walking", "eating", "smoking", "discussion"]

  raise( ValueError, "Unrecognized action: %d" % action )


def get_srnn_gts( actions, model, test_set, data_mean, data_std, dim_to_ignore, one_hot, to_euler=True ):
  """
  Get the ground truths for srnn's sequences, and convert to Euler angles.
  (the error is always computed in Euler angles).

  Args
    actions: a list of actions to get ground truths for.
    model: training model we are using (we only use the "get_batch" method).
    test_set: dictionary with normalized training data.
    data_mean: d-long vector with the mean of the training data.
    data_std: d-long vector with the standard deviation of the training data.
    dim_to_ignore: dimensions that we are not using to train/predict.
    one_hot: whether the data comes with one-hot encoding indicating action.
    to_euler: whether to convert the angles to Euler format or keep thm in exponential map

  Returns
    srnn_gts_euler: a dictionary where the keys are actions, and the values
      are the ground_truth, denormalized expected outputs of srnns's seeds.
  """
  srnn_gts_euler = {}

  for action in actions:

    srnn_gt_euler = []
    _, _, srnn_expmap = model.get_batch_srnn( test_set, action )

    # expmap -> rotmat -> euler
    for i in np.arange( srnn_expmap.shape[0] ):
      denormed = data_utils_srnn.unNormalizeData(srnn_expmap[i,:,:], data_mean, data_std, dim_to_ignore, actions, one_hot )

      if to_euler:
        for j in np.arange( denormed.shape[0] ):
          for k in np.arange(3,97,3):
            denormed[j,k:k+3] = data_utils_srnn.rotmat2euler( data_utils_srnn.expmap2rotmat( denormed[j,k:k+3] ))

      srnn_gt_euler.append( denormed );

    # Put back in the dictionary
    srnn_gts_euler[action] = srnn_gt_euler

  return srnn_gts_euler


def sample():
  """Sample predictions for srnn's seeds"""
#   actions = define_actions( args.action )

  if True:
    # === Create the model ===
    print("Creating %d layers of %d units." % (args.num_layers, args.size))
    sampling     = True
    model = create_model(actions, sampling)
    if not args.use_cpu:
        model = model.cuda()
    print("Model created")

    # Load all the data
    # train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use = read_all_data(
    #   actions, args.seq_length_in, args.seq_length_out, args.data_dir, not args.omit_one_hot )

    # === Read and denormalize the gt with srnn's seeds, as we'll need them
    # many times for evaluation in Euler Angles ===
    srnn_gts_expmap = get_srnn_gts( actions, model, test_set, data_mean,
                              data_std, dim_to_ignore, not args.omit_one_hot, to_euler=False )
    srnn_gts_euler = get_srnn_gts( actions, model, test_set, data_mean,
                              data_std, dim_to_ignore, not args.omit_one_hot )

    # Clean and create a new h5 file of samples
    SAMPLES_FNAME = 'samples.h5'
    try:
      os.remove( SAMPLES_FNAME )
    except OSError:
      pass

    # Predict and save for each action
    for action in actions:

      # Make prediction with srnn' seeds
      encoder_inputs, decoder_inputs, decoder_outputs = model.get_batch_srnn( test_set, action )

      encoder_inputs = torch.from_numpy(encoder_inputs).float()
      decoder_inputs = torch.from_numpy(decoder_inputs).float()
      decoder_outputs = torch.from_numpy(decoder_outputs).float()
      if not args.use_cpu:
        encoder_inputs = encoder_inputs.cuda()
        decoder_inputs = decoder_inputs.cuda()
        decoder_outputs = decoder_outputs.cuda()

      srnn_poses = model(encoder_inputs, decoder_inputs)

      srnn_loss = (srnn_poses - decoder_outputs)**2
      srnn_loss.cpu().data.numpy()
      srnn_loss = srnn_loss.mean()

      srnn_poses = srnn_poses.cpu().data.numpy()
      srnn_poses = srnn_poses.transpose([1,0,2])

      srnn_loss = srnn_loss.cpu().data.numpy()
      # denormalizes too
      srnn_pred_expmap = data_utils_srnn.revert_output_format(srnn_poses, data_mean, data_std, dim_to_ignore, actions, not args.omit_one_hot )

      # Save the samples
      with h5py.File( SAMPLES_FNAME, 'a' ) as hf:
        for i in np.arange(8):
          # Save conditioning ground truth
          node_name = 'expmap/gt/{1}_{0}'.format(i, action)
          hf.create_dataset( node_name, data=srnn_gts_expmap[action][i] )
          # Save prediction
          node_name = 'expmap/preds/{1}_{0}'.format(i, action)
          hf.create_dataset( node_name, data=srnn_pred_expmap[i] )

      # Compute and save the errors here
      mean_errors = np.zeros( (len(srnn_pred_expmap), srnn_pred_expmap[0].shape[0]) )

      for i in np.arange(8):

        eulerchannels_pred = srnn_pred_expmap[i]

        for j in np.arange( eulerchannels_pred.shape[0] ):
          for k in np.arange(3,97,3):
            eulerchannels_pred[j,k:k+3] = data_utils_srnn.rotmat2euler(
              data_utils_srnn.expmap2rotmat( eulerchannels_pred[j,k:k+3] ))

        eulerchannels_pred[:,0:6] = 0

        # Pick only the dimensions with sufficient standard deviation. Others are ignored.
        idx_to_use = np.where( np.std( eulerchannels_pred, 0 ) > 1e-4 )[0]

        euc_error = np.power( srnn_gts_euler[action][i][:,idx_to_use] - eulerchannels_pred[:,idx_to_use], 2)
        euc_error = np.sum(euc_error, 1)
        euc_error = np.sqrt( euc_error )
        mean_errors[i,:] = euc_error

      mean_mean_errors = np.mean( mean_errors, 0 )
      print( action )
      print( ','.join(map(str, mean_mean_errors.tolist() )) )

      with h5py.File( SAMPLES_FNAME, 'a' ) as hf:
        node_name = 'mean_{0}_error'.format( action )
        hf.create_dataset( node_name, data=mean_mean_errors )

  return



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

    p3d_src = p3d_h36.clone()[:, :, dim_used]

    p3d_out_all = net_pred(p3d_src*1000, input_n=in_n, output_n=10, itera=itera)
    p3d_out_all = p3d_out_all*0.001

    p3d_out_all = p3d_out_all[:, seq_in:].transpose(1, 2).reshape([batch_size, 10 * itera, -1])[:, :out_n]

    p3d_out = p3d_h36.clone()[:, in_n:in_n + out_n]
    p3d_out[:, :, dim_used] = p3d_out_all
    p3d_out[:, :, index_to_ignore] = p3d_out[:, :, index_to_equal]
    p3d_out = p3d_out.reshape([-1, out_n, 32, 3])

    p3d_h36 = p3d_h36.reshape([-1, in_n + out_n, 32, 3])
    
    mpjpe_p3d_h36 = torch.sum(torch.mean(torch.norm(p3d_h36[:, in_n:] - p3d_out, dim=3), dim=2), dim=0)

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