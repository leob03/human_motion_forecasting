from utils import h36motion3d as datasets
from model import AttModel
from utils.opt import Options
from utils import util
from utils import log

from torch.utils.data import DataLoader
import torch
from torch.utils.tensorboard import SummaryWriter
# tensorboard --logdir=logs --port=8000
# http://localhost:8000/#timeseries
import torch.nn as nn
import numpy as np
import time
import h5py
import torch.optim as optim

import torch.autograd as autograd

def gen_velocity(m):
    dm = m[:, 1:] - m[:, :-1]
    return dm

# def visualize_dataset(data_loader):
#     for batch_idx, batch_data in enumerate(data_loader):
#         print("batch_data.shape=", batch_data.shape)
#         inputs = batch_data
#         for sample in inputs:
#             print(sample.shape)
#             print(sample)
#             break
#         break

def main(opt):
    lr_now = opt.lr_now
    start_epoch = 1
    nb_iter = 0
    # print("nb_iter = ", nb_iter)
    #Don't forget to change the log_dir name
    writer = SummaryWriter(log_dir='logs_HRI_2')
    # opt.is_eval = True
    print('>>> create models')
    in_features = opt.in_features  # 66
    d_model = opt.d_model
    kernel_size = opt.kernel_size
    net_pred = AttModel.AttModel(in_features=in_features, kernel_size=kernel_size, d_model=d_model,
                                 num_stage=opt.num_stage, dct_n=opt.dct_n)
    net_pred.cuda()

    optimizer = optim.Adam(filter(lambda x: x.requires_grad, net_pred.parameters()), lr=opt.lr_now)
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in net_pred.parameters()) / 1000000.0))

    if opt.is_load or opt.is_eval:
        model_path_len = './{}/ckpt_best.pth.tar'.format(opt.ckpt)
        print(">>> loading ckpt len from '{}'".format(model_path_len))
        ckpt = torch.load(model_path_len)
        start_epoch = ckpt['epoch'] + 1
        err_best = ckpt['err']
        lr_now = ckpt['lr']
        net_pred.load_state_dict(ckpt['state_dict'])
        # net.load_state_dict(ckpt)
        # optimizer.load_state_dict(ckpt['optimizer'])
        # lr_now = util.lr_decay_mine(optimizer, lr_now, 0.2)
        print(">>> ckpt len loaded (epoch: {} | err: {})".format(ckpt['epoch'], ckpt['err']))

    print('>>> loading datasets')

    if not opt.is_eval:
        # dataset = datasets.Datasets(opt, split=0)
        # actions = ["walking", "eating", "smoking", "discussion", "directions",
        #            "greeting", "phoning", "posing", "purchases", "sitting",
        #            "sittingdown", "takingphoto", "waiting", "walkingdog",
        #            "walkingtogether"]
        dataset = datasets.Datasets(opt, split=0)
        print('>>> Training dataset length: {:d}'.format(dataset.__len__()))
        data_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0, pin_memory=True)
        # for batch_idx, batch_data in enumerate(data_loader):
        #     print("batch_data.shape=", batch_data.shape)
        #     inputs = batch_data
        #     for sample in inputs:
        #         print(sample.shape)
        #         print(sample.reshape(-1,32,3)[0,:])
        #         break
        #     break
        valid_dataset = datasets.Datasets(opt, split=1)
        print('>>> Validation dataset length: {:d}'.format(valid_dataset.__len__()))
        valid_loader = DataLoader(valid_dataset, batch_size=opt.test_batch_size, shuffle=True, num_workers=0,
                                  pin_memory=True)

    test_dataset = datasets.Datasets(opt, split=2)
    print('>>> Testing dataset length: {:d}'.format(test_dataset.__len__()))
    test_loader = DataLoader(test_dataset, batch_size=opt.test_batch_size, shuffle=False, num_workers=0,
                             pin_memory=True)

    # evaluation
    if opt.is_eval:
        # print(test_loader)
        ret_test, nb_iter = run_model(net_pred, writer, nb_iter, is_train=3, data_loader=test_loader, opt=opt)
        ret_log = np.array([])
        head = np.array([])
        for k in ret_test.keys():
            ret_log = np.append(ret_log, [ret_test[k]])
            head = np.append(head, [k])
        log.save_csv_log(opt, head, ret_log, is_create=True, file_name='test_walking')
        # print('testing error: {:.3f}'.format(ret_test['m_p3d_h36']))

    # training
    if not opt.is_eval:
        err_best = 1000
        for epo in range(start_epoch, opt.epoch + 1):
            is_best = False
            # if epo % opt.lr_decay == 0:
            lr_now = util.lr_decay_mine(optimizer, lr_now, 0.1 ** (1 / opt.epoch))
            print('>>> training epoch: {:d}'.format(epo))
            with torch.autograd.detect_anomaly():
                ret_train, nb_iter = run_model(net_pred, writer, nb_iter, optimizer, is_train=0, data_loader=data_loader, epo=epo, opt=opt)            
            writer.add_scalar('Error/train', ret_train['m_p3d_h36'], epo)
            print('train error: {:.3f}'.format(ret_train['m_p3d_h36']))
            ret_valid, nb_iter = run_model(net_pred, writer, nb_iter, is_train=1, data_loader=valid_loader, opt=opt, epo=epo)
            writer.add_scalar('Error/validation', ret_valid['m_p3d_h36'], epo)
            print('validation error: {:.3f}'.format(ret_valid['m_p3d_h36']))
            ret_test, nb_iter = run_model(net_pred, writer, nb_iter, is_train=3, data_loader=test_loader, opt=opt, epo=epo)
            writer.add_scalar('Error/testing_frame10', ret_test['#10'], epo)
            # print('testing error: {:.3f}'.format(ret_test['#10']))
            writer.add_scalar('Error/testing_mean', ret_test['m_p3d_h36'], epo)
            print('testing error: {:.3f}'.format(ret_test['m_p3d_h36']))
            ret_log = np.array([epo, lr_now])
            head = np.array(['epoch', 'lr'])
            for k in ret_train.keys():
                ret_log = np.append(ret_log, [ret_train[k]])
                head = np.append(head, [k])
            for k in ret_valid.keys():
                ret_log = np.append(ret_log, [ret_valid[k]])
                head = np.append(head, ['valid_' + k])
            for k in ret_test.keys():
                ret_log = np.append(ret_log, [ret_test[k]])
                head = np.append(head, ['test_' + k])
            log.save_csv_log(opt, head, ret_log, is_create=(epo == 1))
            if ret_valid['m_p3d_h36'] < err_best:
                err_best = ret_valid['m_p3d_h36']
                is_best = True
            log.save_ckpt({'epoch': epo,
                           'lr': lr_now,
                           'err': ret_valid['m_p3d_h36'],
                           'state_dict': net_pred.state_dict(),
                           'optimizer': optimizer.state_dict()},
                          is_best=is_best, opt=opt)
    writer.close()


def run_model(net_pred, writer, nb_iter, optimizer=None, is_train=0, data_loader=None, epo=1, opt=None):
    if is_train == 0:
        net_pred.train()
    else:
        net_pred.eval()

    l_p3d = 0
    # if is_train <= 1: #during training and validation
    m_p3d_h36 = 0
    m_p3d_h36_bis = np.zeros([opt.output_n])
    # else:
    titles = np.array(range(opt.output_n)) + 1
    #     m_p3d_h36 = np.zeros([opt.output_n])
    n = 0
    in_n = opt.input_n
    out_n = opt.output_n
    dim_used = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25,
                         26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                         46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68,
                         75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92])
    seq_in = opt.kernel_size #=M=10
    # joints at same loc
    joint_to_ignore = np.array([16, 20, 23, 24, 28, 31])
    index_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
    joint_equal = np.array([13, 19, 22, 13, 27, 30])
    index_to_equal = np.concatenate((joint_equal * 3, joint_equal * 3 + 1, joint_equal * 3 + 2))

    itera = 1
    idx = np.expand_dims(np.arange(seq_in + out_n), axis=1) + (
            out_n - seq_in + np.expand_dims(np.arange(itera), axis=0))
    st = time.time()
    for i, (p3d_h36) in enumerate(data_loader):
        # print(i)
        batch_size, seq_n, _ = p3d_h36.shape
        # when only one sample in this batch
        if batch_size == 1 and is_train == 0:
            continue
        n += batch_size
        bt = time.time()
        p3d_h36 = p3d_h36.float().cuda()    #p3d_h36.shape = (batch_size, seq_n=50+10=60, 96)
        p3d_sup = p3d_h36.clone()[:, :, dim_used][:, -out_n - seq_in:].reshape(
            [-1, seq_in + out_n, len(dim_used) // 3, 3])
        #p3d_sup.shape = (batch_size, seq_in + out_n, 22, 3), the last M+T=20 frames
        p3d_src = p3d_h36.clone()[:, :, dim_used] #the whole sequence
        p3d_out_all = net_pred(p3d_src, input_n=in_n, output_n=out_n, itera=itera)  #the residual vector of predictions from N-M to N+T frames, the last M+T=20 frames (only dim to use)
        #p3d_out_all.shape = (batch_size, seq_in + out_n, itera=1, 66)

        p3d_out = p3d_h36.clone()[:, in_n:in_n + out_n] # the [N,N+T] frames with all the joints (10 last ones), p3d_out.shape = (batch_size, out_n, 96)
        p3d_out[:, :, dim_used] = p3d_out_all[:, seq_in:, 0] #the dimension to use are replaced by the predictions
        p3d_out[:, :, index_to_ignore] = p3d_out[:, :, index_to_equal] #the joints at same location are replaced by the same value
        p3d_out = p3d_out.reshape([-1, out_n, 32, 3]) #p3d_out.shape = (batch_size, out_n=10, 32, 3)

        p3d_h36 = p3d_h36.reshape([-1, in_n + out_n, 32, 3])    #p3d_h36.shape = (batch_size, seq_n=50+10=60, 32, 3)

        p3d_out_all = p3d_out_all.reshape([batch_size, seq_in + out_n, itera, len(dim_used) // 3, 3])   #p3d_out_all.shape = (batch_size, seq_in + out_n, itera=1, 22, 3)

        # 2d joint loss:
        grad_norm = 0
        if is_train == 0: #training
            b, f = p3d_out_all.shape[0], p3d_out_all.shape[1]
            loss_p3d = torch.mean(torch.norm(p3d_out_all[:, :, 0].reshape(-1,3) - p3d_sup.reshape(-1,3), 2, 1))
            
            # loss = torch.mean(torch.norm(motion_pred - h36m_motion_target, 2, 1))

            motion_pred = p3d_out_all[:, :, 0].reshape(b,f,22,3)
            dmotion_pred = gen_velocity(motion_pred)
            motion_gt = p3d_sup.reshape(b,f,22,3)
            dmotion_gt = gen_velocity(motion_gt)
            dloss = torch.mean(torch.norm((dmotion_pred - dmotion_gt).reshape(-1,3), 2, 1))
            loss = loss_p3d + dloss

            writer.add_scalar('Loss/angle', loss.detach().cpu().numpy(), nb_iter)
            nb_iter += 1
            # print(nb_iter)

            loss_all = loss
            optimizer.zero_grad()
            loss_all.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(list(net_pred.parameters()), max_norm=opt.max_norm)
            optimizer.step()
            # update log values
            l_p3d += loss_p3d.cpu().data.numpy() * batch_size

        if is_train <= 1:  # if is validation or train simply output the overall mean error
            mpjpe_p3d_h36 = torch.mean(torch.norm(p3d_h36[:, in_n:in_n + out_n] - p3d_out, dim=3)) #Xreal-Xpred on the last 10 frames (on all unused dimensions the two are equal)
            #mpjpe_p3d_h36 is a scalar
            m_p3d_h36 += mpjpe_p3d_h36.cpu().data.numpy() * batch_size
            #m_p3d_h36 is a scalar
        else: #testing
            mpjpe_p3d_h36_bis = torch.sum(torch.mean(torch.norm(p3d_h36[:, in_n:] - p3d_out, dim=3), dim=2), dim=0)
            m_p3d_h36_bis += mpjpe_p3d_h36_bis.cpu().data.numpy()
            mpjpe_p3d_h36 = torch.mean(torch.norm(p3d_h36[:, in_n:in_n + out_n] - p3d_out, dim=3)) #Xreal-Xpred on the last 10 frames (on all unused dimensions the two are equal)
            m_p3d_h36 += mpjpe_p3d_h36.cpu().data.numpy() * batch_size
        if i % 1000 == 0:
            print('{}/{}|bt {:.3f}s|tt{:.0f}s|gn{}'.format(i + 1, len(data_loader), time.time() - bt,
                                                           time.time() - st, grad_norm))
    ret = {}
    if is_train == 0:
        if n!=0:
            ret["l_p3d"] = l_p3d / n
        else:
            ret["l_p3d"] = l_p3d

    if is_train <= 1:
        if n!=0:
            ret["m_p3d_h36"] = m_p3d_h36 / n
        else:
            ret["m_p3d_h36"] = m_p3d_h36
    else:
        if n!=0:
            m_p3d_h36 = m_p3d_h36 / n
            m_p3d_h36_bis = m_p3d_h36_bis / n
        else:
            m_p3d_h36 = m_p3d_h36
            m_p3d_h36_bis = m_p3d_h36_bis
        ret["m_p3d_h36"] = m_p3d_h36
        for j in range(out_n):
            ret["#{:d}".format(titles[j])] = m_p3d_h36_bis[j]
    return ret, nb_iter


if __name__ == '__main__':
    option = Options().parse()
    main(option)