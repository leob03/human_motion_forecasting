from torch.nn import Module
from torch import nn
import torch
# import model.transformer_base
import math
from model import GCN
import utils.util as util
import numpy as np

class LSTMMemoryModule(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(LSTMMemoryModule, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)

    def forward(self, input, hidden=None):
        # Reshape the input to have sequence length of 1 and input_size of (N*66, 1, 10)

        # Pass the input through the LSTM layer
        lstm_output, hidden = self.lstm(input, hidden)
        # lstm_output.shape = [32*66, 1, 50]

        return lstm_output, hidden

class RNNAttModel(Module):

    def __init__(self, in_features=48, kernel_size=5, d_model=512, num_stage=2, dct_n=10):
        super(RNNAttModel, self).__init__()

        self.kernel_size = kernel_size
        self.d_model = d_model
        # self.seq_in = seq_in
        self.dct_n = dct_n
        # ks = int((kernel_size + 1) / 2)
        assert kernel_size == 10

        self.convQ = nn.Sequential(nn.Conv1d(in_channels=in_features, out_channels=d_model, kernel_size=6,
                                             bias=False),
                                   nn.ReLU(),
                                   nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=5,
                                             bias=False),
                                   nn.ReLU())

        self.convK = nn.Sequential(nn.Conv1d(in_channels=in_features, out_channels=d_model, kernel_size=6,
                                             bias=False),
                                   nn.ReLU(),
                                   nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=5,
                                             bias=False),
                                   nn.ReLU())

        self.gcn = GCN.GCN(input_feature=(dct_n) * 2, hidden_feature=d_model, p_dropout=0.3,
                           num_stage=num_stage,
                           node_n=in_features)
        
        input_size = 10
        hidden_size = 50

        self.lstm_module = LSTMMemoryModule(input_size, hidden_size)
        self.hidden_state = None

    def forward(self, src, output_n=25, input_n=10, itera=1):
        """

        :param src: [batch_size,seq_len,feat_dim]
        :param output_n:
        :param input_n:
        :param frame_n:
        :param dct_n:
        :param itera:
        :return:
        """
        dct_n = self.dct_n #nb of dct components
        #src.shape = [32,?,66] where ? is the number of frames and 48 or 66 is the number of joints which is 16*3 or 22*3
        src = src[:, :input_n]  # [bs,in_n=10,dim]
        bs = src.shape[0] #batch_size

        src_tmp = src.clone()
        #src_tmp.shape = [32,N,66] = [32,10,66]
        src_tmp = src_tmp.reshape([src_tmp.shape[0]*src_tmp.shape[2],1, src_tmp.shape[1]])
        #src_tmp.shape = [32*66,1,10]
        output_lstm, self.hidden_state = self.lstm_module(src_tmp, self.hidden_state)
        src_tmp = output_lstm.reshape([bs, 50, -1]).detach()
        #src_tmp.shape = [32,N,66] = [32,50,66]
        
        src_key_tmp = src_tmp.transpose(1, 2)[:, :, :(50 - output_n)].clone()
        #src_key_tmp.shape = [32,66,N-T]

        src_query_tmp = src_tmp.transpose(1, 2)[:, :, -self.kernel_size:].clone()
        #src_query_tmp.shape = [32,66,M]

        dct_m, idct_m = util.get_dct_matrix(self.kernel_size + output_n)
        dct_m = torch.from_numpy(dct_m).float().cuda()
        #dct_m.shape = [M + T,M + T]
        idct_m = torch.from_numpy(idct_m).float().cuda()

        vn = 50 - self.kernel_size - output_n + 1 #= N - M - T + 1 = 31, number of subsequences
        vl = self.kernel_size + output_n #= M + T = 20, length of each subsequence

        idx = np.expand_dims(np.arange(vl), axis=0) + \
              np.expand_dims(np.arange(vn), axis=1)
        src_value_tmp = src_tmp[:, idx].clone().reshape(
            [bs * vn, vl, -1])
        #src_tmp[:, idx].shape = [32, 31, 20, 66]
        #src_value_tmp.shape = [32*31, 20, 66] = [992, 20, 66]
        src_value_tmp = torch.matmul(dct_m[:dct_n].unsqueeze(dim=0), src_value_tmp).reshape(
            [bs, vn, dct_n, -1]).transpose(2, 3).reshape(
            [bs, vn, -1])
        #src_value_tmp.shape = [32, vn, 66*20] = [32, 31, 1320]
        #the Values after being turned into dct components

        idx = list(range(-self.kernel_size, 0, 1)) + [-1] * output_n 
        #idx is a list of 10 elements, each element is a number from -10 to -1 and then 10 times -1

        outputs = []

        key_tmp = self.convK(src_key_tmp / 1000.0)
        #key_tmp.shape = [32,512 or 256,31] 31 = N - M - T + 1 = 31, number of keys and values
        for i in range(itera):
            query_tmp = self.convQ(src_query_tmp / 1000.0)
            #query_tmp.shape = [32,512,1]
            score_tmp = torch.matmul(query_tmp.transpose(1, 2), key_tmp) + 1e-15
            #score_tmp.shape = [32,1,31]
            att_tmp = score_tmp / (torch.sum(score_tmp, dim=2, keepdim=True))
            #att_tmp.shape = [32,1,31]
            dct_att_tmp = torch.matmul(att_tmp, src_value_tmp)[:, 0].reshape(
                [bs, -1, dct_n])
            #torch.matmul(att_tmp, src_value_tmp).shape = [32, 1, 1320]
            #dct_att_tmp.shape = [32,?,dct_n] = [32,66,20]
            input_gcn = src_tmp[:, idx] #the padded sequence of input frames
            #idx = [-T, ..., -4, -3, -2, -1, ..., -1]
            #input_gcn.shape = [32, M+T, 66] = [32, 20, 66]
            dct_in_tmp = torch.matmul(dct_m[:dct_n].unsqueeze(dim=0), input_gcn).transpose(1, 2) #convert input to dct components
            #dct_in_tmp.shape = [32,66,20]
            dct_in_tmp = torch.cat([dct_in_tmp, dct_att_tmp], dim=-1) #concatenate the input and the values weighted by attention
            #dct_in_tmp.shape = [32,66,40]
            dct_out_tmp = self.gcn(dct_in_tmp)
            #dct_out_tmp.shape = [32,66,40]
            out_gcn = torch.matmul(idct_m[:, :dct_n].unsqueeze(dim=0),
                                   dct_out_tmp[:, :, :dct_n].transpose(1, 2)) #convert back to 3D coordinates (from frequency domain to space domain)
            #out_gcn.shape = [32,20,66]
            outputs.append(out_gcn.unsqueeze(2))
            if itera > 1:
                # update key-value query
                out_tmp = out_gcn.clone()[:, 0 - output_n:] #the ten predicted frames 
                #out_tmp.shape = [32,T,66] = [32,10,66]
                src_tmp = torch.cat([src_tmp, out_tmp], dim=1) #the whole sequence is updated with the predicted frames
                #src_tmp.shape = [32,N + T,66] = [32,60,66]
                vn = 1 - 2 * self.kernel_size - output_n #= 1 - 2*M - T = 1 - 2*10 - 10 = -29
                #I think it should be:
                #vn = 1 - self.kernel_size - output_n
                vl = self.kernel_size + output_n #= M + T = 20
                idx_dct = np.expand_dims(np.arange(vl), axis=0) + \
                          np.expand_dims(np.arange(vn, -self.kernel_size - output_n + 1), axis=1)
                #idx_dct = np.expand_dims(np.arange(vl), axis=0) + \
                  #         np.expand_dims(np.arange(vn, - output_n + 1), axis=1)
                #idx_dct.shape = [10,20]
                src_key_tmp = src_tmp[:, idx_dct[0, :-1]].transpose(1, 2)
                #src_key_tmp.shape = [32, 66, 19] 19 because after convK 19 -> 10 and the new N is N+10 so we want ten new keys (altough I don't agree with the way he chose the keys) 
                key_new = self.convK(src_key_tmp / 1000.0)
                #key_tmp.shape = [32, 256, 31+(itera-1)*10]
                key_tmp = torch.cat([key_tmp, key_new], dim=2)
                #key_tmp.shape = [32, 256, 31+(itera)*10] these are the keys for the next iteration

                src_dct_tmp = src_tmp[:, idx_dct].clone().reshape(
                    [bs * self.kernel_size, vl, -1])
                src_dct_tmp = torch.matmul(dct_m[:dct_n].unsqueeze(dim=0), src_dct_tmp).reshape(
                    [bs, self.kernel_size, dct_n, -1]).transpose(2, 3).reshape(
                    [bs, self.kernel_size, -1])
                src_value_tmp = torch.cat([src_value_tmp, src_dct_tmp], dim=1)
                #these are the values for the next iteration

                src_query_tmp = src_tmp[:, -self.kernel_size:].transpose(1, 2)
                #this is the query for the next iteration, just the ten last frames

        outputs = torch.cat(outputs, dim=2)
        #outputs.shape = [32,20,itera=3,66]
        return outputs, self.hidden_state