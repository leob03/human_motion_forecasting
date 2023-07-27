from torch.nn import Module
from torch import nn
import torch
# import model.transformer_base
import math
from model import GCN
import utils.util as util
import numpy as np
import torch.nn.functional as F

# torch.autograd.set_detect_anomaly(True)

#parameters

# input = 20
# input_attention = 40
# output_attention = 20
# key_frames = 10
# values_frames = 20
# query_frames = 10


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

    def forward(self, src, memory, output_n=10, input_n=50, itera=1):
        """

        :param src: [batch_size,seq_len,feat_dim]
        :param output_n:
        :param input_n:
        :param frame_n:
        :param dct_n:
        :param itera:
        :return:
        """
        dct_n = self.dct_n #nb of dct components, here ? is the number of frames and 48 or 66 is the number of joints which is 16*3 or 22*3
        #src.shape = [32,?,66] 
        src = src[:, :input_n]  # [bs,in_n=20,dim]
        input_sequence = src
        #input_sequence.shape = [32,N,66] = [32,in_n=20,66]
        bs = src.shape[0] #batch_size

        dct_m, idct_m = util.get_dct_matrix(self.kernel_size + output_n)
        dct_m = torch.from_numpy(dct_m).float().cuda()
        idct_m = torch.from_numpy(idct_m).float().cuda()
        #dct_m.shape = [M + T,M + T] = [20,20]

        #memory.shape = [32,20,66]
        past_memory1 = memory[:bs,:dct_n,:].clone()
        # past_memory2 = memory[:,dct_n:,:]
        #past_memory1.shape = [32,20,66]

        past_memory_idct = torch.matmul(idct_m[:, :dct_n].unsqueeze(dim=0),
                                        past_memory1)
        #past_memory_idct.shape = [32,20,66]

        # past_memory2_idct = torch.matmul(idct_m[:, :dct_n].unsqueeze(dim=0),
        #                                 past_memory2[:, :, :dct_n].transpose(1, 2))
        # past_memory = torch.cat([past_memory1_idct, past_memory2_idct], dim=2)
        #past_memory.shape = [32,66,40]

        attention_sequence = torch.cat([past_memory_idct, input_sequence], dim=1)        #attention_sequence.shape = [32,40,66]

        input_keys = attention_sequence.transpose(1, 2)
        #input_keys.shape = [32,66,50-T=40]

        concat_sequence = torch.cat([attention_sequence, input_sequence[:,10:,:]], dim=1)
        #concat_sequence.shape = [32,50,66]

        input_query = concat_sequence.transpose(1,2)[:, :, -self.kernel_size:]
        #input_query.shape = [32,66,M=10]

        vn = 50 - self.kernel_size - output_n + 1 #= N - T - M + 1 = 31, number of subsequences
        vl = self.kernel_size + output_n #= M + T = 20, length of each subsequence

        idx = np.expand_dims(np.arange(vl), axis=0) + \
              np.expand_dims(np.arange(vn), axis=1)

        values = concat_sequence[:, idx].reshape(
            [bs * vn, vl, -1])
        #concat_sequence[:, idx].shape = [32, 31, 20, 66]   #31 is the number of subsequences, 20 is the length of each subsequence
        #values.shape = [32*31, 20, 66] = [992, 20, 66]
        values_dct = torch.matmul(dct_m[:dct_n].clone().unsqueeze(dim=0), values).reshape(
            [bs, vn, dct_n, -1]).transpose(2, 3).reshape(
            [bs, vn, -1])
        #values_dct.shape = [32, vn, 66*20] = [32, 31, 1320]
        #the Values after being turned into dct components

        idx = list(range(-self.kernel_size, 0, 1)) + [-1] * output_n 
        #idx is a list of 10 elements, each element is a number from -10 to -1 and then 10 times -1

        outputs = []

        keys = self.convK(input_keys / 1000.0).clone()
        #keys.shape = [32, 256, 31] 31 = N - M - T + 1 = 31, number of keys and values
        for i in range(itera):
            scaled_input_query = input_query / 1000.0
            query = self.convQ(scaled_input_query)
            #query.shape = [32,256,1]
            alignment_scores = torch.matmul(query.transpose(1, 2), keys) + 1e-15
            #alignment_scores.shape = [32,1,31]
            att_weights = alignment_scores / (torch.sum(alignment_scores, dim=2, keepdim=True))
            #att_weights.shape = [32,1,31]
            dct_att_output = torch.matmul(att_weights, values_dct)[:, 0].reshape(
                [bs, -1, dct_n])
            #torch.matmul(att_weights, values_dct).shape = [32, 1, 1320]
            #dct_att_output.shape = [32,66,20]
            if itera < 2 and memory.shape[0]==32:
                memory = dct_att_output.clone().transpose(1, 2).detach()
                #memory.shape = [32,20,66]
            input_gcn = input_sequence[:, idx] #the padded sequence of input frames
            #idx = [-T, ..., -4, -3, -2, -1, ..., -1]
            #input_gcn.shape = [32, M+T, 66] = [32, 20, 66]
            dct_input_gcn = torch.matmul(dct_m[:dct_n].unsqueeze(dim=0), input_gcn).transpose(1, 2) #convert input to dct components
            #dct_input_gcn.shape = [32,66,20]
            dct_input_gcn_concat = torch.cat([dct_input_gcn, dct_att_output], dim=-1) #concatenate the input and the values weighted by attention
            #dct_input_gcn_concat.shape = [32,66,40]
            dct_out_gcn = self.gcn(dct_input_gcn_concat)
            #dct_out_gcn.shape = [32,66,40]
            out_gcn = torch.matmul(idct_m[:, :dct_n].unsqueeze(dim=0),
                                   dct_out_gcn[:, :, :dct_n].transpose(1, 2)) #convert back to 3D coordinates (from frequency domain to space domain)
            #out_gcn.shape = [32,20,66]
            outputs.append(out_gcn.unsqueeze(2))
            if itera > 1:
                # update key-value query
                pred = out_gcn.clone()[:, 0 - output_n:] #the ten predicted frames 
                #pred.shape = [32,T,66] = [32,10,66]
                attention_sequence = torch.cat([attention_sequence, pred], dim=1) #the whole sequence is updated with the predicted frames
                #attention_sequence.shape = [32,N + T,66] = [32,60,66]

                vn = 1 - 2 * self.kernel_size - output_n #= 1 - 2*M - T = 1 - 2*10 - 10 = -29
                #I think it should be:
                #vn = 1 - self.kernel_size - output_n
                vl = self.kernel_size + output_n #= M + T = 20

                idx_dct = np.expand_dims(np.arange(vl), axis=0) + \
                          np.expand_dims(np.arange(vn, -self.kernel_size - output_n + 1), axis=1)
                #idx_dct = np.expand_dims(np.arange(vl), axis=0) + \
                  #         np.expand_dims(np.arange(vn, - output_n + 1), axis=1)
                #idx_dct.shape = [10,20]

                new_input_keys = attention_sequence[:, idx_dct[0, :-1]].transpose(1, 2)
                #new_input_keys.shape = [32, 66, 19] 19 because after convK 19 -> 10 and the new N is N+10 so we want ten new keys (altough I don't agree with the way he chose the keys) 
                
                new_keys = self.convK(new_input_keys / 1000.0).clone()
                #new_keys.shape = [32, 256, 10]
                #keys.shape = [32, 256, 31+(itera-1)*10]
                keys = torch.cat([keys, new_keys], dim=2)
                #keys.shape = [32, 256, 31+(itera)*10] these are the keys for the next iteration

                new_values = attention_sequence[:, idx_dct].clone().reshape(
                    [bs * self.kernel_size, vl, -1])
                new_values_dct = torch.matmul(dct_m[:dct_n].unsqueeze(dim=0), new_values).reshape(
                    [bs, self.kernel_size, dct_n, -1]).transpose(2, 3).reshape(
                    [bs, self.kernel_size, -1])
                values_dct = torch.cat([values_dct, new_values_dct], dim=1)
                #these are the values for the next iteration

                input_query = attention_sequence[:, -self.kernel_size:].transpose(1, 2)
                #this is the query for the next iteration, just the ten last frames

        outputs = torch.cat(outputs, dim=2)
        #outputs.shape = [32,20,itera=3,66]
        return outputs, memory