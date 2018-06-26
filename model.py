import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.autograd import Variable

import numpy as np
import math

class LayerNorm(nn.Module):

    def __init__(self, d_model, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(d_model))
        self.b_2 = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

def position_encoding(n_position, d_pos_vec):
    """
    :param n_position: position of each of sentence
    :param d_pos_vec: dimension of sentence vector
    :return: 
    """
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / d_pos_vec) for j in range(d_pos_vec)]
        if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

    position_enc[0:, 0::2] = np.sin(position_enc[0:, 0::2]) # dim 2i
    position_enc[0:, 1::2] = np.cos(position_enc[0:, 1::2]) # dim 2i+1
    return torch.from_numpy(position_enc).type(torch.FloatTensor)

class BLSTMEncoder(nn.Module):

    def __init__(self, embed_size, lstm_dim, dropout=0.0):
        super(BLSTMEncoder, self).__init__()
        self.lstm_enc = nn.LSTM(embed_size, lstm_dim, num_layers=1, bidirectional=True, dropout=dropout)

    def use_cuda(self):
        return 'cuda' in str(type(self.lstm_enc.bias_hh_l0.data))

    def forward(self, sent_tuple):
        """
        :param sent_tuple: (sent, sent_len)
        # sent: Variable (seq_len, batch_size, embed_size)
        # sent_len: numpy array (1, batch_size)
        :return: (batch_size, embed_size)
        """
        sent, sent_len = sent_tuple

        # sort by length
        sent_len, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
        idx_unsort = np.argsort(idx_sort)

        idx_sort = torch.from_numpy(idx_sort).cuda() if self.use_cuda() else torch.from_numpy(idx_sort)
        sent = sent.index_select(1, Variable(idx_sort))

        # padding
        sent_packed = pack_padded_sequence(sent, sent_len)
        sent_output = self.lstm_enc(sent_packed)[0]
        sent_output = pad_packed_sequence(sent_output)[0]

        # unsort by length
        idx_unsort = torch.from_numpy(idx_unsort).cuda() if self.use_cuda() else torch.from_numpy(idx_unsort)
        sent_output = sent_output.index_select(1, Variable(idx_unsort))

        # max pooling
        emb = torch.max(sent_output, 0)[0]
        if emb.ndimension() == 3:
            emb = emb.squeeze(0)
            assert emb.ndimension() == 2
        return emb

class BLSTMNet(nn.Module):

    def __init__(self, embed_size, lstm_dim, fc_dim, num_classes, lstm_dropout=0.1):
        super(BLSTMNet, self).__init__()
        self.encoder = BLSTMEncoder(embed_size, lstm_dim, lstm_dropout)
        self.classifier = nn.Sequential(
                nn.Linear(2*lstm_dim, fc_dim),
                nn.Linear(fc_dim, fc_dim),
                nn.Linear(fc_dim, num_classes)
                )

    def forward(self, sent_tuple):
        enc_output = self.encoder(sent_tuple)
        output = self.classifier(enc_output)
        return output

class Attention(nn.Module):

    def __init__(self, d_model, attn_dropout=0.1):
        super(Attention, self).__init__()
        self.scale = 1 / math.sqrt(d_model)
        self.dropout = nn.Dropout(attn_dropout)
        # self.layer_norm = LayerNorm(d_model)

    def forward(self, q, k, v, attn_mask=None):
        """
        :param : (batch, sent_num, num_filters)
        :return: (batch, sent_num, num_filters)
        """
        # residual = q
        # attn = torch.bmm(q, k.transpose(0, 1)) * self.scale
        attn = torch.matmul(q, k.transpose(0, 1)) * self.scale
        if attn_mask is not None:
            attn.data.masked_fill_(attn_mask, -1e10)
            # attn = attn.masked_fill(attn_mask, -1e10)

        attn = F.softmax(attn, dim=1)
        attn = self.dropout(attn)
        # output = torch.bmm(attn, v)
        output = torch.matmul(attn, v)
        return output, attn

class OneHeadAttn(nn.Module):

    def __init__(self, d_model, d_k, d_v, dropout=0.1):
        super(OneHeadAttn, self).__init__()
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Parameter(torch.FloatTensor(d_model, d_k))
        self.w_ks = nn.Parameter(torch.FloatTensor(d_model, d_k))
        self.w_vs = nn.Parameter(torch.FloatTensor(d_model, d_v))

        self.attention = Attention(d_k)
        self.layer_norm = LayerNorm(d_k)
        self.dropout = nn.Dropout(dropout)

        init.xavier_normal(self.w_qs)
        init.xavier_normal(self.w_ks)
        init.xavier_normal(self.w_vs)

    def forward(self, q, k, v, attn_mask=None):

        residual = q

        # treat the result as a (n_head * mb_size) size batch
        q_s = torch.matmul(q, self.w_qs)
        k_s = torch.matmul(k, self.w_ks)
        v_s = torch.matmul(v, self.w_vs)

        # perform attention, result size = (n_head * mb_size) x len_q x d_v
        outputs, attns = self.attention(q_s, k_s, v_s, attn_mask=attn_mask)
        outputs = self.dropout(outputs)

        return self.layer_norm(outputs + residual), attns


class BLSTMAttnNet(nn.Module):

    def __init__(self, embed_size, lstm_dim, fc_dim, num_classes,
                 max_sent_len=26, lstm_dropout=0.0,attn_dropout=0.0):
        super(BLSTMAttnNet, self).__init__()
        self.max_sent_len = max_sent_len
        self.encoder = BLSTMEncoder(embed_size, lstm_dim, lstm_dropout)
        self.d_model = 2 * lstm_dim
        # self.attention = Attention(d_model, attn_dropout)
        # positional encoding
        self.position_enc = nn.Embedding(max_sent_len, self.d_model, padding_idx=0)
        self.position_enc.weight.data = position_encoding(max_sent_len, self.d_model)
        self.attention = OneHeadAttn(self.d_model, self.d_model, self.d_model, attn_dropout)
        self.decoder = nn.Sequential(
            nn.Linear(self.d_model, fc_dim),
            nn.Linear(fc_dim, fc_dim),
            nn.Linear(fc_dim, num_classes)
        )

    def forward(self, sent_tuple):
        """
        :param sent_tuple: (sent, sent_len) in a dialogue
        :return: 
        """
        sent_enc = self.encoder(sent_tuple)
        _, sent_len = sent_tuple

        # attention mask
        len_mask = np.ones((self.max_sent_len, 1))
        len_mask[len(sent_len):] = 0
        attn_mask = np.matmul(len_mask, len_mask.transpose())
        attn_mask = torch.from_numpy(attn_mask)
        attn_mask = torch.eq(attn_mask, 0).cuda() if torch.cuda.is_available() else torch.eq(attn_mask, 0)

        # positional enc
        pos = torch.LongTensor(range(len(sent_len)))
        pos = Variable(pos).cuda() if torch.cuda.is_available() else Variable(pos)
        pos_enc = self.position_enc(pos)
        sent_enc += pos_enc

        # padding
        enc_pad = nn.ConstantPad2d((0, 0, 0, self.max_sent_len-len(sent_len)), 0)
        sent_enc = enc_pad(sent_enc)

        # sent_enc = sent_enc.data
        attn_output, attn = self.attention(sent_enc, sent_enc, sent_enc, attn_mask)
        # fully connect layer
        logit = self.decoder(attn_output)
        logit = logit[: len(sent_len)]

        return logit

class LabelSmoothing(nn.Module):

    def __init__(self, num_classes, eps=0.1):
        super(LabelSmoothing, self).__init__()
        self.eps = eps
        self.num_classes = num_classes

    def forward(self, target):
        """
        :param target: Variable LongTensor
        :return: 
        """
        smooth = np.array([1-self.eps])
        scale = Variable(torch.from_numpy(smooth))
        result = scale * target.type(torch.DoubleTensor) + self.eps / self.num_classes
        return result



