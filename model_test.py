import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
from Customized_S3DG_base import S3DG_base
from Customized_S3DG_small import S3DG_small

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout, device):
        super(MultiHeadAttention, self).__init__()

        self.d_K = d_model//n_head
        self.n_head = n_head
        self.d_model = d_model

        assert (self.n_head*self.d_K == self.d_model), "Embed size needs to be divisible by heads"
        self.WQ = nn.Linear(self.d_model, self.d_model, bias=True)
        self.WK = nn.Linear(self.d_model, self.d_model, bias=True)
        self.WV = nn.Linear(self.d_model, self.d_model, bias=True)
        # The W0
        self.fc_out  = nn.Linear(self.d_K*self.n_head, d_model)
        self.dropout = nn.Dropout(p = dropout)
        self.scale   = torch.sqrt(torch.FloatTensor([self.d_K])).to(device)

    def forward(self, Q_ipt, K_ipt, V_ipt, Mask = None):
        BS = Q_ipt.shape[0]
        l_Q, l_K, l_V = Q_ipt.shape[1], K_ipt.shape[1], V_ipt.shape[1]
        Q = self.WQ(Q_ipt)
        K = self.WK(K_ipt)
        V = self.WV(V_ipt) # (BS, seq_len, d_model)

        # Split inputs to n_heads
        Queries = Q.reshape(BS, l_Q, self.n_head, self.d_K).permute(0, 2, 1, 3)
        Keys    = K.reshape(BS, l_K, self.n_head, self.d_K).permute(0, 2, 1, 3)
        Values  = V.reshape(BS, l_V, self.n_head, self.d_K).permute(0, 2, 1, 3)
        #dim = [BS, n_head, seq_len, d_k]

        # Input #
        e = torch.matmul(Queries, Keys.permute(0,1,3,2)) / self.scale

        if Mask is not None :
            e = e.masked_fill(Mask == 0, float("-1e20"))    # replace b in a, (a, b)
        alpha = torch.softmax(e, dim= -1)
        #dim = [BS, n_head, l_Q, l_K]
        out = torch.matmul(alpha, Values)
        #dim = [BS, n_head, l_Q, d_K]

        out = out.permute(0,2,1,3).contiguous()
        #dim = [BS, l_Q, n_head, d_K]
        out = out.view(BS, -1, self.d_model)
        out = self.dropout(self.fc_out(out))
        #dim = [BS, l_Q, d_model]

        return out

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, dropout, device, forward_expansion = 2):
        super(TransformerBlock, self).__init__()

        self.MHA = MultiHeadAttention(d_model, n_head, dropout=dropout, device = device)
        self.Lnorm1 = nn.LayerNorm(d_model)     # Layernorm : example별 normalization
                                                # Expansion : 일반 convnet의 filter 증가 - 감소와 비슷한 효과!
        self.FFNN   = nn.Sequential(
            nn.Linear(d_model, forward_expansion*d_model),
            nn.GELU(),
            nn.Dropout(p = dropout),
            nn.Linear(forward_expansion*d_model, d_model)
        )
        self.dropout = nn.Dropout(p=dropout)
        self.Lnorm2 = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, Mask):
        attention = self.MHA(Q, K, V, Mask)
        X = self.Lnorm1(self.dropout(attention) + Q) # Since the Q == input from the Emb layer!
        forward = self.FFNN(X)
        out = self.Lnorm2(self.dropout(forward) + X)

        return out

class Encoder(nn.Module):
    def __init__(self,
                 gloss_vocab_size,
                 embed_size,                        # d_model!
                 n_layers,
                 n_heads,
                 device,
                 forward_expansion,
                 dropout,
                 max_len
                 ):
        super(Encoder, self).__init__()

        self.embed_size = embed_size
        self.device     = device
        #self.WE         = nn.Linear(1024, embed_size)
        self.PE         = nn.Embedding(max_len, embed_size)
        #self.pool       = nn.AdaptiveMaxPool2d((55, embed_size))
        self.fc_out     = nn.Linear(embed_size, gloss_vocab_size) # For gloss
        #self.softmax    = nn.Softmax(dim = -1)

        self.Layers     = nn.ModuleList(
            [
                TransformerBlock(embed_size, n_heads, dropout=dropout,
                                 device=device, forward_expansion=forward_expansion)
                for i in range(n_layers)
            ]
        )
        self.dropout    = nn.Dropout(dropout)
        self.scale      = torch.sqrt(torch.FloatTensor([embed_size])).to(device)

    def forward(self, X, Mask):
        BS, seq_len, emb_dim = X.shape

        Position = torch.arange(0,seq_len).expand(BS, seq_len).to(self.device) # expand : https://seducinghyeok.tistory.com/9

        out = self.dropout(X * self.scale + self.PE(Position))
        #print("after encoder enc : ", out.shape, Mask.shape)

        for layer in (self.Layers):
            out = layer(out, out, out, Mask)            # Since we're doing self MHA in Encoder

        # gloss output
        #pool_out      = self.pool(out)
        predict_gloss = self.fc_out(self.dropout(out))
        #predict_gloss = self.softmax(predict_gloss)

        return out, predict_gloss



class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_head, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.MHA  = MultiHeadAttention(d_model,n_head, dropout=dropout,device=device)
        self.norm = nn.LayerNorm(d_model)
        self.transformer_block = TransformerBlock(d_model, n_head,dropout=dropout,
                                device=device, forward_expansion=forward_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, K, V, source_mask, target_mask):
        '''
        :param X: GT input @ training phase
        :param K: From Encoder features
        :param V: From Encoder features
        :param source_mask: Padding된 것끼리 계산 안하도록
        :param target_mask: LHA mask in Enc-Dec attention
        :return:
        '''
        #print(X.shape, target_mask.shape)
        decoder_attn = self.MHA(X, X, X, target_mask)   # Input : Target, self attention 단계
        Q            = self.norm(self.dropout(decoder_attn) + X)
        out          = self.transformer_block(Q, K, V, source_mask)

        return out

class Decoder(nn.Module):
    def __init__(self,
                 target_vocab_size,
                 gloss_vocab_size,
                 embed_size,
                 n_layers,
                 n_heads,
                 device,
                 forward_expansion,
                 dropout,
                 max_len):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.WE = nn.Embedding(target_vocab_size, embed_size)
        self.PE = nn.Embedding(max_len, embed_size)
        self.Layers = nn.ModuleList([
            DecoderBlock(embed_size, n_heads, forward_expansion, dropout, device)
            for i in range(n_layers)
        ])
        #self.gloss_decoder =\
        #DecoderBlock(embed_size, n_heads, forward_expansion, dropout, device)
        self.fc_out  = nn.Linear(embed_size, target_vocab_size)
        #self.fc_gloss = nn.Linear(embed_size, gloss_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.scale   = torch.sqrt(torch.FloatTensor([embed_size])).to(device)
        #self.softmax = nn.Softmax(dim = -1)

    def forward(self, X, enc_out, source_mask, target_mask):
        BS, seq_len = X.shape

        Position = torch.arange(0,seq_len).expand(BS, seq_len).to(self.device)
        out = self.dropout((self.WE(X) * self.scale + self.PE(Position)))

        #gloss_attn = self.gloss_decoder(out, enc_out, enc_out, source_mask, target_mask)
        #predict_gloss = self.fc_gloss(gloss_attn)

        #print("after decoder enc : ", out.shape)
        for layer in self.Layers:
            out = layer(out, enc_out, enc_out, source_mask, target_mask)    # Q : Decoder // K,V : Encoder

        out = self.fc_out(out)
        #out = self.softmax(out)

        return out



class SLT_Transformer(nn.Module):
    def __init__(self,
                 gloss_vocab_size,
                 target_vocab_size,
                 source_pad_idx,
                 target_pad_idx,
                 embed_size = 512,
                 n_layers = 2,
                 forward_expansion = 4,
                 n_heads = 8,
                 dropout = 0.2,
                 device = "cuda",
                 max_len_enc =112,
                 max_len_dec =55
                 ):
        super(SLT_Transformer, self).__init__()

        self.S3D     = S3DG_small(embed_size, device, dropout = 0.25)
        self.Encoder = Encoder(gloss_vocab_size, embed_size,
                               n_layers, n_heads, device, forward_expansion,
                               dropout, max_len_enc)
        self.Decoder = Decoder(target_vocab_size, gloss_vocab_size, embed_size,
                               n_layers, n_heads, device, forward_expansion,
                               dropout, max_len_dec)
        self.source_pad_idx = source_pad_idx
        self.target_pad_idx = target_pad_idx
        self.device = device

    def make_source_mask(self, source):
        # source : [BS, T, C, H, W]
        source_mask = (source[:,:,0] != self.source_pad_idx).unsqueeze(1).unsqueeze(2)
        #print("source shape : ", source.shape)
        #print("mask shape : ", source_mask.shape)
        # (BS, 1, 1, source_length)
        return source_mask.to(self.device)

    def make_target_mask(self, target):
        BS, target_len = target.shape
        #print("target shape : ", target.shape)
        target_mask = torch.tril(torch.ones((target_len, target_len))).expand(
            BS, 1, target_len, target_len)
        return target_mask.to(self.device)

    def forward(self, source, target):

        S3D_feature = self.S3D(source)
        source_mask = self.make_source_mask(S3D_feature)
        target_mask = self.make_target_mask(target)

        # Outputs : glosses and translations
        enc_feature, predict_gloss\
                            = self.Encoder(S3D_feature, source_mask)
        predict_translation = self.Decoder(target, enc_feature, source_mask, target_mask)
        #print(target_mask)

        return predict_translation, predict_gloss

'''
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # 1 : sos / 0 : pad / 2 : eos
    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(device)
    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)

    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10
    model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, device=device).to(
        device
    )
    out = model(x, trg[:, :-1])
    print(out.shape)'''

if __name__ == "__main__":
    model = SLT_Transformer(gloss_vocab_size=1024, target_vocab_size=2048,
                            source_pad_idx=0, target_pad_idx=0, embed_size=512)
    src_mask = model.make_source_mask(torch.Tensor([1,33,52,80,13,43,26,767,9,0,0,0]))
    print(src_mask)