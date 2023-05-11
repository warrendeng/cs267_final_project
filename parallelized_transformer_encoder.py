import pdb
from torch import nn
import torch
import torch.nn.functional as F
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=256, n_head=4, d_qkv=32, dropout=0.1, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_qkv = d_qkv

        self.w_q = nn.Parameter(torch.Tensor(n_head, d_model, d_qkv))
        self.w_k = nn.Parameter(torch.Tensor(n_head, d_model, d_qkv))
        self.w_v = nn.Parameter(torch.Tensor(n_head, d_model, d_qkv))
        self.w_o = nn.Parameter(torch.Tensor(n_head, d_qkv, d_model))
        nn.init.xavier_normal_(self.w_q)
        nn.init.xavier_normal_(self.w_k)
        nn.init.xavier_normal_(self.w_v)
        nn.init.xavier_normal_(self.w_o)


        self.linear = nn.Linear(n_head * d_qkv, d_model)
        self.dropout = dropout
        self.layer_norm = nn.LayerNorm(d_model)
  
    def forward(self, x):

        residual = x

        q = torch.einsum("hmq,blm->bhlq",self.w_q,x)
        k = torch.einsum("hmq,blm->bhlq",self.w_k,x)
        v = torch.einsum("hmq,blm->bhlq",self.w_v,x)

        # print(q.shape, k.shape)
        s = torch.einsum("bhiq, bhjq->bhij", q, k)/(self.d_qkv**0.5)


        s = F.softmax(s, dim=-1) # b,h,l,l
        s = F.dropout(s, self.dropout)
        # print(s.shape)

        attention = torch.einsum("bhij,bhjq->bhiq",s,v)
        attention = torch.einsum("bhlq, hqm->blm", attention, self.w_o)

        output =  F.dropout(attention, self.dropout)

        return self.layer_norm(residual+output)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, d_context, dropout=0.1):
        super().__init__()

        self.linear1 = nn.Linear(d_model + d_context * 2, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x):

        residual = x
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        x = self.dropout(x)
        x = self.layer_norm(x+residual)

        return x

class AddPositionalEncoding(nn.Module):
    def __init__(self, d_model=256, input_dropout=0.1, timing_dropout=0.1,
                max_len=512):
        super().__init__()
        self.timing_table = nn.Parameter(torch.FloatTensor(max_len, d_model))
        nn.init.normal_(self.timing_table)
        self.input_dropout = nn.Dropout(input_dropout)
        self.timing_dropout = nn.Dropout(timing_dropout)
    
    def forward(self, x):

        x = self.input_dropout(x)
        timing = self.timing_table[None, :x.shape[1], :]
        timing = self.timing_dropout(timing)
        return x + timing


from torch.nn.modules.activation import MultiheadAttention
class CustomTransformerEncoder(nn.Module):
    def __init__(self, d_model=256, d_ff=1024, d_context = 32, n_layers=4, n_head=4, d_qkv=32,
                dropout=0.1):
        super().__init__()


        self.main_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.main_layers.append(MultiHeadAttention(d_model=d_model+2*d_context, n_head=n_head, d_qkv=d_qkv, dropout=dropout))
            self.main_layers.append(PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, d_context=d_context, dropout=dropout))
        
        self.context_layers_1 = nn.ModuleList()
        for _ in range(n_layers):
            self.context_layers_1.append(MultiHeadAttention(d_model=d_model, n_head=n_head, d_qkv=d_qkv, dropout=dropout))
            self.context_layers_1.append(PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, d_context=0, dropout=dropout))
            self.context_layers_1.append(nn.Linear(d_model, d_context))

        self.context_layers_2 = nn.ModuleList()
        for _ in range(n_layers):
            self.context_layers_2.append(MultiHeadAttention(d_model=d_model, n_head=n_head, d_qkv=d_qkv, dropout=dropout))
            self.context_layers_2.append(PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, d_context=0, dropout=dropout))
            self.context_layers_2.append(nn.Linear(d_model, d_context))
        
        self.n_layers = n_layers

    def forward(self, c1, c2, m): # three chunks: left context, right context, main
        

        c1 = self.context_layers_1[0](c1)
        c1 = self.context_layers_1[1](c1)
        c1_m = self.context_layers_1[2](c1)  # compressed left context for main chunk

        c2 = self.context_layers_2[0](c2)
        c2 = self.context_layers_2[1](c2)
        c2_m = self.context_layers_2[2](c2)  # compressed right context for main chunk

        for i in range(1, self.n_layers):
            m = self.main_layers[i*2-2](torch.cat((c1_m, m, c2_m), 1))
            m = self.main_layers[i*2-1](m)

            c1 = self.context_layers_1[i*3](c1)
            c1 = self.context_layers_1[i*3+1](c1)
            c1_m = self.context_layers_1[i*3+2](c1)

            c2 = self.context_layers_2[i*3](c2)
            c2 = self.context_layers_2[i*3+1](c2)
            c2_m = self.context_layers_2[i*3+2](c2) 

        m = self.main_layers[-2](torch.cat((c1_m, m, c2_m), 1))
        m = self.main_layers[-1](m)
        return m

class ContextEncoder(nn.Module):
    def __init__(self, d_model=256, d_ff=1024, d_context = 32, n_layers=4, n_head=4, d_qkv=32,
                dropout=0.1, device='cuda:0'):
        super().__init__()

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(MultiHeadAttention(d_model=d_model, n_head=n_head, d_qkv=d_qkv, dropout=dropout).to(device))
            self.layers.append(PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, d_context=0, dropout=dropout).to(device))
            self.layers.append(nn.Linear(d_model, d_context).to(device))

    def forward(self, x):
        outputs = []
        for i in range(self.n_layers):
            x = self.layers[i*3](x)
            x = self.layers[i*3+1](x)
            x = self.layers[i*3+2](x)
            outputs.append(x)
        return outputs


class MainEncoder(nn.Module):
    def __init__(self, d_model=256, d_ff=1024, d_context = 32, n_layers=4, n_head=4, d_qkv=32,
                dropout=0.1, device='cuda:1'):
        super().__init__()

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(MultiHeadAttention(d_model=d_model+2*d_context, n_head=n_head, d_qkv=d_qkv, dropout=dropout).to(device))
            self.layers.append(PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, d_context=d_context, dropout=dropout).to(device))

    def forward(self, x, c1s, c2s):
        for i in range(self.n_layers):
            x = self.layers[i*2](torch.cat((c1s[i], x, c2s[i]), 1))
            x = self.layers[i*2+1](x)
        return x


class TransformerPOSTaggingModel(POSTaggingModel):
    def __init__(self):
        super().__init__()
        d_model = 256
        d_context = 32
        self.add_timing = AddPositionalEncoding(d_model)
        self.context_encoder1 = ContextEncoder(device='cuda:0')
        self.context_encoder2 = ContextEncoder(device='cuda:1')
        self.main_encoder = MainEncoder(device='cuda:2')
        self.context_embedding1 = nn.Embedding(30522, d_model).to('cuda:0')
        self.context_embedding2 = nn.Embedding(30522, d_model).to('cuda:1')
        self.main_embedding1 = nn.Embedding(30522, d_model).to('cuda:2')
        self.layer_norm = nn.LayerNorm(d_model).to('cuda:2')
        self.linear = nn.Linear(d_model, 1).to('cuda:2')


    def encode(self, batch_c1, batch_c2, batch_m):
        c1 = self.context_embedding1(batch_c1)
        c1 = self.add_timing(c1)

        c2 = self.context_embedding2(batch_c2)
        c2 = self.add_timing(c2)

        m = self.main_embedding1(batch_m)
        m = self.add_timing(m)

        # Parallelize context computations
        c1s = self.context_encoder1(c1)  
        c2s = self.context_encoder2(c2)  

        # Move context outputs to GPU 2
        c1s = [x.to('cuda:2') for x in c1s]
        c2s = [x.to('cuda:2') for x in c2s]

        m = self.main_encoder(m, c1s, c2s)

        x = self.layer_norm(m)
        x = self.linear(x)
        return x
