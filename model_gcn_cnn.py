import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch.nn.modules.transformer import _get_clones, _get_activation_fn
from torch_geometric.nn import GINConv, global_add_pool, GCNConv
import numpy as np
from math import sqrt



d_model=120
dim_feedforward = 512
n_heads = 4
vocab_size=26
n_layers=4


class TransformerEncoder(nn.Module):
    __constants__ = ['norm']
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
    def forward(self, src):
        output = src
        for mod in self.layers:
            output,attn = mod(output)
        if self.norm is not None:
            output = self.norm(output)
        return output,attn

class TransformerEncoderLayer(nn.Module):
    __constants__ = ['batch_first']
    def __init__(self, d_model, nhead, dim_feedforward=dim_feedforward, dropout=0.1, activation="relu",
                 layer_norm_eps=1e-5, batch_first=True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model,nhead,dropout=dropout,batch_first=batch_first)

        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src):
        src2,attn = self.self_attn(src, src, src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src,attn[:,0,:]

def get_sinusoid_encoding_table(n_position, d_model):
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_model)
    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_model)]
    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.FloatTensor(sinusoid_table)


class EnhancedPositional(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.static = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(vocab_size, d_model), freeze=True)
        self.dynamic = nn.Embedding(vocab_size, d_model)  # Learned positions
        
    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return self.static(positions) + self.dynamic(positions)
    
class AttentionPool(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.query = nn.Parameter(torch.randn(d_model))
        self.scale = 1 / sqrt(d_model)
        
    def forward(self, x):
        # x: [Batch, Seq, Dim]
        attn_weights = torch.matmul(x, self.query) * self.scale
        attn_weights = F.softmax(attn_weights, dim=1)
        return torch.sum(attn_weights.unsqueeze(-1) * x, dim=1)
    

class DilatedCNN(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.conv1 = nn.Conv1d(d_model, d_model, 3, dilation=1, padding=1)
        self.conv2 = nn.Conv1d(d_model, d_model, 3, dilation=2, padding=2)

        self.adaptive_pool = nn.AdaptiveMaxPool1d(1)
        
    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        return self.adaptive_pool(x).squeeze(-1)

class CrossModalAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.proj_q = nn.Linear(d_model, d_model)
        self.proj_kv = nn.Linear(d_model*2, d_model*2)
        
    def forward(self, protein, ligand):
        # protein: [B, S, D], ligand: [B, D]
        q = self.proj_q(protein)
        ligand_expanded = ligand.unsqueeze(1).expand(-1, protein.size(1), -1)
        kv = self.proj_kv(torch.cat([protein, ligand_expanded], -1))
        k, v = kv.chunk(2, dim=-1)
        
        attn = F.scaled_dot_product_attention(q, k, v)
        return attn.mean(1)

class DeepTTG(torch.nn.Module):
    def __init__(self, n_output=1,MLP_dim=96, dropout=0.1,
                 c_feature=108,vocab_size=vocab_size,d_model =d_model,n_heads = n_heads,n_layers=n_layers,gcn_hidden=64):
        super(DeepTTG, self).__init__()
        # TransformerEncoder for extracting protein features
        self.encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=n_heads)
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=n_layers)

        self.src_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(vocab_size, d_model),freeze=True)
        # self.pos_emb = EnhancedPositional(d_model)
        # self.domain_emb = nn.Embedding(3, d_model)

        self.attention_pool = AttentionPool(d_model)
        self.cnn_protein = DilatedCNN(d_model)
        self.cross_attn = CrossModalAttention(d_model)

        # self.dropout = nn.Dropout(dropout)
        # self.relu = nn.ReLU()
        self.n_output = n_output
        # GIN model for extracting compound features
        nn1 = Sequential(Linear(c_feature, MLP_dim), ReLU(), Linear(MLP_dim, MLP_dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(MLP_dim)
        nn2 = Sequential(Linear(MLP_dim, MLP_dim), ReLU(), Linear(MLP_dim, MLP_dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(MLP_dim)
        nn3 = Sequential(Linear(MLP_dim, MLP_dim), ReLU(), Linear(MLP_dim, MLP_dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(MLP_dim)
        nn4 = Sequential(Linear(MLP_dim, MLP_dim), ReLU(), Linear(MLP_dim, MLP_dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(MLP_dim)

        self.fc1_c = Linear(MLP_dim, 120)
        self.poc_fc = Linear(120, 60)


        self.gcn1 = GCNConv(c_feature, gcn_hidden)
        self.gcn2 = GCNConv(gcn_hidden, gcn_hidden)
        self.fc_gcn = Linear(gcn_hidden, 120)

        self.fc1 = nn.Linear(120, 64)
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, self.n_output)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # learnable weights for sum
        self.alpha = nn.Parameter(torch.tensor(0.5))  # weight for GIN path
        self.beta = nn.Parameter(torch.tensor(0.5))   # weight for GCN path

        self.pro_poc_proj      = nn.Linear(480 + 60, 120)  


        # self.final_attn = nn.MultiheadAttention(embed_dim=660, num_heads=1, batch_first=True)
        self.final_cross_attn  = nn.MultiheadAttention(embed_dim=120, num_heads=1, batch_first=True)


        
        # 


    def forward(self, data):
        x, edge_index , batch = data.x, data.edge_index,data.batch
        target = data.protein
        pocket = data.pocket

        # Feature extraction from the compound graphs
        x1 =  F.relu(self.conv1(x, edge_index))
        x1 = self.bn1(x1)
        x1 =  F.relu(self.conv2(x1, edge_index))
        x1 = self.bn2(x1)
        x1 =  F.relu(self.conv3(x1, edge_index))
        x1 = self.bn3(x1)
        x1 =  F.relu(self.conv4(x1, edge_index))
        x1 = self.bn4(x1)
        x1 = global_add_pool(x1, batch)
        x1 =  F.relu(self.fc1_c(x1))
        gin_feat = self.dropout(x1)

        x2 = F.relu(self.gcn1(x, edge_index))
        x2 = F.relu(self.gcn2(x2, edge_index))
        x2 = global_add_pool(x2, batch)
        gcn_feat = F.relu(self.fc_gcn(x2))
        gcn_feat = self.dropout(gcn_feat)


        src_emb = self.src_emb(target)
        pos_emb = self.pos_emb(target)
        transformer_input = src_emb + pos_emb 
        transformer_out,_ = self.transformer_encoder(transformer_input)


        cls_pool = transformer_out[:, 0, :]
        attn_pool = self.attention_pool(transformer_out)
        cnn_pool = self.cnn_protein(transformer_out)

        graph_feat = self.alpha * gin_feat + self.beta * gcn_feat
        cross_feat = self.cross_attn(transformer_out, graph_feat)

        pocket_emb = self.src_emb(pocket) + self.pos_emb(pocket)
        pocket_out,_ = self.transformer_encoder(pocket_emb)
        poc = self.poc_fc(pocket_out[:, 0, :])



        pro = torch.cat([cls_pool, attn_pool, cnn_pool, cross_feat], dim=1)
        # x = torch.cat([pro, poc, graph_feat], dim=1)
        qp = torch.cat([pro, poc], dim=1) 
        q  = self.pro_poc_proj(qp) 
        q  = q.unsqueeze(1)
        kv = graph_feat.unsqueeze(1) 
        attn_out, _ = self.final_cross_attn(q, kv, kv) 

        # attn_out, _ = self.final_attn(x.unsqueeze(1), x.unsqueeze(1), x.unsqueeze(1))  # [batch,1,660]
        x = attn_out.squeeze(1)  # [batch,660]


        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        return self.out(x)

       
