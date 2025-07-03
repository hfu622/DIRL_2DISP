import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# TODO：更改网络参数，隐藏层和head
# class PieceSelector(nn.Module):
#     def __init__(self, out_dim):
#         super(PieceSelector, self).__init__()
#         self.layer1 = nn.Linear(64, 32)
#         self.transformer = nn.TransformerEncoder(
#             nn.TransformerEncoderLayer(d_model=32, nhead=8, dim_feedforward=64),
#             num_layers=1
#         )
#         self.decoder = nn.Linear(32, out_dim)
#
#     def forward(self, x):  # 输入是pieces的顶点列表和当前bin信息
#         x = torch.tensor(x, dtype=torch.float32, device=device)
#
#         x = torch.relu(self.layer1(x))
#         x = x.unsqueeze(0)
#         x = self.transformer(x)
#         x = x.squeeze(0)
#
#         logit = self.decoder(x)
#         return logit

# NOTE：新的模型，加了参数层面的random
# 增加复杂度？
class PieceSelector(nn.Module):
    def __init__(self, out_dim):
        super(PieceSelector, self).__init__()
        self.layer1 = nn.Linear(64, 32)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=32, nhead=8, dim_feedforward=64),
            num_layers=2
        )
        # self.decoder = nn.Linear(32, out_dim)
        self.decoder_mu = nn.Linear(32, out_dim)
        self.decoder_logstd = nn.Linear(32, out_dim)

    def forward(self, x):  # 输入是pieces的顶点列表和当前bin信息
        x = torch.tensor(x, dtype=torch.float32, device=device)

        x = torch.relu(self.layer1(x))
        x = x.unsqueeze(0)
        x = self.transformer(x)
        x = x.squeeze(0)

        # logit = self.decoder(x)
        mu=self.decoder_mu(x)
        log_std=self.decoder_logstd(x)
        log_std = torch.clamp(log_std, max=10)
        logit=mu + torch.randn_like(log_std) * torch.exp(log_std)
        return logit


class AngleSelector(nn.Module):
    def __init__(self, out_dim):
        super(AngleSelector, self).__init__()
        self.layer1 = nn.Linear(64, 16)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=16, nhead=8, dim_feedforward=16),
            num_layers=1
        )
        self.decoder = nn.Linear(16, out_dim)

    def forward(self, x):  # 输入是bin和pieces位置信息以及当前零件信息
        x = torch.tensor(x, dtype=torch.float32, device=device)

        x = torch.relu(self.layer1(x))
        x = x.unsqueeze(0)
        x = self.transformer(x)
        x = x.squeeze(0)

        logit = self.decoder(x)

        return logit


class XSelector(nn.Module):
    def __init__(self, out_dim):
        super(XSelector, self).__init__()
        self.layer1 = nn.Linear(64, 64)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=64, nhead=8, dim_feedforward=256),
            num_layers=2
        )
        self.decoder = nn.Linear(64, out_dim)  # 输出维度为底板的宽度

    def forward(self, x):  # 输入是bin信息加上当前零件（旋转后）信息
        x = torch.tensor(x, dtype=torch.float32, device=device)

        x = torch.relu(self.layer1(x))
        x = x.unsqueeze(0)
        x = self.transformer(x)
        x = x.squeeze(0)

        logit = self.decoder(x)
        return logit


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(128, 32)  # NOTE：将输入改成64维
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=32, nhead=8, dim_feedforward=32),
            num_layers=1
        )
        self.value_head = nn.Linear(32, 1)

    def forward(self, x):  # 输入的是state，包括bin和pieces信息
        x = torch.tensor(x, dtype=torch.float32, device=device)
        x = torch.relu(self.layer1(x))
        x = x.unsqueeze(0)
        x = self.transformer(x)
        x = x.squeeze(0)
        value = self.value_head(x)
        return value


class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNModel, self).__init__()
        # self.conv1 = GCNConv(input_dim, hidden_dim)
        # self.conv2 = GCNConv(hidden_dim, output_dim)
        # 改成一层
        self.conv = GCNConv(input_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # x = self.conv1(x, edge_index)
        # x = torch.relu(x)
        # x = self.conv2(x, edge_index)
        x = torch.relu(self.conv(x, edge_index))
        return x


class LS_selector(nn.Module):
    def __init__(self, out_dim):
        super(LS_selector, self).__init__()
        self.layer1 = nn.Linear(64, 64)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=2, dim_feedforward=64)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)

        self.decoder_layer = nn.TransformerDecoderLayer(d_model=64, nhead=2, dim_feedforward=64)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=2)

        self.decoder_output = nn.Linear(64, out_dim)

    def forward(self, x):  # 输入是当前placement的编码信息
        x = torch.tensor(x, dtype=torch.float32, device=device)

        x = torch.relu(self.layer1(x))
        x = x.unsqueeze(0)
        x_enc = self.transformer_encoder(x)

        tgt = torch.zeros_like(x_enc)

        x_dec = self.transformer_decoder(tgt, memory=x_enc)
        x_dec = x_dec.squeeze(0)

        logit = self.decoder_output(x_dec)
        return logit


class LS_critic(nn.Module):
    def __init__(self):
        super(LS_critic, self).__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=128, nhead=8, dim_feedforward=256),
            num_layers=2
        )
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):  # 输入的是state，包括bin和pieces信息
        x = torch.tensor(x, dtype=torch.float32, device=device)
        x_len = x.shape[-1]
        embedding = nn.Linear(x_len, 128).to(device)
        x = torch.relu(embedding(x))
        x = x.unsqueeze(0)
        x = self.transformer(x)
        x = x.squeeze(0)
        value = self.value_head(x)
        return value
