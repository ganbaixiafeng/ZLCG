import torch
from torch import nn


class Graph_FusionMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, graph_num, first, configs):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.first = first

        self.fusion_num_layer = configs.fusion_num_layer
        self.fusion_dim = configs.fusion_dim
        self.fusion_dropout = configs.fusion_dropout
        self.if_forward = configs.if_forward
        self.node_dim = configs.node_dim
        self.if_feedback = configs.if_feedback
        self.nhead = configs.nhead

        self.if_time_in_day = configs.if_time_in_day
        self.if_day_in_week = configs.if_day_in_week
        self.if_day_in_month = configs.if_day_in_month
        self.if_day_in_year = configs.if_day_in_year
        self.temp_dim = configs.temp_dim
        self.time_of_day_size = configs.time_of_day_size
        self.day_of_week_size = configs.day_of_week_size
        self.day_of_month_size = configs.day_of_month_size
        self.day_of_year_size = configs.day_of_year_size

        if self.if_time_in_day:
            self.time_in_day_emb = nn.Parameter(torch.empty(self.time_of_day_size, self.temp_dim))
            nn.init.xavier_uniform_(self.time_in_day_emb)
        if self.if_day_in_week:
            self.day_in_week_emb = nn.Parameter(torch.empty(self.day_of_week_size, self.temp_dim))
            nn.init.xavier_uniform_(self.day_in_week_emb)
        if self.if_day_in_month:
            self.day_in_month_emb = nn.Parameter(torch.empty(self.day_of_month_size, self.temp_dim))
            nn.init.xavier_uniform_(self.day_in_month_emb)
        if self.if_day_in_year:
            self.day_in_year_emb = nn.Parameter(torch.empty(self.day_of_year_size, self.temp_dim))
            nn.init.xavier_uniform_(self.day_in_year_emb)

        self.fusion_graph_model = nn.Sequential(
            *[MultiLayerPerceptron(input_dim=self.input_dim,
                                   hidden_dim=self.hidden_dim,
                                   dropout=self.fusion_dropout)
              for _ in range(self.fusion_num_layer)],
        )
        self.fusion_forward_linear = nn.Linear(in_features=self.hidden_dim, out_features=self.fusion_dim, bias=True)

        self.fusion_model = nn.Sequential(
            *[MultiLayerPerceptron(input_dim=self.fusion_dim,
                                   hidden_dim=self.fusion_dim,
                                   dropout=self.fusion_dropout)
              for _ in range(self.fusion_num_layer)],
            nn.Linear(in_features=self.fusion_dim, out_features=self.out_dim, bias=True),
        )

        if not self.first and self.if_feedback and self.if_forward:
            self.forward_att = nn.MultiheadAttention(embed_dim=self.fusion_dim,
                                                     num_heads=self.nhead,
                                                     batch_first=True)
            self.forward_fc = nn.Sequential(
                nn.Linear(in_features=self.fusion_dim, out_features=self.node_dim, bias=True),
                nn.Sigmoid()
            )

    def forward(self, x_enc, x_mark_enc, time_series_emb, predict_emb, node_emb, hidden_emb):
        tem_emb = []
        if self.if_time_in_day:
            t_i_d_data = (x_mark_enc[..., 0] + 0.5) * (self.time_of_day_size - 1)
            t_i_d_data = t_i_d_data.unsqueeze(-1).expand(-1, -1, x_enc.shape[-1])
            tem_emb.append(self.time_in_day_emb[(t_i_d_data[:, -1, :]).type(torch.LongTensor)])
        if self.if_day_in_week:
            d_i_w_data = (x_mark_enc[..., 1] + 0.5) * (self.day_of_week_size - 1)
            d_i_w_data = d_i_w_data.unsqueeze(-1).expand(-1, -1, x_enc.shape[-1])
            tem_emb.append(self.day_in_week_emb[(d_i_w_data[:, -1, :]).type(torch.LongTensor)])
        if self.if_day_in_month:
            d_i_m_data = (x_mark_enc[..., 2] + 0.5) * (self.day_of_month_size - 1)
            d_i_m_data = d_i_m_data.unsqueeze(-1).expand(-1, -1, x_enc.shape[-1])
            tem_emb.append(self.day_in_month_emb[(d_i_m_data[:, -1, :]).type(torch.LongTensor)])
        if self.if_day_in_year:
            d_i_y_data = (x_mark_enc[..., 3] + 0.5) * self.day_of_year_size
            d_i_y_data = d_i_y_data.unsqueeze(-1).expand(-1, -1, x_enc.shape[-1])
            tem_emb.append(self.day_in_year_emb[(d_i_y_data[:, -1, :]).type(torch.LongTensor)])

        if not self.first and self.if_feedback and self.if_forward:
            node_emb_val = node_emb[0]
            hidden_val = hidden_emb[0]
            hidden_val = self.forward_att(hidden_val, hidden_val, hidden_val)[0]
            hidden_val = self.forward_fc(hidden_val)
            node_emb = [node_emb_val * hidden_val]

        forward_emb = torch.cat(time_series_emb + predict_emb + node_emb + tem_emb, dim=2)
        hidden = self.fusion_graph_model(forward_emb)
        hidden = self.fusion_forward_linear(hidden)
        predict = self.fusion_model(hidden)
        return predict, [hidden], node_emb


class GraphMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.act_fn = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        return x + self.fc2(x)


class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.2) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=True),
        )

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        hidden = self.fc(input_data)
        hidden = hidden + input_data
        return hidden
