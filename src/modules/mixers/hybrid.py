import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class HybridMixer(nn.Module):
    def __init__(self, args):
        super(HybridMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        self.u_dim = int(np.prod(args.agent_own_state_size))
        self.embed_dim = args.mixing_embed_dim

        # -------- Qatten 部分 --------
        self.n_query_embedding_layer1 = args.n_query_embedding_layer1
        self.n_query_embedding_layer2 = args.n_query_embedding_layer2
        self.n_key_embedding_layer1 = args.n_key_embedding_layer1
        self.n_attention_head = args.n_attention_head

        self.query_embedding_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.state_dim, self.n_query_embedding_layer1),
                nn.ReLU(),
                nn.Linear(self.n_query_embedding_layer1, self.n_query_embedding_layer2)
            ) for _ in range(self.n_attention_head)
        ])

        self.key_embedding_layers = nn.ModuleList([
            nn.Linear(self.u_dim, self.n_key_embedding_layer1)
            for _ in range(self.n_attention_head)
        ])

        self.scaled_product_value = np.sqrt(self.n_query_embedding_layer2)

        # -------- Mixer 部分 --------
        self.hyper_w = nn.Sequential(
            nn.Linear(self.state_dim, args.hypernet_embed),
            nn.ReLU(),
            nn.Linear(args.hypernet_embed, self.n_agents)
        )

        self.hyper_b = nn.Sequential(
            nn.Linear(self.state_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1)
        )

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        states = states.reshape(-1, self.state_dim)
        us = self._get_us(states)

        # Qatten-style attention 权重计算
        q_lambda_list = []
        for i in range(self.n_attention_head):
            query = self.query_embedding_layers[i](states).view(-1, 1, self.n_query_embedding_layer2)
            key = self.key_embedding_layers[i](us).view(-1, self.n_agents, self.n_key_embedding_layer1)
            key = key.permute(0, 2, 1)  # (bs, key_dim, n_agents)
            attn = th.matmul(query, key) / self.scaled_product_value  # (bs, 1, n_agents)
            q_lambda = F.softmax(attn, dim=-1)
            q_lambda_list.append(q_lambda)

        # 多头注意力加权 Q 值融合
        q_lambda_all = th.stack(q_lambda_list, dim=1).mean(1)  # (bs, 1, n_agents)
        attn_q = th.bmm(agent_qs, q_lambda_all.transpose(1, 2))  # (bs, 1, 1)

        # Mixer-style 加权Q值（非注意力）
        w = th.abs(self.hyper_w(states)).view(-1, 1, self.n_agents)
        w = F.softmax(w, dim=-1)  # normalize
        mix_q = th.bmm(agent_qs, w.transpose(1, 2))  # (bs, 1, 1)

        # 融合注意力与混合权重 Q 值
        final_q = 0.5 * (attn_q + mix_q) + self.hyper_b(states).view(-1, 1, 1)
        return final_q.view(bs, -1, 1)

    def _get_us(self, states):
        agent_own_state_size = self.args.agent_own_state_size
        with th.no_grad():
            us = states[:, :agent_own_state_size * self.n_agents].reshape(-1, agent_own_state_size)
        return us
 