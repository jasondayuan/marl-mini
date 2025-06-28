import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GraphMixer(nn.Module):
    def __init__(self, args):
        super(GraphMixer, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        self.embed_dim = args.mixing_embed_dim

        # 节点特征转换：将 Q 值嵌入高维空间
        self.agent_fc1 = nn.Linear(1, self.embed_dim)
        self.agent_fc2 = nn.Linear(self.embed_dim, self.embed_dim)

        # 邻接矩阵生成器：从 state 估计 agent 间连接
        self.adj_fc = nn.Sequential(
            nn.Linear(self.state_dim, self.n_agents * self.n_agents),
            nn.Sigmoid()  # 生成归一化的邻接矩阵
        )

        # 全局读出层（类似 Mixer 最后一层加权）
        self.readout = nn.Sequential(
            nn.Linear(self.embed_dim, 1)
        )

        # 状态相关偏置项（与 QMIX 类似）
        self.bias_layer = nn.Sequential(
            nn.Linear(self.state_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1)
        )

    def forward(self, agent_qs, states):
        bs, t, _ = agent_qs.shape
        agent_qs = agent_qs.view(bs * t, self.n_agents, 1)
        states = states.view(bs * t, self.state_dim)

        # 1. 节点嵌入
        x = F.relu(self.agent_fc1(agent_qs))      # (bs*t, n_agents, embed)
        x = F.relu(self.agent_fc2(x))             # (bs*t, n_agents, embed)

        # 2. 邻接矩阵估计
        adj = self.adj_fc(states).view(-1, self.n_agents, self.n_agents)
        mask = 1 - th.eye(self.n_agents).to(agent_qs.device).unsqueeze(0)
        adj = adj * mask  # 去掉自环连接

        # 3. 图消息传递（Graph Aggregation）
        agg = th.bmm(adj, x) / (adj.sum(dim=-1, keepdim=True) + 1e-6)
        node_embed = x + agg  # 残差连接

        # 4. 全局汇总
        readout = self.readout(node_embed).sum(dim=1)  # (bs*t, 1)
        bias = self.bias_layer(states)                 # (bs*t, 1)

        q_tot = readout + bias
        return q_tot.view(bs, t, 1)
