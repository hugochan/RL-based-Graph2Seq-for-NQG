import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.generic_utils import to_cuda
from .common import GatedFusion, GRUStep
from ..utils.constants import VERY_SMALL_NUMBER, INF


class GraphNN(nn.Module):
    def __init__(self, config):
        super(GraphNN, self).__init__()
        print('[ Using {}-hop GraphNN ]'.format(config['graph_hops']))
        self.device = config['device']
        hidden_size = config['hidden_size']
        self.graph_direction = config.get('graph_direction', 'all')
        assert self.graph_direction in ('all', 'forward', 'backward')
        self.graph_type = config['graph_type']
        self.graph_hops = config['graph_hops']
        self.linear_max = nn.Linear(hidden_size, hidden_size, bias=False)

        if self.graph_type in ('static', 'hybrid_sep'):
            # Static graph
            self.static_graph_mp = GraphMessagePassing(config)
            self.static_gru_step = GRUStep(hidden_size, hidden_size)
            if self.graph_direction == 'all':
                self.static_gated_fusion = GatedFusion(hidden_size)
        if self.graph_type in ('dynamic', 'hybrid_sep'):
            # Dynamic graph
            self.graph_learner = GraphLearner(config['gl_input_size'], hidden_size, topk=config['graph_learner_topk'], \
                                                    num_pers=config['graph_learner_num_pers'], device=self.device)
            self.dynamic_gru_step = GRUStep(hidden_size, hidden_size)
            if self.graph_direction == 'all':
                self.dynamic_gated_fusion = GatedFusion(hidden_size)
        if self.graph_type == 'static':
            self.graph_update = self.static_graph_update
        elif self.graph_type == 'dynamic':
            self.graph_update = self.dynamic_graph_update
        elif self.graph_type == 'hybrid':
            self.graph_update = self.hybrid_graph_update
            self.graph_learner = GraphLearner(config['gl_input_size'], hidden_size, topk=config['graph_learner_topk'], \
                                                    num_pers=config['graph_learner_num_pers'], device=self.device)
            self.static_graph_mp = GraphMessagePassing(config)
            self.hybrid_gru_step = GRUStep(hidden_size, hidden_size // 4 * 4)
            self.linear_kernels = nn.ModuleList([nn.Linear(hidden_size, hidden_size // 4, bias=False) for _ in range(4)])
        elif self.graph_type == 'static_gcn':
            self.graph_update = self.static_gcn
            self.gcn_linear = nn.ModuleList([nn.Linear(hidden_size, hidden_size, bias=False) for _ in range(self.graph_hops)])
        else:
            raise RuntimeError('Unknown graph_type: {}'.format(self.graph_type))

        print('[ Using graph type: {} ]'.format(self.graph_type))
        print('[ Using graph direction: {} ]'.format(self.graph_direction))

    def forward(self, node_state, edge_vec, adj, node_mask=None, raw_node_vec=None):
        node_state, graph_embedding = self.graph_update(node_state, edge_vec, adj, node_mask=node_mask, raw_node_vec=raw_node_vec)
        return node_state, graph_embedding

    def static_graph_update(self, node_state, edge_vec, adj, node_mask=None, raw_node_vec=None):
        '''Static graph update'''
        node2edge, edge2node = adj

        # Shape: (batch_size, num_edges, num_nodes)
        node2edge = to_cuda(torch.stack([torch.Tensor(x.A) for x in node2edge], dim=0), self.device)
        # Shape: (batch_size, num_nodes, num_edges)
        edge2node = to_cuda(torch.stack([torch.Tensor(x.A) for x in edge2node], dim=0), self.device)

        for _ in range(self.graph_hops):
            bw_agg_state = self.static_graph_mp.mp_func(node_state, edge_vec, node2edge, edge2node)
            fw_agg_state = self.static_graph_mp.mp_func(node_state, edge_vec, edge2node.transpose(1, 2), node2edge.transpose(1, 2))
            if self.graph_direction == 'all':
                agg_state = self.static_gated_fusion(fw_agg_state, bw_agg_state)
                node_state = self.static_gru_step(node_state, agg_state)
            elif self.graph_direction == 'forward':
                node_state = self.static_gru_step(node_state, fw_agg_state)
            else:
                node_state = self.static_gru_step(node_state, bw_agg_state)

        graph_embedding = self.graph_maxpool(node_state, node_mask).unsqueeze(0)
        return node_state.transpose(0, 1), graph_embedding

    def dynamic_graph_update(self, node_state, edge_vec, adj, node_mask=None, raw_node_vec=None):
        '''Dynamic graph update'''
        assert raw_node_vec is not None
        node2edge, edge2node = adj

        dynamic_adjacency_matrix = self.graph_learner(raw_node_vec, node_mask)
        bw_dynamic_adjacency_matrix = torch.softmax(dynamic_adjacency_matrix.transpose(-1, -2), dim=-1)
        fw_dynamic_adjacency_matrix = torch.softmax(dynamic_adjacency_matrix, dim=-1)
        for _ in range(self.graph_hops):
            bw_agg_state = self.aggregate(node_state, bw_dynamic_adjacency_matrix)
            fw_agg_state = self.aggregate(node_state, fw_dynamic_adjacency_matrix)
            if self.graph_direction == 'all':
                agg_state = self.dynamic_gated_fusion(bw_agg_state, fw_agg_state)
                node_state = self.dynamic_gru_step(node_state, agg_state)
            elif self.graph_direction == 'forward':
                node_state = self.dynamic_gru_step(node_state, fw_agg_state)
            else:
                node_state = self.dynamic_gru_step(node_state, bw_agg_state)

        graph_embedding = self.graph_maxpool(node_state, node_mask).unsqueeze(0)
        return node_state.transpose(0, 1), graph_embedding

    def hybrid_graph_update(self, node_state, edge_vec, adj, node_mask=None, raw_node_vec=None):
        assert raw_node_vec is not None
        node2edge, edge2node = adj

        # Shape: (batch_size, num_edges, num_nodes)
        node2edge = to_cuda(torch.stack([torch.Tensor(x.A) for x in node2edge], dim=0), self.device)
        # Shape: (batch_size, num_nodes, num_edges)
        edge2node = to_cuda(torch.stack([torch.Tensor(x.A) for x in edge2node], dim=0), self.device)

        dynamic_adjacency_matrix = self.graph_learner(raw_node_vec, node_mask)
        bw_dynamic_adjacency_matrix = torch.softmax(dynamic_adjacency_matrix.transpose(-1, -2), dim=-1)
        fw_dynamic_adjacency_matrix = torch.softmax(dynamic_adjacency_matrix, dim=-1)
        for _ in range(self.graph_hops):
            # Dynamic
            bw_dyn_agg_state = self.aggregate(node_state, bw_dynamic_adjacency_matrix)
            fw_dyn_agg_state = self.aggregate(node_state, fw_dynamic_adjacency_matrix)
            # Static
            bw_sta_agg_state = self.static_graph_mp.mp_func(node_state, edge_vec, node2edge, edge2node)
            fw_sta_agg_state = self.static_graph_mp.mp_func(node_state, edge_vec, edge2node.transpose(1, 2), node2edge.transpose(1, 2))

            # Combine
            agg_state = torch.cat([self.linear_kernels[i](x) for i, x in enumerate([bw_dyn_agg_state, fw_dyn_agg_state, bw_sta_agg_state, fw_sta_agg_state])], -1)
            node_state = self.hybrid_gru_step(node_state, agg_state)

        graph_embedding = self.graph_maxpool(node_state, node_mask).unsqueeze(0)
        return node_state.transpose(0, 1), graph_embedding

    def graph_maxpool(self, node_state, node_mask=None):
        # Maxpool
        # Shape: (batch_size, hidden_size, num_nodes)
        node_embedding_p = self.linear_max(node_state).transpose(-1, -2)
        graph_embedding = F.max_pool1d(node_embedding_p, kernel_size=node_embedding_p.size(-1)).squeeze(-1)
        return graph_embedding

    def aggregate(self, node_state, weighted_adjacency_matrix):
        return torch.bmm(weighted_adjacency_matrix, node_state)

    def static_gcn(self, node_state, edge_vec, adj, node_mask=None, raw_node_vec=None):
        '''Static GCN update'''
        node2edge, edge2node = adj

        # Shape: (batch_size, num_edges, num_nodes)
        node2edge = to_cuda(torch.stack([torch.Tensor(x.A) for x in node2edge], dim=0), self.device)
        # Shape: (batch_size, num_nodes, num_edges)
        edge2node = to_cuda(torch.stack([torch.Tensor(x.A) for x in edge2node], dim=0), self.device)

        adj = torch.bmm(edge2node, node2edge)
        adj = adj + adj.transpose(1, 2)
        adj = adj + to_cuda(torch.eye(adj.shape[1], adj.shape[2]), self.device)
        adj = torch.clamp(adj, max=1)

        rowsum = torch.sum(adj, -1)
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_mat_inv_sqrt = torch.stack([torch.diagflat(d_inv_sqrt[i]) for i in range(d_inv_sqrt.shape[0])], dim=0)

        adj = torch.bmm(d_mat_inv_sqrt, torch.bmm(adj, d_mat_inv_sqrt))

        for _ in range(self.graph_hops):
            node_state = F.relu(self.gcn_linear[_](torch.bmm(adj, node_state)))

        graph_embedding = self.graph_maxpool(node_state, node_mask).unsqueeze(0)
        return node_state.transpose(0, 1), graph_embedding


class GraphMessagePassing(nn.Module):
    def __init__(self, config):
        super(GraphMessagePassing, self).__init__()
        self.config = config
        hidden_size = config['hidden_size']
        if config['message_function'] == 'edge_mm':
            self.edge_weight_tensor = torch.Tensor(config['num_edge_types'], hidden_size * hidden_size)
            self.edge_weight_tensor = nn.Parameter(nn.init.xavier_uniform_(self.edge_weight_tensor))
            self.mp_func = self.msg_pass_edge_mm
        elif config['message_function'] == 'edge_network':
            self.edge_network = torch.Tensor(config['edge_embed_dim'], hidden_size, hidden_size)
            self.edge_network = nn.Parameter(nn.init.xavier_uniform_(self.edge_network))
            self.mp_func = self.msg_pass_edge_network
        elif config['message_function'] == 'edge_pair':
            self.linear_edge = nn.Linear(config['edge_embed_dim'], hidden_size, bias=False)
            self.mp_func = self.msg_pass
        elif config['message_function'] == 'no_edge':
            self.mp_func = self.msg_pass
        else:
            raise RuntimeError('Unknown message_function: {}'.format(config['message_function']))

    def msg_pass(self, node_state, edge_vec, node2edge, edge2node):
        node2edge_emb = torch.bmm(node2edge, node_state) # batch_size x num_edges x hidden_size
        if edge_vec is not None and self.config['message_function'] == 'edge_pair':
            node2edge_emb = node2edge_emb + self.linear_edge(edge_vec)

        # Add self-loop
        norm_ = torch.sum(edge2node, 2, keepdim=True) + 1
        agg_state = (torch.bmm(edge2node, node2edge_emb) + node_state) / norm_
        return agg_state

    def msg_pass_maxpool(self, node_state, edge_vec, node2edge, edge2node, fc_maxpool):
        node2edge_emb = torch.bmm(node2edge, node_state) # batch_size x num_edges x hidden_size
        node2edge_emb = fc_maxpool(node2edge_emb)
        # Expand + mask
        # batch_size x num_nodes x num_edges x hidden_size
        node2edge_emb = node2edge_emb.unsqueeze(1) * edge2node.unsqueeze(-1) - (1 - edge2node).unsqueeze(-1) * INF
        node2edge_emb = node2edge_emb.view(-1, node2edge_emb.size(-2), node2edge_emb.size(-1)).transpose(-1, -2)
        agg_state = F.max_pool1d(node2edge_emb, kernel_size=node2edge_emb.size(-1)).squeeze(-1).view(node_state.size())
        agg_state = agg_state * (torch.sum(edge2node, dim=-1, keepdim=True) != 0).float()
        return agg_state

    def msg_pass_edge_mm(self, node_state, edge_vec, node2edge, edge2node):
        node2edge_emb = torch.bmm(node2edge, node_state) # batch_size x num_edges x hidden_size

        new_node2edge_emb = []
        for i in range(node2edge_emb.size(1)):
            edge_weight = F.embedding(edge_vec[:, i], self.edge_weight_tensor).view(-1, node_state.size(-1), node_state.size(-1)) # batch_size x hidden_size x hidden_size
            new_node2edge_emb.append(torch.matmul(edge_weight, node2edge_emb[:, i].unsqueeze(-1)).squeeze(-1))

        new_node2edge_emb = torch.stack(new_node2edge_emb, dim=1) # batch_size x num_edges x hidden_size

        # Add self-loop
        norm_ = torch.sum(edge2node, 2, keepdim=True) + 1
        agg_state = (torch.bmm(edge2node, new_node2edge_emb) + node_state) / norm_ # TODO: apply LP to node_state itself
        return agg_state

    def msg_pass_edge_network(self, node_state, edge_vec, node2edge, edge2node):
        node2edge_emb = torch.bmm(node2edge, node_state) # batch_size x num_edges x hidden_size

        new_node2edge_emb = []
        for i in range(node2edge_emb.size(1)):
            edge_weight = torch.mm(edge_vec[:, i], self.edge_network.view(self.edge_network.size(0), -1)).view((-1,) + self.edge_network.shape[-2:])
            new_node2edge_emb.append(torch.matmul(edge_weight, node2edge_emb[:, i].unsqueeze(-1)).squeeze(-1))

        new_node2edge_emb = torch.stack(new_node2edge_emb, dim=1) # batch_size x num_edges x hidden_size

        # Add self-loop
        norm_ = torch.sum(edge2node, 2, keepdim=True) + 1
        agg_state = (torch.bmm(edge2node, new_node2edge_emb) + node_state) / norm_ # TODO: apply LP to node_state itself
        return agg_state

class GraphLearner(nn.Module):
    def __init__(self, input_size, hidden_size, topk=10, num_pers=16, device=None):
        super(GraphLearner, self).__init__()
        self.device = device
        self.topk = topk
        self.linear_sim = nn.Linear(input_size, hidden_size, bias=False)

    def forward(self, context, ctx_mask):
        """
        Parameters
        :context, (batch_size, ctx_size, dim)
        :ctx_mask, (batch_size, ctx_size)

        Returns
        :adjacency_matrix, (batch_size, ctx_size, ctx_size)
        """
        context_fc = torch.relu(self.linear_sim(context))
        attention = torch.matmul(context_fc, context_fc.transpose(-1, -2))

        if ctx_mask is not None:
            attention = attention.masked_fill_(1 - ctx_mask.byte().unsqueeze(1), -INF)
            attention = attention.masked_fill_(1 - ctx_mask.byte().unsqueeze(-1), -INF)

        weighted_adjacency_matrix = self.build_knn_neighbourhood(attention, self.topk)
        return weighted_adjacency_matrix

    def build_knn_neighbourhood(self, attention, topk):
        topk = min(topk, attention.size(-1))
        knn_val, knn_ind = torch.topk(attention, topk, dim=-1)
        weighted_adjacency_matrix = to_cuda((-INF * torch.ones_like(attention)).scatter_(-1, knn_ind, knn_val), self.device)
        return weighted_adjacency_matrix
