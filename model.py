import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import binary_search


torch.set_num_threads(3)
class T_aT_R1_GCN_SSL(nn.Module):
    def __init__(self, entity_num, time_num, relation_num, emb_dim, temperature, device__, aT_ratio, rel_ratio,
                 random_seed, time_split_window_size, init_time_span_weight):
        super(T_aT_R1_GCN_SSL, self).__init__()
        self.device_0 = torch.device(device__ if torch.cuda.is_available() else "cpu")
        torch.manual_seed(random_seed)
        rgcn_ratio = rel_ratio  # 论文公式(3)
        static_ratio = aT_ratio  # 论文C 1)部分：TempGCN嵌入et 和 aTempGCN嵌入e 的权重比例，得到融合嵌入h: (1-λ)et + λe = h
        self.lambda_ = torch.tensor([static_ratio]).to(self.device_0)
        self.lambda_2 = torch.tensor([rgcn_ratio]).to(self.device_0)
        self.temperature = temperature
        self.time_num = time_num
        self.emb_dim = emb_dim
        self.global_node_embedding_layer = nn.Embedding(entity_num, emb_dim, sparse=True).to(
            self.device_0)  # 全局嵌入层：嵌入矩阵entity_num行
        self.node_embedding_layer = nn.Embedding(entity_num * time_num, emb_dim, sparse=True).to(
            self.device_0)  # 时间嵌入层：嵌入矩阵entity_num * time_num行
        self.edge_embedding_layer = nn.Embedding(relation_num, emb_dim, sparse=True).to(
            self.device_0)  # 关系嵌入层：嵌入矩阵relation_num行
        # 初始化权重
        nn.init.xavier_normal_(self.global_node_embedding_layer.weight.data)
        nn.init.xavier_normal_(self.node_embedding_layer.weight.data)
        nn.init.xavier_normal_(self.edge_embedding_layer.weight.data)

        self.base_window_size = int(time_split_window_size)
        self.historical_window_size_parameters = nn.Parameter((torch.ones(relation_num, 1) * 0.7).to(self.device_0), requires_grad=True)
        self.future_window_size_parameters = nn.Parameter((torch.ones(relation_num, 1) * 0.7).to(self.device_0), requires_grad=True)

        self.relation_time_span = nn.Parameter(torch.tensor(init_time_span_weight).to(self.device_0), requires_grad=True)


    def forward(self, h_g, f_g, SSL_g, Glob_g, Rel_g, seed_nodes, relation_batch, neighbor_batch_size, h_ratio, f_ratio):
        h_g.readonly(True)
        f_g.readonly(True)
        Glob_g.readonly(True)
        seed_node_batch_size = len(seed_nodes)
        original_nodes = seed_nodes // torch.tensor(self.time_num)  # 去除entity_id的时间属性
        """ aTempGCN """
        Global_seed, Global_seed_idx = torch.unique(original_nodes, sorted=False, return_inverse=True)
        Global_batch = dgl.contrib.sampling.NeighborSampler(Glob_g,
                                                            batch_size=seed_node_batch_size,
                                                            expand_factor=neighbor_batch_size,  # 采样所有邻居
                                                            neighbor_type='in',
                                                            shuffle=False,
                                                            num_hops=2,  # 采样两跳
                                                            seed_nodes=Global_seed,
                                                            add_self_loop=False)
        for Global_flow in Global_batch:
            break
        Global_flow.copy_from_parent()
        Global_node_unique, Global_node_index = torch.unique(torch.cat([Global_flow.layers[0].data['node_idx'],
                                                                        Global_flow.layers[1].data['node_idx'],
                                                                        Global_flow.layers[2].data['node_idx']]),
                                                             sorted=False, return_inverse=True)
        Glob_node_emb = self.global_node_embedding_layer(Global_node_unique.to(self.device_0))  # 获得全局嵌入
        Glob_len0 = len(Global_flow.layers[0].data['node_idx'])  # layer 0 的节点数
        Glob_len1 = len(Global_flow.layers[1].data['node_idx'])  # layer 1 的节点数
        Glob_rel_unique, Glob_rel_index = torch.unique(torch.cat([Global_flow.blocks[0].data['relation_idx'],
                                                                  Global_flow.blocks[1].data['relation_idx']]),
                                                       sorted=False, return_inverse=True)
        Glob_len2 = len(Global_flow.blocks[0].data['relation_idx'])  # layer 0 - layer 1的边数
        self.Glob_rel_emb = self.edge_embedding_layer(Glob_rel_unique.to(self.device_0))  # 获得关系嵌入

        Global_flow.layers[0].data['node_emb'] = Glob_node_emb[Global_node_index[:Glob_len0]]  # 初始化layer 0节点的嵌入信息
        Global_flow.blocks[0].data['unique_idx'] = Glob_rel_index[:Glob_len2]
        Global_flow.layers[0].data['out_degree_sqrt'] = torch.sqrt(
            Global_flow.layer_out_degree(0).type(torch.FloatTensor)).unsqueeze(1).to(
            self.device_0)  # degree normalization
        Global_flow.layers[1].data['in_degree_sqrt'] = torch.sqrt(
            Global_flow.layer_in_degree(1).type(torch.FloatTensor)).unsqueeze(1).to(
            self.device_0)  # degree normalization
        Global_flow.block_compute(block_id=0, message_func=self.msg_Global,
                                  reduce_func=self.reduce_GCN)  # 消息传递、聚合layer 0的节点信息
        Global_flow.layers[1].data['node_emb'] = Global_flow.layers[1].data['reduced'] + Glob_node_emb[
            Global_node_index[Glob_len0:Glob_len0 + Glob_len1]]  # 利用layer 0的聚合信息更新layer 1节点信息
        Global_flow.blocks[1].data['unique_idx'] = Glob_rel_index[Glob_len2:]
        Global_flow.layers[1].data['out_degree_sqrt'] = torch.sqrt(
            Global_flow.layer_out_degree(1).type(torch.FloatTensor)).unsqueeze(1).to(
            self.device_0)  # degree normalization
        Global_flow.layers[2].data['in_degree_sqrt'] = torch.sqrt(
            Global_flow.layer_in_degree(2).type(torch.FloatTensor)).unsqueeze(1).to(
            self.device_0)  # degree normalization
        Global_flow.block_compute(block_id=1, message_func=self.msg_Global,
                                  reduce_func=self.reduce_GCN)  # 消息传递、聚合layer 1的节点信息
        Glob_n = Global_flow.layers[2].data['reduced'] + Glob_node_emb[
            Global_node_index[Glob_len0 + Glob_len1:]]  # 利用layer 1的聚合信息更新layer 2的节点信息

        """ history TempGCN """
        GCN_batch = dgl.contrib.sampling.NeighborSampler(h_g,
                                                         batch_size=seed_node_batch_size,
                                                         expand_factor=neighbor_batch_size,
                                                         neighbor_type='in',
                                                         shuffle=False,
                                                         num_hops=2,
                                                         seed_nodes=seed_nodes,  # 采样有时间信息的节点id
                                                         add_self_loop=False)
        for node_flow in GCN_batch:
            break
        node_flow.copy_from_parent()
        node_unique, node_index = torch.unique(torch.cat([node_flow.layers[0].data['node_idx'],
                                                          node_flow.layers[1].data['node_idx'],
                                                          node_flow.layers[2].data['node_idx']]), sorted=False,
                                               return_inverse=True)
        node_emb = self.node_embedding_layer(node_unique.to(self.device_0))
        len0 = len(node_flow.layers[0].data['node_idx'])
        len1 = len(node_flow.layers[1].data['node_idx'])  # 第1跳节点数量 = Layer 1的结点数
        rel_unique, rel_index = torch.unique(torch.cat([node_flow.blocks[0].data['relation_idx'],
                                                        node_flow.blocks[1].data['relation_idx'],
                                                        relation_batch]), sorted=False, return_inverse=True)
        len2 = len(node_flow.blocks[0].data['relation_idx'])
        len3 = len(node_flow.blocks[1].data['relation_idx'])
        self.rel_emb = self.edge_embedding_layer(rel_unique.to(self.device_0))  # 获得关系嵌入
        # 动态窗口
        historical_weight_clamped = torch.clamp(self.historical_window_size_parameters, 0, 1)
        self.dynamic_window = 1 / (1 + torch.exp(node_flow.blocks[0].data['window_size'].to(self.device_0) -
                                                 self.base_window_size * historical_weight_clamped[node_flow.blocks[0].data['relation_idx'].cpu().tolist()].squeeze()))
        node_flow.layers[0].data['node_emb'] = node_emb[node_index[:len0]]  # layer 0节点的时间嵌入
        node_flow.blocks[0].data['unique_idx'] = rel_index[:len2]
        node_flow.layers[0].data['out_degree_sqrt'] = torch.sqrt(
            node_flow.layer_out_degree(0).type(torch.FloatTensor)).unsqueeze(1).to(
            self.device_0)  # degree normalization
        node_flow.layers[1].data['in_degree_sqrt'] = torch.sqrt(
            node_flow.layer_in_degree(1).type(torch.FloatTensor)).unsqueeze(1).to(self.device_0)  # degree normalization
        node_flow.block_compute(block_id=0, message_func=self.msg_dynamic_window_GCN,
                                reduce_func=self.reduce_GCN)  # 消息传递、聚合layer 0节点信息
        node_flow.layers[1].data['node_emb'] = node_flow.layers[1].data['reduced'] + node_emb[
            node_index[len0:len0 + len1]]  # 利用邻域节点信息更新layer 1节点信息
        node_flow.blocks[1].data['unique_idx'] = rel_index[len2:len2 + len3]
        node_flow.layers[1].data['out_degree_sqrt'] = torch.sqrt(
            node_flow.layer_out_degree(1).type(torch.FloatTensor)).unsqueeze(1).to(
            self.device_0)  # degree normalization
        node_flow.layers[2].data['in_degree_sqrt'] = torch.sqrt(
            node_flow.layer_in_degree(2).type(torch.FloatTensor)).unsqueeze(1).to(self.device_0)  # degree normalization
        self.dynamic_window = 1 / (1 + torch.exp(node_flow.blocks[1].data['window_size'].to(self.device_0) - self.base_window_size*historical_weight_clamped[node_flow.blocks[1].data['relation_idx'].cpu().tolist()].squeeze()))
        node_flow.block_compute(block_id=1, message_func=self.msg_dynamic_window_GCN,
                                reduce_func=self.reduce_GCN)  # 消息传递、聚合layer 1节点信息
        h_n = node_flow.layers[2].data['reduced'] + node_emb[node_index[len0 + len1:]]  # 利用邻域节点信息更新layer 2节点信息

        """ future TempGCN """
        GCN_batch = dgl.contrib.sampling.NeighborSampler(f_g,
                                                         batch_size=seed_node_batch_size,
                                                         expand_factor=neighbor_batch_size,
                                                         neighbor_type='in',
                                                         shuffle=False,
                                                         num_hops=2,
                                                         seed_nodes=seed_nodes,  # 采样有时间信息的节点id
                                                         add_self_loop=False)
        for node_flow in GCN_batch:
            break
        node_flow.copy_from_parent()
        node_unique, node_index = torch.unique(torch.cat([node_flow.layers[0].data['node_idx'],
                                                          node_flow.layers[1].data['node_idx'],
                                                          node_flow.layers[2].data['node_idx']]), sorted=False,
                                               return_inverse=True)
        node_emb = self.node_embedding_layer(node_unique.to(self.device_0))
        len0 = len(node_flow.layers[0].data['node_idx'])
        len1 = len(node_flow.layers[1].data['node_idx'])  # 第1跳节点数量 = Layer 1的结点数
        rel_unique, rel_index = torch.unique(torch.cat([node_flow.blocks[0].data['relation_idx'],
                                                        node_flow.blocks[1].data['relation_idx'],
                                                        relation_batch]), sorted=False, return_inverse=True)
        len2 = len(node_flow.blocks[0].data['relation_idx'])
        len3 = len(node_flow.blocks[1].data['relation_idx'])
        self.rel_emb = self.edge_embedding_layer(rel_unique.to(self.device_0))  # 获得关系嵌入
        # 动态窗口
        future_weight_clamped = torch.clamp(self.future_window_size_parameters, 0, 1)
        self.dynamic_window = 1 / (1 + torch.exp(node_flow.blocks[0].data['window_size'].to(self.device_0) - self.base_window_size * future_weight_clamped[node_flow.blocks[0].data['relation_idx'].cpu().tolist()].squeeze()))

        node_flow.layers[0].data['node_emb'] = node_emb[node_index[:len0]]  # layer 0节点的时间嵌入
        node_flow.blocks[0].data['unique_idx'] = rel_index[:len2]
        node_flow.layers[0].data['out_degree_sqrt'] = torch.sqrt(
            node_flow.layer_out_degree(0).type(torch.FloatTensor)).unsqueeze(1).to(
            self.device_0)  # degree normalization
        node_flow.layers[1].data['in_degree_sqrt'] = torch.sqrt(
            node_flow.layer_in_degree(1).type(torch.FloatTensor)).unsqueeze(1).to(self.device_0)  # degree normalization
        node_flow.block_compute(block_id=0, message_func=self.msg_dynamic_window_GCN,
                                reduce_func=self.reduce_GCN)  # 消息传递、聚合layer 0节点信息
        node_flow.layers[1].data['node_emb'] = node_flow.layers[1].data['reduced'] + node_emb[
            node_index[len0:len0 + len1]]  # 利用邻域节点信息更新layer 1节点信息
        node_flow.blocks[1].data['unique_idx'] = rel_index[len2:len2 + len3]
        node_flow.layers[1].data['out_degree_sqrt'] = torch.sqrt(
            node_flow.layer_out_degree(1).type(torch.FloatTensor)).unsqueeze(1).to(
            self.device_0)  # degree normalization
        node_flow.layers[2].data['in_degree_sqrt'] = torch.sqrt(
            node_flow.layer_in_degree(2).type(torch.FloatTensor)).unsqueeze(1).to(self.device_0)  # degree normalization
        self.dynamic_window = 1 / (1 + torch.exp(node_flow.blocks[1].data['window_size'].to(self.device_0) - self.base_window_size*future_weight_clamped[node_flow.blocks[1].data['relation_idx'].cpu().tolist()].squeeze()))
        node_flow.block_compute(block_id=1, message_func=self.msg_dynamic_window_GCN,
                                reduce_func=self.reduce_GCN)  # 消息传递、聚合layer 1节点信息
        f_n = node_flow.layers[2].data['reduced'] + node_emb[node_index[len0 + len1:]]  # 利用邻域节点信息更新layer 2节点信息

        """ 融合 【全局嵌入】 和 【局部嵌入】：论文C不分1)处，h(t) = (1-λ)et + λe """
        t_n = (h_ratio*h_n)+(f_ratio*f_n)
        n = Glob_n[Global_seed_idx] * self.lambda_ + t_n * (1 - self.lambda_)

        """ RelGCN """
        rel_seed, rel_seed_idx = torch.unique(relation_batch, sorted=False, return_inverse=True)
        rel_batch = dgl.contrib.sampling.NeighborSampler(Rel_g,
                                                         batch_size=seed_node_batch_size,
                                                         expand_factor=neighbor_batch_size,
                                                         neighbor_type='in',
                                                         shuffle=False,
                                                         num_hops=1,
                                                         seed_nodes=rel_seed,
                                                         add_self_loop=False)
        for rel_flow in rel_batch:
            break
        rel_flow.copy_from_parent()
        edge_unique, edge_index = torch.unique(torch.cat([rel_flow.layers[0].data['relation_idx'],
                                                          rel_flow.layers[1].data['relation_idx']]), sorted=False,
                                               return_inverse=True)

        self.time_span = rel_flow.blocks[0].data['time_span'].to(self.device_0)
        edge_emb = self.edge_embedding_layer(edge_unique.to(self.device_0))  # 获得关系嵌入
        len0 = len(rel_flow.layers[0].data['relation_idx'])
        rel_flow.layers[0].data['rel_emb'] = edge_emb[edge_index[:len0]]
        rel_flow.layers[0].data['out_degree_sqrt'] = torch.sqrt(
            rel_flow.layer_out_degree(0).type(torch.FloatTensor)).unsqueeze(1).to(self.device_0)  # degree normalization
        rel_flow.layers[1].data['in_degree_sqrt'] = torch.sqrt(
            rel_flow.layer_in_degree(1).type(torch.FloatTensor)).unsqueeze(1).to(self.device_0)  # degree normalization
        rel_flow.block_compute(block_id=0, message_func=self.msg_rel, reduce_func=self.reduce_rel)
        e = rel_flow.layers[1].data['reduced'] * self.lambda_2 + edge_emb[edge_index[len0:]] * (
                    1 - self.lambda_2)  # 公式(3)

        return n, e[rel_seed_idx]


    def forward_SSL(self, h_g, f_g, SSL_g, Glob_g, seed_nodes, relation_batch, neighbor_batch_size, batch_size, seed_idx,h_ratio,f_ratio):
        """
            g：时间分割图 Train_time_split_Graph
            Glob_g：全局静态图 Train_Global_Graph
            SSL_g：window_graph，用于SSL任务
            seed_nodes：种子节点，用于采样
                包括去重后的 头正样本、尾实体正样本、SSL任务的头实体负样本、SSL任务的尾实体负样本
            seed_idx：种子节点索引
                头正样本（batch_size个）、尾实体正样本（batch_size个）、SSL任务的头实体负样本（batch_size * neg_num个）、SSL任务的尾实体负样本（batch_size * neg_num个）
            neighbor_batch_size：邻居采样的扩展因子，即每个节点抽样的邻居数量
        """
        h_g.readonly(True)
        f_g.readonly(True)
        SSL_g.readonly(True)
        seed_node_batch_size = len(seed_nodes)
        # 前 2 * batch_size 个节点，即 【头、尾实体的正样本】
        SSL_seed, SSL_seed_idx = torch.unique(seed_nodes[seed_idx[:batch_size * 2]], sorted=False, return_inverse=True)

        """ SSL任务采样：获得种子节点 e[t] ∈ SSL_send 在不同时间的表示：e[t-w], ..., e[t-1], e[t+1], ..., e[t+w] """
        # 进行1阶邻居采样（所有一阶邻居），根据提供的种子节点SSL_seed生成子图。
        SSL_batch = dgl.contrib.sampling.NeighborSampler(SSL_g,
                                                         batch_size=seed_node_batch_size,
                                                         # 从seed_nodes数组中取出用于邻居采样的种子节点数量
                                                         expand_factor=neighbor_batch_size,
                                                         # 从每个种子节点出发时邻居的最大采样数量，参数为num_of_ent，等于所有1阶邻居
                                                         neighbor_type='in',  # 采样入边邻居
                                                         shuffle=False,
                                                         num_hops=1,  # 1跳
                                                         seed_nodes=SSL_seed,  # 用采样的种子节点
                                                         add_self_loop=False)
        # 获得生成器的数据
        for SSL_flow in SSL_batch:
            break
        # 创建GCN采样的种子节点
        # GCN_seed：S、O的正样本实体（2*batch_szie个） + SSL任务的S、O负样本实体 + S、O在其他时间戳的实体
        GCN_seed, GCN_seed_idx = torch.unique(torch.cat([seed_nodes,  # len = batch_size
                                                         SSL_flow.layer_parent_nid(1),  # len = len(SSL_seed)
                                                         SSL_flow.layer_parent_nid(0)]),
                                              sorted=False, return_inverse=True)
        GCN_seed_batch_size = len(GCN_seed)

        """ Historical 相邻时间快照采样：用于获得实体的时间嵌入 """
        # 采样两跳邻居
        GCN_batch = dgl.contrib.sampling.NeighborSampler(h_g,
                                                         batch_size=GCN_seed_batch_size,
                                                         expand_factor=neighbor_batch_size,
                                                         neighbor_type='in',
                                                         shuffle=False,
                                                         num_hops=2,
                                                         seed_nodes=GCN_seed,
                                                         add_self_loop=False)
        for node_flow in GCN_batch:
            break
        node_flow.copy_from_parent()  # 把原始图中的特征复制到node_flow中
        node_unique, node_index = torch.unique(torch.cat([node_flow.layers[0].data['node_idx'],
                                                          node_flow.layers[1].data['node_idx'],
                                                          node_flow.layers[2].data['node_idx']]), sorted=False,
                                               return_inverse=True)
        node_emb = self.node_embedding_layer(node_unique.to(self.device_0))  # 获得实体嵌入
        len0 = len(node_flow.layers[0].data['node_idx'])  # 第2跳节点数量 = Layer 0的结点数
        len1 = len(node_flow.layers[1].data['node_idx'])  # 第1跳节点数量 = Layer 1的结点数
        rel_unique, rel_index = torch.unique(torch.cat([node_flow.blocks[0].data['relation_idx'],
                                                        node_flow.blocks[1].data['relation_idx']]), sorted=False,
                                             return_inverse=True)
        len2 = len(node_flow.blocks[0].data['relation_idx'])  # 第2跳关系数量 = Block 0的边数
        self.rel_emb = self.edge_embedding_layer(rel_unique.to(self.device_0)).detach()
        node_flow.layers[0].data['node_emb'] = node_emb[node_index[:len0]]  # 初始化layer 0的节点嵌入
        node_flow.blocks[0].data['unique_idx'] = rel_index[:len2]
        node_flow.layers[0].data['out_degree_sqrt'] = torch.sqrt(
            node_flow.layer_out_degree(0).type(torch.FloatTensor)).unsqueeze(1).to(
            self.device_0)  # degree normalization
        node_flow.layers[1].data['in_degree_sqrt'] = torch.sqrt(
            node_flow.layer_in_degree(1).type(torch.FloatTensor)).unsqueeze(1).to(self.device_0)  # degree normalization
        # time sensitive
        historical_weight_clamped = torch.clamp(self.historical_window_size_parameters, 0, 1)
        self.dynamic_window = 1 / (1 + torch.exp(
            node_flow.blocks[0].data['window_size'].to(self.device_0) - self.base_window_size * historical_weight_clamped[
                node_flow.blocks[0].data['relation_idx'].cpu().tolist()].squeeze()))

        node_flow.block_compute(block_id=0, message_func=self.msg_dynamic_window_GCN, reduce_func=self.reduce_GCN)  # 第0-1层消息传递、聚合
        node_flow.layers[1].data['node_emb'] = node_flow.layers[1].data['reduced'] + node_emb[
            node_index[len0:len0 + len1]]  # 更新layer 1的节点嵌入
        node_flow.blocks[1].data['unique_idx'] = rel_index[len2:]
        node_flow.layers[1].data['out_degree_sqrt'] = torch.sqrt(
            node_flow.layer_out_degree(1).type(torch.FloatTensor)).unsqueeze(1).to(
            self.device_0)  # degree normalization
        node_flow.layers[2].data['in_degree_sqrt'] = torch.sqrt(
            node_flow.layer_in_degree(2).type(torch.FloatTensor)).unsqueeze(1).to(self.device_0)  # degree normalization
        self.dynamic_window = 1 / (1 + torch.exp(node_flow.blocks[1].data['window_size'].to(self.device_0) -
                                                 self.base_window_size*
                                                 historical_weight_clamped[node_flow.blocks[1].data['relation_idx'].cpu().tolist()].squeeze()))
        node_flow.block_compute(block_id=1, message_func=self.msg_dynamic_window_GCN, reduce_func=self.reduce_GCN)  # 第1-2层消息传递、聚合
        h_n = node_flow.layers[2].data['reduced'] + node_emb[node_index[len0 + len1:]]  # 更新layer 2的节点嵌入

        """ Future 相邻时间快照采样：用于获得实体的时间嵌入 """
        # 采样两跳邻居
        GCN_batch = dgl.contrib.sampling.NeighborSampler(f_g,
                                                         batch_size=GCN_seed_batch_size,
                                                         expand_factor=neighbor_batch_size,
                                                         neighbor_type='in',
                                                         shuffle=False,
                                                         num_hops=2,
                                                         seed_nodes=GCN_seed,
                                                         add_self_loop=False)
        for node_flow in GCN_batch:
            break
        node_flow.copy_from_parent()  # 把原始图中的特征复制到node_flow中
        node_unique, node_index = torch.unique(torch.cat([node_flow.layers[0].data['node_idx'],
                                                          node_flow.layers[1].data['node_idx'],
                                                          node_flow.layers[2].data['node_idx']]), sorted=False,
                                               return_inverse=True)
        node_emb = self.node_embedding_layer(node_unique.to(self.device_0))  # 获得实体嵌入
        len0 = len(node_flow.layers[0].data['node_idx'])  # 第2跳节点数量 = Layer 0的结点数
        len1 = len(node_flow.layers[1].data['node_idx'])  # 第1跳节点数量 = Layer 1的结点数
        rel_unique, rel_index = torch.unique(torch.cat([node_flow.blocks[0].data['relation_idx'],
                                                        node_flow.blocks[1].data['relation_idx']]), sorted=False,
                                             return_inverse=True)
        len2 = len(node_flow.blocks[0].data['relation_idx'])  # 第2跳关系数量 = Block 0的边数
        self.rel_emb = self.edge_embedding_layer(rel_unique.to(self.device_0)).detach()
        node_flow.layers[0].data['node_emb'] = node_emb[node_index[:len0]]  # 初始化layer 0的节点嵌入
        node_flow.blocks[0].data['unique_idx'] = rel_index[:len2]
        node_flow.layers[0].data['out_degree_sqrt'] = torch.sqrt(
            node_flow.layer_out_degree(0).type(torch.FloatTensor)).unsqueeze(1).to(
            self.device_0)  # degree normalization
        node_flow.layers[1].data['in_degree_sqrt'] = torch.sqrt(
            node_flow.layer_in_degree(1).type(torch.FloatTensor)).unsqueeze(1).to(self.device_0)  # degree normalization

        # time sensitive
        future_weight_clamped = torch.clamp(self.future_window_size_parameters, 0, 1)
        self.dynamic_window = 1 / (1 + torch.exp(
            node_flow.blocks[0].data['window_size'].to(self.device_0) - self.base_window_size * future_weight_clamped[
                node_flow.blocks[0].data['relation_idx'].cpu().tolist()].squeeze()))

        node_flow.block_compute(block_id=0, message_func=self.msg_dynamic_window_GCN, reduce_func=self.reduce_GCN)  # 第0-1层消息传递、聚合
        node_flow.layers[1].data['node_emb'] = node_flow.layers[1].data['reduced'] + node_emb[
            node_index[len0:len0 + len1]]  # 更新layer 1的节点嵌入
        node_flow.blocks[1].data['unique_idx'] = rel_index[len2:]
        node_flow.layers[1].data['out_degree_sqrt'] = torch.sqrt(
            node_flow.layer_out_degree(1).type(torch.FloatTensor)).unsqueeze(1).to(
            self.device_0)  # degree normalization
        node_flow.layers[2].data['in_degree_sqrt'] = torch.sqrt(
            node_flow.layer_in_degree(2).type(torch.FloatTensor)).unsqueeze(1).to(self.device_0)  # degree normalization

        self.dynamic_window = 1 / (1 + torch.exp(node_flow.blocks[1].data['window_size'].to(self.device_0) -
                                                 self.base_window_size*
                                                 future_weight_clamped[node_flow.blocks[1].data['relation_idx'].cpu().tolist()].squeeze()))

        node_flow.block_compute(block_id=1, message_func=self.msg_dynamic_window_GCN, reduce_func=self.reduce_GCN)  # 第1-2层消息传递、聚合
        f_n = node_flow.layers[2].data['reduced'] + node_emb[node_index[len0 + len1:]]  # 更新layer 2的节点嵌入
        
        t_n = (h_ratio*h_n) + (f_ratio*f_n)

        SSL_flow.layers[0].data['node_emb'] = t_n[GCN_seed_idx[seed_node_batch_size + len(SSL_seed):]]  # 初始化第一跳的节点嵌入
        SSL_flow.layers[1].data['node_emb'] = t_n[GCN_seed_idx[seed_node_batch_size: seed_node_batch_size + len(SSL_seed)]]  # 初始化种子节点的节点嵌入
        SSL_flow.block_compute(block_id=0, message_func=self.msg_SSL, reduce_func=self.reduce_SSL)

        temp_entity_embs = t_n[GCN_seed_idx[:seed_node_batch_size]]
        pos_sim = SSL_flow.layers[1].data['sim'][SSL_seed_idx]

        return temp_entity_embs, pos_sim


    def msg_SSL(self, edges):
        return {'window': edges.src['node_emb']}

    def reduce_SSL(self, nodes):
        """ 公式(9)：实体在t时的嵌入 与 其在相邻时间嵌入 的相似度 """
        shape = torch.tensor([nodes.batch_size(), 10])
        # nodes.mailbox['window'].shape = (batch_size , num_of_src, emb_dim)
        # nodes.data['node_emb'].shape = (batch_size, emb_dim)
        # similarity.shape = (batch_size, num_of_src)
        similarity = torch.bmm(nodes.mailbox['window'], (nodes.data['node_emb']).unsqueeze(2)).squeeze(
            2)  # batch matrix multiplication
        # 在similarity的第二个维度后面，填充 k = 10-num_of_src 个 -inf  (若k小于0则删除)
        # 最后的similarity.shape = (batch_size, 10)
        return {'sim': F.pad(similarity, (0, (shape - torch.tensor(similarity.shape))[1], 0, 0), value=float("-inf"))}

    def msg_GCN(self, edges):  # out degree
        """ 公式(1)：消息传递 """
        return {'m': (edges.src['node_emb'] * self.rel_emb[edges.data['unique_idx']]) / (
                    edges.src['out_degree_sqrt'] * edges.dst['in_degree_sqrt'])}

    def msg_dynamic_window_GCN(self, edges):  # out degree
        """ 公式(1)：消息传递 """
        A = (edges.src['node_emb'] * self.rel_emb[edges.data['unique_idx']])
        A *= self.dynamic_window[:, None]
        return {'m': A / (edges.src['out_degree_sqrt'] * edges.dst['in_degree_sqrt'])}
        # return {'m': (edges.src['node_emb'] * self.rel_emb[edges.data['unique_idx']] * self.dynamic_window) / (
        #             edges.src['out_degree_sqrt'] * edges.dst['in_degree_sqrt'])}

    def reduce_GCN(self, nodes):  # in degree
        """ 公式(1)：聚合 """
        return {'reduced': nodes.mailbox['m'].sum(1)}


    def msg_Global(self, edges):  # out degree
        return {'m': (edges.src['node_emb'] * self.Glob_rel_emb[edges.data['unique_idx']]) / (
                    edges.src['out_degree_sqrt'] * edges.dst['in_degree_sqrt'])}

    def reduce_Global(self, nodes):  # in degree
        return {'reduced': nodes.mailbox['m'].sum(1)}

    def msg_rel(self, edges):
        weight = torch.clamp(self.relation_time_span[edges.src['relation_idx'], edges.dst['relation_idx']], 0, 1)
        weight = 1/(1+torch.exp(self.time_span - self.base_window_size*weight))
        A = edges.src['rel_emb']*weight[:, None]
        return {'m': A / (edges.src['out_degree_sqrt'] * edges.dst['in_degree_sqrt'])}

    def reduce_rel(self, nodes):
        return {'reduced': nodes.mailbox['m'].sum(1)}

