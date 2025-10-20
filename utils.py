import torch
import numpy as np
import dgl

""" 公式（4） """
def distmult(s, r, o):
    return torch.sum(s * r * o, dim=-1)

def transE(head, relation, tail):
    score = head + relation - tail
    score = - torch.norm(score, p=1, dim=-1)
    return score

def complex(head, relation, tail):
    re_head, im_head = torch.chunk(head, 2, dim=-1)
    re_relation, im_relation = torch.chunk(relation, 2, dim=-1)
    re_tail, im_tail = torch.chunk(tail, 2, dim=-1)
    re_score = re_head * re_relation - im_head * im_relation
    im_score = re_head * im_relation + im_head * re_relation
    score = re_score * re_tail + im_score * im_tail
    return score.sum(dim = -1)

def save_total_model(epoch, model, optimizer, filename):
    filename_list = list()
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    # 检测PyTorch的版本
    torch_version = torch.__version__.split('.')
    major_version = int(torch_version[0])
    minor_version = int(torch_version[1])
    if major_version > 1 or (major_version == 1 and minor_version >= 6):
        torch.save(state, filename)
        torch.save(state, 'torch1.3.0_'+filename, _use_new_zipfile_serialization=False)
        filename_list.append(filename)
        filename_list.append('torch1.3.0_'+filename)
    else:
        torch.save(state, filename)
        filename_list.append(filename)
    return filename_list

def save_total_model_2(epoch, model, optimizer_dense, optimizer_sparse, filename):
    filename_list = list()
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer_dense': optimizer_dense.state_dict(),
        'optimizer_sparse': optimizer_sparse.state_dict()
    }
    # 检测PyTorch的版本
    torch_version = torch.__version__.split('.')
    major_version = int(torch_version[0])
    minor_version = int(torch_version[1])
    if major_version > 1 or (major_version == 1 and minor_version >= 6):
        torch.save(state, filename)
        torch.save(state, 'torch1.3.0_'+filename, _use_new_zipfile_serialization=False)
        filename_list.append(filename)
        filename_list.append('torch1.3.0_'+filename)
    else:
        torch.save(state, filename)
        filename_list.append(filename)
    return filename_list

""" 公式（8） """
def self_supervised_loss(pos, neg):
    return -((torch.logsumexp(pos,1)-torch.logsumexp(torch.cat([pos, neg],1),1)).sum())

def self_supervised_loss_no_neg(pos):
    return (2-2*((torch.mean(pos,1)))).sum()

""" 公式（5）（6） """
def stabilized_log_softmax(pos, neg):
    return pos - torch.logsumexp(torch.cat([neg,pos.unsqueeze(1)],1), 1)

def stabilized_log_softmax_with_temperature(pos, neg, temp):
    return pos/temp - torch.logsumexp(torch.cat([neg,pos.unsqueeze(1)],1)/temp, 1)

def stabilized_NLL_with_temperature(positive, o_negative, s_negative, temp):
    return -(torch.sum(stabilized_log_softmax_with_temperature(positive, o_negative, temp)) + torch.sum(stabilized_log_softmax_with_temperature(positive, s_negative, temp)))

""" 公式（5）（6） """
def stabilized_NLL(positive, o_negative, s_negative):
    return -(torch.sum(stabilized_log_softmax(positive, o_negative)) + torch.sum(stabilized_log_softmax(positive, s_negative)))

""" 公式（5）（6） """
def stabilized_NLL_noTail(positive, o_negative):
    return -(torch.sum(stabilized_log_softmax(positive, o_negative)))

def binary_search(list_, key):
  low = 0
  high = len(list_)-1
  while high >= low:
    mid = (low+high)//2
    if key<list_[mid]:
      high = mid-1
    elif key == list_[mid]:
      return mid
    else:
      low = mid+1
  return low

def MRR(ranks):
    sum_ = 0
    for rank in ranks:
        sum_+= 1/rank
    return sum_ / len(ranks)

def HitsK(ranks, k):
    sum_ = 0
    for rank in ranks:
        if rank <= k:
            sum_+=1
    return (sum_/len(ranks))*100

def MRR_single(ranks):
    MRRs = []
    sum_ = 0
    for rank in ranks:
        res = 1 / rank
        sum_+= res
        MRRs.append(res)
    return sum_ / len(ranks), MRRs

def HitsK_single(ranks, k):
    sum_ = 0
    Hits = []
    for rank in ranks:
        if rank <= k:
            sum_+=1
            Hits.append(1)
        else:
            Hits.append(0)
    return (sum_/len(ranks))*100, Hits

def eval_rank(ranks):
    return MRR(ranks), HitsK(ranks, 1), HitsK(ranks, 3), HitsK(ranks,5), HitsK(ranks,10)

def eval_rank_single(ranks):
    MRR_res, MRR_list = MRR_single(ranks)
    H1_res, H1_list = HitsK_single(ranks, 1)
    H3_res, H3_list = HitsK_single(ranks, 3)
    return MRR_res, MRR_list, H1_res, H1_list, H3_res, H3_list, HitsK(ranks,5), HitsK(ranks,10)

def print_metrics(object_data_ranks, subject_data_ranks):
    mrr, h1, h3, h5, h10 = eval_rank(object_data_ranks)
    mrr_, h1_, h3_, h5_, h10_ = eval_rank(subject_data_ranks)
    print('tail_prediction: MRR',round(mrr,4),'Hits@1', round(h1,4),'Hits@3', round(h3,4),'Hits@5', round(h5,4),'Hits@10', round(h10,4))
    print('head_prediction: MRR',round(mrr_,4),'Hits@1', round(h1_,4),'Hits@3', round(h3_,4),'Hits@5', round(h5_,4),'Hits@10', round(h10_,4))
    print('average:         MRR',round((mrr+mrr_)/2,4),'Hits@1', round((h1+h1_)/2,4),'Hits@3', round((h3+h3_)/2,4),'Hits@5', round((h5+h5_)/2,4),'Hits@10', round((h10+h10_)/2,4))
    return round((mrr+mrr_)/2,4), round((h1+h1_)/2,4), round((h3+h3_)/2,4), round((h10+h10_)/2,4)

def print_metrics_single(object_data_ranks, subject_data_ranks):
    mrr, mrr_list, h1, h1_list, h3, h3_list, h5, h10 = eval_rank_single(object_data_ranks)
    mrr_, mrr_list_, h1_, h1_list_, h3_, h3_list_, h5_, h10_ = eval_rank_single(subject_data_ranks)
    print('tail_prediction: MRR',round(mrr,4),'Hits@1', round(h1,4),'Hits@3', round(h3,4),'Hits@5', round(h5,4),'Hits@10', round(h10,4))
    print('head_prediction: MRR',round(mrr_,4),'Hits@1', round(h1_,4),'Hits@3', round(h3_,4),'Hits@5', round(h5_,4),'Hits@10', round(h10_,4))
    print('average:         MRR',round((mrr+mrr_)/2,4),'Hits@1', round((h1+h1_)/2,4),'Hits@3', round((h3+h3_)/2,4),'Hits@5', round((h5+h5_)/2,4),'Hits@10', round((h10+h10_)/2,4))
    result = {'mrr_list': mrr_list, 'mrr_list_': mrr_list_, 'h1_list': h1_list, 'h1_list_': h1_list_, 'h3_list': h3_list, 'h3_list_': h3_list_}
    return round((mrr+mrr_)/2,4), round((h1+h1_)/2,4), round((h3+h3_)/2,4), round((h10+h10_)/2,4), result

def print_mrr(object_data_ranks, subject_data_ranks):
    mrr, h1, h3, h5, h10 = eval_rank(object_data_ranks)
    mrr_, h1_, h3_, h5_, h10_ = eval_rank(subject_data_ranks)
    print('tail_prediction:',round(mrr,4), 'head_prediction:',round(mrr_,4), 'average:',round((mrr+mrr_)/2,4))

""" key在list_中的排名 """
def rank(list_, key):
    try: 
        return len(list_) - list_.index(key)
    except: 
        return len(list_) - binary_search(list_,key) + 1

def print_hms(time):
    if time / 3600 > 1:
        print("{:.1f}h".format(time / 3600), end =" ")
        time %= 3600
    if time / 60 > 1:
        print("{:.1f}m".format(time / 60), end =" ")
        time %= 60
    print("{:.1f}s".format(time))

def query_satisfies_rules(head, rel, tail, time, query_rules_list, sorted_quadruples, time_split_fact_idx, window_size):
    """
        对于规则 r1 , r2 → rel
            查询 (head, r1, ?, t1) 是否存在
            查询 (?, r2, tail, t2) 是否存在
    """

    condition_tuple_rules = query_rules_list[rel]['condition_tuple_rules']
    condition_triplet_rules = query_rules_list[rel]['condition_triplet_rules']

    satisfies_tuple_rules = list()
    satisfies_triplet_rules = list()

    start_time = max(0, time - window_size)
    end_time = time
    window_quadruples = sorted_quadruples[time_split_fact_idx[start_time]: time_split_fact_idx[end_time + 1]]

    # 二元组规则
    for query_rel1, _ in condition_tuple_rules:
        query_window_quadruples = [p for s,p,o,_ in window_quadruples if (s == head and o == tail and p==query_rel1)]
        for rel_1 in query_window_quadruples:
            satisfies_tuple_rules.append((rel_1, rel))

    # 三元组规则
    for query_rel_1, query_rel_2, _ in condition_triplet_rules:
        # (head, r1, ?, t1)
        query_quadruples_1 = [(s, p, o, t) for s, p, o, t in window_quadruples if (s==head and p==query_rel_1)]
        for head_1, rel_1, tail_1, time_1 in query_quadruples_1:
            window_quadruples_2 = window_quadruples = sorted_quadruples[time_split_fact_idx[time_1 - 1]: time_split_fact_idx[end_time + 1]]
            # (?, r2, tail, t2)
            query_quadruples_2 = [(s, p, o, t) for s, p, o, t in window_quadruples_2 if (s == tail_1 and o==tail and p==query_rel_2)]
            for _, rel_2, _, _ in query_quadruples_2:
                satisfies_triplet_rules.append((rel_1, rel_2, rel))

    return satisfies_tuple_rules, satisfies_triplet_rules,

def top_k_unmasked_elements(score_list, mask_list, k):
    """
    返回score_list中未被mask_list过滤的top K元素及其原始位置索引

    参数:
    score_list (list): 包含评分数据的列表
    mask_list (list): 包含布尔值的列表，指示哪些元素被过滤
    k (int): 需要返回的前K个元素

    返回:
    top_k_elements (list): 包含前K个元素的列表
    top_k_indices (list): 包含前K个元素原始位置索引的列表
    """
    if len(score_list) != len(mask_list):
        raise ValueError("score_list和mask_list的长度必须相同")

    # 过滤掉被mask的元素
    unmasked_elements = [(score, idx) for idx, (score, mask) in enumerate(zip(score_list, mask_list)) if mask]

    # 按照score降序排序，并获取前K个元素
    unmasked_elements.sort(key=lambda x: x[0], reverse=False)   # 最小的K个
    top_k_elements = [elem[0] for elem in unmasked_elements[:k]]
    top_k_indices = [elem[1] for elem in unmasked_elements[:k]]

    return top_k_elements, top_k_indices

def sort_unmaksed_score_with_indices(entities_score, entities_filter_mask):
    # 假设我们已经有 objects_score 和 o_filter_mask
    entities_score = entities_score.cpu().numpy()
    entities_filter_mask = np.array(entities_filter_mask)

    # 将o_filter_mask为False的位置的objects_score值设置为负无穷小
    entities_score[~entities_filter_mask] = -np.inf

    # 获取objects_score按升序排序的索引
    sorted_indices = np.argsort(entities_score)

    # 按照排序后的索引重新排列objects_score，将负无穷小的值放在最前
    sorted_score = entities_score[sorted_indices]

    # 显示排序后的objects_score和原顺序索引
    return sorted_indices, sorted_score.tolist()

def cycle_socre(cycle_query_list, train_and_augment_data_query, s, r, o, t):
    std_e_threshold = 5
    count_threshold = 8
    gap_threshold = 4

    hit_right = False
    hit_left = False
    cycle_score = -1
    if (s, r, o) in cycle_query_list:
        fre = cycle_query_list[(s, r, o)]['cycle']
        std_e = cycle_query_list[(s, r, o)]['std_e']
        count = cycle_query_list[(s, r, o)]['count']
        if std_e <= std_e_threshold and count >= count_threshold:  # todo: 超参数
            if (s, r, o) in train_and_augment_data_query:
                if t - fre >= 0:  # 上一个周期
                    for t_ in train_and_augment_data_query[(s, r, o)]:
                        if t_ <= t:  # 在当前时间戳之前
                            gap = max(t - fre, t_) - min(t - fre, t_)  # 训练集+增强数据中，存在距离上一个周期4个时间窗口内的数据
                            if gap <= gap_threshold:
                                hit_left = True
                                break
                else:
                    hit_left = True
                if t + fre < 365:  # 下一个周期
                    for t_ in train_and_augment_data_query[(s, r, o)]:
                        if t_ > t:  # 在当前时间戳之后
                            gap = max(t + fre, t_) - min(t + fre, t_)
                            if gap <= gap_threshold:
                                hit_right = True
                                break
                else:
                    hit_right = True

                if hit_left or hit_right:
                    cycle_score = (999-std_e)
    return cycle_score


""" test """

def get_historical_subgraph(graph, current_time):
    # 获取绝对时间小于或等于当前时间的边
    historical_mask = graph.edata['absolute_time'] <= current_time
    # 将布尔掩码转换为索引列表
    historical_indices = torch.nonzero(historical_mask, as_tuple=False).squeeze()
    # 使用掩码创建子图
    historical_subgraph_ = graph.edge_subgraph(historical_indices, preserve_nodes=True)
    historical_subgraph_.copy_from_parent()
    # 再次筛选：考虑绝对扩散时间
    diffusion_mask = historical_subgraph_.edata['absolute_diffusion_time'] <= current_time
    diffusion_indices = torch.nonzero(diffusion_mask, as_tuple=False).squeeze()
    historical_subgraph = historical_subgraph_.edge_subgraph(diffusion_indices, preserve_nodes=True)
    historical_subgraph.copy_from_parent()
    return historical_subgraph


def get_future_subgraph(graph, current_time):
    # 获取绝对时间小于或等于当前时间的边
    future_mask = graph.edata['absolute_time'] >= current_time
    # 将布尔掩码转换为索引列表
    future_indices = torch.nonzero(future_mask, as_tuple=False).squeeze()
    # 使用掩码创建子图
    future_subgraph_ = graph.edge_subgraph(future_indices, preserve_nodes=True)
    future_subgraph_.copy_from_parent()
    # 再次筛选：考虑绝对扩散时间
    diffusion_mask = future_subgraph_.edata['absolute_diffusion_time'] >= current_time
    diffusion_indices = torch.nonzero(diffusion_mask, as_tuple=False).squeeze()
    future_subgraph = future_subgraph_.edge_subgraph(diffusion_indices, preserve_nodes=True)
    future_subgraph.copy_from_parent()
    return future_subgraph


def _idx_from_mask(mask):
    """安全地从布尔掩码获取一维索引张量。"""
    return torch.nonzero(mask, as_tuple=True)[0].long()

def _sample_indices(idxs, keep_ratio=None, max_keep=None, generator=None, weights=None):
    """
    从索引张量idxs中抽取一个子集。

    参数:
    - keep_ratio: 按比例抽取 (0到1)。
    - max_keep: 保留的最大数量。
    - weights: 用于无放回加权抽样的权重。
    - generator: 用于复现的torch随机数生成器。

    返回:
    - 选中的idxs子集 (1D LongTensor)。
    """
    n = idxs.numel()
    if n == 0:
        return idxs

    # 确定最终要保留的数量 k
    k = n
    if keep_ratio is not None and 0.0 <= keep_ratio < 1.0:
        k = max(0, min(n, int(round(n * float(keep_ratio)))))
    if max_keep is not None:
        k = min(k, int(max_keep))
    
    if k >= n: # 如果要保留全部或更多，直接返回原集合
        return idxs
    if k <= 0: # 如果不保留任何元素，返回空集
        return idxs.new_empty((0,), dtype=idxs.dtype)

    # 权重抽样 (无放回)
    if weights is not None:
        w = torch.as_tensor(weights, dtype=torch.float32, device=idxs.device)
        if w.numel() != n:
            raise ValueError("weights的长度必须与idxs的长度相等")
        # 使用Gumbel-Max技巧进行高效的无放回加权抽样
        g = -torch.log(-torch.log(torch.rand_like(w, generator=generator).clamp_(min=1e-12, max=1-1e-12)))
        score = torch.log(w.clamp_min(1e-12)) + g
        sel = score.topk(k, largest=True).indices
        return idxs[sel]

    # 随机无放回抽样
    perm = torch.randperm(n, generator=generator, device=idxs.device)
    return idxs[perm[:k]]

def get_historical_subgraph_partial(
    graph, current_time,
    keep_valid_ratio=1.0,        # [新增] [0,1]，保留有效边的比例
    max_valid_keep=None,         # [新增] 有效边保留数量的上限
    keep_invalid_ratio=0.0,      # [0,1]，保留无效边的比例
    max_invalid_keep=None,       # 无效边保留数量的上限
    seed=None,
    strategy='random'            # 'random' 或 'closest'
):
    """
    获取历史子图，并根据比例对有效和无效边进行采样。
    """
    g = graph
    gen = None
    if seed is not None:
        gen = torch.Generator(device=g.device if hasattr(g, 'device') else 'cpu')
        gen.manual_seed(int(seed))

    # 1. 筛选出候选边：事件时间戳 <= 当前时间
    primary_mask = g.edata['absolute_time'] <= current_time
    primary_eids = _idx_from_mask(primary_mask)

    # 2. 在候选边中区分有效/无效
    adt = g.edata['absolute_diffusion_time'][primary_eids]
    valid_mask_local = adt <= current_time
    valid_eids_all = primary_eids[valid_mask_local]
    invalid_eids_all = primary_eids[~valid_mask_local]

    # 3. [核心修改] 对有效边进行采样，实现“删除部分有效边”
    keep_valid = _sample_indices(
        valid_eids_all,
        keep_ratio=keep_valid_ratio,
        max_keep=max_valid_keep,
        generator=gen
    )

    # 4. 对无效边进行采样，实现“加入部分无效边”
    keep_invalid = torch.tensor([], dtype=torch.long, device=g.device)
    if keep_invalid_ratio > 0 or (max_invalid_keep is not None and max_invalid_keep > 0):
        weights = None
        if strategy == 'closest':
            dist = (adt[~valid_mask_local].float() - float(current_time)).abs()
            weights = 1.0 / (1.0 + dist)
        
        keep_invalid = _sample_indices(
            invalid_eids_all,
            keep_ratio=keep_invalid_ratio,
            max_keep=max_invalid_keep,
            generator=gen,
            weights=weights
        )

    # 5. 合并两部分保留的边，构建子图
    keep_eids = torch.unique(torch.cat([keep_valid, keep_invalid], dim=0))
    subg = g.edge_subgraph(keep_eids, preserve_nodes=True)
    
    if hasattr(subg, 'copy_from_parent'):
        subg.copy_from_parent()
        
    return subg


def get_future_subgraph_partial(
    graph, current_time,
    keep_valid_ratio=1.0,        # [新增]
    max_valid_keep=None,         # [新增]
    keep_invalid_ratio=0.0,
    max_invalid_keep=None,
    seed=None,
    strategy='random'
):
    """
    获取未来子图，并根据比例对有效和无效边进行采样。
    """
    g = graph
    gen = None
    if seed is not None:
        gen = torch.Generator(device=g.device if hasattr(g, 'device') else 'cpu')
        gen.manual_seed(int(seed))

    # 1. 筛选出候选边：事件时间戳 >= 当前时间
    primary_mask = g.edata['absolute_time'] >= current_time
    primary_eids = _idx_from_mask(primary_mask)

    # 2. 在候选边中区分有效/无效
    adt = g.edata['absolute_diffusion_time'][primary_eids]
    valid_mask_local = adt >= current_time
    valid_eids_all = primary_eids[valid_mask_local]
    invalid_eids_all = primary_eids[~valid_mask_local]

    # 3. [核心修改] 对有效边进行采样
    keep_valid = _sample_indices(
        valid_eids_all,
        keep_ratio=keep_valid_ratio,
        max_keep=max_valid_keep,
        generator=gen
    )
    
    # 4. 对无效边进行采样
    keep_invalid = torch.tensor([], dtype=torch.long, device=g.device)
    if keep_invalid_ratio > 0 or (max_invalid_keep is not None and max_invalid_keep > 0):
        weights = None
        if strategy == 'closest':
            dist = (adt[~valid_mask_local].float() - float(current_time)).abs()
            weights = 1.0 / (1.0 + dist)
            
        keep_invalid = _sample_indices(
            invalid_eids_all,
            keep_ratio=keep_invalid_ratio,
            max_keep=max_invalid_keep,
            generator=gen,
            weights=weights
        )

    # 5. 合并并构建子图
    keep_eids = torch.unique(torch.cat([keep_valid, keep_invalid], dim=0))
    subg = g.edge_subgraph(keep_eids, preserve_nodes=True)
    
    if hasattr(subg, 'copy_from_parent'):
        subg.copy_from_parent()

    return subg