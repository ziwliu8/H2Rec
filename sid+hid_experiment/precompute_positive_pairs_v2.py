"""
预计算InfoNCE正样本对矩阵 - V2版本
基于实验结果优化：
1. 删除码本重叠策略（效果不佳）
2. 重点优化时间窗口策略（加权+过滤）
3. 添加PMI/PPMI加权
4. 添加时间衰减
"""
import os
import json
import pickle
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict, Counter
import argparse
import math


def load_semantic_codes(semantic_code_path):
    """加载语义代码"""
    with open(semantic_code_path, 'r') as f:
        semantic_codes = json.load(f)
    print(f"[加载] 语义代码: {len(semantic_codes)} 个物品")
    n_codebooks = len(semantic_codes[1]) if len(semantic_codes) > 1 and semantic_codes[1] else 0
    print(f"  - 码本层数: {n_codebooks}")
    return semantic_codes


def load_interaction_data(dataset_path):
    """加载交互数据"""
    print(f"加载交互数据: {dataset_path}")
    
    User = defaultdict(list)
    usernum = 0
    itemnum = 0
    
    with open(os.path.join(dataset_path, 'handled/inter.txt'), 'r') as f:
        for line in f:
            parts = line.rstrip().split()
            if len(parts) < 2:
                continue
            
            u = int(parts[0])
            i = int(parts[1])
            usernum = max(u, usernum)
            itemnum = max(i, itemnum)
            User[u].append(i)
    
    print(f"  用户数: {usernum}")
    print(f"  物品数: {itemnum}")
    print(f"  交互数: {sum(len(v) for v in User.values())}")
    
    return User, usernum, itemnum


def compute_exact_codebook_matches(semantic_codes, max_pairs_per_item=100):
    """
    找出所有码字完全相同的物品对（语义完全一致）
    这些是高质量的正样本，应该作为额外的正样本添加到行为正样本中
    
    Args:
        semantic_codes: 语义代码列表
        max_pairs_per_item: 每个物品最大正样本数
    
    Returns:
        exact_match_pairs: {item_id: [exact_match_item_ids]}
        exact_match_weights: {item_id: {match_id: weight}}
    """
    print(f"\n计算完全码字重合的物品对...")
    print(f"  最大正样本数: {max_pairs_per_item}")
    
    # 构建码字到物品的映射
    code_to_items = defaultdict(list)
    
    n_items = len(semantic_codes)
    for item_id in tqdm(range(n_items), desc="构建码字索引"):
        if semantic_codes[item_id]:
            # 将码字列表转换为tuple作为key
            code_tuple = tuple(semantic_codes[item_id])
            code_to_items[code_tuple].append(item_id)
    
    # 找出每个码字对应的所有物品（完全匹配）
    exact_match_pairs = {}
    exact_match_weights = {}
    total_pairs = 0
    collision_groups = 0
    
    for code_tuple, items in code_to_items.items():
        if len(items) > 1:
            collision_groups += 1
            # 这些物品的码字完全相同
            for item_i in items:
                # 将其他相同码字的物品作为正样本
                matches = [item_j for item_j in items if item_j != item_i]
                
                # 限制数量
                if len(matches) > max_pairs_per_item:
                    matches = np.random.choice(matches, max_pairs_per_item, replace=False).tolist()
                
                if matches:
                    exact_match_pairs[item_i] = matches
                    # 完全匹配的权重设为最高（1.0）
                    exact_match_weights[item_i] = {match_id: 1.0 for match_id in matches}
                    total_pairs += len(matches)
    
    print(f"  找到 {collision_groups} 个码字组（多个物品共享同一码字）")
    print(f"  涉及 {len(exact_match_pairs)} 个物品")
    print(f"  总正样本对数: {total_pairs}")
    avg_per_group = total_pairs / collision_groups if collision_groups > 0 else 0
    print(f"  平均每组: {avg_per_group:.2f} 个物品")
    
    return exact_match_pairs, exact_match_weights


def compute_item_popularity(User, itemnum):
    """计算物品流行度"""
    popularity = Counter()
    for user_items in User.values():
        for item in user_items:
            popularity[item] += 1
    
    # 归一化
    total = sum(popularity.values())
    popularity = {k: v/total for k, v in popularity.items()}
    
    return popularity


def compute_pmi_weighted_pairs(User, window_size=3, min_cooccur=2, 
                                use_ppmi=True, max_pairs_per_item=50):
    """
    计算基于PMI/PPMI加权的窗口正样本对
    
    Args:
        User: 用户交互字典
        window_size: 时间窗口大小
        min_cooccur: 最小共现次数（质量过滤）
        use_ppmi: 是否使用PPMI（Positive PMI，截断负值）
        max_pairs_per_item: 每个物品最大正样本数
    """
    print(f"\n计算PMI加权正样本对...")
    print(f"  窗口大小: {window_size}")
    print(f"  最小共现次数: {min_cooccur}")
    print(f"  使用PPMI: {use_ppmi}")
    
    # 统计共现次数和单独出现次数
    co_occurrence = defaultdict(lambda: defaultdict(int))
    item_counts = defaultdict(int)
    total_pairs = 0
    
    for user_id, items in tqdm(User.items(), desc="统计共现"):
        if len(items) < 2:
            continue
        
        # 滑动窗口
        for i in range(len(items)):
            item_i = items[i]
            item_counts[item_i] += 1
            
            window_start = max(0, i - window_size)
            window_end = min(len(items), i + window_size + 1)
            
            for j in range(window_start, window_end):
                if i != j:
                    item_j = items[j]
                    co_occurrence[item_i][item_j] += 1
                    total_pairs += 1
    
    # 总的窗口对数
    N = total_pairs
    
    # 计算PMI/PPMI并构建正样本对
    positive_pairs = {}
    positive_weights = {}
    
    print("  计算PMI权重...")
    for item_i, co_items in tqdm(co_occurrence.items()):
        pairs = []
        weights = []
        
        for item_j, count_ij in co_items.items():
            # 质量过滤：至少共现min_cooccur次
            if count_ij < min_cooccur:
                continue
            
            # 计算PMI: log(P(i,j) / (P(i) * P(j)))
            # P(i,j) = count(i,j) / N
            # P(i) = count(i) / N
            p_ij = count_ij / N
            p_i = item_counts[item_i] / N
            p_j = item_counts[item_j] / N
            
            pmi = math.log(p_ij / (p_i * p_j) + 1e-10)
            
            # PPMI：截断负值
            if use_ppmi:
                pmi = max(0, pmi)
            
            if pmi > 0:  # 只保留正相关的
                pairs.append(item_j)
                weights.append(pmi)
        
        if pairs:
            # 限制数量：保留权重最高的top-k
            if len(pairs) > max_pairs_per_item:
                # 按权重排序
                sorted_indices = np.argsort(weights)[::-1][:max_pairs_per_item]
                pairs = [pairs[i] for i in sorted_indices]
                weights = [weights[i] for i in sorted_indices]
            
            # 归一化权重到[0.3, 1.0]
            weights = np.array(weights)
            if weights.max() > weights.min():
                weights = (weights - weights.min()) / (weights.max() - weights.min())
                weights = 0.3 + 0.7 * weights
            else:
                weights = np.ones_like(weights) * 0.7
            
            positive_pairs[item_i] = pairs
            positive_weights[item_i] = {item_j: w for item_j, w in zip(pairs, weights)}
    
    total_pairs = sum(len(v) for v in positive_pairs.values())
    print(f"  找到 {len(positive_pairs)} 个物品有相似物品")
    print(f"  总正样本对数: {total_pairs}")
    print(f"  平均每物品: {total_pairs/len(positive_pairs):.2f}")
    
    return positive_pairs, positive_weights


def compute_time_decayed_pairs(User, window_size=3, decay_factor=0.8, 
                                 min_weight=0.3, max_pairs_per_item=50):
    """
    计算带时间衰减的正样本对
    距离anchor越近，权重越高
    
    Args:
        User: 用户交互字典
        window_size: 时间窗口大小
        decay_factor: 衰减因子（0-1）
        min_weight: 最小权重
        max_pairs_per_item: 每个物品最大正样本数
    """
    print(f"\n计算时间衰减正样本对...")
    print(f"  窗口大小: {window_size}")
    print(f"  衰减因子: {decay_factor}")
    
    # 统计带权重的共现
    weighted_co_occurrence = defaultdict(lambda: defaultdict(list))
    
    for user_id, items in tqdm(User.items(), desc="计算时间衰减"):
        if len(items) < 2:
            continue
        
        for i in range(len(items)):
            item_i = items[i]
            
            window_start = max(0, i - window_size)
            window_end = min(len(items), i + window_size + 1)
            
            for j in range(window_start, window_end):
                if i != j:
                    item_j = items[j]
                    
                    # 计算距离权重：距离越近权重越高
                    distance = abs(i - j)
                    weight = decay_factor ** (distance - 1)
                    weight = max(min_weight, weight)
                    
                    weighted_co_occurrence[item_i][item_j].append(weight)
    
    # 聚合权重：取平均
    positive_pairs = {}
    positive_weights = {}
    
    for item_i, co_items in weighted_co_occurrence.items():
        pairs = []
        weights = []
        
        for item_j, weight_list in co_items.items():
            avg_weight = np.mean(weight_list)
            pairs.append(item_j)
            weights.append(avg_weight)
        
        # 限制数量
        if len(pairs) > max_pairs_per_item:
            sorted_indices = np.argsort(weights)[::-1][:max_pairs_per_item]
            pairs = [pairs[i] for i in sorted_indices]
            weights = [weights[i] for i in sorted_indices]
        
        positive_pairs[item_i] = pairs
        positive_weights[item_i] = {item_j: w for item_j, w in zip(pairs, weights)}
    
    total_pairs = sum(len(v) for v in positive_pairs.values())
    print(f"  找到 {len(positive_pairs)} 个物品有相似物品")
    print(f"  总正样本对数: {total_pairs}")
    
    return positive_pairs, positive_weights


def filter_by_popularity(positive_pairs, positive_weights, popularity, 
                          popularity_threshold=0.01, downsample_ratio=0.5):
    """
    过滤热门物品，减少流行度偏置
    
    Args:
        positive_pairs: 正样本对字典
        positive_weights: 正样本权重字典
        popularity: 物品流行度字典
        popularity_threshold: 热门物品阈值
        downsample_ratio: 热门物品降采样比例
    """
    print(f"\n过滤热门物品...")
    print(f"  流行度阈值: {popularity_threshold}")
    print(f"  降采样比例: {downsample_ratio}")
    
    filtered_pairs = {}
    filtered_weights = {}
    
    for item_i, pairs in positive_pairs.items():
        new_pairs = []
        new_weights = {}
        
        for item_j in pairs:
            # 热门物品降采样
            if popularity.get(item_j, 0) > popularity_threshold:
                if np.random.rand() > downsample_ratio:
                    continue
            
            new_pairs.append(item_j)
            if item_i in positive_weights:
                new_weights[item_j] = positive_weights[item_i].get(item_j, 0.5)
        
        if new_pairs:
            filtered_pairs[item_i] = new_pairs
            filtered_weights[item_i] = new_weights
    
    old_total = sum(len(v) for v in positive_pairs.values())
    new_total = sum(len(v) for v in filtered_pairs.values())
    print(f"  过滤前: {old_total} 对")
    print(f"  过滤后: {new_total} 对")
    print(f"  过滤比例: {(old_total - new_total) / old_total * 100:.1f}%")
    
    return filtered_pairs, filtered_weights


def save_positive_pairs(positive_pairs, positive_weights, save_path):
    """保存正样本对和权重"""
    print(f"\n保存到: {save_path}")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 保存正样本对
    with open(save_path, 'wb') as f:
        pickle.dump(positive_pairs, f)
    
    # 保存权重
    weights_path = save_path.replace('.pkl', '_weights.pkl')
    with open(weights_path, 'wb') as f:
        pickle.dump(positive_weights, f)
    
    print(f"  正样本对: {save_path}")
    print(f"  权重文件: {weights_path}")
    print("保存完成！")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='yelp')
    parser.add_argument('--semantic_code_file', type=str,
                        default='yelp.code.rqvae_orig.4_128.emb1536.Cosine.recon.pca0.json',
                        help='语义代码文件')
    parser.add_argument('--window_size', type=int, default=3,
                        help='时间窗口大小')
    parser.add_argument('--min_cooccur', type=int, default=2,
                        help='最小共现次数（质量过滤）')
    parser.add_argument('--max_pairs_per_item', type=int, default=50,
                        help='每个物品最大正样本数')
    parser.add_argument('--use_pmi', action='store_true',
                        help='使用PMI/PPMI加权')
    parser.add_argument('--use_time_decay', action='store_true',
                        help='使用时间衰减')
    parser.add_argument('--decay_factor', type=float, default=0.8,
                        help='时间衰减因子')
    parser.add_argument('--filter_popular', action='store_true',
                        help='过滤热门物品')
    parser.add_argument('--popularity_threshold', type=float, default=0.01,
                        help='热门物品阈值')
    parser.add_argument('--add_exact_matches', action='store_true',
                        help='添加完全码字重合的物品作为高质量正样本（强烈推荐）')
    parser.add_argument('--output_file', type=str, default=None)
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("预计算InfoNCE正样本对矩阵 - V2")
    print("=" * 60)
    print(f"数据集: {args.dataset}")
    print(f"窗口大小: {args.window_size}")
    print(f"使用PMI加权: {args.use_pmi}")
    print(f"使用时间衰减: {args.use_time_decay}")
    print(f"过滤热门物品: {args.filter_popular}")
    print(f"添加完全码字匹配: {args.add_exact_matches}")
    print("=" * 60)
    
    # 路径设置
    dataset_path = f'./data/{args.dataset}'
    
    # 自动生成输出文件名
    if args.output_file is None:
        filename_parts = [args.dataset, 'positive_pairs_v2']
        filename_parts.append(f'window{args.window_size}')
        if args.use_pmi:
            filename_parts.append('pmi')
        if args.use_time_decay:
            filename_parts.append(f'decay{args.decay_factor}')
        if args.filter_popular:
            filename_parts.append('filtered')
        if args.add_exact_matches:
            filename_parts.append('exactmatch')
        output_file = '.'.join(filename_parts) + '.pkl'
        output_path = os.path.join(dataset_path, 'handled', output_file)
    else:
        output_path = args.output_file
    
    print(f"输出路径: {output_path}\n")
    
    # 1. 加载数据
    User, usernum, itemnum = load_interaction_data(dataset_path)
    
    # 加载语义代码（如果需要完全匹配）
    semantic_codes = None
    if args.add_exact_matches:
        semantic_code_path = os.path.join(dataset_path, 'handled', args.semantic_code_file)
        semantic_codes = load_semantic_codes(semantic_code_path)
    
    # 2. 计算物品流行度
    popularity = compute_item_popularity(User, itemnum)
    
    # 3. 计算正样本对（根据配置选择策略）
    if args.use_pmi:
        positive_pairs, positive_weights = compute_pmi_weighted_pairs(
            User,
            window_size=args.window_size,
            min_cooccur=args.min_cooccur,
            use_ppmi=True,
            max_pairs_per_item=args.max_pairs_per_item
        )
    elif args.use_time_decay:
        positive_pairs, positive_weights = compute_time_decayed_pairs(
            User,
            window_size=args.window_size,
            decay_factor=args.decay_factor,
            max_pairs_per_item=args.max_pairs_per_item
        )
    else:
        # 默认：简单窗口共现（等权）
        print("\n使用默认窗口共现策略（等权）")
        from precompute_positive_pairs import compute_consecutive_interaction_pairs
        positive_pairs = compute_consecutive_interaction_pairs(User, args.window_size)
        positive_weights = {k: {j: 0.5 for j in v} for k, v in positive_pairs.items()}
    
    # 4. 添加完全码字匹配的正样本（可选但强烈推荐）
    if args.add_exact_matches and semantic_codes is not None:
        exact_pairs, exact_weights = compute_exact_codebook_matches(
            semantic_codes,
            max_pairs_per_item=args.max_pairs_per_item
        )
        
        # 合并：完全匹配的物品有最高权重
        print(f"\n合并行为正样本和完全码字匹配正样本...")
        print(f"  合并前行为正样本: {sum(len(v) for v in positive_pairs.values())} 对")
        print(f"  完全匹配正样本: {sum(len(v) for v in exact_pairs.values())} 对")
        
        all_items = set(positive_pairs.keys()) | set(exact_pairs.keys())
        for item_id in all_items:
            # 行为正样本
            behavior_set = set(positive_pairs.get(item_id, []))
            behavior_weights = positive_weights.get(item_id, {})
            
            # 完全匹配正样本
            exact_set = set(exact_pairs.get(item_id, []))
            exact_wts = exact_weights.get(item_id, {})
            
            # 合并：并集
            merged_set = behavior_set | exact_set
            merged_weights = {}
            
            for pos_item in merged_set:
                # 如果同时在两个集合中，取最大权重
                w1 = behavior_weights.get(pos_item, 0)
                w2 = exact_wts.get(pos_item, 0)
                merged_weights[pos_item] = max(w1, w2)
            
            # 更新
            if merged_set:
                positive_pairs[item_id] = list(merged_set)
                positive_weights[item_id] = merged_weights
        
        print(f"  合并后总正样本: {sum(len(v) for v in positive_pairs.values())} 对")
    
    # 5. 过滤热门物品（可选）
    if args.filter_popular:
        positive_pairs, positive_weights = filter_by_popularity(
            positive_pairs,
            positive_weights,
            popularity,
            popularity_threshold=args.popularity_threshold
        )
    
    # 6. 保存
    save_positive_pairs(positive_pairs, positive_weights, output_path)
    
    # 6. 统计信息
    print("\n" + "=" * 60)
    print("预计算完成！统计信息:")
    print("=" * 60)
    
    n_items = len(positive_pairs)
    n_pairs = sum(len(v) for v in positive_pairs.values())
    avg_pairs = n_pairs / n_items if n_items > 0 else 0
    
    print(f"  有正样本的物品数: {n_items}")
    print(f"  总正样本对数: {n_pairs}")
    print(f"  平均每物品正样本数: {avg_pairs:.2f}")
    
    # 权重统计
    all_weights = []
    for item_weights in positive_weights.values():
        all_weights.extend(item_weights.values())
    
    if all_weights:
        print(f"\n权重统计:")
        print(f"  最小权重: {min(all_weights):.3f}")
        print(f"  最大权重: {max(all_weights):.3f}")
        print(f"  平均权重: {np.mean(all_weights):.3f}")
        print(f"  中位数权重: {np.median(all_weights):.3f}")
    
    print("\n" + "=" * 60)
    print("使用方法：")
    print("=" * 60)
    print(f"在训练时添加参数:")
    print(f"  --positive_pairs_file {os.path.basename(output_path)}")
    print(f"  --use_positive_weights  # 启用权重矩阵")
    print("=" * 60)


if __name__ == '__main__':
    main()

