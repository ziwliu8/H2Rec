# 将训练好的码本转换为item embedding查找表
import os
import sys
import torch
import pickle
import json
import numpy as np

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.RQVAEEmbedding import RQVAECodebookEmbedding


def create_item_embedding_lookup_table(
    trained_codebooks_path: str,
    semantic_code_file: str,
    output_path: str,
    embedding_dim: int = 128,
    fusion_method: str = 'sum',
    device: str = 'cpu'
):
    """
    从训练好的码本创建所有item的embedding查找表
    
    Args:
        trained_codebooks_path: 训练后的码本权重文件路径
        semantic_code_file: 语义代码文件路径
        output_path: 输出的embedding查找表路径
        embedding_dim: embedding维度
        fusion_method: 融合方法
        device: 计算设备
    """
    print(f"[CodebookConverter] 开始创建embedding查找表...")
    print(f"  - 训练码本: {trained_codebooks_path}")
    print(f"  - 语义代码: {semantic_code_file}")
    print(f"  - 输出路径: {output_path}")
    
    # 1. 创建临时的RQVAECodebookEmbedding来加载训练好的码本
    temp_rqvae = RQVAECodebookEmbedding(
        codebook_file=trained_codebooks_path,
        semantic_code_file=semantic_code_file,
        embedding_dim=embedding_dim,
        padding_idx=0,
        fusion_method=fusion_method,
        learnable=False,  # 不需要可学习，只用于计算
        device=device
    )
    
    print(f"[CodebookConverter] RQVAE embedding创建完成")
    print(f"  - 码本数量: {temp_rqvae.n_codebooks}")
    print(f"  - 物品数量: {temp_rqvae.n_items}")
    
    # 2. 计算所有item的embedding
    print(f"[CodebookConverter] 开始计算所有item的embedding...")
    
    # 创建所有item的ID tensor (从1开始，跳过padding 0)
    all_item_ids = torch.arange(1, temp_rqvae.n_items + 1, device=device)
    
    # 分批计算以避免内存溢出
    batch_size = 1000
    all_embeddings = []
    
    temp_rqvae.eval()
    with torch.no_grad():
        for i in range(0, len(all_item_ids), batch_size):
            batch_ids = all_item_ids[i:i+batch_size]
            batch_embs = temp_rqvae(batch_ids)  # [batch_size, embedding_dim]
            all_embeddings.append(batch_embs.cpu().numpy())
    
    # 3. 合并所有embedding
    item_embeddings = np.concatenate(all_embeddings, axis=0)  # [n_items, embedding_dim]
    
    print(f"[CodebookConverter] Embedding计算完成")
    print(f"  - 总item数: {item_embeddings.shape[0]}")
    print(f"  - Embedding维度: {item_embeddings.shape[1]}")
    
    # 4. 保存为查找表
    lookup_table_data = {
        'embeddings': item_embeddings,
        'n_items': temp_rqvae.n_items,
        'embedding_dim': embedding_dim,
        'fusion_method': fusion_method,
        'source_codebooks': trained_codebooks_path,
        'source_semantic_codes': semantic_code_file,
        'creation_info': {
            'n_codebooks': temp_rqvae.n_codebooks,
            'codebook_dims': temp_rqvae.codebook_dims,
            'codebook_sizes': temp_rqvae.codebook_sizes
        }
    }
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(lookup_table_data, f)
    
    print(f"[CodebookConverter] Embedding查找表已保存到: {output_path}")
    print(f"[CodebookConverter] 转换完成！")
    
    return output_path


def load_embedding_lookup_table(lookup_table_path: str):
    """加载embedding查找表"""
    with open(lookup_table_path, 'rb') as f:
        data = pickle.load(f)
    return data


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='将训练好的码本转换为embedding查找表')
    parser.add_argument('--trained_codebooks', type=str, required=True, 
                       help='训练后的码本权重文件路径')
    parser.add_argument('--semantic_codes', type=str, required=True,
                       help='语义代码文件路径')
    parser.add_argument('--output', type=str, required=True,
                       help='输出的embedding查找表路径')
    parser.add_argument('--embedding_dim', type=int, default=128,
                       help='embedding维度')
    parser.add_argument('--fusion_method', type=str, default='sum',
                       choices=['sum', 'mean', 'concat'],
                       help='码本融合方法')
    parser.add_argument('--device', type=str, default='cpu',
                       help='计算设备')
    
    args = parser.parse_args()
    
    create_item_embedding_lookup_table(
        trained_codebooks_path=args.trained_codebooks,
        semantic_code_file=args.semantic_codes,
        output_path=args.output,
        embedding_dim=args.embedding_dim,
        fusion_method=args.fusion_method,
        device=args.device
    )
