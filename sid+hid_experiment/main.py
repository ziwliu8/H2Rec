# here put the import lib
import os
import argparse
import torch

from generators.generator import Seq2SeqGeneratorAllUser
from generators.generator import GeneratorAllUser
from generators.bert_generator import BertGeneratorAllUser
from generators.semantic_generator import SemanticSeq2SeqGeneratorAllUser, SemanticGeneratorAllUser
from trainers.sequence_trainer import SeqTrainer
from trainers.semantic_trainer import SemanticTrainer
from utils.utils import set_seed
from utils.logger import Logger


parser = argparse.ArgumentParser()

# Required parameters
parser.add_argument("--model_name", 
                    default='SASRec_seq',
                    choices=[
                    "SASRec_seq", 
                    "SemanticSASRec_seq",
                    "SemanticSASRec",
                    "Bert4Rec",
                    "GRU4Rec_seq",
                    "DualTrisRecSASRec_seq",
                    "DualTrisRecSASRec",
                    "s_h_SASRec_seq",
                    "s_h_SASRec",
                    "LLM_SASRec_seq",
                    "LLM_SASRec",
                    "DualLLMSASRec",
                    "LLMESR_SASRec",
                    ],
                    type=str, 
                    required=False,
                    help="model name")
parser.add_argument("--dataset", 
                    default="yelp", 
                    choices=["yelp", "fashion", "beauty", "instrument", "games"],  
                    help="Choose the dataset")
parser.add_argument("--inter_file",
                    default="inter",
                    type=str,
                    help="the name of interaction file")
parser.add_argument("--demo", 
                    default=False, 
                    action='store_true', 
                    help='whether run demo')
parser.add_argument("--pretrain_dir",
                    type=str,
                    default="sasrec_seq",
                    help="the path that pretrained model saved in")
parser.add_argument("--output_dir",
                    default='./saved/',
                    type=str,
                    required=False,
                    help="The output directory where the model checkpoints will be written.")
parser.add_argument("--check_path",
                    default='',
                    type=str,
                    help="the save path of checkpoints for different running")
parser.add_argument("--do_test",
                    default=False,
                    action="store_true",
                    help="whehther run the test on the well-trained model")
parser.add_argument("--do_emb",
                    default=False,
                    action="store_true",
                    help="save the user embedding derived from the SRS model")
parser.add_argument("--do_group",
                    default=False,
                    action="store_true",
                    help="conduct the group test")
parser.add_argument("--keepon",
                    default=False,
                    action="store_true",
                    help="whether keep on training based on a trained model")
parser.add_argument("--keepon_path",
                    type=str,
                    default="normal",
                    help="the path of trained model for keep on training")
parser.add_argument("--clip_path",
                    type=str,
                    default="",
                    help="the path to save the CLIP-pretrained embedding and adapter")
parser.add_argument("--ts_user",
                    type=int,
                    default=10,
                    help="the threshold to split the short and long seq")
parser.add_argument("--ts_item",
                    type=int,
                    default=20,
                    help="the threshold to split the long-tail and popular items")

# Model parameters
parser.add_argument("--hidden_size",
                    default=64,
                    type=int,
                    help="the hidden size of embedding")
parser.add_argument("--trm_num",
                    default=2,
                    type=int,
                    help="the number of transformer layer")
parser.add_argument("--num_heads",
                    default=1,
                    type=int,
                    help="the number of heads in Trm layer")
parser.add_argument("--num_layers",
                    default=1,
                    type=int,
                    help="the number of GRU layers")
parser.add_argument("--cl_scale",
                    type=float,
                    default=0.1,
                    help="the scale for contastive loss")
parser.add_argument("--mask_crop_ratio",
                    type=float,
                    default=0.3,
                    help="the mask/crop ratio for CL4SRec")
parser.add_argument("--tau",
                    default=1,
                    type=float,
                    help="the temperature for contrastive loss")
parser.add_argument("--sse_ratio",
                    default=0.4,
                    type=float,
                    help="the sse ratio for SSE-PT model")
parser.add_argument("--dropout_rate",
                    default=0.5,
                    type=float,
                    help="the dropout rate")
parser.add_argument("--max_len",
                    default=200,
                    type=int,
                    help="the max length of input sequence")
parser.add_argument("--mask_prob",
                    type=float,
                    default=0.4,
                    help="the mask probability for training Bert model")
parser.add_argument("--aug",
                    default=False,
                    action="store_true",
                    help="whether augment the sequence data")
parser.add_argument("--aug_seq",
                    default=False,
                    action="store_true",
                    help="whether use the augmented data")
parser.add_argument("--aug_seq_len",
                    default=0,
                    type=int,
                    help="the augmented length for each sequence")
parser.add_argument("--aug_file",
                    default="inter",
                    type=str,
                    help="the augmentation file name")
parser.add_argument("--train_neg",
                    default=1,
                    type=int,
                    help="the number of negative samples for training")
parser.add_argument("--test_neg",
                    default=100,
                    type=int,
                    help="the number of negative samples for test")
parser.add_argument("--suffix_num",
                    default=5,
                    type=int,
                    help="the suffix number for augmented sequence")
parser.add_argument("--prompt_num",
                    default=2,
                    type=int,
                    help="the number of prompts")
parser.add_argument("--freeze",
                    default=False,
                    action="store_true",
                    help="whether freeze the pretrained architecture when finetuning")
parser.add_argument("--pg",
                    default="length",
                    choices=['length', 'attention'],
                    type=str,
                    help="choose the prompt generator")
parser.add_argument("--use_cross_att",
                    default=False,
                    action="store_true",
                    help="whether add a cross-attention to interact the dual-view")
parser.add_argument("--alpha",
                    default=0.1,
                    type=float,
                    help="the weight of auxiliary loss")
parser.add_argument("--user_sim_func",
                    default="kd",
                    type=str,
                    help="the type of user similarity function to derive the loss")
parser.add_argument("--item_reg",
                    default=False,
                    action="store_true",
                    help="whether regularize the item embedding by CL")
parser.add_argument("--beta",
                    default=0.1,
                    type=float,
                    help="the weight of regulation loss")
parser.add_argument("--sim_user_num",
                    default=10,
                    type=int,
                    help="the number of similar users for enhancement")
parser.add_argument("--split_backbone",
                    default=False,
                    action="store_true",
                    help="whether use a split backbone")
parser.add_argument("--co_view",
                    default=False,
                    action="store_true",
                    help="only use the collaborative view")
parser.add_argument("--se_view",
                    default=False,
                    action="store_true",
                    help="only use the semantic view")

# Semantic ID parameters
parser.add_argument("--use_semantic_id",
                    default=False,
                    action="store_true",
                    help="whether use semantic ID instead of hash ID")
parser.add_argument("--semantic_code_file",
                    default="beauty.code.fixed_rq.12_256.pca128.json",
                    type=str,
                    help="the filename of semantic codes")
parser.add_argument("--semantic_encoder_type",
                    default="rqvae",
                    choices=["rqvae", "pqvae"],
                    type=str,
                    help="quantizer type used to produce semantic IDs")

# RQVAE/Semantic embedding parameters
parser.add_argument("--semantic_fusion_method",
                    default="mean",
                    choices=["sum", "mean", "concat"],
                    type=str,
                    help="fusion method for semantic codebook embeddings")
parser.add_argument("--fusion_method",
                    default="sum", 
                    choices=["sum", "mean", "concat"],
                    type=str,
                    help="fusion method for codebook embeddings (alias for semantic_fusion_method)")
parser.add_argument("--learnable_codebooks",
                    default=False,
                    action="store_true",
                    help="whether codebook embeddings are learnable")
parser.add_argument("--freeze_codebooks",
                    default=False,
                    action="store_true",
                    help="whether freeze codebook embeddings")
parser.add_argument("--use_decoder",
                    default=False,
                    action="store_true",
                    help="whether use RQVAE decoder for item representations")
parser.add_argument("--freeze_decoder",
                    default=False,
                    action="store_true",
                    help="whether freeze decoder parameters")
parser.add_argument("--embedding_pooling",
                    default="mean",
                    choices=["mean", "sum", "concat"],
                    type=str,
                    help="pooling method for embeddings")

# Multi-view enhancement parameters (for DualTrisRec)
# 注：已删除mv_align_weight, hier_consistency_weight, align_temperature
# 这些辅助损失与InfoNCE冲突/重复，已从模型中移除

parser.add_argument("--use_dynamic_gate",
                    default=False,
                    action="store_true",
                    help="whether to use dynamic hierarchical gating for multi-view fusion")

# InfoNCE contrastive learning parameters (for DualTrisRec) - V2升级版
parser.add_argument("--use_infonce_loss",
                    default=False,
                    action="store_true",
                    help="whether use InfoNCE contrastive loss to align sid and hash embeddings")
parser.add_argument("--infonce_weight",
                    default=0.1,
                    type=float,
                    help="weight for InfoNCE contrastive loss")
parser.add_argument("--infonce_temperature",
                    default=0.5,
                    type=float,
                    help="temperature parameter for InfoNCE loss")
parser.add_argument("--codebook_overlap_threshold",
                    default=2,
                    type=int,
                    help="threshold for codebook overlap to consider items as positive pairs")
parser.add_argument("--consecutive_window",
                    default=3,
                    type=int,
                    help="window size for consecutive interactions to identify positive pairs")
parser.add_argument("--positive_pairs_file",
                    default=None,
                    type=str,
                    help="precomputed positive pairs file (pickle format) for faster training")

# InfoNCE V2: 正样本权重和质量控制
parser.add_argument("--use_positive_weights",
                    default=False,
                    action="store_true",
                    help="whether use weighted positive samples (PMI/time-decay weights)")
parser.add_argument("--use_dynamic_filter",
                    default=False,
                    action="store_true",
                    help="whether dynamically filter weak positive samples based on sequence representation")
parser.add_argument("--positive_filter_ratio",
                    default=0.6,
                    type=float,
                    help="ratio of positive samples to keep after dynamic filtering (keep top-k%)")
parser.add_argument("--positive_min_sim",
                    default=0.1,
                    type=float,
                    help="minimum similarity threshold for positive samples in dynamic filtering")

# InfoNCE V2: 硬负样本采样
parser.add_argument("--use_hard_negatives",
                    default=False,
                    action="store_true",
                    help="whether use hard negative samples (semantically similar but not co-occurred)")
parser.add_argument("--n_hard_negatives",
                    default=5,
                    type=int,
                    help="number of hard negative samples per anchor")

# InfoNCE V2: 去偏处理
parser.add_argument("--use_debias",
                    default=False,
                    action="store_true",
                    help="whether use debiasing to reduce false negative impact")

# ViewCL: 视角一致性对比学习参数（防止过度依赖主导粒度）
parser.add_argument("--use_view_cl",
                    default=False,
                    action="store_true",
                    help="whether use View Consistency Contrastive Learning to prevent over-relying on dominant granularities")
parser.add_argument("--view_cl_weight",
                    default=0.1,
                    type=float,
                    help="weight for ViewCL loss")
parser.add_argument("--view_cl_mask_ratio",
                    default=0.5,
                    type=float,
                    help="ratio of views to mask in ViewCL (0.5 = mask half of views)")
parser.add_argument("--view_cl_mask_strategy",
                    default="low_weight",
                    choices=["low_weight", "high_weight", "random"],
                    type=str,
                    help="strategy for selecting views to mask: low_weight (mask ignored views), high_weight (more challenging), random")
parser.add_argument("--view_cl_temperature",
                    default=1.0,
                    type=float,
                    help="temperature for ViewCL contrastive loss")

# PCR_CA: Item-level和User-level InfoNCE参数
parser.add_argument("--use_item_infonce",
                    default=False,
                    action="store_true",
                    help="whether use item-level InfoNCE loss to align sid and hash item embeddings (PCR_CA)")
parser.add_argument("--item_infonce_weight",
                    default=0.1,
                    type=float,
                    help="weight for item-level InfoNCE loss")
parser.add_argument("--item_infonce_temperature",
                    default=0.5,
                    type=float,
                    help="temperature parameter for item-level InfoNCE loss")
parser.add_argument("--use_user_infonce",
                    default=False,
                    action="store_true",
                    help="whether use user-level InfoNCE loss to align sid and hash user preferences (PCR_CA)")
parser.add_argument("--user_infonce_weight",
                    default=0.1,
                    type=float,
                    help="weight for user-level InfoNCE loss")
parser.add_argument("--user_infonce_temperature",
                    default=0.5,
                    type=float,
                    help="temperature parameter for user-level InfoNCE loss")

# InfoNCE V2: 序列增强对比（CL4SRec风格）
parser.add_argument("--use_seq_augment",
                    default=False,
                    action="store_true",
                    help="whether use sequence augmentation contrastive learning (CL4SRec style)")
parser.add_argument("--seq_augment_weight",
                    default=0.1,
                    type=float,
                    help="weight for sequence augmentation contrastive loss")
parser.add_argument("--use_projection_head",
                    default=False,
                    action="store_true",
                    help="whether use projection head for sequence-level contrastive learning")
parser.add_argument("--projection_dim",
                    default=128,
                    type=int,
                    help="projection dimension for sequence-level contrastive learning")

# LLM-related parameters
parser.add_argument("--llm_file",
                    default=None,
                    type=str,
                    help="path to LLM embeddings file")
parser.add_argument("--use_llm_emb",
                    default=False,
                    action='store_true',
                    help="use pretrained LLM embeddings")
parser.add_argument("--freeze_llm_emb",
                    default=False,
                    action='store_true',
                    help="freeze LLM embeddings during training")

# MSA (Masked Sequence Alignment) Loss parameters
parser.add_argument("--use_msa_loss",
                    default=False,
                    action='store_true',
                    help="use MSA loss for cross-modal alignment")
parser.add_argument("--msa_weight",
                    default=0.1,
                    type=float,
                    help="weight for MSA loss")
parser.add_argument("--msa_temperature",
                    default=0.5,
                    type=float,
                    help="temperature for MSA loss")

# CL (Contrastive Learning) Loss parameters
parser.add_argument("--use_cl_loss",
                    default=False,
                    action='store_true',
                    help="use CL loss for contrastive learning")
parser.add_argument("--cl_weight",
                    default=0.1,
                    type=float,
                    help="weight for CL loss")
parser.add_argument("--cl_temperature",
                    default=0.07,
                    type=float,
                    help="temperature for CL loss")

# MLM (Masked Language Modeling) Loss parameters
parser.add_argument("--use_mlm_loss",
                    default=False,
                    action='store_true',
                    help="use MLM loss for masked code prediction")
parser.add_argument("--mlm_weight",
                    default=0.1,
                    type=float,
                    help="weight for MLM loss")
parser.add_argument("--mlm_mask_prob",
                    default=0.15,
                    type=float,
                    help="mask probability for MLM")


# Other parameters
parser.add_argument("--train_batch_size",
                    default=512,
                    type=int,
                    help="Total batch size for training.")
parser.add_argument("--lr",
                    default=0.001,
                    type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--l2",
                    default=0,
                    type=float,
                    help='The L2 regularization')
parser.add_argument("--num_train_epochs",
                    default=100,
                    type=float,
                    help="Total number of training epochs to perform.")
parser.add_argument("--lr_dc_step",
                    default=1000,
                    type=int,
                    help='every n step, decrease the lr')
parser.add_argument("--lr_dc",
                    default=0,
                    type=float,
                    help='how many learning rate to decrease')
parser.add_argument("--patience",
                    type=int,
                    default=20,
                    help='How many steps to tolerate the performance decrease while training')
parser.add_argument("--watch_metric",
                    type=str,
                    default='NDCG@10',
                    help="which metric is used to select model.")
parser.add_argument('--seed',
                    type=int,
                    default=42,
                    help="random seed for different data split")
parser.add_argument("--no_cuda",
                    action='store_true',
                    help="Whether not to use CUDA when available")
parser.add_argument('--gpu_id',
                    default=0,
                    type=int,
                    help='The device id.')
parser.add_argument('--num_workers',
                    default=0,
                    type=int,
                    help='The number of workers in dataloader')
parser.add_argument("--log", 
                    default=False,
                    action="store_true",
                    help="whether create a new log file")

torch.autograd.set_detect_anomaly(False)

args = parser.parse_args()
set_seed(args.seed) # fix the random seed
args.output_dir = os.path.join(args.output_dir, args.dataset)
args.pretrain_dir = os.path.join(args.output_dir, args.pretrain_dir)
args.output_dir = os.path.join(args.output_dir, args.model_name)
args.keepon_path = os.path.join(args.output_dir, args.keepon_path)
args.output_dir = os.path.join(args.output_dir, args.check_path)    # if check_path is none, then without check_path


def main():

    log_manager = Logger(args)  # initialize the log manager
    logger, writer = log_manager.get_logger()    # get the logger
    args.now_str = log_manager.get_now_str()

    device = torch.device("cuda:"+str(args.gpu_id) if torch.cuda.is_available()
                          and not args.no_cuda else "cpu")


    os.makedirs(args.output_dir, exist_ok=True)

    # 选择训练器和数据生成器
    semantic_models = [
        'SemanticSASRec', 'SemanticSASRec_seq',
        'DualTrisRecSASRec', 'DualTrisRecSASRec_seq',
        's_h_SASRec', 's_h_SASRec_seq'
    ]
    
    if args.use_semantic_id or args.model_name.startswith('Semantic') or args.model_name in semantic_models:
        # 使用semantic trainer
        trainer = SemanticTrainer(args, logger, writer, device)
    else:
        # generator is used to manage dataset
        if args.model_name in ['GRU4Rec_seq']:
            generator = GeneratorAllUser(args, logger, device)
        elif args.model_name in ["Bert4Rec"]:
            generator = BertGeneratorAllUser(args, logger, device)
        elif args.model_name in ["SASRec_seq", "LLM_SASRec_seq", "LLM_SASRec", "DualLLMSASRec", "LLMESR_SASRec"]:
            generator = Seq2SeqGeneratorAllUser(args, logger, device)
        else:
            raise ValueError(f"Unsupported model: {args.model_name}")

        trainer = SeqTrainer(args, logger, writer, device, generator)

    if args.do_test:
        trainer.test()
    elif args.do_emb:
        trainer.save_user_emb()
    elif args.do_group:
        trainer.test_group()
    else:
        trainer.train()

    log_manager.end_log()   # delete the logger threads



if __name__ == "__main__":

    main()



