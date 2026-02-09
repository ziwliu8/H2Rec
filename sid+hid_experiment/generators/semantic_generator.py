# 支持semantic ID的数据生成器
import os
import time
import pickle
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from generators.data import SeqDataset, SeqDatasetAllUser, Seq2SeqDataset, Seq2SeqDatasetAllUser
from utils.utils import unzip_data, concat_data


class SemanticGenerator(object):
    """支持semantic ID的数据生成器"""

    def __init__(self, args, logger, device):
        self.args = args
        self.aug_file = args.aug_file
        self.inter_file = args.inter_file
        self.dataset = args.dataset
        self.num_workers = args.num_workers
        self.bs = args.train_batch_size
        self.logger = logger
        self.device = device
        self.aug_seq = args.aug_seq

        # semantic ID相关配置
        self.use_semantic_id = getattr(args, 'use_semantic_id', False)
        self.semantic_code_file = getattr(args, 'semantic_code_file', None)
        
        # 如果使用semantic ID，验证文件存在
        if self.use_semantic_id and self.semantic_code_file:
            semantic_path = os.path.join('./data', self.dataset, 'handled', self.semantic_code_file)
            if not os.path.exists(semantic_path):
                raise FileNotFoundError(f"Semantic code file not found: {semantic_path}")

        self.logger.info("Loading dataset ... ")
        start = time.time()
        self._load_dataset()
        end = time.time()
        self.logger.info("Dataset is loaded: consume %.3f s" % (end - start))

    def _load_semantic_mapping(self):
        """加载semantic ID映射"""
        if not self.use_semantic_id or not self.semantic_code_file:
            return None
        
        semantic_path = os.path.join('./data', self.dataset, 'handled', self.semantic_code_file)
        self.logger.info(f"Loading semantic codes from: {semantic_path}")
        
        with open(semantic_path, 'r') as f:
            semantic_codes = json.load(f)
        
        # semantic_codes[0]是空列表，semantic_codes[i]对应原始item_id=i的semantic code
        # 创建从hash_id到semantic_code的映射
        hash_to_semantic = {}
        for hash_id in range(1, len(semantic_codes)):
            if semantic_codes[hash_id]:  # 确保不是空列表
                hash_to_semantic[hash_id] = tuple(semantic_codes[hash_id])  # 转为tuple便于使用
        
        self.logger.info(f"Loaded {len(hash_to_semantic)} semantic code mappings")
        return hash_to_semantic
    
    def _convert_to_semantic_ids(self, user_data, hash_to_semantic):
        """将hash ID转换为semantic ID"""
        if not hash_to_semantic:
            return user_data
        
        # 不进行ID转换，而是保持原始的hash ID
        # 在模型层面通过CodebookEmbedding来处理semantic code的映射
        self.logger.info(f"Keeping original hash IDs, semantic mapping will be handled in model layer")
        return user_data

    def _load_dataset(self):
        '''Load train, validation, test dataset'''

        usernum = 0
        itemnum = 0
        User = defaultdict(list)    # default value is a blank list
        user_train = {}
        user_valid = {}
        user_test = {}
        
        # 加载semantic ID映射
        hash_to_semantic = self._load_semantic_mapping()
        
        # assume user/item index starting from 1
        f = open('./data/%s/handled/%s.txt' % (self.dataset, 'inter'), 'r')
        for line in f:  # use a dict to save all sequences of each user
            u, i = line.rstrip().split(' ')
            u = int(u)
            i = int(i)
            usernum = max(u, usernum)
            itemnum = max(i, itemnum)
            User[u].append(i)
        f.close()
        
        # 如果使用semantic ID，转换数据
        if self.use_semantic_id and hash_to_semantic:
            self.logger.info("Converting hash IDs to semantic IDs...")
            User = self._convert_to_semantic_ids(User, hash_to_semantic)
        
        self.user_num = usernum
        self.item_num = itemnum

        for user in tqdm(User):
            nfeedback = len(User[user]) - self.args.aug_seq_len
            if nfeedback < 3:
                user_train[user] = User[user]
                user_valid[user] = []
                user_test[user] = []
            else:
                user_train[user] = User[user][:-2]
                user_valid[user] = []
                user_valid[user].append(User[user][-2])
                user_test[user] = []
                user_test[user].append(User[user][-1])
        
        self.train = user_train
        self.valid = user_valid
        self.test = user_test

    def make_trainloader(self):
        train_dataset = unzip_data(self.train, aug=self.args.aug, aug_num=self.args.aug_seq_len)
        
        # 根据模型名称选择合适的数据集
        if self.args.model_name.endswith('_seq'):
            self.train_dataset = Seq2SeqDataset(self.args, train_dataset, self.item_num, self.args.max_len, self.args.train_neg)
        else:
            self.train_dataset = SeqDataset(train_dataset, self.item_num, self.args.max_len, self.args.train_neg)

        train_dataloader = DataLoader(self.train_dataset,
                                      sampler=RandomSampler(self.train_dataset),
                                      batch_size=self.bs,
                                      num_workers=self.num_workers)

        return train_dataloader

    def make_evalloader(self, test=False):
        if test:
            eval_dataset = concat_data([self.train, self.valid, self.test])
        else:
            eval_dataset = concat_data([self.train, self.valid])

        self.eval_dataset = SeqDataset(eval_dataset, self.item_num, self.args.max_len, self.args.test_neg)
        eval_dataloader = DataLoader(self.eval_dataset,
                                    sampler=SequentialSampler(self.eval_dataset),
                                    batch_size=100,
                                    num_workers=self.num_workers)
        
        return eval_dataloader

    def get_user_item_num(self):
        return self.user_num, self.item_num
    
    def get_item_pop(self):
        """get item popularity according to item index. return a np-array"""
        all_data = concat_data([self.train, self.valid, self.test])
        
        # 找到实际的最大item ID
        max_item_id = 0
        for items in all_data:
            if items:  # 确保不是空列表
                max_item_id = max(max_item_id, max(items))
        
        # 使用实际最大ID+1作为数组大小，确保不会越界
        pop = np.zeros(max_item_id + 1)
        
        for items in all_data:
            if items:  # 确保不是空列表
                for item in items:
                    if 0 <= item <= max_item_id:  # 安全检查
                        pop[item] += 1

        return pop
    
    def get_user_len(self):
        """get sequence length according to user index. return a np-array"""
        all_data = concat_data([self.train, self.valid])
        lens = []

        for user in all_data:
            lens.append(len(user))

        return np.array(lens)


class SemanticGeneratorAllUser(SemanticGenerator):
    """支持semantic ID的全用户数据生成器"""

    def __init__(self, args, logger, device):
        super().__init__(args, logger, device)
    
    def make_trainloader(self):
        train_dataset = unzip_data(self.train, aug=self.args.aug, aug_num=self.args.aug_seq_len)
        self.train_dataset = SeqDatasetAllUser(self.args, train_dataset, self.item_num, self.args.max_len, self.args.train_neg)

        train_dataloader = DataLoader(self.train_dataset,
                                      sampler=RandomSampler(self.train_dataset),
                                      batch_size=self.bs,
                                      num_workers=self.num_workers)
        
        return train_dataloader


class SemanticSeq2SeqGeneratorAllUser(SemanticGenerator):
    """支持semantic ID的序列到序列全用户数据生成器"""
    
    def __init__(self, args, logger, device):
        super().__init__(args, logger, device)
    
    def make_trainloader(self):
        train_dataset = unzip_data(self.train, aug=self.args.aug, aug_num=self.args.aug_seq_len)
        self.train_dataset = Seq2SeqDatasetAllUser(self.args, train_dataset, self.item_num, self.args.max_len, self.args.train_neg)

        train_dataloader = DataLoader(self.train_dataset,
                                      sampler=RandomSampler(self.train_dataset),
                                      batch_size=self.bs,
                                      num_workers=self.num_workers)
        
        return train_dataloader
