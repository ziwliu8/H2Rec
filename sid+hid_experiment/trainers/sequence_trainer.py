# here put the import lib
import os
import time
import torch
import numpy as np
from tqdm import tqdm
from trainers.trainer import Trainer
from utils.utils import metric_report, metric_len_report, record_csv, metric_pop_report
from utils.utils import metric_len_5group, metric_pop_5group


class SeqTrainer(Trainer):

    def __init__(self, args, logger, writer, device, generator):

        super().__init__(args, logger, writer, device, generator)
    

    def _train_one_epoch(self, epoch):

        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        train_time = []

        self.model.train()
        prog_iter = tqdm(self.train_loader, leave=False, desc='Training')

        for batch in prog_iter:

            batch = tuple(t.to(self.device) for t in batch)

            train_start = time.time()
            inputs = self._prepare_train_inputs(batch)
            loss = self.model(**inputs)
            loss.backward()

            tr_loss += loss.item()
            nb_tr_examples += 1
            nb_tr_steps += 1

            # Display loss
            prog_iter.set_postfix(loss='%.4f' % (tr_loss / nb_tr_steps))

            self.optimizer.step()
            self.optimizer.zero_grad()

            train_end = time.time()
            train_time.append(train_end-train_start)

        self.writer.add_scalar('train/loss', tr_loss / nb_tr_steps, epoch)



    def eval(self, epoch=0, test=False):

        print('')
        if test:
            self.logger.info("\n----------------------------------------------------------------")
            self.logger.info("********** Running test **********")
            desc = 'Testing'
            model_state_dict = torch.load(os.path.join(self.args.output_dir, 'pytorch_model.bin'))
            self.model.load_state_dict(model_state_dict['state_dict'])
            self.model.to(self.device)
            test_loader = self.test_loader
        
        else:
            self.logger.info("\n----------------------------------")
            self.logger.info("********** Epoch: %d eval **********" % epoch)
            desc = 'Evaluating'
            test_loader = self.valid_loader
        
        self.model.eval()
        pred_rank = torch.empty(0).to(self.device)
        seq_len = torch.empty(0).to(self.device)
        target_items = torch.empty(0).to(self.device)

        for batch in tqdm(test_loader, desc=desc):

            batch = tuple(t.to(self.device) for t in batch)
            inputs = self._prepare_eval_inputs(batch)
            seq_len = torch.cat([seq_len, torch.sum(inputs["seq"]>0, dim=1)])
            target_items = torch.cat([target_items, inputs["pos"]])
            
            with torch.no_grad():

                inputs["item_indices"] = torch.cat([inputs["pos"].unsqueeze(1), inputs["neg"]], dim=1)
                pred_logits = -self.model.predict(**inputs)

                per_pred_rank = torch.argsort(torch.argsort(pred_logits))[:, 0]
                pred_rank = torch.cat([pred_rank, per_pred_rank])

        self.logger.info('')
        #res_dict = metric_report(pred_rank.detach().cpu().numpy())
        #res_len_dict = metric_len_report(pred_rank.detach().cpu().numpy(), seq_len.detach().cpu().numpy(), aug_len=self.args.aug_seq_len, args=self.args)
        #res_pop_dict = metric_pop_report(pred_rank.detach().cpu().numpy(), target_items.detach().cpu().numpy(), self.item_pop, self.args.ts_item, args=self.args)
        #from utils.utils import metric_report, metric_len_report, metric_pop_report, record_csv
        from utils.utils import metric_report, metric_len_report, metric_pop_5group_dict, record_csv
        res_dict = metric_report(pred_rank.detach().cpu().numpy())
        res_len_dict = metric_len_report(pred_rank.detach().cpu().numpy(), seq_len.detach().cpu().numpy(), aug_len=self.args.aug_seq_len, args=self.args)
        #res_pop_dict = metric_pop_report(pred_rank.detach().cpu().numpy(), target_items.detach().cpu().numpy(), self.item_pop, self.args.ts_item)
        res_pop_dict = metric_pop_5group_dict(pred_rank.detach().cpu().numpy(), self.item_pop, target_items.detach().cpu().numpy())

        self.logger.info("Overall Performance:")
        for k, v in res_dict.items():
            if not test:
                self.writer.add_scalar('Test/{}'.format(k), v, epoch)
            self.logger.info('\t %s: %.5f' % (k, v))

        if test:
            self.logger.info("User Group Performance:")
            for k, v in res_len_dict.items():
                if not test:
                    self.writer.add_scalar('Test/{}'.format(k), v, epoch)
                self.logger.info('\t %s: %.5f' % (k, v))
            self.logger.info("Item Group Performance:")
            for k, v in res_pop_dict.items():
                if not test:
                    self.writer.add_scalar('Test/{}'.format(k), v, epoch)
                self.logger.info('\t %s: %.5f' % (k, v))
        
        res_dict = {**res_dict, **res_len_dict, **res_pop_dict}

        if test:
            record_csv(self.args, res_dict)
        
        return res_dict
    


    def save_user_emb(self):
        """保存用户嵌入表示，支持log_feats特征保存"""
        
        # 检查模型文件是否存在
        model_path = os.path.join(self.args.output_dir, 'pytorch_model.bin')
        if not os.path.exists(model_path):
            self.logger.error(f"模型文件不存在: {model_path}")
            self.logger.error("请先训练模型或指定正确的模型路径")
            return None
        
        self.logger.info(f"正在加载模型权重: {model_path}")
        
        try:
            model_state_dict = torch.load(model_path, map_location=self.device)
            if 'state_dict' in model_state_dict:
                self.model.load_state_dict(model_state_dict['state_dict'])
                self.logger.info(f"成功加载模型权重 (epoch: {model_state_dict.get('epoch', 'unknown')})")
            else:
                self.model.load_state_dict(model_state_dict)
                self.logger.info("成功加载模型权重")
        except Exception as e:
            self.logger.error(f"加载模型权重失败: {e}")
            return None
        
        # 为了避免CUDA问题，将模型移到CPU进行推理
        self.logger.info("将模型移动到CPU以避免CUDA问题")
        cpu_device = torch.device('cpu')
        self.model.to(cpu_device)
        self.model.eval()
        
        # 清理CUDA缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        test_loader = self.test_loader
        user_emb = torch.empty(0).to(cpu_device)
        user_ids = torch.empty(0).to(cpu_device)
        desc = 'Extracting User Embeddings'

        for batch in tqdm(test_loader, desc=desc):
            try:
                # 将数据移到CPU设备
                batch = tuple(t.to(cpu_device) for t in batch)
                inputs = self._prepare_eval_inputs(batch)
                
                with torch.no_grad():
                    # 检查模型是否有get_log_feats方法（用于LLM_SASRec）
                    if hasattr(self.model, 'get_log_feats'):
                        per_user_emb = self.model.get_log_feats(**inputs)
                    else:
                        # 回退到原有的get_user_emb方法
                        per_user_emb = self.model.get_user_emb(**inputs)
                    
                    user_emb = torch.cat([user_emb, per_user_emb], dim=0)
                    
                    # 如果输入中有user_id，则保存用户ID映射
                    if 'user_id' in inputs:
                        user_ids = torch.cat([user_ids, inputs['user_id']], dim=0)
                
                # 定期清理内存
                if len(user_emb) % 1000 == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
            except Exception as e:
                self.logger.error(f"处理批次时出错: {e}")
                import traceback
                self.logger.error(f"详细错误: {traceback.format_exc()}")
                continue
        
        user_emb = user_emb.detach().cpu().numpy()
        
        # 检查是否成功提取到嵌入
        if user_emb.shape[0] == 0:
            self.logger.error("未能成功提取任何用户嵌入！所有批次都失败了。")
            return None
        
        self.logger.info(f"成功提取 {user_emb.shape[0]} 个用户的嵌入，维度: {user_emb.shape[1]}")
        
        # 构建保存路径和文件名
        import pickle
        save_dir = f"./data/{self.args.dataset}/handled/"
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存用户嵌入
        emb_file = os.path.join(save_dir, f"{self.args.model_name}_user_emb.pkl")
        pickle.dump(user_emb, open(emb_file, "wb"))
        self.logger.info(f"User embeddings saved to: {emb_file}")
        self.logger.info(f"Embedding shape: {user_emb.shape}")
        
        # 如果有用户ID映射，也保存它
        if len(user_ids) > 0:
            user_ids = user_ids.detach().cpu().numpy()
            id_file = os.path.join(save_dir, f"{self.args.model_name}_user_ids.pkl")
            pickle.dump(user_ids, open(id_file, "wb"))
            self.logger.info(f"User ID mapping saved to: {id_file}")
            
            # 创建用户ID到嵌入索引的映射字典
            user_id_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
            mapping_file = os.path.join(save_dir, f"{self.args.model_name}_user_mapping.pkl")
            pickle.dump(user_id_to_idx, open(mapping_file, "wb"))
            self.logger.info(f"User ID to index mapping saved to: {mapping_file}")
        
        # 为了兼容你的使用方式，创建一个完整的嵌入矩阵（包含padding）
        # 假设用户ID从1开始，0为padding
        if len(user_ids) > 0:
            max_user_id = int(user_ids.max())
            # 创建完整的用户嵌入矩阵，索引对应用户ID
            full_user_emb = np.zeros((max_user_id + 1, user_emb.shape[1]))  # +1 因为包含ID=0的padding
            for idx, user_id in enumerate(user_ids):
                full_user_emb[int(user_id)] = user_emb[idx]
            
            full_emb_file = os.path.join(save_dir, f"{self.args.model_name}_full_user_emb.pkl")
            pickle.dump(full_user_emb, open(full_emb_file, "wb"))
            self.logger.info(f"Full user embedding matrix saved to: {full_emb_file}")
            self.logger.info(f"Full embedding shape: {full_user_emb.shape}")
        
        return emb_file


    
    def test_group(self):

        print('')
        self.logger.info("\n----------------------------------------------------------------")
        self.logger.info("********** Running Group test **********")
        desc = 'Testing'
        model_state_dict = torch.load(os.path.join(self.args.output_dir, 'pytorch_model.bin'))
        self.model.load_state_dict(model_state_dict['state_dict'])
        self.model.to(self.device)
        test_loader = self.test_loader
        
        self.model.eval()
        pred_rank = torch.empty(0).to(self.device)
        seq_len = torch.empty(0).to(self.device)
        target_items = torch.empty(0).to(self.device)

        for batch in tqdm(test_loader, desc=desc):

            batch = tuple(t.to(self.device) for t in batch)
            inputs = self._prepare_eval_inputs(batch)
            seq_len = torch.cat([seq_len, torch.sum(inputs["seq"]>0, dim=1)])
            target_items = torch.cat([target_items, inputs["pos"]])
            
            with torch.no_grad():

                inputs["item_indices"] = torch.cat([inputs["pos"].unsqueeze(1), inputs["neg"]], dim=1)
                pred_logits = -self.model.predict(**inputs)

                per_pred_rank = torch.argsort(torch.argsort(pred_logits))[:, 0]
                pred_rank = torch.cat([pred_rank, per_pred_rank])

        self.logger.info('')
        res_dict = metric_report(pred_rank.detach().cpu().numpy())
        # res_len_dict = metric_len_report(pred_rank.detach().cpu().numpy(), seq_len.detach().cpu().numpy(), aug_len=self.args.aug_seq_len, args=self.args)
        # res_pop_dict = metric_pop_report(pred_rank.detach().cpu().numpy(), self.item_pop, target_items.detach().cpu().numpy(), args=self.args)
        hr_len, ndcg_len, count_len = metric_len_5group(pred_rank.detach().cpu().numpy(), seq_len.detach().cpu().numpy(), [5, 10, 15, 20])
        hr_pop, ndcg_pop, count_pop = metric_pop_5group(pred_rank.detach().cpu().numpy(), self.item_pop,  target_items.detach().cpu().numpy(), [10, 30, 60, 100])

        self.logger.info("Overall Performance:")
        for k, v in res_dict.items():
            self.logger.info('\t %s: %.5f' % (k, v))

        self.logger.info("User Group Performance:")
        for i, (hr, ndcg) in enumerate(zip(hr_len, ndcg_len)):
            self.logger.info('The %d Group: HR %.4f, NDCG %.4f' % (i, hr, ndcg))
        self.logger.info("Item Group Performance:")
        for i, (hr, ndcg) in enumerate(zip(hr_pop, ndcg_pop)):
            self.logger.info('The %d Group: HR %.4f, NDCG %.4f' % (i, hr, ndcg))
        
        
        return res_dict
    


class CL4SRecTrainer(SeqTrainer):

    def __init__(self, args, logger, writer, device, generator):
        
        super().__init__(args, logger, writer, device, generator)


    def _train_one_epoch(self, epoch):

        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        train_time = []

        self.model.train()
        prog_iter = tqdm(self.train_loader, leave=False, desc='Training')

        for batch in prog_iter:

            batch = tuple(t.to(self.device) for t in batch)

            train_start = time.time()
            seq, pos, neg, positions, aug1, aug2 = batch
            seq, pos, neg, positions, aug1, aug2 = seq.long(), pos.long(), neg.long(), positions.long(), aug1.long(), aug2.long()
            aug = (aug1, aug2)
            loss = self.model(seq, pos, neg, positions, aug)
            loss.backward()

            tr_loss += loss.item()
            nb_tr_examples += 1
            nb_tr_steps += 1

            # Display loss
            prog_iter.set_postfix(loss='%.4f' % (tr_loss / nb_tr_steps))

            self.optimizer.step()
            self.optimizer.zero_grad()

            train_end = time.time()
            train_time.append(train_end-train_start)

        self.writer.add_scalar('train/loss', tr_loss / nb_tr_steps, epoch)



class SSEPTTrainer(Trainer):

    def __init__(self, args, logger, writer, device, generator):

        super().__init__(args, logger, writer, device, generator)
    

    def _train_one_epoch(self, epoch):

        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        train_time = []

        self.model.train()
        prog_iter = tqdm(self.train_loader, leave=False, desc='Training')

        for batch in prog_iter:

            batch = tuple(t.to(self.device) for t in batch)

            train_start = time.time()
            seq_user, pos_user, neg_user, seq, pos, neg, positions = batch
            seq, pos, neg, positions = seq.long(), pos.long(), neg.long(), positions.long()
            seq_user, pos_user, neg_user = seq_user.long(), pos_user.long(), neg_user.long()
            loss = self.model(seq_user, pos_user, neg_user, seq, pos, neg, positions)
            loss.backward()

            tr_loss += loss.item()
            nb_tr_examples += 1
            nb_tr_steps += 1

            # Display loss
            prog_iter.set_postfix(loss='%.4f' % (tr_loss / nb_tr_steps))

            self.optimizer.step()
            self.optimizer.zero_grad()

            train_end = time.time()
            train_time.append(train_end-train_start)

        self.writer.add_scalar('train/loss', tr_loss / nb_tr_steps, epoch)



    def eval(self, epoch=0, test=False):

        print('')
        if test:
            self.logger.info("\n----------------------------------------------------------------")
            self.logger.info("********** Running test **********")
            desc = 'Testing'
            model_state_dict = torch.load(os.path.join(self.args.output_dir, 'pytorch_model.bin'))
            try:
                self.model.load_state_dict(model_state_dict['state_dict'])
            except:
                self.model.load_state_dict(model_state_dict)
            self.model.to(self.device)
            test_loader = self.test_loader
        
        else:
            self.logger.info("\n----------------------------------")
            self.logger.info("********** Epoch: %d eval **********" % epoch)
            desc = 'Evaluating'
            test_loader = self.valid_loader
        
        self.model.eval()
        pred_rank = torch.empty(0).to(self.device)
        seq_len = torch.empty(0).to(self.device)

        for batch in tqdm(test_loader, desc=desc):

            batch = tuple(t.to(self.device) for t in batch)
            seq_user, pos_user, neg_user, seq, pos, neg, positions = batch
            seq, pos, neg, positions = seq.long(), pos.long(), neg.long(), positions.long()
            seq_user, pos_user, neg_user = seq_user.long(), pos_user.long(), neg_user.long()
            seq_len = torch.cat([seq_len, torch.sum(seq>0, dim=1)])

            with torch.no_grad():

                pred_logits = -self.model.predict(seq_user, seq, torch.cat([pos_user.unsqueeze(1), neg_user], dim=1), torch.cat([pos.unsqueeze(1), neg], dim=1), positions)

                per_pred_rank = torch.argsort(torch.argsort(pred_logits))[:, 0]
                pred_rank = torch.cat([pred_rank, per_pred_rank])

        self.logger.info('')
        res_dict = metric_report(pred_rank.detach().cpu().numpy())
        res_len_dict = metric_len_report(pred_rank.detach().cpu().numpy(), seq_len.detach().cpu().numpy(), aug_len=self.args.aug_seq_len)
        
        for k, v in res_dict.items():
            if not test:
                self.writer.add_scalar('Test/{}'.format(k), v, epoch)
            self.logger.info('%s: %.5f' % (k, v))
        for k, v in res_len_dict.items():
            if not test:
                self.writer.add_scalar('Test/{}'.format(k), v, epoch)
            self.logger.info('%s: %.5f' % (k, v))
        
        res_dict = {**res_dict, **res_len_dict}

        if test:
            record_csv(self.args, res_dict)
        
        return res_dict
