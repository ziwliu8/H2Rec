# 支持semantic ID的训练器
import os
import time
import torch
from tqdm import tqdm, trange
from utils.earlystop import EarlyStoppingNew
from utils.utils import get_n_params
from models.SemanticSASRec import SemanticSASRec_seq, SemanticSASRec
from models.s_h_SASRec import s_h_SASRec_seq, s_h_SASRec
from models.Bert4Rec import Bert4Rec
from models.GRU4Rec import GRU4Rec_seq
from generators.semantic_generator import SemanticGenerator, SemanticGeneratorAllUser, SemanticSeq2SeqGeneratorAllUser


class SemanticTrainer(object):
    """支持semantic ID的训练器"""

    def __init__(self, args, logger, writer, device):
        self.args = args
        self.logger = logger
        self.writer = writer
        self.device = device
        self.start_epoch = 0

        # 创建semantic generator
        self._create_generator()
        self.user_num, self.item_num = self.generator.get_user_item_num()

        self.logger.info('Loading Model: ' + args.model_name)
        self._create_model()
        logger.info('# of model parameters: ' + str(get_n_params(self.model)))

        self._set_optimizer()
        self._set_scheduler()
        self._set_stopper()

        if args.keepon:
            self._load_pretrained_model()

        self.loss_func = torch.nn.BCEWithLogitsLoss()
        
        self.train_loader = self.generator.make_trainloader()
        self.valid_loader = self.generator.make_evalloader()
        self.test_loader = self.generator.make_evalloader(test=True)

        # get item pop and user len
        self.item_pop = self.generator.get_item_pop()
        self.user_len = self.generator.get_user_len()
        
        
        # 如果模型支持流行度感知，更新物品流行度信息
        if hasattr(self.model, 'update_item_popularity'):
            self.model.update_item_popularity(self.item_pop)

        self.watch_metric = args.watch_metric

    def _create_generator(self):
        """创建数据生成器"""
        if self.args.model_name.endswith('_seq'):
            if 'alluser' in self.args.model_name.lower():
                self.generator = SemanticSeq2SeqGeneratorAllUser(self.args, self.logger, self.device)
            else:
                self.generator = SemanticGenerator(self.args, self.logger, self.device)
        else:
            if 'alluser' in self.args.model_name.lower():
                self.generator = SemanticGeneratorAllUser(self.args, self.logger, self.device)
            else:
                self.generator = SemanticGenerator(self.args, self.logger, self.device)

    def _create_model(self):
        '''create your model'''
        if self.args.model_name == "SemanticSASRec_seq":
            self.model = SemanticSASRec_seq(self.user_num, self.item_num, self.device, self.args)
        elif self.args.model_name == "SemanticSASRec":
            self.model = SemanticSASRec(self.user_num, self.item_num, self.device, self.args)
        elif self.args.model_name == "s_h_SASRec_seq":
            self.model = s_h_SASRec_seq(self.user_num, self.item_num, self.device, self.args)
        elif self.args.model_name == "s_h_SASRec":
            self.model = s_h_SASRec(self.user_num, self.item_num, self.device, self.args)
        elif self.args.model_name == "DualTrisRecSASRec_seq":
            from models.DualTrisRec import DualTrisRecSASRec_seq
            self.model = DualTrisRecSASRec_seq(self.user_num, self.item_num, self.device, self.args)
        elif self.args.model_name == "DualTrisRecSASRec":
            from models.DualTrisRec import DualTrisRecSASRec
            self.model = DualTrisRecSASRec(self.user_num, self.item_num, self.device, self.args)
        elif self.args.model_name == "SASRec_seq":
            # 如果使用semantic ID但模型名还是原来的，自动使用semantic版本
            if getattr(self.args, 'use_semantic_id', False):
                self.model = SemanticSASRec_seq(self.user_num, self.item_num, self.device, self.args)
            else:
                from models.SASRec import SASRec_seq
                self.model = SASRec_seq(self.user_num, self.item_num, self.device, self.args)
        elif self.args.model_name == "GRU4Rec_seq":
            self.model = GRU4Rec_seq(self.user_num, self.item_num, self.device, self.args)
        elif self.args.model_name == "llmesr_bert4rec":
            self.model = Bert4Rec(self.user_num, self.item_num, self.device, self.args)
        else:
            raise ValueError(f"Unknown model name: {self.args.model_name}")
        
        self.model.to(self.device)

    def _load_pretrained_model(self):
        self.logger.info("Loading the trained model for keep on training ... ")
        checkpoint_path = os.path.join(self.args.keepon_path, 'pytorch_model.bin')

        model_dict = self.model.state_dict()
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        pretrained_dict = checkpoint['state_dict']

        # filter out required parameters
        new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
        model_dict.update(new_dict)
        # Print: how many parameters are loaded from the checkpoint
        self.logger.info('Total loaded parameters: {}, update: {}'.format(len(pretrained_dict), len(new_dict)))
        self.model.load_state_dict(model_dict)  # load model parameters
        self.optimizer.load_state_dict(checkpoint['optimizer']) # load optimizer
        self.scheduler.load_state_dict(checkpoint['scheduler']) # load scheduler
        self.start_epoch = checkpoint['epoch']  # load epoch

    def _set_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                          lr=self.args.lr,
                                          weight_decay=self.args.l2)

    def _set_scheduler(self):
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=self.args.lr_dc_step,
                                                         gamma=self.args.lr_dc)

    def _set_stopper(self):
        self.stopper = EarlyStoppingNew(patience=self.args.patience, 
                                     verbose=False,
                                     path=self.args.output_dir,
                                     trace_func=self.logger)

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
        return sum(train_time)
    
    def _prepare_train_inputs(self, data):
        """Prepare the inputs as a dict for training"""
        assert len(self.generator.train_dataset.var_name) == len(data)
        inputs = {}
        for i, var_name in enumerate(self.generator.train_dataset.var_name):
            inputs[var_name] = data[i]

        return inputs
    
    def _prepare_eval_inputs(self, data):
        """Prepare the inputs as a dict for evaluation"""
        inputs = {}
        assert len(self.generator.eval_dataset.var_name) == len(data)
        for i, var_name in enumerate(self.generator.eval_dataset.var_name):
            inputs[var_name] = data[i]

        return inputs

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
        
        # 导入评估函数
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

    def train(self):
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        self.logger.info("\n----------------------------------------------------------------")
        self.logger.info("********** Running training **********")
        self.logger.info("  Batch size = %d", self.args.train_batch_size)
        res_list = []
        train_time = []

        for epoch in trange(self.start_epoch, self.start_epoch + int(self.args.num_train_epochs), desc="Epoch"):

            t = self._train_one_epoch(epoch)
            train_time.append(t)

            # evaluate on validation per epoch
            if (epoch % 1) == 0:
                metric_dict = self.eval(epoch=epoch)
                res_list.append(metric_dict)
                self.stopper(metric_dict[self.watch_metric], epoch, model_to_save, self.optimizer, self.scheduler)

                if self.stopper.early_stop:
                    break
        
        best_epoch = self.stopper.best_epoch
        best_res = res_list[best_epoch - self.start_epoch]
        self.logger.info('')
        self.logger.info('The best epoch is %d' % best_epoch)
        self.logger.info('The best results are NDCG@10: %.5f, HR@10: %.5f' %
                    (best_res['NDCG@10'], best_res['HR@10']))
        
        res = self.eval(test=True)

        # 如果需要保存embeddings
        if getattr(self.args, 'save_embeddings', False) and hasattr(self.model, 'save_item_embeddings'):
            embeddings_path = getattr(self.args, 'embeddings_save_path', '')
            if embeddings_path:
                self.logger.info(f"保存item embeddings到: {embeddings_path}")
                # 加载最佳模型状态
                model_state_dict = torch.load(os.path.join(self.args.output_dir, 'pytorch_model.bin'))
                self.model.load_state_dict(model_state_dict['state_dict'])
                self.model.to(self.device)
                # 保存embeddings
                self.model.save_item_embeddings(embeddings_path)
            else:
                self.logger.warning("save_embeddings为True但未提供embeddings_save_path")
        
        # 如果需要保存训练后的码本权重
        if getattr(self.args, 'save_codebooks', False) and hasattr(self.model, 'save_codebooks'):
            codebooks_path = getattr(self.args, 'codebooks_save_path', '')
            if codebooks_path:
                self.logger.info(f"保存训练后的码本权重到: {codebooks_path}")
                # 加载最佳模型状态
                model_state_dict = torch.load(os.path.join(self.args.output_dir, 'pytorch_model.bin'))
                self.model.load_state_dict(model_state_dict['state_dict'])
                self.model.to(self.device)
                # 保存码本权重
                self.model.save_codebooks(codebooks_path)
            else:
                self.logger.warning("save_codebooks为True但未提供codebooks_save_path")

        return res, best_epoch

    def test(self):
        """Do test directly. Set the output dir as the path that save the checkpoint"""
        res = self.eval(test=True)
        return res, -1

    def get_model(self):
        return self.model
    
    def get_model_param_num(self):
        total_num = sum(p.numel() for p in self.model.parameters())
        trainable_num = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        freeze_num = total_num - trainable_num

        return freeze_num, trainable_num
