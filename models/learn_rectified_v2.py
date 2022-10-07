### learning scripts
import torch
from torch import nn
from tqdm import tqdm
import pdb
from models.model_rectified_v2 import MoeLike
import os
from distillation import Linf_PGD, Linf_distillation
import json
from utils.utils_data import VisiualLoader,  DistillationLoader
import torch.optim as optim
from eval.eval import pgd, robust_eval, black_box, transfer_box, transfer_box_all
import numpy as np
from line_profiler import LineProfiler


class Learning(object):
    def __init__(self, args, logger, train_loader, test_loader):
        self.args = args
        self.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
        self.model = MoeLike(args, self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.logger = logger
        self.global_epoch = 0
        self.pickle_record = {"train":{}, "valid":{}, "robust_valid": {}, "white_box": {}, "black_box": {}, "eval": {}}
        self.all_args_save(args)
        self.set_optim(args)
        self.attack_cfg = {'eps': self.args.attack_eps, 
                'alpha': self.args.alpha,
                'steps': 10,
                'is_targeted': False,
                'rand_start': self.args.rand_start}

        self.distill_cfg = {'eps': self.args.attack_eps, 
                           'alpha': self.args.distill_alpha,
                           'steps': self.args.distill_steps,
                           'layer': self.args.distill_layer,
                           'rand_start': self.args.distill_rand_start,
                           'before_relu': True,
                           'momentum': self.args.distill_momentum}

    def all_args_save(self, args):
        with open(os.path.join(self.args.log_dir, "args.json"), "w") as f:
            json.dump(args.__dict__, f, indent = 2)

    def set_optim(self, args):
        if args.optim_type == "sgd":
            self.optims = [optim.SGD(model.parameters(), lr=args.lr_expert, momentum=args.momentum,
                            weight_decay=args.weight_decay) for model in self.model.experts]
            self.scheds = [optim.lr_scheduler.MultiStepLR(optim_i, milestones=args.intervals, gamma=args.gamma) for optim_i in self.optims]
            if "gate_base" in args.train_type or "moe" in args.train_type:
                self.optim_gate = optim.SGD(self.model.gate.parameters(), lr=args.lr_gate, momentum=args.momentum,
                                weight_decay=args.weight_decay) 
                self.sched_gate = optim.lr_scheduler.MultiStepLR(self.optim_gate, milestones=args.intervals, gamma=args.gamma) 

        elif args.optim_type == "adam":
            self.optims = [optim.Adam(model.parameters(), lr=args.lr_expert, 
                            weight_decay=args.weight_decay) for model in self.model.experts]
            self.scheds = [optim.lr_scheduler.MultiStepLR(optim_i, milestones=args.intervals, gamma=args.gamma) for optim_i in self.optims]
            if "gate_base" in args.train_type:
                self.optim_gate = optim.Adam(self.model.gate.parameters(), lr=args.lr_gate,
                                weight_decay=args.weight_decay) 
                self.sched_gate = optim.lr_scheduler.MultiStepLR(self.optim_gate, milestones=args.intervals, gamma=args.gamma)            
        # self.optim_moelike = optim.SGD(self.model.parameters(), lr=args.lr_expert, momentum=args.momentum,
        #                 weight_decay=args.weight_decay) 
        # self.sched_moelike = optim.lr_scheduler.MultiStepLR(self.optim_moelike, milestones=args.intervals, gamma=args.gamma) 


    def get_epoch_iterator(self):
        iterator = tqdm(list(range(1,self.args.total_epoches+1)), total=self.args.total_epoches, desc='Epoch',
                        leave=True, position=1)
        return iterator
    
    def get_batch_iterator(self):
        loader = DistillationLoader(self.train_loader, self.train_loader)
        iterator = tqdm(loader, desc='Batch', leave=False, position=2)
        return iterator

    def acc_compute(self, prob, Y):
        y_pred = prob.data.max(1)[1]
        acc = torch.mean((y_pred==Y).float()).item()
        return acc

    def update_para(self, losses, loss_soft):
        if self.args.train_type == "normal":
            for i in range(self.args.num_experts):
                self.optims[i].zero_grad()
                losses[i].backward() 
                self.optims[i].step()   
        elif self.args.train_type == "gate_base" or self.args.train_type == "moe" :
            self.optim_gate.zero_grad()   
            for i in range(self.args.num_experts):
                self.optims[i].zero_grad()
            loss_soft.backward()
            self.optim_gate.step()     
            for i in range(self.args.num_experts):
                self.optims[i].step()             
        elif self.args.train_type == "loss_base":
            for i in range(self.args.num_experts):
                self.optims[i].zero_grad()
            loss_soft.backward()
            for i in range(self.args.num_experts):
                self.optims[i].step()              
        else:
            self.logger.info("error train type.")

    def generate_distill_data(self, Xs, Xts, Ys):
        if self.args.distill_data and (not self.args.distill_fixed_layer):
            self.logger.info('Randomly choosing a layer for distillation...')
            self.distill_cfg['layer'] = np.random.randint(1, self.args.depth)
        elif self.args.distill_data and ( self.args.distill_fixed_layer):
            self.logger.info('choosing a fixed layer for distillation...') 

        distilled_data_list = []
        distilled_label_list = []
        for m in self.model.experts:
            temp = Linf_distillation(m, Xs, Xts, **self.distill_cfg)
            distilled_data_list.append(temp)
            distilled_label_list.append(Ys)
        Xs = torch.cat(distilled_data_list, dim = 0)
        Ys = torch.cat(distilled_label_list, dim = 0)
        return Xs, Ys

    def generate_adver_data(self, Xs, Ys):
        if self.args.train_type == "loss_base" and self.args.plus_at_type == "one_by_one":
            adv_inputs_list = []
            adv_outputs_list = []
            for m in self.model.experts:
                temp = Linf_PGD(m, Xs, Ys, **self.attack_cfg)
                adv_inputs_list.extend(temp)
                adv_outputs_list.extend(Ys)
            Xs_adv = torch.stack(adv_inputs_list)
            Ys_adv = torch.stack(adv_outputs_list)
        else:
            Xs_adv = Linf_PGD(self.model, Xs, Ys, **self.attack_cfg)
            Ys_adv = Ys 
        rand_list = torch.randperm(Xs_adv.size(0))
        Xs_adv = Xs_adv[rand_list]
        Ys_adv = Ys_adv[rand_list]
        return Xs_adv, Ys_adv


    def train(self, epoch):
        for m in self.model.experts:
            m.train()

        batch_iter = self.get_batch_iterator()

        for iteration, (Xs, Ys, Xts, Yts) in enumerate(batch_iter):
            self.pickle_record["train"][str(epoch)][str(iteration)] = {}
            Xs, Ys = Xs.to(self.device), Ys.to(self.device)
            Xts = Xts.to(self.device)

            if self.args.distill_data:
                Xs, Ys = self.generate_distill_data( Xs, Xts, Ys)
            loss_originals, loss_experts, loss_soft, loss, aux_loss, aux_acc, acc_originals, probs, acc = self.model(X = Xs, Y = Ys, if_attack=False)


            if self.args.plus_at:
                Xs_adv, Ys_adv = self.generate_adver_data(Xs, Ys)
                adv_loss_originals, adv_loss_experts, adv_loss_soft, adv_loss, adv_aux_loss, adv_aux_acc, adv_acc_originals, adv_probs, adv_acc = self.model(X = Xs_adv, Y = Ys_adv, if_attack=False)


                self.update_para(loss_originals+adv_loss_originals, loss_soft + adv_loss_soft)
                adv_loss_originals = [loss.item() for loss in adv_loss_originals]
                adv_loss_soft = adv_loss_soft.item()
                loss_originals = [loss.item() for loss in loss_originals]
                loss_soft = loss_soft.item()
                content = [adv_loss_originals, adv_loss_experts, adv_loss_soft,adv_loss, adv_aux_loss, adv_aux_acc, adv_acc_originals,  adv_acc]
                for idx, name in enumerate(["adv_loss_originals", "adv_loss_experts", "adv_loss_soft", "adv_loss",  "adv_aux_loss", "adv_aux_acc", "adv_acc_originals",  "adv_acc",]):
                    self.pickle_record["train"][str(epoch)][str(iteration)][name] = content[idx]

                self.logger.info("training: epoch: {}, iteration: {}, loss_orignals: {}, loss_experts: {},  loss_soft: {}, loss: {},  aux_loss: {}, aux_acc: {},  acc_originals: {},acc: {}, \
                 adv_loss_originals: {}, adv_loss_experts: {}, adv_loss_soft: {}, adv_loss: {}, adv_aux_loss: {}, adv_aux_acc:{}, adv_acc_originals: {},  adv_acc: {}.".format( epoch, iteration, loss_originals,\
                loss_experts, loss_soft, loss, aux_loss, aux_acc, acc_originals,  acc, adv_loss_originals, adv_loss_experts, adv_loss_soft, adv_loss, adv_aux_loss, adv_aux_acc, adv_acc_originals, adv_acc ))

            else:
                self.update_para(loss_originals, loss_soft)

                loss_originals = [loss.item() for loss in loss_originals]
                loss_soft = loss_soft.item()
                content = [loss_originals, loss_experts, loss_soft, loss, aux_loss, aux_acc, acc_originals,  acc]
                for idx, name in enumerate(["loss_originals", "loss_experts", "loss_soft", "loss",  "aux_loss", "aux_acc" ,"acc_originals",  "acc"]):
                    self.pickle_record["train"][str(epoch)][str(iteration)][name] = content[idx]

                self.logger.info("training: epoch: {}, iteration: {}, loss_orignals: {}, loss_experts: {},  loss_soft: {}, loss: {}, aux_loss: {}, aux_acc: {}, acc_originals: {}, acc: {}, \
                .".format(epoch, iteration, loss_originals,loss_experts,  loss_soft, loss,  aux_loss, aux_acc, acc_originals,acc))


        if "gate_base" in self.args.train_type or "moe" in self.args.train_type:
            for i in range(self.args.num_experts):
                self.scheds[i].step()
            self.sched_gate.step()
        else:
            for i in range(self.args.num_experts):
                self.scheds[i].step()



    def train_attack(self, epoch):
        self.pickle_record[self.args.attack_type][str(epoch)] = {}
        print("-"*100 + str(self.args.attack_eps))
        if self.args.attack_type == "white_box":
            # for adapt_attack_weight in [0.2, 0.4, 0.6, 0.8, 1.0]:
            #     self.args.adapt_attack_weight = adapt_attack_weight
            #     adv_Xs, Ys = pgd(self.model, self.args, self.device)
            #     self.robust_valid( epoch = epoch, Xs = adv_Xs, Ys = Ys, attack = True)        
            
            adv_Xs, Ys = pgd(self.model, self.args, self.device)
            self.robust_valid( epoch = epoch, Xs = adv_Xs, Ys = Ys, attack = True) 


        elif self.args.attack_type == "black_box":
            self.args.black_datafolder = "" 
            if self.args.black_mode_clean:
                self.args.black_datafolder = os.path.join(self.args.black_datafolder, "clean")
            else:
                self.args.black_datafolder = os.path.join(self.args.black_datafolder, "eps_{}".format(self.args.attack_eps))
                if self.args.black_method != "mpgd":    
                    self.args.black_datafolder = os.path.join(self.args.black_datafolder, "from_baseline{}_{}_{}_steps_100".format(self.args.black_surro_num_experts, self.args.attack_loss_fn, self.args.black_method))
                else:
                    self.args.black_datafolder = os.path.join(self.args.black_datafolder, "from_baseline{}_{}_{}_steps_100_{}".format(self.args.black_surro_num_experts, self.args.attack_loss_fn, self.args.black_method, self.args.black_random_start_seed))

            adv_Xs = torch.load(os.path.join(self.model.args.black_datafolder, 'inputs.pt')).to(self.device)
            Ys = torch.load(os.path.join(self.model.args.black_datafolder, 'labels.pt')).to(self.device)
            self.logger.info("the black-box evaluation is beginning (the adv data is from {}).".format(self.model.args.black_datafolder))
            self.robust_valid( epoch = epoch, Xs = adv_Xs, Ys = Ys, attack = True)        
        else:
            self.logger.info("error attack_type, ignored~")
            pass


    def save(self, ckptname):
        model_dicts = {}
        os.makedirs(self.args.model_dir, exist_ok = True)
        filepath = os.path.join(self.args.model_dir, str(ckptname))
        for i in range(self.args.num_experts):
            model_dicts["expert_"+str(i)] = {
                "epoch": self.global_epoch,
                "model": self.model.experts[i].state_dict(),
                "optim": self.optims[i].state_dict()}
        if "gate_base" in self.args.train_type or "moe" in self.args.train_type:
            model_dicts["gate"] = {
                    "epoch": self.global_epoch,
                    "model": self.model.gate.state_dict(),
                    "optim": self.optim_gate.state_dict()}
        with open(filepath, 'wb+') as f:
            torch.save(model_dicts, f)
        self.logger.info("=> model saved checkpoint '{}' (epoch {})".format(filepath, self.global_epoch))  


    def load(self, ckptname = "last"):
        if  ckptname == None:
            ckpts = os.listdir(self.args.model_dir)
            if not ckpts:
                self.logger.info("=> no checkpoint found")
                exit()
            ckpts = [int(ckpt) for ckpt in ckpts]
            ckpts.sort(reverse=True)
            ckptname = str(ckpts[0])
            filepath = os.path.join(self.args.model_dir, str(ckptname))
        else:
            filepath = ckptname
        
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                checkpoint = torch.load(f, map_location=self.device)
            for i in range(self.args.num_experts):
                self.model.experts[i].load_state_dict(checkpoint["expert_" + str(i)]['model'])
                # self.optims[i].load_state_dict(checkpoint["expert_" + str(i)]['optim'])
                # self.global_epoch = checkpoint["expert_" + str(i)]['epoch']
            if "gate_base" in self.args.train_type or self.args.train_type == "moe" :
                self.model.gate.load_state_dict(checkpoint["gate"]['model'])
                self.optim_gate.load_state_dict(checkpoint["gate"]['optim'])
            self.logger.info("=> model loaded checkpoint '{} (epoch {})'".format(filepath, self.global_epoch))
        else:
            self.logger.info("=> no checkpoint found at '{}', start re-training".format(filepath))


    def robust_valid(self, epoch, Xs = None, Ys = None, attack = False):
        for m in self.model.experts:
            m.eval()
        with torch.no_grad():
            if attack == False:
                if str(self.args.train_type) not in self.pickle_record["robust_valid"].keys():
                    self.pickle_record["robust_valid"][str(self.args.train_type)] = {}
                if str(epoch) not in self.pickle_record["robust_valid"][str(self.args.train_type)].keys():
                    self.pickle_record["robust_valid"][str(self.args.train_type)][str(epoch)] = {}

                for iteration, (Xs, Ys) in enumerate(self.test_loader):
                    Xs, Ys = Xs.to(self.device), Ys.to(self.device)
                    loss_originals, loss_experts, loss_soft, loss, aux_loss, aux_acc, acc_originals, probs, acc =  self.model(X = Xs, Y = Ys, if_attack=False)
                loss_originals = [loss.item() for loss in loss_originals]
                loss_soft = loss_soft.item()
                content = [loss_originals, loss_experts, loss_soft, loss, acc_originals, acc, aux_loss, aux_acc, loss_experts]
                for idx, name in enumerate(["loss_originals", "loss_experts", "loss_soft", "loss", "acc_originals", "acc", "aux_loss", "aux_acc", "loss_experts"]):
                    self.pickle_record["robust_valid"][str(self.args.train_type)][str(epoch)][name] = content[idx]
                self.logger.info("valid_type: {}, epoch: {}, iteration: {}, loss_orignals: {}, loss_experts: {}, loss_soft: {}, loss: {}, aux_loss: {}, aux_acc: {}, acc_originals: {}, acc: {}.".format( \
                self.args.train_type, epoch, iteration, loss_originals, loss_experts, loss_soft, loss, aux_loss, aux_acc, acc_originals, acc))

            else:
                loss_originals, loss_experts, loss_soft, loss, aux_loss, aux_acc, acc_originals, probs, acc =  self.model(X = Xs, Y = Ys, if_attack=False)
                loss_originals = [loss.item() for loss in loss_originals]
                loss_soft = loss_soft.item()                  
                content = [loss_originals, loss_experts, loss_soft, loss, aux_loss, aux_acc, acc_originals, acc]
                for idx, name in enumerate(["loss_originals", "loss_experts", "loss_soft", "loss", "aux_loss",  "aux_acc", "acc_originals", "acc"]):
                    self.pickle_record[self.args.attack_type][str(epoch)][name] = content[idx]
                self.logger.info("attack_type: {}, epoch: {}, valid_type: {}, attack eps: {}, loss_orignals: {},loss_experts: {}, loss_soft: {}, loss: {}, aux_loss: {}, aux_acc: {}, acc_originals: {}, acc: {}.".format( \
                self.args.attack_type, epoch,  self.args.train_type, self.args.attack_eps, loss_originals,loss_experts, loss_soft, loss, aux_loss, aux_acc, acc_originals, acc))


    def run(self):
        ini_train_type = ""
        ini_train_type = self.args.train_type
        epoch_iter = self.get_epoch_iterator()
        for epoch in epoch_iter:
            self.pickle_record["train"][str(epoch)] = {}

            self.train(epoch)
            if (epoch) % self.args.test_interval_epoches == 0:
                for robust_valid in ["normal", ini_train_type]:
                    self.args.train_type = robust_valid
                    self.robust_valid(epoch)
            self.args.train_type = ini_train_type

            if self.args.plus_at:
                ###### training attack to measure the resultes
                if (epoch) % self.args.attack_interval_epoches == 0:
                        self.args.attack_type = "white_box"
                        self.args.attack_loss_fn = "xent"
                        self.train_attack(epoch)


                ###### eval the robust results
                # if (epoch) % self.args.robust_eval_interval_epoches == 0:
                #     self.pickle_record["eval"][str(epoch)] = {}
                #     for m in self.model.experts:
                #         m.eval()
                #     for eval_type in ["white_box"]:
                #         results = robust_eval(eval_type, self.model, self.args, self.device, self.logger)
                #         self.pickle_record["eval"][str(epoch)][eval_type] = results
                if (epoch) % self.args.save_interval_epoches == 0:
                    self.save(epoch)
           
            self.global_epoch+=1


    def evaluate(self, epoch, evaluate_type):
        self.logger.info("robust evaluation...")
        self.model.eval()
        for m in self.model.experts:
            m.eval()
        with torch.no_grad():
            if evaluate_type == "robust_valid":
                self.logger.info("evaluating on all clean test data")
                self.robust_valid(epoch)
            elif evaluate_type == "trans_box":
                if str(epoch) not in self.pickle_record["trans_box"].keys():
                    self.pickle_record["trans_box"][str(epoch)] = {}
                if str(epoch) not in self.pickle_record["trans_box_all"].keys():
                    self.pickle_record["trans_box_all"][str(epoch)] = {}

                acc = transfer_box(self.model, self.args, self.device, self.logger)
                self.pickle_record["trans_box"][str(epoch)]["acc"] = acc
                acc = transfer_box_all(self.model, self.args, self.device, self.logger)
                self.pickle_record["trans_box_all"][str(epoch)]["acc"] = acc
            elif evaluate_type == "black_box":
                if str(epoch) not in self.pickle_record["black_box"].keys():
                    self.pickle_record["black_box"][str(epoch)] = {}  

                acc = black_box(self.model, self.args, self.device, self.logger)
                print("eps: {}, acc: {}".format(self.args.attack_eps, acc))
                self.pickle_record[evaluate_type][str(epoch)]["acc"] = acc

        if evaluate_type == "trans_white_box":

            self.white_box_evaluate(epoch, evaluate_type = "white_box")
            with torch.no_grad():
                if str(epoch) not in self.pickle_record["trans_box"].keys():
                    self.pickle_record["trans_box"][str(epoch)] = {}
                if str(epoch) not in self.pickle_record["trans_box_all"].keys():
                    self.pickle_record["trans_box_all"][str(epoch)] = {}

                acc = transfer_box(self.model, self.args, self.device, self.logger)
                self.pickle_record["trans_box"][str(epoch)]["acc"] = acc
                acc = transfer_box_all(self.model, self.args, self.device, self.logger)
                self.pickle_record["trans_box_all"][str(epoch)]["acc"] = acc
