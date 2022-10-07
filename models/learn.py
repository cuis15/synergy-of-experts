### learning scripts
import torch
from torch import nn
from tqdm import tqdm
import pdb
from models.model import MoeLike
import os
from distillation import Linf_PGD, Linf_distillation
import json
from utils.utils_data import VisiualLoader,  DistillationLoader
import torch.optim as optim
from eval.eval_wbox import white_box, pgd
import numpy as np 

class Learning(object):
    def __init__(self, args, logger, train_loader, test_loader):
        self.args = args
        self.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
        self.model = MoeLike(args, self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.logger = logger
        self.global_epoch = 0
        self.pickle_record = {"train":{}, "valid":{}, "robust_valid": {}, "white_box": {}, "black_box": {}, "transfer_attack": {}, "diversity": {}}
        self.all_args_save(args)
        self.set_optim(args)

    def all_args_save(self, args):
        with open(os.path.join(self.args.log_dir, "args.json"), "w") as f:
            json.dump(args.__dict__, f, indent = 2)

    def set_optim(self, args):
        self.optims = [optim.SGD(model.parameters(), lr=args.lr_expert, momentum=args.momentum,
                        weight_decay=args.weight_decay) for model in self.model.experts]
        self.optim_gate = optim.SGD(self.model.gate.parameters(), lr=args.lr_gate, momentum=args.momentum,
                        weight_decay=args.weight_decay) 
        self.scheds = [optim.lr_scheduler.MultiStepLR(optim_i, milestones=args.intervals, gamma=args.gamma) for optim_i in self.optims]
        self.sched_gate = optim.lr_scheduler.MultiStepLR(self.optim_gate, milestones=args.intervals, gamma=args.gamma) 
        
    def get_epoch_iterator(self):
        iterator = tqdm(list(range(1,self.args.total_epoches+1)), total=self.args.total_epoches, desc='Epoch',
                        leave=True, position=1)
        return iterator
    
    def get_batch_iterator(self, loader):
        loader = DistillationLoader(loader, loader)
        iterator = tqdm(loader, desc='Batch', leave=False, position=2)
        return iterator


    def acc_compute(self, prob, Y):
        # pdb.set_trace()
        y_pred = prob.data.max(1)[1]
        acc = torch.mean((y_pred==Y).float()).item()
        return acc


    def train(self, epoch):
        for i in range(self.args.num_experts):
            self.model.experts[i].train()
        self.attack_cfg = {'eps': self.args.attack_eps, 
                'alpha': self.args.alpha,
                'steps': self.args.attack_steps,
                'is_targeted': self.args.is_targeted,
                'rand_start': self.args.rand_start}
        self.distill_cfg = {'eps': self.args.distill_eps, 
                           'alpha': self.args.distill_alpha,
                           'steps': self.args.distill_steps,
                           'layer': self.args.distill_layer,
                           'rand_start': self.args.distill_rand_start,
                           'before_relu': True,
                           'momentum': self.args.distill_momentum}

        batch_iter = self.get_batch_iterator(self.train_loader)

        for m in self.model.experts:
            m.train()
        if not self.args.distill_fixed_layer:
            self.logger.info('Randomly choosing a layer for distillation...')
            self.distill_cfg['layer'] = np.random.randint(1, self.args.depth)


        for iteration, (Xs, Ys, Xts, Yts) in enumerate(batch_iter):
            self.pickle_record["train"][str(epoch)][str(iteration)] = {}
            Xs, Ys = Xs.to(self.device), Ys.to(self.device)
            Xts, Yts = Xts.to(self.device), Yts.to(self.device)

            if self.args.distill_data:
                distilled_data_list = []
                distilled_label_list = []
                for m in self.model.experts:
                    temp = Linf_distillation(m, Xs, Xts, **self.distill_cfg)
                    distilled_data_list.append(temp)
                    distilled_label_list.append(Ys)

                Xs = torch.cat(distilled_data_list, dim = 0)
                Ys = torch.cat(distilled_label_list, dim = 0)

            losses, loss_experts, loss_soft,acc_experts, probs = self.model(X = Xs, Y = Ys, if_attack=False)
            acc = self.acc_compute(probs, Ys)

            if self.args.plus_at:
                adv_inputs_list = []
                adv_outputs_list = []
                for m in self.model.experts:
                    temp = Linf_PGD(m, Xs, Ys, **self.attack_cfg)
                    adv_inputs_list.extend(temp)
                    adv_outputs_list.extend(Ys)
                adv_losses, adv_loss_experts, adv_loss_soft, adv_acc_experts, adv_probs = self.model(X = torch.stack(adv_inputs_list), Y = torch.stack(adv_outputs_list), if_attack=False)
                adv_acc = self.acc_compute(adv_probs, torch.stack(adv_outputs_list))
                self.pickle_record["train"][str(epoch)][str(iteration)]["adv_losses_original"] = [loss.item() for loss in adv_losses]
                self.pickle_record["train"][str(epoch)][str(iteration)]["adv_loss_experts"] = [loss.item() for loss in adv_loss_experts]
                self.pickle_record["train"][str(epoch)][str(iteration)]["adv_acc_experts"] = [acc for acc in adv_acc_experts]
                self.pickle_record["train"][str(epoch)][str(iteration)]["adv_loss_soft"] = adv_loss_soft.item()
                self.pickle_record["train"][str(epoch)][str(iteration)]["adv_acc"] = adv_acc 
                self.logger.info("training: epoch: {}, iteration: {}, losses_orignal: {}, loss_experts: {}, acc_experts: {}, loss_soft: {}, acc: {}, \
                 adv_losses_original: {}, adv_loss_experts: {}, adv_loss_soft: {}, adv_acc: {}.".format( epoch, iteration, [loss.item() for loss in losses],\
                [loss.item() for loss in loss_experts], [acc for acc in adv_acc_experts], loss_soft.item(), acc, [loss.item() for loss in adv_losses], [loss.item() for loss in adv_loss_experts], \
                    adv_loss_soft.item(), adv_acc ))   

                if self.args.train_type == "loss_base":
                    for i in range(self.args.num_experts):
                        self.optims[i].zero_grad()
                    (loss_soft + adv_loss_soft).backward()
                    for i in range(self.args.num_experts):
                        self.optims[i].step()  

                elif  self.args.train_type == "kl_base":     
                    for i in range(self.args.num_experts):
                        self.optims[i].zero_grad()
                        (loss_experts[i] + adv_loss_experts[i]).backward()
                        self.optims[i].step()                           

            else:
                if self.args.train_type == "loss_base":
                    for i in range(self.args.num_experts):
                        self.optims[i].zero_grad()
                    (loss_soft).backward()
                    for i in range(self.args.num_experts):
                        self.optims[i].step()  

                elif  self.args.train_type == "kl_base":     
                    for i in range(self.args.num_experts):
                        self.optims[i].zero_grad()
                        (loss_experts[i]).backward()
                        self.optims[i].step()   
                self.logger.info("training: epoch: {}, iteration: {}, losses_orignal: {}, loss_experts: {}, acc_experts: {}, loss_soft: {}, acc: {}, \
                .".format(epoch, iteration,  [loss.item() for loss in losses],\
                [loss.item() for loss in loss_experts], [acc for acc in acc_experts], loss_soft.item(), acc))
            self.pickle_record["train"][str(epoch)][str(iteration)]["losses_original"] = [loss.item() for loss in losses]
            self.pickle_record["train"][str(epoch)][str(iteration)]["loss_experts"] = [loss.item() for loss in loss_experts]
            self.pickle_record["train"][str(epoch)][str(iteration)]["acc_experts"] = [acc for acc in acc_experts]
            self.pickle_record["train"][str(epoch)][str(iteration)]["loss_soft"] = loss_soft.item()
            self.pickle_record["train"][str(epoch)][str(iteration)]["acc"] = acc
            
        for i in range(self.args.num_experts):
            self.scheds[i].step()



    def valid(self, epoch):
        with torch.no_grad():
            for iteration, (Xs, Ys) in enumerate(self.test_loader):
                Xs, Ys = Xs.to(self.device), Ys.to(self.device)
                losses, loss_experts, loss_soft, acc_experts, probs = self.model(Xs, Ys, if_attack=False)
                acc = self.acc_compute(probs, Ys)
                try:
                    self.pickle_record["valid"][str(epoch)][str(iteration)] = {}
                except KeyError:
                    self.pickle_record["valid"][str(epoch)] = {}
                    self.pickle_record["valid"][str(epoch)][str(iteration)] = {}
                self.pickle_record["valid"][str(epoch)][str(iteration)]["losses_original"] = [loss.item() for loss in losses]
                self.pickle_record["valid"][str(epoch)][str(iteration)]["loss_experts"] = [loss.item() for loss in loss_experts]
                self.pickle_record["valid"][str(epoch)][str(iteration)]["acc_experts"] = [acc for acc in acc_experts]
                self.pickle_record["valid"][str(epoch)][str(iteration)]["loss_gate"] = loss_soft.item()
                self.pickle_record["valid"][str(epoch)][str(iteration)]["acc"] = acc
                self.logger.info("valid: epoch: {}, iteration: {}, losses_orignal: {}, loss_experts: {},acc_experts: {}, loss_soft: {}, acc: {} \
                 .".format( epoch, iteration, [loss.item() for loss in losses], [loss.item() for loss in loss_experts], [acc for acc in acc_experts], loss_soft.item(), acc))



    def attack(self, epoch):
        

        if self.args.attack_loss_fn not in self.pickle_record[self.args.attack_type].keys():
            self.pickle_record[self.args.attack_type][self.args.attack_loss_fn] = {}
        if str(self.args.attack_eps) not in self.pickle_record[self.args.attack_type][self.args.attack_loss_fn].keys():
            self.pickle_record[self.args.attack_type][self.args.attack_loss_fn][str(self.args.attack_eps)] = {}
        self.pickle_record[self.args.attack_type][self.args.attack_loss_fn][str(self.args.attack_eps)][str(epoch)] = {}


        if self.args.attack_type == "white_box":
            adv_Xs, Ys = pgd(self.model, self.args, self.device)
            self.logger.info("the white-box adv samples have been generated.")
            robust_measure, experts_outputs, Ys =  self.robust_valid( epoch = epoch, Xs = adv_Xs, Ys = Ys, attack = True)        
            return robust_measure, experts_outputs, Ys

        elif self.args.attack_type == "black_box":

            self.args.black_datafolder = "/home/cuisen/workspace/adv_train/DVERGE-main/data/transfer_adv_examples" 
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
            robust_measure, experts_outputs, Ys =  self.robust_valid( epoch = epoch, Xs = adv_Xs, Ys = Ys, attack = True)        
            return robust_measure, experts_outputs, Ys

        else:
            self.logger.info("error attack_type, ignored~")
            pass


    def transfer(self, epoch):
        adv_model_samples = []
        transfer_matrix = np.zeros((self.args.num_experts, self.args.num_experts))
        robust_matrix = []
        minimal_loss_matrix = []
        max_confidence_matrix = []
        for m in self.model.experts:
            adv_Xs, Ys = pgd(m, self.args, self.device)
            adv_model_samples.append([adv_Xs, Ys])
        for i, (adv_Xs, Ys) in enumerate(adv_model_samples):
            for j, m in enumerate(self.model.experts):
                if j == i:
                    outputs = m(adv_Xs)
                    _, pred = outputs.max(1)
                    # assert pred.eq(Ys).all()
                    transfer_matrix[i, i] = torch.mean((pred==Ys).float()).item()
                else:
                    outputs = m(adv_Xs)
                    _, pred = outputs.max(1)                
                    transfer_matrix[i, j] = torch.mean((pred==Ys).float()).item()
            self.args.train_type = "loss_base"
            loss_original, loss_experts, loss_soft, acc_experts, probs  = self.model(X = adv_Xs, Y = Ys, if_attack = False)
            minimal_loss_matrix.append(self.acc_compute(probs, Ys))
            self.args.kl_select_type = "argmin"    
            kl_loss, loss_original, loss_experts, experts_outputs, probs, loss_final = self.model.kl_loss_logit_compute( adv_Xs, Ys)
            robust_matrix.append(self.acc_compute(probs, Ys))
            self.args.valid_type = "max_confidence"
            robust_measure, experts_outputs, loss_original, probs, loss_experts, loss_final   = self.model.valid(X = adv_Xs, Y = Ys, valid_type=self.args.valid_type)
            max_confidence_matrix.append(self.acc_compute(probs, Ys))
        
        return transfer_matrix, minimal_loss_matrix, robust_matrix, max_confidence_matrix


    def save(self, ckptname = None):
        model_dicts = {}
        if ckptname == None:
            ckptname = str(self.global_epoch)
        os.makedirs(self.args.model_dir, exist_ok = True)
        filepath = os.path.join(self.args.model_dir, str(ckptname))
        for i in range(self.args.num_experts):
            model_dicts["expert_"+str(i)] = {
                "epoch": self.global_epoch,
                "model": self.model.experts[i].state_dict(),
                "optim": self.optims[i].state_dict()}
        model_dicts["gate"] = {
                "epoch": self.global_epoch,
                "model": self.model.gate.state_dict(),
                "optim": self.optim_gate.state_dict()}
        with open(filepath, 'wb+') as f:
            torch.save(model_dicts, f)
        self.logger.info("=> model saved checkpoint '{}' (epoch {})".format(filepath, self.global_epoch))  


    def load(self, ckptname = "last"):
        if ckptname == 'last':
            ckpts = os.listdir(self.args.model_dir)
            if not ckpts:
                self.logger.info("=> no checkpoint found")
                exit()
            ckpts = [int(ckpt) for ckpt in ckpts]
            ckpts.sort(reverse=True)
            ckptname = str(ckpts[0])
        filepath = os.path.join(self.args.model_dir, str(ckptname))
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                checkpoint = torch.load(f, map_location=self.device)
            for i in range(self.args.num_experts):
                self.model.experts[i].load_state_dict(checkpoint["expert_" + str(i)]['model'])
                self.optims[i].load_state_dict(checkpoint["expert_" + str(i)]['optim'])
                self.global_epoch = checkpoint["expert_" + str(i)]['epoch']

            self.model.gate.load_state_dict(checkpoint["gate"]['model'])
            self.optim_gate.load_state_dict(checkpoint["gate"]['optim'])
            self.logger.info("=> model loaded checkpoint '{} (epoch {})'".format(filepath, self.global_epoch))
        else:
            self.logger.info("=> no checkpoint found at '{}', start re-training".format(filepath))


    def load_pre_train_model(self, ckptname = "epoch_200.pth"):
        try:
            filepath = os.path.join(self.args.pre_train_model_dir, str(ckptname))

            with open(filepath, 'rb') as f:
                checkpoint = torch.load(f, map_location=self.device)
            for i in range(self.args.num_experts):
                try:
                    self.model.experts[i].load_state_dict(checkpoint["model_"+str(i)])
                except:
                    self.model.experts[i].load_state_dict(checkpoint["model"+str(i)])
            self.logger.info("=> model loaded checkpoint '{}'.".format(filepath, self.global_epoch))
        except Exception as e:
            self.logger.info(e)
            self.logger.info("=> no checkpoint found at '{}'".format(filepath))
            exit()


    def pre_train(self, epoch, if_train_experts = True, if_train_gate = False):
        self.attack_cfg = {'eps': self.args.attack_eps, 
                'alpha': self.args.alpha,
                'steps': self.args.steps,
                'is_targeted': self.args.is_targeted,
                'rand_start': self.args.rand_start}
        if if_train_experts:
            for i in range(self.args.num_experts):
                self.model.experts[i].train()
        else:
            for i in range(self.args.num_experts):
                self.model.experts[i].eval()      
        if if_train_gate:      
            self.model.gate.train()
        else:
            self.model.gate.eval()
        batch_iter = self.get_batch_iterator(self.train_loader)
        self.loss_func = nn.CrossEntropyLoss()
        for iteration, (Xs, Ys) in enumerate(batch_iter):
            
            self.pickle_record["train"][str(epoch)][str(iteration)] = {}
            Xs, Ys = Xs.to(self.device), Ys.to(self.device)
            outputs = self.model(X = Xs, Y = Ys, if_attack=False, pre_train_experts=True)
            loss = [self.loss_func(outputs[i], Ys) for i in range(self.args.num_experts)]
            acc = [self.acc_compute(outputs[i], Ys) for i in range(self.args.num_experts)]

            if self.args.plus_at:
                adv_loss = []
                adv_acc = []
                for idx,  m in enumerate(self.model.experts):
                    temp = Linf_PGD(m, Xs, Ys, **self.attack_cfg)
                    outputs = self.model(temp, Ys, if_attack=False, pre_train_experts=True)
                    adv_loss.append(self.loss_func(outputs[idx], Ys))
                    adv_acc.append(self.acc_compute(outputs[idx], Ys))

                self.pickle_record["train"][str(epoch)][str(iteration)]["adv_loss"] = [l.item() for l in adv_loss]
                self.pickle_record["train"][str(epoch)][str(iteration)]["adv_acc"] = adv_acc 
                self.logger.info("training: epoch: {}, iteration: {},  loss: {}, acc: {}, adv_loss: {}, adv_acc: \
                     {}.".format( epoch, iteration, [l.item() for l in loss], acc, [l.item() for l in adv_loss], adv_acc))  

                if if_train_experts:             
                    for i in range(self.args.num_experts):
                        (loss[i] + adv_loss[i]).backward()
                        self.optims[i].step()
                        self.optims[i].zero_grad()    
                if if_train_gate:
                    self.optim_gate.step()
                    self.optim_gate.zero_grad()

            else:
                if if_train_experts:
                    for i in range(self.args.num_experts):
                        self.optims[i].zero_grad()
                        loss[i].backward()
                        self.optims[i].step()
                        self.optims[i].zero_grad()
                if if_train_gate: 
                    self.optim_gate.zero_grad() 
                    self.optim_gate.step()         
                    self.optim_gate.zero_grad()              
                self.logger.info("training: epoch: {}, iteration: {}, loss: {},  acc: {}.".format(epoch, iteration, [l.item() for l in loss], acc))
 
            self.pickle_record["train"][str(epoch)][str(iteration)]["loss"] = [l.item() for l in loss]
            self.pickle_record["train"][str(epoch)][str(iteration)]["acc"] = acc
            
        if if_train_experts:
            for i in range(self.args.num_experts):
                self.scheds[i].step()
        if if_train_gate:
            self.sched_gate.step()



    def robust_valid(self, epoch, Xs = None, Ys = None, attack = False):
        with torch.no_grad():
            if attack == False:
                if str(self.args.valid_type) not in self.pickle_record["robust_valid"].keys():
                    self.pickle_record["robust_valid"][str(self.args.valid_type)] = {}
                if str(epoch) not in self.pickle_record["robust_valid"][str(self.args.valid_type)].keys():
                    self.pickle_record["robust_valid"][str(self.args.valid_type)][str(epoch)] = {}

                for iteration, (Xs, Ys) in enumerate(self.test_loader):
                    Xs, Ys = Xs.to(self.device), Ys.to(self.device)
                    robust_measure, experts_outputs, loss_original, probs_final, loss_experts, loss_final  = self.model.valid(X = Xs, Y = Ys, valid_type=self.args.valid_type)
                    acc_original = [self.acc_compute(prob, Ys) for prob in experts_outputs]
                    acc_final = self.acc_compute(probs_final, Ys)
                    self.pickle_record["robust_valid"][str(self.args.valid_type)][str(epoch)]["losses_original"] = loss_original
                    self.pickle_record["robust_valid"][str(self.args.valid_type)][str(epoch)]["loss_final"] = loss_final
                    self.pickle_record["robust_valid"][str(self.args.valid_type)][str(epoch)]["acc_original"] = acc_original
                    self.pickle_record["robust_valid"][str(self.args.valid_type)][str(epoch)]["acc_final"] = acc_final
                    self.logger.info("valid_type: {}, epoch: {}, iteration: {}, losses_orignal: {}, loss_final: {}, acc_original: {}, acc_final: {}.".format(self.args.valid_type, epoch, iteration, loss_original, loss_final, acc_original, acc_final))


            else:
                robust_measure, experts_outputs, loss_original, probs_final, loss_experts, loss_final  = self.model.valid(X = Xs, Y = Ys, valid_type=self.args.valid_type)
                acc_original = [self.acc_compute(prob, Ys) for prob in experts_outputs]
                acc_final = self.acc_compute(probs_final, Ys)
                if self.args.attack_type == "white_box":
                    pgd_steps = self.args.attack_steps
                    attack_type = "white_box"
                elif self.args.attack_type == "black_box":
                    attack_type = "black_box"
                    pgd_steps = 100
                self.pickle_record[attack_type][self.args.attack_loss_fn][str(self.args.attack_eps)][str(epoch)]["losses_original"] = loss_original
                self.pickle_record[attack_type][self.args.attack_loss_fn][str(self.args.attack_eps)][str(epoch)]["loss_final"] = loss_final
                self.pickle_record[attack_type][self.args.attack_loss_fn][str(self.args.attack_eps)][str(epoch)]["acc_original"] = acc_original
                self.pickle_record[attack_type][self.args.attack_loss_fn][str(self.args.attack_eps)][str(epoch)]["acc_final"] = acc_final               

                self.logger.info("attack_type: {}, valid_type: {}, attack eps: {}, pgd_steps: {}, losses_orignal: {}, loss_final: {}, acc_original: {}, acc_final: {}.".format(self.args.attack_type, \
                    self.args.valid_type, self.args.attack_eps, pgd_steps, loss_original, loss_final, acc_original, acc_final))
            return robust_measure, experts_outputs, Ys


    def run(self):
        ### regular training
        if self.args.fusion_type == "regular":
            if self.args.pre_train_model_dir != "":
                self.load_pre_train_model(ckptname = "epoch_200.pth")
            epoch_iter = self.get_epoch_iterator()
            for epoch in epoch_iter:
                self.pickle_record["train"][str(epoch)] = {}

                self.train(epoch)
                if (epoch) % self.args.test_interval_epoches == 0:

                    if self.args.valid_type not in self.pickle_record["robust_valid"].keys():
                        self.pickle_record["robust_valid"][self.args.valid_type] = {}
                    self.pickle_record["robust_valid"][self.args.valid_type][str(self.global_epoch)] = {}

                    robust_measure, experts_outputs, Ys = self.robust_valid(self.global_epoch)

                if (epoch) % self.args.attack_interval_epoches == 0:
                    robust_measure, experts_outputs, Ys = self.attack(epoch)

                self.global_epoch+=1
            self.save()

        ### train gate
        # for epoch in range(self.args.total_epoches, 200+self.args.total_epoches):
        #     self.pickle_record["train"][str(epoch)] = {}
        #     self.train(epoch, if_train_experts=False, if_train_gate=True)
        #     if (epoch) % self.args.test_interval_epoches == 0:
        #         self.pickle_record["valid"][str(epoch)] = {}
        #         self.valid(epoch)
        #     if (epoch) % self.args.attack_interval_epoches == 0:
        #         self.pickle_record[self.args.attack_type][str(epoch)] = {}
        #         self.attack(epoch, attack_type="white_box")
        #     self.global_epoch+=1
        # self.save(epoch)


        ### train fusion_type == coupling
        # epoch_iter = self.get_epoch_iterator()
        # for epoch in epoch_iter:
        #     self.pickle_record["train"][str(epoch)] = {}
        #     self.train_coupling(epoch)
        #     if (epoch) % self.args.test_interval_epoches == 0:
        #         self.pickle_record["valid"][str(epoch)] = {}
        #         self.valid(epoch)
        #     if (epoch) % self.args.attack_interval_epoches == 0:
        #         self.pickle_record[self.args.attack_type][str(epoch)] = {}
        #         self.attack(epoch, attack_type="white_box")
        #     self.global_epoch+=1
        # self.save(epoch)

        # if self.args.fusion_type == "pre_train_experts":
        #     epoch_iter = self.get_epoch_iterator()
        #     for epoch in epoch_iter:
        #         self.pickle_record["train"][str(epoch)] = {}
        #         self.pre_train(epoch)
        #         if (epoch) % self.args.test_interval_epoches == 0:
        #             self.pickle_record["valid"][str(epoch)] = {}
        #             self.valid(epoch)
        #         if (epoch) % self.args.attack_interval_epoches == 0:
        #             self.pickle_record[self.args.attack_type][str(epoch)] = {}
        #             self.attack(epoch, attack_type="white_box")
        #             for i in range(self.args.num_experts):
        #                 self.optims[i].zero_grad()
        #             self.optim_gate.zero_grad()   

        #         self.global_epoch+=1
        #     self.save(epoch)

        # if self.args.fusion_type == "robust_evaluation":
        #     self.load()
        #     self.pickle_record["robust_valid"][self.args.valid_type] = {}
        #     self.pickle_record["robust_valid"][self.args.valid_type][str(self.global_epoch)] = {}
        #     self.robust_valid(self.global_epoch)

        # clean data when loading the pre trained model
        # if self.args.fusion_type == "pre_train_robust_evaluation":
        #     self.load_pre_train_model()
        #     self.pickle_record["robust_valid"][self.args.valid_type] = {}
        #     self.pickle_record["robust_valid"][self.args.valid_type][str(self.global_epoch)] = {}
        #     self.robust_valid(self.global_epoch)


        # white box attach when loading the pre trained model
        if self.args.fusion_type == "pre_train_attack_robust_evaluation":
            self.load_pre_train_model()
            self.attack(self.global_epoch, attack_type="white_box") 


        # error analysis
        if self.args.fusion_type == "error_analysis":
            self.pickle_record["robust_valid"][self.args.valid_type] = {}
            self.pickle_record["robust_valid"][self.args.valid_type][str(self.global_epoch)] = {}            
            self.robust_valid(self.global_epoch)
