import logging
import os, sys
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
sys.path.append(os.getcwd())
import argparse, random
from tqdm import tqdm
import pandas as pd
import json
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from utils.utils_data import get_testloader
from advertorch.attacks import GradientSignAttack, LinfBasicIterativeAttack, LinfPGDAttack, MomentumIterativeAttack
from advertorch.attacks.utils import multiple_mini_batch_attack, attack_whole_dataset
from distillation import Linf_PGD
from advertorch.utils import to_one_hot
import numpy as np 
from utils.utils_func import setup_seed
import pdb
from autoattack import AutoAttack 


# https://github.com/BorealisAI/advertorch/blob/master/advertorch/utils.py
class CarliniWagnerLoss(nn.Module):
    """
    Carlini-Wagner Loss: objective function #6.
    Paper: https://arxiv.org/pdf/1608.04644.pdf
    """
    def __init__(self, conf=50.):
        super(CarliniWagnerLoss, self).__init__()
        self.conf = conf

    def forward(self, input, target):
        """
        :param input: pre-softmax/logits.
        :param target: true labels.
        :return: CW loss value.
        """
        num_classes = input.size(1)
        label_mask = to_one_hot(target, num_classes=num_classes).float()
        correct_logit = torch.sum(label_mask * input, dim=1)
        wrong_logit = torch.max((1. - label_mask) * input, dim=1)[0]
        loss = -F.relu(correct_logit - wrong_logit + self.conf).sum()
        return loss


def pgd(model, args, device):
    if args.subset_num:
        with open("subset_idx.json", "r") as f:
            subset_idx = json.load(f)
        testloader = get_testloader(args, subset_idx=subset_idx)
    else:
        testloader = get_testloader(args)

    adv_Xs = []
    adv_Ys = []
    for (Xs, Ys) in testloader:
        Xs, Ys = Xs.to(device), Ys.to(device)
        x_adv = Xs.detach() + 0.001 * torch.randn(Xs.shape).detach().to(device)
        x_adv.requires_grad_()
        for k in range(args.attack_steps):


            ######## adaptive attacks
            # output, aux_loss = model(X = x_adv, Y = Ys)
            # with torch.enable_grad():
            #     loss_adv = nn.CrossEntropyLoss()(output, Ys) if args.attack_loss_fn == 'xent' else CarliniWagnerLoss(conf=args.cw_conf)(output, Ys)
            # (loss_adv + args.adapt_attack_weight * aux_loss).backward()
            ########


            ######## without adaptive attacks
            output = model(X = x_adv, Y = Ys)
            with torch.enable_grad():
                loss_adv = nn.CrossEntropyLoss()(output, Ys) if args.attack_loss_fn == 'xent' else CarliniWagnerLoss(conf=args.cw_conf)(output, Ys)
            loss_adv.backward()
            ########


            eta = args.wbox_lr * x_adv.grad.sign()
            x_adv.data = x_adv.data + eta.data 
            x_adv.data = torch.min(torch.max(x_adv.data, Xs - args.attack_eps), Xs + args.attack_eps)
            x_adv.data = torch.clamp(x_adv.data, 0.0, 1.0)
            x_adv.grad.zero_()

        adv_Xs.append(x_adv)
        adv_Ys.append(Ys)
    return  torch.cat(adv_Xs, dim = 0),  torch.cat(adv_Ys, dim = 0)


def adaptative_attack_r6(model, args, device):
    if args.subset_num:
        with open("subset_idx.json", "r") as f:
            subset_idx = json.load(f)
        testloader = get_testloader(args, subset_idx=subset_idx)
    else:
        testloader = get_testloader(args)
    adv_Xs = []
    adv_Ys = []
    for (Xs, Ys) in testloader:
        Xs, Ys = Xs.to(device), Ys.to(device)
        x_adv = Xs.detach() + 0.001 * torch.randn(Xs.shape).detach().to(device)
        x_adv.requires_grad_()
        for k in range(args.attack_steps):
            output = model(X = x_adv, Y = Ys)
            with torch.enable_grad():
                loss_adv = nn.CrossEntropyLoss()(output, Ys) if args.attack_loss_fn == 'xent' else CarliniWagnerLoss(conf=args.cw_conf)(output, Ys)
            loss_adv.backward()
            eta = args.wbox_lr * x_adv.grad.sign()
            x_adv.data = x_adv.data + eta.data 
            x_adv.data = torch.min(torch.max(x_adv.data, Xs - args.attack_eps), Xs + args.attack_eps)
            x_adv.data = torch.clamp(x_adv.data, 0.0, 1.0)
            x_adv.grad.zero_()
        adv_Xs.append(x_adv)
        adv_Ys.append(Ys)
    return  torch.cat(adv_Xs, dim = 0),  torch.cat(adv_Ys, dim = 0)    


def adaptative_attack(model, args, device):
    if args.subset_num:
        with open("subset_idx.json", "r") as f:
            subset_idx = json.load(f)
        testloader = get_testloader(args, subset_idx=subset_idx)
    else:
        testloader = get_testloader(args)
    adv_Xs = []
    adv_Ys = []
    for (Xs, Ys) in testloader:
        Xs, Ys = Xs.to(device), Ys.to(device)
        x_adv = Xs.detach() + 0.001 * torch.randn(Xs.shape).detach().to(device)
        x_adv.requires_grad_()
        for k in range(args.attack_steps):
            output = model(X = x_adv, Y = Ys)
            with torch.enable_grad():
                loss_adv = nn.CrossEntropyLoss()(output, Ys) if args.attack_loss_fn == 'xent' else CarliniWagnerLoss(conf=args.cw_conf)(output, Ys)
            loss_adv.backward()
            eta = args.wbox_lr * x_adv.grad.sign()
            x_adv.data = x_adv.data + eta.data 
            x_adv.data = torch.min(torch.max(x_adv.data, Xs - args.attack_eps), Xs + args.attack_eps)
            x_adv.data = torch.clamp(x_adv.data, 0.0, 1.0)
            x_adv.grad.zero_()
        adv_Xs.append(x_adv)
        adv_Ys.append(Ys)
    return  torch.cat(adv_Xs, dim = 0),  torch.cat(adv_Ys, dim = 0)    


def white_box(model, args, device, logger):
    if args.subset_num:
        with open("subset_idx.json", "r") as f:
            subset_idx = json.load(f)
        testloader = get_testloader(args, subset_idx=subset_idx)
    else:
        testloader = get_testloader(args)
    test_iter = tqdm(testloader, desc='Batch', leave=False, position=2)
    result = {}
    loss_fn = nn.CrossEntropyLoss() if args.attack_loss_fn == 'xent' else CarliniWagnerLoss(conf=args.cw_conf)
    if args.wbox_type_pgd:
        adversary = LinfPGDAttack(
                    model, loss_fn=loss_fn, eps=args.attack_eps,
                    nb_iter=args.attack_steps, eps_iter=args.attack_eps/5, rand_init=True, clip_min=0., clip_max=1.,
                    targeted=False)
        _, label, pred, advpred = attack_whole_dataset(adversary, test_iter, device=device)
        acc = (label == advpred).sum().item() / len(label)
        logger.info("white-box type: {}, attack-steps: {}, attack-eps: {}, acc: {}".format("pgd", args.attack_steps, args.attack_eps, acc))
        result["pgd"] = acc
    if args.wbox_type_bim:
        adversary = LinfBasicIterativeAttack(
                    model, loss_fn=loss_fn, eps=args.attack_eps,
                    nb_iter=args.attack_steps, eps_iter=args.attack_eps/5, clip_min=0., clip_max=1.,
                    targeted=False)
        _, label, pred, advpred = attack_whole_dataset(adversary, test_iter, device=device)
        acc = (label == advpred).sum().item() / len(label)
        logger.info("white-box type: {}, attack-steps: {}, attack-eps: {}, acc: {}".format("bim", args.attack_steps, args.attack_eps, acc))
        result["bim"] = acc
    if args.wbox_type_fgsm:
        adversary = GradientSignAttack(
            model, loss_fn=loss_fn, eps=args.attack_eps, clip_min=0., clip_max=1., targeted=False)
        _, label, pred, advpred = attack_whole_dataset(adversary, test_iter, device=device)
        acc = (label == advpred).sum().item() / len(label)
        logger.info("white-box type: {}, attack-steps: {}, attack-eps: {}, acc: {}".format("fgsm", args.attack_steps, args.attack_eps, acc))
        result["fgsm"] = acc
    if args.wbox_type_mim:
        adversary = MomentumIterativeAttack(
            model, loss_fn=loss_fn, eps=args.attack_eps, eps_iter=args.attack_eps/5, decay_factor = 0.9,
            clip_min=0., clip_max=1., targeted=False)
        _, label, pred, advpred = attack_whole_dataset(adversary, test_iter, device=device)    
        acc = (label == advpred).sum().item() / len(label)
        logger.info("white-box type: {}, attack-steps: {}, attack-eps: {}, acc: {}".format("mim", args.attack_steps, args.attack_eps, acc))
        result["mim"] = acc
    return result 

def black_box(model, args, device, logger):
    adversary = AutoAttack(model, norm = "Linf", eps = args.attack_eps, version = 'standard', device = device, verbose= False)
    if args.subset_num:
        with open("subset_idx.json", "r") as f:
            subset_idx_ori = json.load(f)
            # subset_idx = []
            # for i in subset_idx_ori:
            #     if i + args.seed <=9999:
            #         subset_idx.append(i + args.seed)
            #     else:
            #         subset_idx.append(i - args.seed)
            subset_idx = [i+args.seed for i in range(1000)]


        testloader = get_testloader(args, subset_idx=subset_idx)
    else:
        testloader = get_testloader(args)

    results = {}
    for iteration, (Xs, Ys) in enumerate(testloader):
        Xs = Xs.to(device); Ys = Ys.to(device)
        results[str(iteration)] = {}
        correct_ind = torch.ones([Xs.shape[0]]).float().to(device)
        # for black_type in  ['apgd-ce', 'apgd-t', 'fab-t', 'square']:
        for black_type in  ['square']: 
            adversary.attacks_to_run = [black_type]
            adv_Xs = adversary.run_standard_evaluation_individual(Xs, Ys, bs = 100)
            pred_output = model(adv_Xs[black_type])    
            _, predicted = pred_output.max(1)
            tmp_acc = predicted.eq(Ys).float()
            results[str(iteration)][black_type] = torch.mean(tmp_acc).item()
            correct_ind = correct_ind * tmp_acc
        results[str(iteration)]["black_attack"] = torch.mean(correct_ind).item()

    acc = np.mean([ results[str(i)]["black_attack"]  for i in range(10)])
    logger.info("auto-attack, acc: {}.".format(acc))
    
    return acc


def transfer_box(model, args, device, logger):
    tmp_data_loader = ""
    correct = torch.ones([1000]).to(device)
    predictions = []
    for attack_type in ['mpgd', "sgm", "mdi2"]:
        tmp_data_loader = os.path.join(args.black_datafolder, "eps_{}".format(args.attack_eps))
        if args.black_method == "mpgd":    
            tmp_data_loader = os.path.join(tmp_data_loader, "from_baseline{}_{}_{}_steps_100_0".format(args.black_surro_num_experts, args.attack_loss_fn, args.black_method))
        elif args.black_method == "sgm":
            tmp_data_loader = os.path.join(tmp_data_loader, "from_baseline{}_{}_{}_0.2_steps_100".format(args.black_surro_num_experts, args.attack_loss_fn, args.black_method))
        elif args.black_method == "mdi2":
            tmp_data_loader = os.path.join(tmp_data_loader, "from_baseline{}_{}_{}_0.5_steps_100".format(args.black_surro_num_experts, args.attack_loss_fn, args.black_method))
        adv_Xs = torch.load(os.path.join(tmp_data_loader, 'inputs.pt'), map_location = device).to(device)
        Ys = torch.load(os.path.join(tmp_data_loader, 'labels.pt'),  map_location = device).to(device)
        pred_output = model(adv_Xs)
        _, predicted = pred_output.max(1)
        predictions.append(predicted.eq(Ys).float())
    acc = torch.mean(torch.stack(predictions), dim = 0, keepdim = False)
    acc1 = torch.mean((acc == 1).float()).item()
    acc2 = torch.mean((acc > 0).float()).item()

    logger.info("transfer-box, acc: {}, {}.".format(acc1, acc2))
    return acc1

def transfer_box_all(model, args, device, logger):
    tmp_data_loader = ""
    correct = torch.ones([1000]).to(device)
    predictions = []
    tmp_data_loader = os.path.join(args.black_datafolder, "eps_{}".format(args.attack_eps))
    all_trans_data = os.listdir(tmp_data_loader)
    for tmp_data_dir in all_trans_data:
        if tmp_data_dir[0] != ".":
            adv_Xs = torch.load(os.path.join(tmp_data_loader, tmp_data_dir, 'inputs.pt'), map_location = device).to(device)
            Ys = torch.load(os.path.join(tmp_data_loader, tmp_data_dir, 'labels.pt'), map_location = device).to(device)
            with torch.no_grad():
                pred_output = model(adv_Xs)
            _, predicted = pred_output.max(1)
            predictions.append(predicted.eq(Ys).float())
    acc = torch.mean(torch.stack(predictions), dim = 0, keepdim = False)
    acc1 = torch.mean((acc == 1).float()).item()
    acc2 = torch.mean((acc > 0).float()).item()
    logger.info("transfer-box, acc: {}, {}.".format(acc1, acc2))
    return acc1

def robust_eval(eval_type, model, args, device, logger):
    if eval_type == "white_box":
        results = white_box(model, args, device, logger)
    elif eval_type == "black_box":
        results = black_box(model, args, device, logger)
    elif eval_type == "transfer_box":
        results = transfer_box(model, args, device, logger)
    return results

