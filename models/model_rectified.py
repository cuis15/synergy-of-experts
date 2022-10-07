# https://github.com/NVlabs/AdaBatch/blob/master/models/cifar/resnet.py
'''Resnet for cifar dataset. 
Ported form 
https://github.com/facebook/fb.resnet.torch
and
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
(c) YANG, Wei 
'''
import torch.nn as nn
import math
import torch
import torch.nn.functional as F
import numpy as np
import pdb
from advertorch.utils import NormalizeByChannelMeanStd

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, leaky_relu=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True) if not leaky_relu else nn.LeakyReLU(0.1, True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, before_relu=False, intermediate=False):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)

        if intermediate:
            return out if before_relu else self.relu(out)

        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        return out if before_relu else self.relu(out)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, leaky_relu=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True) if not leaky_relu else nn.LeakyReLU(0.1, True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, before_relu=False):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        return out if before_relu else self.relu(out)


class ResNet_Aux(nn.Module):
    def __init__(self, args):
        super(ResNet_Aux, self).__init__()
        self.args = args
        self.depth = args.depth
        self.leaky_relu = args.leaky_relu
        # Model type specifies number of layers for CIFAR-10 model
        assert (self.depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (self.depth - 2) // 6
        self.n = n

        block = Bottleneck if self.depth >=44 else BasicBlock

        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True) if not self.leaky_relu else nn.LeakyReLU(0.1, True)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.avgpool = nn.AvgPool2d(8)  # original 8
        self.fc = nn.Linear(64 * block.expansion, args.num_classes)
        if self.args.two_branch and self.args.train_type == "loss_base":
            self.fc_aux = nn.Linear(64 * block.expansion, args.out_dim)
        elif self.args.two_branch and "gate_base" in self.args.train_type:
            self.fc_gate = nn.Linear(64 * block.expansion, args.hidden_dim)
        # self.sm = nn.Softmax(dim=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, leaky_relu=self.leaky_relu))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, leaky_relu=self.leaky_relu))

        return nn.Sequential(*layers)

    def forward(self, x, if_attack = True):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)    # 32x32

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        output = self.fc(x)
        if if_attack:
            return output
        elif self.args.two_branch and "gate_base" in self.args.train_type :
            out_gate  = self.fc_gate(x)
            return (output, out_gate)
        elif self.args.two_branch and self.args.train_type == "loss_base":
            aux_con  = self.fc_aux(x)
            return (output, aux_con)
        else:
            return (output, 0)
            
    
    def get_features(self, x, layer, before_relu=False):
        layers_per_block = 2 * self.n

        x = self.conv1(x)
        x = self.bn1(x)

        if layer == 1:
            return x

        x = self.relu(x)

        if layer > 1 and layer <= 1 + layers_per_block:
            relative_layer = layer - 1 
            x = self.layer_block_forward(x, self.layer1, relative_layer, before_relu=before_relu)
            return x

        x = self.layer1(x)
        if layer > 1 + layers_per_block and layer <= 1 + 2*layers_per_block:
            relative_layer = layer - (1 + layers_per_block)
            x = self.layer_block_forward(x, self.layer2, relative_layer, before_relu=before_relu)
            return x
        
        x = self.layer2(x)
        if layer > 1 + 2*layers_per_block and layer <= 1 + 3*layers_per_block:
            relative_layer = layer - (1 + 2*layers_per_block)
            x = self.layer_block_forward(x, self.layer3, relative_layer, before_relu=before_relu)
            return x
        
        x = self.layer3(x)
        if layer == self.depth:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)
        else:
            raise ValueError('layer {:d} is out of index!'.format(layer))
    
    def layer_block_forward(self, x, layer_block, relative_layer, before_relu=False):
        out = x
        if relative_layer == 1:
            return layer_block[0](out, before_relu, intermediate=True)

        if relative_layer == 2:
            return layer_block[0](out, before_relu, intermediate=False)
        
        out = layer_block[0](out)
        if relative_layer == 3:
            return layer_block[1](out, before_relu, intermediate=True)

        if relative_layer == 4:
            return layer_block[1](out, before_relu, intermediate=False)
        
        out = layer_block[1](out)
        if relative_layer == 5:
            return layer_block[2](out, before_relu, intermediate=True)

        if relative_layer == 6:
            return layer_block[2](out, before_relu, intermediate=False)

        raise ValueError('relative_layer is invalid')

    def infer(self, x):
        return self.forward(x).max(1, keepdim=True)[1]

def resnet(**kwargs):
    """
    Constructs a ResNet model.
    """
    return ResNet(**kwargs)


class Tower(nn.Module):
    def __init__(self, args):
        super(Tower, self).__init__()
        self.fc1 = nn.Linear(args.hidden_dim * args.num_experts, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.num_experts)
        self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(0.4)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        # out = self.softmax(out)
        # out = torch.sigmoid(out)
        return out


class SimpleTower(nn.Module):
    def __init__(self, args):
        super(SimpleTower, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(args.hidden_dim * args.num_experts, args.num_experts)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.fc1(x)
        # out = self.softmax(out)
        # out = torch.sigmoid(out)
        return out 


class ModelWrapper(nn.Module):
    def __init__(self, model, normalizer):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.normalizer = normalizer

    def forward(self, X, Y = None, if_attack = True):
        x = self.normalizer(X)
        return self.model(x, if_attack)
    
    def get_features(self, x, layer, before_relu=True):
        x = self.normalizer(x)
        return self.model.get_features(x, layer, before_relu)


def get_expert(model, device):
    mean = torch.tensor([0.4914, 0.4822, 0.4465], dtype=torch.float32).to(device)
    std = torch.tensor([0.2023, 0.1994, 0.2010], dtype=torch.float32).to(device)
    normalizer = NormalizeByChannelMeanStd(mean=mean, std=std)
    model = ModelWrapper(model, normalizer).to(device)
    return model


class MoeLike(nn.Module):
    def __init__(self, args, device):
        super(MoeLike, self).__init__()
        self.args = args
        self.device = device
        self.if_Rcon_attack = True
        if args.model_type == "ResNet20":
            self.experts = [get_expert(ResNet_Aux(args).to(device), device) for i in range(args.num_experts)]

        if "gate_base" in args.train_type  and args.tower_type == "simple":
            self.gate = SimpleTower(args).to(device)
        elif "gate_base" in args.train_type and args.tower_type == "multiple":
            self.gate = Tower(args).to(device)
        self.log_softmax_func = nn.LogSoftmax(dim=1)
        self.nll_loss_func = nn.NLLLoss(size_average = False, reduce = False, reduction = None)
        self.loss_func = nn.CrossEntropyLoss()
        self.BCEcriterion = nn.BCELoss(reduction='none')
        self.softmax = nn.Softmax(dim=1)

    def acc_compute(self, prob, Y):
            # pdb.set_trace()
            y_pred = prob.data.max(1)[1]
            acc = torch.mean((y_pred==Y).float()).item()
            return acc


    def forward(self, X, Y = None,  if_attack = True): # gate is for mainfold capture
        outputs_all = [e(X, if_attack = False) for e in self.experts]
        experts_outputs = [out[0] for out in outputs_all]

        if if_attack:   
            if self.args.train_type == "gate_base_hard":
                assert self.args.two_branch
                gate_outputs = torch.cat( [ out[1] for out in outputs_all] , dim = 1)
                gate_outputs = self.gate(gate_outputs).unsqueeze(1)
                gate_outputs = self.softmax(gate_outputs)
                experts_cat = torch.stack(experts_outputs).permute(1,0,2)
                predicted_output = torch.bmm(gate_outputs, experts_cat).squeeze()


            if self.args.train_type == "gate_base_soft":
                assert self.args.two_branch
                gate_outputs = torch.cat( [ out[1] for out in outputs_all] , dim = 1)
                gate_outputs = self.gate(gate_outputs).unsqueeze(1)
                gate_outputs = self.softmax(gate_outputs)
                experts_cat = torch.stack(experts_outputs).permute(1,0,2)
                predicted_output = torch.bmm(gate_outputs, experts_cat).squeeze()


            elif (self.if_Rcon_attack and self.args.two_branch and self.args.train_type == "loss_base"):

                # ### old version
                # robust_output_aux = [out[1].sigmoid().squeeze() for out in outputs_all]
                # expert_idx = torch.argmax(torch.stack(robust_output_aux), dim = 0).detach() # 
                # expert_idx = expert_idx * self.args.batch_size + torch.tensor([i for i in range(X.size(0))]).to(self.device)
                # predicted_output = torch.index_select(outputs, dim = 0, index = expert_idx)

                ### new version
                assert self.args.two_branch == True
                robust_output_s = [torch.softmax(out * self.args.tempC, dim=1) for out in experts_outputs]
                robust_con_pre = [out.max(1)[0] for out in robust_output_s]
                robust_output_aux = [out[1].sigmoid().squeeze() for out in outputs_all]
                robust_detector = [ robust_con_pre[idx] * robust_output_aux[idx] for idx in range(self.args.num_experts)]
                expert_idx_pre = torch.argmax(torch.stack(robust_detector, dim = 0), dim = 0).detach()
                expert_idx = expert_idx_pre * X.size(0) + torch.tensor([i for i in range(X.size(0))]).to(self.device)
                predicted_output = torch.index_select(torch.cat(experts_outputs, dim = 0), dim = 0, index = expert_idx)

            else:
                losses = [self.nll_loss_func(self.log_softmax_func(eo), Y) for eo in experts_outputs]
                losses_detach = torch.stack(losses)
                expert_idx = torch.argmin(losses_detach, dim = 0).detach()
                expert_idx = expert_idx * X.size(0) + torch.tensor([i for i in range(X.size(0))]).to(self.device)
                predicted_output = torch.index_select(torch.cat(experts_outputs, dim = 0), dim = 0, index = expert_idx)
            
            return predicted_output      

        else: 

            if self.args.two_branch and self.args.train_type == "loss_base":
                robust_output_s = [torch.softmax(out * self.args.tempC, dim=1) for out in experts_outputs]
                robust_con_pre = [out.max(1)[0] for out in robust_output_s]
                robust_output_aux = [out[1].sigmoid().squeeze() for out in outputs_all]
                robust_output_s_ = [torch.softmax(out * self.args.tempC_trueonly, dim=1) for out in experts_outputs]
                robust_con_y = [ out[torch.tensor(range(X.size(0))), Y].detach() for  out in  robust_output_s_]
                correct_index = [ torch.where(out.max(1)[1] == Y)[0] for out in experts_outputs ]
                for i in range(self.args.num_experts):
                    robust_con_pre[i][correct_index[i]] = robust_con_pre[i][correct_index[i]].detach()
                robust_detector = [ robust_con_pre[idx] * robust_output_aux[idx] for idx in range(self.args.num_experts)]
                aux_loss = [ self.BCEcriterion(robust_detector[idx], robust_con_y[idx]) for idx in range(self.args.num_experts)]

                loss_original = [self.loss_func(eo, Y) for eo in experts_outputs]
                losses = [self.nll_loss_func(self.log_softmax_func(eo), Y) for eo in experts_outputs] # [expert_num, batch_size]
                losses_detach = torch.stack(losses)
                expert_idx_pre = torch.argmin(losses_detach, dim = 0).detach() # 
                expert_idx = expert_idx_pre * X.size(0) + torch.tensor([i for i in range(X.size(0))]).to(self.device)
                loss_soft = torch.mean(torch.sum(F.softmax(-losses_detach, dim = 0).detach() * losses_detach, dim = 0))
                aux_loss = torch.mean(torch.stack(aux_loss))
                loss_soft += self.args.adaptivetrainlambda * aux_loss
                loss_experts = torch.index_select(torch.cat(losses, dim = 0), dim = 0, index = expert_idx)
                loss_each_experts = [torch.mean(loss_experts[torch.where(expert_idx_pre == i)[0]])  for i in range(self.args.num_experts)]
                acc_experts = [self.acc_compute(eo, Y) for eo in experts_outputs]
                probs = torch.index_select(torch.cat(experts_outputs, dim = 0), dim = 0, index = expert_idx)
                return loss_original, loss_each_experts, loss_soft, aux_loss, acc_experts, probs


            elif self.args.two_branch and self.args.train_type == "gate_base_hard":
                loss_original = [self.loss_func(eo, Y) for eo in experts_outputs]
                acc_experts = [self.acc_compute(eo, Y) for eo in experts_outputs]
                experts_cat = torch.stack(experts_outputs).permute(1,0,2)
                gate_outputs = torch.cat( [ self.softmax(out[1]) for out in outputs_all] , dim = 1) # batch_size, 3*hidden_dim
                gate_outputs = self.gate(gate_outputs).unsqueeze(1) # batch_size num-experts batch_size 3 experts
                gate_outputs = self.softmax(gate_outputs)
                pred_output = torch.bmm(gate_outputs, experts_cat).squeeze() # batch_size 3
                loss_soft = self.loss_func(pred_output, Y)
                return loss_original, loss_original, loss_soft, torch.tensor(0), acc_experts, pred_output  


            elif self.args.two_branch and self.args.train_type == "gate_base_soft":
                losses = torch.stack([self.nll_loss_func(self.log_softmax_func(eo), Y) for eo in experts_outputs]) # batch_size * num_experts
                gate_labels = losses.T.argmin(dim = 1).detach()
                gate_outputs = torch.cat( [out[1] for out in outputs_all] , dim = 1) # batch_size, 3*hidden_dim
                aux_loss = self.loss_func(gate_outputs, gate_labels)   
                gate_outputs = self.gate(gate_outputs).unsqueeze(1) # batch_size num-experts batch_size 3 experts             
                loss_original = [self.loss_func(eo, Y) for eo in experts_outputs]
                acc_experts = [self.acc_compute(eo, Y) for eo in experts_outputs]
                expert_idx = gate_labels * X.size(0) + torch.tensor([i for i in range(X.size(0))]).to(self.device)
                pred_output = torch.index_select(torch.cat(experts_outputs, dim = 0), dim = 0, index = expert_idx)
                loss_soft = torch.mean(torch.sum(F.softmax(-losses, dim = 0).detach() * losses, dim = 0))
                loss_soft += self.args.adaptivetrainlambda * aux_loss
                return loss_original, loss_original, loss_soft, aux_loss, acc_experts, pred_output  


            elif (self.args.two_branch == False and self.args.train_type == "loss_base"):
                aux_loss = [0,0,0]
                loss_original = [self.loss_func(eo, Y) for eo in experts_outputs]
                losses = [self.nll_loss_func(self.log_softmax_func(eo), Y) for eo in experts_outputs] # [expert_num, batch_size]
                losses_detach = torch.stack(losses)
                expert_idx_pre = torch.argmin(losses_detach, dim = 0).detach() # 
                expert_idx = expert_idx_pre * X.size(0) + torch.tensor([i for i in range(X.size(0))]).to(self.device)
                loss_soft = torch.mean(torch.sum(F.softmax(-losses_detach, dim = 0).detach() * losses_detach, dim = 0))
                aux_loss = torch.mean(torch.stack(aux_loss))
                loss_soft += self.args.adaptivetrainlambda * aux_loss
                loss_experts = torch.index_select(torch.cat(losses, dim = 0), dim = 0, index = expert_idx)
                loss_each_experts = [torch.mean(loss_experts[torch.where(expert_idx_pre == i)[0]])  for i in range(self.args.num_experts)]
                acc_experts = [self.acc_compute(eo, Y) for eo in experts_outputs]
                probs = torch.index_select(torch.cat(experts_outputs, dim = 0), dim = 0, index = expert_idx)
                return loss_original, loss_each_experts, loss_soft, aux_loss, acc_experts, probs


            elif self.args.train_type == "kl_base":
                d = torch.rand(X.shape).sub(0.5).to(X.device)
                d = self._l2_normalize(d)
                adv_experts_outputs = [e(X+d, if_attack = False) for e in self.experts]
                losses = [self.nll_loss_func(self.log_softmax_func(eo), Y) for eo in experts_outputs] # [expert_num, batch_size]
                kl_loss = torch.stack([torch.sum(F.kl_div( F.log_softmax(adv_experts_outputs[idx], dim=1) , F.softmax(experts_outputs[idx], dim = 1), reduction = 'none'), dim = 1) for idx in range(self.args.num_experts)])
                expert_idx_pre = torch.argmin(kl_loss, dim = 0).detach() # 
                expert_idx = expert_idx_pre * self.args.batch_size + torch.tensor([i for i in range(self.args.batch_size)]).to(self.device)
                loss_experts = torch.index_select(torch.cat(losses, dim = 0), dim = 0, index = expert_idx)
                loss_each_experts = [torch.mean(loss_experts[torch.where(expert_idx_pre == i)[0]])  for i in range(self.args.num_experts)]
                acc_experts = [self.acc_compute(eo, Y) for eo in experts_outputs]
                probs = torch.index_select(torch.cat(experts_outputs, dim = 0), dim = 0, index = expert_idx)
                loss_original = [self.loss_func(eo, Y) for eo in experts_outputs]
                aux_loss = [0,0,0]
                aux_loss = [torch.mean(out) for out in aux_loss]

                return loss_original, loss_each_experts, torch.tensor(0), aux_loss, acc_experts, probs                  

            else:
                print("error train type and two_branch setting")
                exit()


    def _l2_normalize(self, d):
        d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
        d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
        return d


    def acc_compute(self, prob, Y):
        # pdb.set_trace()
        y_pred = prob.data.max(1)[1]
        acc = torch.mean((y_pred==Y).float()).item()
        return acc

    def kl_loss_logit_compute(self, X, Y):
        with torch.no_grad():
            experts_outputs = [e(X, if_attack = False) for e in self.experts]
        preds = [prob.data.max(1)[1] for prob in experts_outputs]
        loss_original = [self.loss_func(eo, Y).item() for eo in experts_outputs]
        losses = [self.nll_loss_func(self.log_softmax_func(eo), Y) for eo in experts_outputs]
        kl_loss = []
        for idx in range(self.args.num_experts):
            d = torch.rand(X.shape).sub(0.5).to(X.device)
            d = self._l2_normalize(d)
            adv_X = (X+d).clone()
            K = self.args.K
            while K:
                adv_X.requires_grad_()
                with torch.enable_grad():
                    adv_outputs = self.experts[idx](adv_X, if_attack = False)
                    loss_adv = nn.CrossEntropyLoss()(adv_outputs, preds[idx]) 
                loss_adv.backward()
                grad = adv_X.grad
                eta = self.args.lr_kl * grad.sign()
                adv_X = adv_X.detach() + eta 
                # adv_X = torch.stack([sample for idx, sample in enumerate(adv_X) if continue_idx[idx]]).to(self.device)
                adv_X = torch.min(torch.max(adv_X, X - self.args.kl_eps), X + self.args.kl_eps)
                adv_X = torch.clamp(adv_X, 0, 1)
                K = K-1
            kl_loss.append(torch.sum(F.kl_div( F.log_softmax(adv_outputs, dim=1) , F.softmax(experts_outputs[idx], dim = 1), reduction = 'none'), dim = 1))

        if self.args.kl_select_type == "argmin":
            expert_idx = torch.argmin(torch.stack(kl_loss), dim = 0)
        elif self.args.kl_select_type == "argmax":
            expert_idx = torch.argmax(torch.stack(kl_loss), dim = 0)
        else:            
            print("error kl select type")
            exit()
        onehot_expert_idx = torch.zeros(expert_idx.size(0), self.args.num_experts).to(self.device)
        onehot_expert_idx.scatter_(1, expert_idx.view(-1, 1), 1) # [batch_size, expert_num] 
        loss_experts = [ (torch.sum(losses[idx] * onehot_expert_idx[:, idx])/torch.sum(onehot_expert_idx[:, idx])).item() for idx in range(self.args.num_experts)]
        probs = torch.bmm(onehot_expert_idx.unsqueeze(1), torch.stack(experts_outputs).permute(1, 0,2)).squeeze(1) # [batch_size, class_num]
        loss_final = self.loss_func(probs, Y)
        return torch.stack(kl_loss), loss_original, loss_experts, experts_outputs, probs, loss_final             


    def kappa_logit_compute(self, X, Y):
        if self.args.kappa_random_type == "none":
            iter_adv = X.detach()
        elif self.args.kappa_random_type == "trades":
            iter_adv = X.detach() + 0.001 * torch.randn(X.shape).to(self.device).detach()
            iter_adv = torch.clamp(iter_adv, 0.0, 1.0)
        elif self.args.kappa_random_type == "mart":
            iter_adv = X.detach() + torch.from_numpy(np.random.uniform(-self.args.kappa_eps, self.args.kappa_eps, X.shape)).float().to(self.device)
            iter_adv = torch.clamp(iter_adv, 0.0, 1.0)
        else:
            print("error kappa_random_type")
        with torch.no_grad():  #  poison all data needs kappa steps
            experts_outputs = [e(X) for e in self.experts]
            preds = [torch.argmax(eo, dim = 1) for eo in experts_outputs]
            loss_original = [self.loss_func(eo, Y).item() for eo in experts_outputs]
            losses = [self.nll_loss_func(self.log_softmax_func(eo), Y) for eo in experts_outputs]
        outputs_kappa = []
        K  = 0
        for idx in range(self.args.num_experts):
            kappa = torch.ones(X.size(0)).to(self.device)
            adv_X = iter_adv.clone()
            K = self.args.K
            while K:
                adv_X.requires_grad_()
                with torch.enable_grad():
                    adv_outputs = self.experts[idx](adv_X, if_attack = False)
                    adv_pred = torch.argmax(adv_outputs, dim = 1)
                    continue_idx = (preds[idx] == adv_pred).float()  ### [1,1,0,1,0,1] # 1 indicates continue; 0 indicated stop.
                    if torch.sum(continue_idx) == 0:
                        break
                    kappa = kappa + continue_idx
                    if self.args.kappa_random_type == "mart" or self.args.kappa_random_type == "none" :
                        loss_adv = nn.CrossEntropyLoss()(adv_outputs, preds[idx]) 
                    elif self.args.kappa_random_type == "trades":
                        criterion_kl = nn.KLDivLoss(size_average=False)
                        loss_adv = criterion_kl(F.log_softmax(adv_outputs, dim=1), F.softmax(experts_outputs[idx] , dim=1)) 
                    else:
                        print("error kappa random_type")
                        exit()
                loss_adv.backward()
                grad = adv_X.grad
                eta = self.args.lr_kappa * grad.sign()
                adv_X = adv_X.detach() + ((eta + self.args.kappa_omega * torch.randn(adv_X.shape).detach().to(self.device)).view(X.size(0), -1).T * continue_idx).T.view(X.shape)
                # adv_X = torch.stack([sample for idx, sample in enumerate(adv_X) if continue_idx[idx]]).to(self.device)
                adv_X = torch.min(torch.max(adv_X, X - self.args.kappa_eps), X + self.args.kappa_eps)
                adv_X = torch.clamp(adv_X, 0, 1)
                K = K-1

            outputs_kappa.append(kappa)
        if self.args.kappa_select_type == "argmin":
            expert_idx = torch.argmin(torch.stack(outputs_kappa), dim = 0)
        elif self.args.kappa_select_type == "argmax":
            expert_idx = torch.argmax(torch.stack(outputs_kappa), dim = 0)
        onehot_expert_idx = torch.zeros(expert_idx.size(0), self.args.num_experts).to(self.device)
        onehot_expert_idx.scatter_(1, expert_idx.view(-1, 1), 1) # [batch_size, expert_num] 

        probs = torch.bmm(onehot_expert_idx.unsqueeze(1), torch.stack(experts_outputs).permute(1, 0,2)).squeeze(1) # [batch_size, class_num]
        loss_final = self.loss_func(probs, Y)
        return torch.stack(outputs_kappa), experts_outputs, loss_original, loss_experts, probs , loss_final        


    def valid(self, X, Y = None, valid_type = "normal"):   # valid_type = [kl_loss, kappa, normal]
        with torch.no_grad():

            outputs = [e(X, if_attack = False) for e in self.experts]
            experts_outputs = [out[0] for out in outputs]

            if valid_type == "gate_base_hard":
                assert (self.args.train_type == "gate_base_hard" and  self.args.two_branch)
                loss_original = [self.loss_func(eo, Y).item() for eo in experts_outputs]
                experts_cat = torch.stack(experts_outputs).permute(1,0,2)
                gate_outputs = torch.cat( [out[1] for out in outputs] , dim = 1) # batch_size, 3*hidden_dim
                gate_outputs = self.gate(gate_outputs).unsqueeze(1) # batch_size num-experts batch_size 3 experts
                gate_outputs = self.softmax(gate_outputs)
                pred_output = torch.bmm(gate_outputs, experts_cat).squeeze() # batch_size 3
                loss_soft = self.loss_func(pred_output, Y).item()
                return 0, experts_outputs, loss_original, pred_output, loss_original, 0, loss_soft

            elif valid_type == "gate_base_soft":
                assert (self.args.train_type == "gate_base_soft" and  self.args.two_branch)
                loss_original = [self.loss_func(eo, Y).item() for eo in experts_outputs]
                experts_cat = torch.stack(experts_outputs).permute(1,0,2)
                gate_outputs = torch.cat( [out[1] for out in outputs] , dim = 1) # batch_size, 3*hidden_dim
                gate_outputs = self.gate(gate_outputs).unsqueeze(1) # batch_size num-experts batch_size 3 experts
                gate_outputs = self.softmax(gate_outputs)
                pred_output = torch.bmm(gate_outputs, experts_cat).squeeze() # batch_size 3
                loss_soft = self.loss_func(pred_output, Y).item()
                return 0, experts_outputs, loss_original, pred_output, loss_original, 0, loss_soft

            elif valid_type == "kl_loss": 
                kl_losses, loss_original, loss_experts, experts_outputs, probs, loss_final = self.kl_loss_logit_compute(X, Y)
                return kl_losses,  experts_outputs, loss_original, probs, loss_experts, 0, loss_final

            elif valid_type == "kappa":
                kappa_outputs, experts_outputs, loss_original, loss_experts, probs = self.kappa_logit_compute(X, Y)
                return kappa_outputs, experts_outputs, loss_original, probs, loss_experts, 0, loss_final     
            elif valid_type == "normal":
                loss_original = [self.loss_func(eo, Y).item() for eo in experts_outputs]
                probs_ensemble = torch.mean(torch.stack(experts_outputs), dim = 0, keepdim=False)
                loss_ensemble = self.loss_func(probs_ensemble, Y).item()
                return 0, experts_outputs, loss_original, probs_ensemble, loss_original, 0, loss_ensemble

            elif valid_type == "max_confidence":
                losses = [self.nll_loss_func(self.log_softmax_func(eo), Y) for eo in experts_outputs]
                loss_original = [self.loss_func(eo, Y).item() for eo in experts_outputs]
                probs = [F.softmax(eo, dim = 1) for eo in experts_outputs]
                probs_max, _ = torch.max(torch.stack(probs), dim = 2)
                expert_idx_pre = torch.argmax(probs_max, dim = 0)
                
                expert_idx = expert_idx_pre * X.size(0) + torch.tensor([i for i in range(X.size(0))]).to(self.device)
                

                probs_max_confidence = torch.index_select(torch.cat(experts_outputs, dim = 0), dim = 0, index = expert_idx)
                loss_experts = torch.index_select(torch.cat(losses, dim = 0), dim = 0, index = expert_idx)
                loss_each_experts = [torch.mean(loss_experts[torch.where(expert_idx_pre == i)[0]]).item()  for i in range(self.args.num_experts)]
                loss_max_confidence = torch.mean(loss_experts).item()
                
                
                # onehot_expert_idx = torch.zeros(expert_idx.size(0), self.args.num_experts).to(self.device)
                # onehot_expert_idx.scatter_(1, expert_idx.view(-1, 1), 1) # [batch_size, expert_num] 
                # probs_max_confidence = torch.bmm(onehot_expert_idx.unsqueeze(1), torch.stack(experts_outputs).permute(1, 0,2)).squeeze(1) # [batch_size, class_num]
                # loss_each_experts = [ (torch.sum(losses[idx] * onehot_expert_idx[:, idx])/torch.sum(onehot_expert_idx[:, idx])).item() for idx in range(self.args.num_experts)]
                # loss_max_confidence = self.loss_func(probs_max_confidence, Y).item()
                return 0, experts_outputs, loss_original, probs_max_confidence, loss_each_experts, 0, loss_max_confidence         

            elif valid_type == "max_entropy":
                losses = [self.nll_loss_func(self.log_softmax_func(eo), Y) for eo in experts_outputs]
                loss_original = [self.loss_func(eo, Y).item() for eo in experts_outputs]
                entropy_list = [ torch.sum(-1* F.softmax(eo, dim = 1) * F.log_softmax(eo, dim = 1), dim = 1)  for eo in experts_outputs]
                expert_idx = torch.argmin(torch.stack(entropy_list), dim = 0)
                onehot_expert_idx = torch.zeros(expert_idx.size(0), self.args.num_experts).to(self.device)
                onehot_expert_idx.scatter_(1, expert_idx.view(-1, 1), 1) # [batch_size, expert_num] 
                probs_max_confidence = torch.bmm(onehot_expert_idx.unsqueeze(1), torch.stack(experts_outputs).permute(1, 0,2)).squeeze(1) # [batch_size, class_num]
                loss_max_confidence = self.loss_func(probs_max_confidence, Y).item()
                loss_experts = [ (torch.sum(losses[idx] * onehot_expert_idx[:, idx])/torch.sum(onehot_expert_idx[:, idx])).item() for idx in range(self.args.num_experts)]
                return 0, experts_outputs, loss_original, probs_max_confidence, loss_experts, 0, loss_max_confidence     


            elif valid_type == "max_margin":
                losses = [self.nll_loss_func(self.log_softmax_func(eo), Y) for eo in experts_outputs]
                loss_original = [self.loss_func(eo, Y).item() for eo in experts_outputs]
                probs = F.softmax(torch.stack(experts_outputs), dim = 2)
                vs, idxs = torch.max( probs, dim = 2)
                for idx in range(self.args.num_experts):
                    for sample in range(len(Y)):
                        probs[idx, sample, idxs[idx, sample]] = 0
                v2s, idx2s = torch.max(probs, dim = 2)
                expert_idx = torch.argmax(vs - v2s, dim = 0)
                onehot_expert_idx = torch.zeros(expert_idx.size(0), self.args.num_experts).to(self.device)
                onehot_expert_idx.scatter_(1, expert_idx.view(-1, 1), 1) # [batch_size, expert_num] 
                probs_max_confidence = torch.bmm(onehot_expert_idx.unsqueeze(1), torch.stack(experts_outputs).permute(1, 0,2)).squeeze(1) # [batch_size, class_num]
                loss_max_confidence = self.loss_func(probs_max_confidence, Y).item()
                loss_experts = [ (torch.sum(losses[idx] * onehot_expert_idx[:, idx])/torch.sum(onehot_expert_idx[:, idx])).item() for idx in range(self.args.num_experts)]
                return 0, experts_outputs, loss_original,probs_max_confidence, loss_experts, 0, loss_max_confidence    
            

            elif valid_type == "r_con":
                assert self.args.two_branch == True
                robust_output_s = [torch.softmax(out * self.args.tempC, dim=1) for out in experts_outputs]
                robust_output_s_ = [torch.softmax(out * self.args.tempC_trueonly, dim=1) for out in experts_outputs]
                robust_con_y = [ out[torch.tensor(range(X.size(0))), Y].detach() for  out in  robust_output_s_]
                robust_con_pre = [out.max(1)[0] for out in robust_output_s]
                loss_original = [self.loss_func(eo, Y).item() for eo in experts_outputs]
                losses = [self.nll_loss_func(self.log_softmax_func(eo), Y) for eo in experts_outputs] # [expert_num, batch_size]
                robust_output_aux = [out[1].sigmoid().squeeze() for out in outputs]
                robust_detector = [ robust_con_pre[idx] * robust_output_aux[idx] for idx in range(self.args.num_experts)]
                expert_idx_pre = torch.argmax(torch.stack(robust_detector, dim = 0), dim = 0)
                expert_idx = expert_idx_pre * X.size(0) + torch.tensor([i for i in range(X.size(0))]).to(self.device)
                probs_output = torch.index_select(torch.cat(experts_outputs, dim = 0), dim = 0, index = expert_idx)
                loss_experts = torch.index_select(torch.cat(losses, dim = 0), dim = 0, index = expert_idx)
                loss_each_experts = [torch.mean(loss_experts[torch.where(expert_idx_pre == i)[0]]).item()  for i in range(self.args.num_experts)]
                loss_max_confidence = torch.mean(loss_experts).item()
                losses_detach = torch.stack(losses)
                expert_idx_loss_pre = torch.argmin(losses_detach, dim = 0).detach()
                aux_loss = torch.mean((expert_idx_pre == expert_idx_loss_pre).float()).item()               
                return 0, experts_outputs, loss_original, probs_output, loss_each_experts, aux_loss, loss_max_confidence

            else:
                print("error valid_type")
                exit()

