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
        assert (self.depth - 2) % 6 == 0, 'depth should be 6n+2' # 2, 8, 14, 20
        n = (self.depth - 2) // 6
        self.n = n
        block = Bottleneck if self.depth >=44 else BasicBlock
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True) if not self.leaky_relu else nn.LeakyReLU(0.1, True)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.avgpool = nn.AvgPool2d(8)  # original 8
        if self.n == 0:
            self.fc = nn.Linear(64 * 4, args.num_classes)
        else:
            self.fc = nn.Linear(64 * block.expansion, args.num_classes)

        if self.args.train_type == "loss_base":
            self.inplanes = 16
            self.layer1_aux = self._make_layer(block, 16, n)
            self.layer2_aux = self._make_layer(block, 32, n, stride=2)
            self.layer3_aux = self._make_layer(block, 64, n, stride=2)            
            if self.n == 0:
                self.fc_aux = nn.Linear(64 * 4, args.out_dim)
            else:
                self.fc_aux = nn.Linear(64 * block.expansion, args.out_dim)
        elif "gate_base" in self.args.train_type:
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

    def forward(self, inputs, if_attack = True):
        inputs = self.conv1(inputs)
        inputs = self.bn1(inputs)
        inputs = self.relu(inputs)    # 32x32
        if self.n != 0:
            features = self.layer1(inputs)  # 32x32
            features = self.layer2(features)  # 16x16
            features = self.layer3(features)  # 8x8

            if self.args.train_type == "loss_base":
                x_aux = self.layer1_aux(inputs)  # 32x32
                x_aux = self.layer2_aux(x_aux)  # 32x32
                x_aux = self.layer3_aux(x_aux)  # 32x32

        x = self.avgpool(features)
        x = x.view(x.size(0), -1)
        output = self.fc(x)
        if if_attack:
            return output
        elif "gate_base" in self.args.train_type:
            out_gate  = self.fc_gate(x)
            return (output, out_gate)
        elif self.args.train_type == "loss_base" and self.args.ppd_type == "simple":
            x_aux = self.avgpool(features)
            x_aux = x_aux.view(x_aux.size(0), -1)
            aux_con  = self.fc_aux(x_aux)
            return (output, aux_con)
        elif self.args.train_type == "loss_base" and self.args.ppd_type == "complex":
            x_aux = self.avgpool(x_aux)
            x_aux = x_aux.view(x_aux.size(0), -1)
            aux_con  = self.fc_aux(x_aux)
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
    normalizer = NormalizeByChannelMeanStd(mean=mean, std=std).to(device)
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

        if "moe" in args.train_type:
            args.num_classes = 3
            self.gate = get_expert(ResNet_Aux(args).to(device), device)
            args.num_classes = 10

        if "gate_base" in args.train_type  and args.tower_type == "simple":
            self.gate = SimpleTower(args).to(device)
        elif "gate_base" in args.train_type and args.tower_type == "multiple":
            self.gate = Tower(args).to(device)
        self.log_softmax_func = nn.LogSoftmax(dim=1)
        self.nll_loss_func = nn.NLLLoss(size_average = False, reduce = False, reduction = None)
        self.loss_func = nn.CrossEntropyLoss()
        self.BCEcriterion = nn.BCELoss(reduction='none')
        self.softmax = nn.Softmax(dim=1)

    def _l2_normalize(self, d):
        d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
        d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
        return d

    def acc_compute(self, prob, Y):
        y_pred = prob.data.max(1)[1]
        acc = torch.mean((y_pred==Y).float()).item()
        return acc

    def forward(self, X, Y = None,  if_attack = True): # gate is for mainfold capture
        outputs_all = [e(X, if_attack = False) for e in self.experts]
        experts_outputs = [out[0] for out in outputs_all]

        if if_attack:
            if self.args.train_type == "gate_base":
                gate_outputs = torch.cat( [ out[1] for out in outputs_all] , dim = 1)
                gate_outputs = self.softmax(self.gate(gate_outputs)).unsqueeze(1) #### # gate_outputs = [batch_size, 1, num_experts]
                experts_cat = torch.stack(experts_outputs).permute(1,0,2) # experts_cat  = [batch_size, num_experts, 10]
                predicted_output = torch.bmm(gate_outputs, experts_cat).squeeze()  # [batch_size,  10] 
                return predicted_output
            elif self.args.train_type == "loss_base":
                robust_output_s = [torch.softmax(out * self.args.tempC, dim=1) for out in experts_outputs]
                robust_con_pre = [out.max(1)[0] for out in robust_output_s]
                robust_output_aux = [out[1].sigmoid().squeeze() for out in outputs_all]
                robust_detector = [ robust_con_pre[idx] * robust_output_aux[idx] for idx in range(self.args.num_experts)]
                expert_idx_pre = torch.argmax(torch.stack(robust_detector, dim = 0), dim = 0).detach()
                expert_idx = expert_idx_pre * X.size(0) + torch.tensor([i for i in range(X.size(0))]).to(self.device)
                predicted_output = torch.index_select(torch.cat(experts_outputs, dim = 0), dim = 0, index = expert_idx)
                
                detected_prob = torch.index_select(torch.cat(robust_detector, dim = 0), dim = 0, index = expert_idx)
                output_prob =  torch.ones_like(detected_prob)
                # output_prob = torch.softmax(predicted_output, dim=1).max(dim = 1)[0]
                aux_loss = torch.mean(self.BCEcriterion(detected_prob, output_prob))
                return predicted_output
            elif self.args.train_type == "moe":
                gate_outputs = self.softmax(self.gate(X)).view(X.size(0), 1, -1) # 1000, 3
                predicted_output =  torch.bmm(gate_outputs, torch.stack(experts_outputs).permute(1,0,2)).squeeze() # 1000, 3, 10
                return predicted_output

            elif self.args.train_type == "normal":
                if self.args.num_experts ==1:
                    predicted_output = experts_outputs[0]
                else:
                    softmax_experts_outputs = [self.softmax(e) for e in experts_outputs]
                    predicted_output = torch.mean(torch.stack(softmax_experts_outputs), dim = 0, keepdim=False)
                    predicted_output = torch.clamp(predicted_output, min=1e-40)
                    predicted_output = torch.log(predicted_output)
            

        else: 
            ########## old loss_base training
            # if self.args.train_type == "loss_base":
            #     robust_output_s = [torch.softmax(out * self.args.tempC, dim=1) for out in experts_outputs]
            #     robust_con_pre = [out.max(1)[0] for out in robust_output_s]
            #     robust_output_aux = [out[1].sigmoid().squeeze() for out in outputs_all]
            #     robust_output_s_ = [torch.softmax(out * self.args.tempC_trueonly, dim=1) for out in experts_outputs]
            #     robust_con_y = [ out[torch.tensor(range(X.size(0))), Y].detach() for  out in  robust_output_s_]
            #     correct_index = [ torch.where(out.max(1)[1] == Y)[0] for out in experts_outputs ]
            #     for i in range(self.args.num_experts):
            #         robust_con_pre[i][correct_index[i]] = robust_con_pre[i][correct_index[i]].detach()
            #     robust_detector = [ robust_con_pre[idx] * robust_output_aux[idx] for idx in range(self.args.num_experts)]
            #     aux_loss = [ self.BCEcriterion(robust_detector[idx], robust_con_y[idx]) for idx in range(self.args.num_experts)]

            #     loss_original = [self.loss_func(eo, Y) for eo in experts_outputs]
            #     losses = [self.nll_loss_func(self.log_softmax_func(eo), Y) for eo in experts_outputs] # [expert_num, batch_size]
            #     losses_detach = torch.stack(losses) # [3, batch_size]
            #     expert_idx_pre = torch.argmin(losses_detach, dim = 0).detach() # 
            #     expert_idx = expert_idx_pre * X.size(0) + torch.tensor([i for i in range(X.size(0))]).to(self.device)
            #     loss_soft = torch.mean(torch.sum(F.softmax(-losses_detach, dim = 0).detach() * losses_detach, dim = 0))
            #     aux_loss = torch.mean(torch.stack(aux_loss))
            #     loss_soft += self.args.adaptivetrainlambda * aux_loss
            #     # loss_soft = torch.mean(torch.stack(loss_original))
            #     loss_experts = torch.index_select(torch.cat(losses, dim = 0), dim = 0, index = expert_idx)
            #     loss_each_experts = [torch.mean(loss_experts[torch.where(expert_idx_pre == i)[0]]).item()  for i in range(self.args.num_experts)]
            #     acc_originals = [self.acc_compute(eo, Y) for eo in experts_outputs]
            #     true_idx = torch.argmax(torch.stack(robust_detector, dim = 0), dim = 0)
            #     true_idx = true_idx * X.size(0) + torch.tensor([i for i in range(X.size(0))]).to(self.device)
            #     probs_output = torch.index_select(torch.cat(experts_outputs, dim = 0), dim = 0, index = true_idx) 
            #     loss = self.loss_func(probs_output, Y).item()
            #     acc_final = self.acc_compute(probs_output, Y)
            #     aux_acc = torch.mean((true_idx == expert_idx).float()).item()
            #     return loss_original, loss_each_experts, loss_soft, loss,  aux_loss.item(), aux_acc, acc_originals, probs_output, acc_final

            ########## adaptative training
            if self.args.train_type == "loss_base":
                robust_output_s = [torch.softmax(out, dim=1) for out in experts_outputs]
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
                losses_detach = torch.stack(losses) # [3, batch_size]
                expert_idx_pre = torch.argmin(losses_detach, dim = 0).detach() # 
                expert_idx = expert_idx_pre * X.size(0) + torch.tensor([i for i in range(X.size(0))]).to(self.device)
                aux_loss = torch.mean(torch.stack(aux_loss))
                # loss_soft = torch.mean(torch.stack(loss_original))
                loss_experts = torch.index_select(torch.cat(losses, dim = 0), dim = 0, index = expert_idx)
                loss_each_experts = [torch.mean(loss_experts[torch.where(expert_idx_pre == i)[0]]).item()  for i in range(self.args.num_experts)]
                acc_originals = [self.acc_compute(eo, Y) for eo in experts_outputs]
                true_idx = torch.argmax(torch.stack(robust_detector, dim = 0), dim = 0)
                true_idx = true_idx * X.size(0) + torch.tensor([i for i in range(X.size(0))]).to(self.device)
                probs_output = torch.index_select(torch.cat(experts_outputs, dim = 0), dim = 0, index = true_idx) 
                loss = self.loss_func(probs_output, Y)
                acc_final = self.acc_compute(probs_output, Y)
                aux_acc = torch.mean((true_idx == expert_idx).float()).item()
                # loss_soft = torch.sum(torch.stack(loss_original)) + self.args.adaptivetrainlambda * aux_loss ### normal train
                loss_soft = torch.mean(torch.sum(nn.Softmax(dim = 0)(torch.stack(robust_detector) * self.args.tempC ).detach() * losses_detach, dim = 0))  +  self.args.adaptivetrainlambda * aux_loss ### predicted best
                # loss_soft = loss +  self.args.adaptivetrainlambda * aux_loss ### predicted best
                # loss_soft = torch.mean(torch.sum(F.softmax(-losses_detach, dim = 0).detach() * losses_detach, dim = 0)) +  self.args.adaptivetrainlambda * aux_loss  ### true best
                return loss_original, loss_each_experts, loss_soft, loss.item(),  aux_loss.item(), aux_acc, acc_originals, probs_output, acc_final


            elif self.args.train_type == "gate_base":
                loss_original = [self.loss_func(eo, Y) for eo in experts_outputs]
                loss_each_experts = [loss.item() for loss in loss_original]
                acc_originals = [self.acc_compute(eo, Y) for eo in experts_outputs]
                gate_outputs = torch.cat( [ out[1] for out in outputs_all] , dim = 1) # batch_size, 
                gate_outputs = self.softmax(self.gate(gate_outputs)).unsqueeze(1) #### # gate_outputs = [batch_size, 1, num_experts]
                experts_cat = torch.stack(experts_outputs).permute(1,0,2) # experts_cat  = [batch_size, num_experts, 10]
                probs = torch.bmm(gate_outputs, experts_cat).squeeze()  # [batch_size,  10] 
                loss_soft = self.loss_func(probs, Y)
                acc_final = self.acc_compute(probs, Y)
                return loss_original, loss_each_experts, loss_soft, loss_soft.item(), 0,0, acc_originals, probs, acc_final

            elif self.args.train_type == "moe":
                loss_original = [self.loss_func(eo, Y) for eo in experts_outputs]
                loss_each_experts = [loss.item() for loss in loss_original]
                acc_originals = [self.acc_compute(eo, Y) for eo in experts_outputs]
                gate_outputs = self.softmax(self.gate(X)).view(X.size(0), 1, -1) # 1000, 3
                probs =  torch.bmm(gate_outputs, torch.stack(experts_outputs).permute(1,0,2)).squeeze() # 1000, 3, 10
                loss_soft = self.loss_func(probs, Y)
                acc_final = self.acc_compute(probs, Y)
                return loss_original, loss_each_experts, loss_soft, loss_soft.item(), 0,0, acc_originals, probs, acc_final
            
            elif self.args.train_type == "normal":
                loss_original = [self.loss_func(eo, Y) for eo in experts_outputs]
                loss_each_experts = [loss.item() for loss in loss_original]
                acc_originals = [self.acc_compute(eo, Y) for eo in experts_outputs]
                if self.args.num_experts ==1:
                    predicted_output = experts_outputs[0]
                else:
                    softmax_experts_outputs = [self.softmax(e) for e in experts_outputs]
                    predicted_output = torch.mean(torch.stack(softmax_experts_outputs), dim = 0, keepdim=False)
                    predicted_output = torch.clamp(predicted_output, min=1e-40)
                    predicted_output = torch.log(predicted_output)
                loss_soft = self.loss_func(predicted_output, Y)
                acc_final = self.acc_compute(predicted_output, Y)
                return loss_original, loss_each_experts, loss_soft, loss_soft.item(), 0,0, acc_originals, predicted_output, acc_final                  

            else:
                print("error train type and two_branch setting")
                exit()
