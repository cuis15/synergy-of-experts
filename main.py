# this is main function.
# main.py
import copy 
import numpy as np
import torch
import os.path as osp
from utils.utils_data import get_loaders
from utils.utils_func import construct_log, get_random_dir_name, setup_seed
from models.learn_rectified_v2_5_13 import Learning
import os
import pdb
import argparse
import pickle
from eval.eval import transfer_box, transfer_box_all

parser = argparse.ArgumentParser()

#### deploy setting
parser.add_argument('--auto_deploy', action='store_true', help='whether auto deploy not')
parser.add_argument('--host_name', type=str, default= "", help='where this try will be executed')
parser.add_argument('--eval_mode', action='store_true', help='if just eval the performance of the saved model ')


#### training setting
parser.add_argument('--lr_expert', type=float, default= 0.1,  help='the learning rate for the experts')
parser.add_argument('--lr_gate', type=float, default= 0.1,  help='the learning rate for the gate')
parser.add_argument('--momentum', type=float, default= 0.9,  help='momentum for the optimizer')
parser.add_argument('--weight_decay', type=float, default= 1e-4,  help='weight_decay for the optimizer')
parser.add_argument('--gamma', type=float, default= 0.1,  help='gamma parameter for the scheduler')
parser.add_argument('--intervals', nargs='*', default=[50,75], type=int, help='learning scheduler milestones')
parser.add_argument('--num_experts', type=int, default= 3,  help='the number of experts')
parser.add_argument('--gpu', type=str, default= 0,  help='the gpu id for training')
parser.add_argument('--batch_size', type=int, default= 128,  help='dataset batch size for training')
parser.add_argument('--seed', type=int, default= 0, help='random seed for training')
parser.add_argument('--add_gaussian', action='store_true',  help='if add gaussian noise for the cifar10 data')
parser.add_argument('--num_workers', type=int, default= 0,  help='num workers for data loader')
parser.add_argument('--plus_at', action = 'store_true',  help='num workers for data loader')
parser.add_argument('--test_interval_epoches', type=int, default= 10,  help='num workers for data loader')
parser.add_argument('--save_interval_epoches', type=int, default= 10,  help='num workers for data loader')
parser.add_argument('--attack_interval_epoches', type=int, default= 10,  help='num workers for data loader')
parser.add_argument('--robust_eval_interval_epoches', type=int, default= 10,  help='num workers for data loader')
parser.add_argument('--total_epoches', type = int, default = 200,  help='the layer depth of the model resnet')
parser.add_argument('--shuffle', action = 'store_true',  help='whether to shuffle the dataset')


#### data setting
parser.add_argument('--dataset', type=str, default= "cifar10",  help='dataset name')
parser.add_argument('--num_classes', type=int, default= 10,  help='dataset name')


#### model setting
parser.add_argument('--model_type', type=str, default= "ResNet20",  help='the model type')
parser.add_argument('--tower_type', type=str, default= "simple",  help='the tower type')
parser.add_argument('--ppd_type', type=str, default= "simple",  help='the ppd type')
parser.add_argument('--optim_type', type=str, default= "sgd",  help='optim selection, adam and sgd')
parser.add_argument('--leaky_relu', action = "store_true",  help='the model structure')
parser.add_argument('--depth', type = int, default = 20,  help='the layer depth of the model resnet')
parser.add_argument('--image_size', type = int, default = 3*32*32,  help='the layer depth of the model resnet')
parser.add_argument('--fusion_type', type=str, default= "regular",  help='the moe structure type [coupling, allocating, ...] ')


#### attack setting
parser.add_argument('--attack_type', default = "white_box", type = str,   help='attack type when training')
parser.add_argument('--attack_eps', type=float, default= 0.02,  help='eps ball for finding the adversarial samples')
parser.add_argument('--alpha', type=float, default= 0.01,  help='alpha for attack')                              
parser.add_argument('--attack_steps', type=int, default= 50,  help='steps for PGD, etc.')
parser.add_argument('--is_targeted', action='store_true',  help='if target attack')
parser.add_argument('--rand_start', action='store_true',  help='if random start given a clean sample when adversarial training')
parser.add_argument('--subset_num', default=1000, type=int, help='number of samples of the subset, will use the full test set if none')
parser.add_argument('--random_start_attack', type=int, default= 1,  help='evaluating the robustness when being attacked')
parser.add_argument('--attack_loss_fn', default="xent", type=str, help='confidence for cw loss function')


## white_box attack
parser.add_argument('--without_wbox', action='store_true',  help='whether to white-box attack during training')
parser.add_argument('--convergence_check', action='store_true',  help='whether to check the convergence when attack')
parser.add_argument('--wbox_type_pgd', action='store_true',  help='white-box attack type')
parser.add_argument('--wbox_type_bim', action='store_true',  help='white-box attack type')
parser.add_argument('--wbox_type_fgsm', action='store_true',  help='white-box attack type')
parser.add_argument('--wbox_type_mim', action='store_true',  help='white-box attack type')
parser.add_argument('--cw_conf', default=.1, type=float, help='confidence for cw loss function')
parser.add_argument('--wbox_lr', type=float, default= 0.01,  help='the white-box learning rate')


## transferability attack
parser.add_argument('--steps_transfer', type=int, default= 100,  help='the PGD steps in transfer attack')
## diversity evaluation


#### dir setting
parser.add_argument('--data_root', type=str, default= "",  help='data set dir root')
parser.add_argument('--outputs_root', type=str, default='', help=" dir name of for saving all experiments")
parser.add_argument('--target_dir', type=str, default='', help=" dir name of for saving the tmp experiment")
parser.add_argument('--log_dir', type=str, default='', help=" the absolute path dir name of for saving the tmp experiment")
parser.add_argument('--log_name', type=str, default='log', help="the log name for saving all log contents")
parser.add_argument('--model_dir', type=str, default='', help=" the absolute path dir name of for saving the model")
parser.add_argument('--pre_train_model_dir', type=str, default='', help=" the absolute path dir name for loading the model") 
parser.add_argument('--save_to_csv', default=False, action="store_true", help='whether to save the attack results in a csv file.')


#### training
parser.add_argument('--train_type', type=str, default= "loss_base",  help='training method: 1.loss_base; 2.kl_base; 3.gate_base_soft; 4.normal')


#### evaluating
parser.add_argument('--valid_type', type=str, default= "max_confidence",  help='evaluation method: 1.kl_loss, 2.kappa, 3.normal, 4.max_confidence, 5.max_entropy, 6.max_margin, 7.loss_base')


### valid kl_loss
parser.add_argument('--kl_select_type', type=str, default= "argmin",  help='select kappa using argmin or argmax')
parser.add_argument('--kl_num_sample', type=int, default= 1,  help='the num samples for evaluating the kl div.')
parser.add_argument('--lr_kl', type=float, default= 0.01,  help='lr kl for updating the adversarial samples')


### valid kappa
parser.add_argument('--kappa_random_type', type=str, default= "none",  help='kappa random type 1. none; 2. trades; 3. mart')
parser.add_argument('--K', type=int, default= 20,  help='the PGD steps for measuring the kappa')
parser.add_argument('--lr_kappa', type=float, default= 0.01,  help='lr kappa for updating the adversarial samples')
parser.add_argument('--kappa_omega', type=float, default= 0.0,  help='lr kappa for updating the adversarial samples')
parser.add_argument('--kappa_select_type', type=str, default= "argmin",  help='select kappa using argmin or argmax')


#### dverge training (used for data augmentation)
parser.add_argument('--distill_data', default=False, action="store_true", help='whether use distilltion data as in DVERGE')
parser.add_argument('--distill_fixed_layer', default=False, action="store_true", help='whether fixing the layer for distillation')
parser.add_argument('--distill_layer', default=20, type=int, help='which layer is used for distillation, only useful when distill-fixed-layer is True')
parser.add_argument('--distill_eps', default=0.07, type=float,  help='perturbation budget for distillation')
parser.add_argument('--distill_alpha', default=0.007, type=float,  help='step size for distillation')
parser.add_argument('--distill_steps', default=10, type=int, help='number of steps for distillation')
parser.add_argument('--distill_rand_start', default=False, action="store_true", help='whether use random start for distillation')
parser.add_argument('--distill_no_momentum', action="store_false", dest='distill_momentum', help='whether use momentum for distillation')


#### black box attack
parser.add_argument('--black_mode_clean', action="store_true", help='whether use clean data in black box attack.')
parser.add_argument('--black_surro_num_experts', type =  int, default= 3,  help='the num of surrogate experts when in black box attack.')
parser.add_argument('--black_method', type =  str, default= "mpgd",  help='method when in black box attack. (mpgd, mdi2_0.5, sgm_0.2)')
parser.add_argument('--black_random_start_seed', type = int, default= 0,  help='random start when in mpgd mode in black box attack. (0,1,2)')
parser.add_argument('--black_datafolder', type = str, default = "",  help='the data folder when downloading the adv data in mpgd mode in black box attack. (0,1,2)')


#### rectified confidence setting
parser.add_argument('--hidden_dim', type =  int, default = 10,  help='the out dim for the RR module (default 1)')
parser.add_argument('--out_dim', type =  int, default = 1,  help='the out dim for the RR module (default 1)')
parser.add_argument('--two_branch', action="store_true", help='if use rectified confidence')
parser.add_argument('--tempC', default=1., type=float)
parser.add_argument('--tempC_trueonly', default=1., type=float) # stop gradient for the confidence term  
parser.add_argument('--adaptivetrainlambda', default=1., type=float, help = "the hyperparameter for training RR module")
parser.add_argument('--plus_at_type', type =  str, default = "together",  help='the type for attack the experts')
parser.add_argument('--adapt_attack_weight', type = float, default = 0.0,  help='the weight of aux_loss used for adaptative attack.')
args = parser.parse_args()


if __name__ == '__main__':
    ##########################
    #### adjust the remote dir
    ##########################
    args.data_root = "data"
    args.outputs_root = 'outputs'


    ########################## specify the GPU
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    torch.cuda.set_device(device)
    ##########################


    if  args.target_dir == "":
        args.log_dir = os.path.join(args.outputs_root, get_random_dir_name())
    elif args.host_name == "":
        args.log_dir = os.path.join(args.outputs_root, args.target_dir, get_random_dir_name() )
    else:
        args.log_dir = os.path.join(args.outputs_root, args.target_dir)
    args.model_dir = os.path.join(args.log_dir, "model_saved")
    logger = construct_log(args)


    ###################
    #### start learning
    ###################
    logger.info("start learning")
    setup_seed(seed = args.seed)
    train_loader, test_loader = get_loaders(args)
    model = Learning(args, logger, train_loader, test_loader)



    if args.auto_deploy:
        os.system("cp -r {} {}".format(os.path.dirname(os.path.realpath(__file__)), args.log_dir))
        logger.info("the source code has been saved")
        try:
            model.run()
        except Exception as e:
            logger.info(e)
        
    else:
        os.system("cp -r {} {}".format(os.path.dirname(os.path.realpath(__file__)), args.log_dir))
        logger.info("the source code has been saved")
        model.run()
        with open(os.path.join(args.log_dir, "pickle.pkl"), "wb") as f:
            pickle.dump(model.pickle_record, f)

