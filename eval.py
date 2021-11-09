import importlib
import os
import warnings
warnings.filterwarnings('ignore')

import argparse



my_parser = argparse.ArgumentParser()
my_parser.add_argument('--dataset_name', type=str, default="breakfast", choices=['breakfast', '50salads', 'gtea'])
my_parser.add_argument('--split', type=int, required=False, help="Comma seperated split number to run evaluation," +\
                                                                  "default = 1,2,3,4 for breakfast and gtea, 1,2,3,4,5 for 50salads")
my_parser.add_argument('--cudad', type=str, help="Cuda device number to run evaluation program in")
my_parser.add_argument('--base_dir', type=str, help="Base directory with groundTruth, features, splits, results directory of dataset")
my_parser.add_argument('--model_path', type=str, default='model')
my_parser.add_argument('--wd', type=float, required=False, help="Provide weigth decay if you want to change from default")
my_parser.add_argument('--lr', type=float, required=False, help="Provide learning rate if you want to change from default")
my_parser.add_argument('--chunk_size', type=int, required=False, help="Provide chunk size to be used if you want to change from default")
my_parser.add_argument('--max_frames', type=int, required=False, 
                        help="Max number of frames to be considered for a given video, beyond it would be considered as new sample")
my_parser.add_argument('--ensem_weights', type=str, required=False,
                        help='Default = \"1,1,1,0,0,0\", provide in similar comma-seperated 6 weights values if required to be changed')
my_parser.add_argument('--ft_file', type=str, required=False, help="Provide feature file dir path if default is not base_dir/features")
my_parser.add_argument('--ft_size', type=int, required=False, help="Default I3D features size = 2048, change if default feature size changes")
args = my_parser.parse_args()

os.environ["WANDB_API_KEY"] = "992b3b1371ba79f48484cfca522b3786d7fa52c2"
# os.environ["WANDB_MODE"] = "dryrun"


import random

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
seed = 42

# Ensure deterministic behavior
def set_seed():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
set_seed()

# Device configuration
os.environ['CUDA_VISIBLE_DEVICES']=args.cudad
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


from util.misc import dotdict
from models.transformer_wt import TransformerEncoderLayer
from models.model import PositionalEncoding, MLP
from util.misc import calculate_mof
from unet_model.testtime_postprocess import PostProcess
import torch.nn.functional as F

config = dotdict(
    epochs=500,
    dataset=args.dataset_name,
    feature_size=2048,
    gamma=0.5,
    step_size=500,
    model_path=args.model_path,
    base_dir =args.base_dir,
    aug=1,
    lps=1)

if args.noaug:
    config.aug = 0

if args.nolps:
    config.lps = 0

config.ensem_weights = [1, 1, 1, 1, 0, 0]
if args.dataset_name == "50Salads":
    config.chunk_size = 20
    config.max_frames_per_video = 960
    config.learning_rate = 3e-4
    config.weight_decay = 1e-3
    config.batch_size = 20
    config.num_class = 19
    config.back_gd = ['action_start', 'action_end']
    config.num_of_splits = 5
    if args.compile_result:
        config.chunk_size = [20]
        config.weights = [1]
        config.eval_true = True
    else:
        config.chunk_size = list(range(20,40))
        config.weights = np.ones(len(config.chunk_size))
        config.eval_true = False
elif args.dataset_name == "breakfast":
    config.chunk_size = 10
    config.max_frames_per_video = 600
    config.learning_rate = 1e-4
    config.weight_decay = 3e-3
    config.batch_size = 100
    config.num_class = 48
    config.back_gd = ['SIL']
    config.num_of_splits = 4
    if args.compile_result:
        config.chunk_size = [10]
        config.weights = [1]
        config.eval_true = True
    else:
        config.chunk_size = list(range(8,16))
        config.weights = np.ones(len(config.chunk_size))
        config.eval_true = False
elif args.dataset_name == "gtea":
    config.chunk_size = 4
    config.max_frames_per_video = 600
    config.learning_rate = 5e-4
    config.weight_decay = 3e-4
    config.batch_size = 11
    config.num_class = 11
    config.back_gd = ['background']
    config.num_of_splits = 4
    config.eval_true = True
    if args.compile_result:
        config.chunk_size = [4]
        config.weights = [1]
    else:
        config.chunk_size = [3, 4, 5] # list(range(20,40))
        config.weights = [1, 3, 1]

    

config.features_file_name=config.base_dir + "/features/"
config.ground_truth_files_dir=config.base_dir + "/groundTruth/"
config.label_id_csv = config.base_dir + "mapping.csv"


def model_pipeline(config):
    acc_list = []
    edit_list = []
    f1_10_list = []
    f1_25_list = []
    f1_50_list = []
    for ele in range(1, config.num_of_splits+1):
        config.output_dir=config.base_dir + "results/split{}_{}_lps{}_aug{}".format(ele,
                                                                                     config.model_path.split(".")[-1],
                                                                                     config.lps,
                                                                                     config.aug)
        if args.wd is not None:
            config.weight_decay = args.wd
            config.output_dir=config.output_dir + "_wd{:.5f}".format(config.weight_decay)

        if args.lr is not None:
            config.learning_rate = args.lr
            config.output_dir=config.output_dir + "_lr{:.6f}".format(config.learning_rate)

        if args.chunk_size is not None:
            config.chunk_size = args.chunk_size
            config.output_dir=config.output_dir + "_chunk{}".format(config.chunk_size)

        if args.max_frames is not None:
            config.max_frames_per_video = args.max_frames    
            config.output_dir=config.output_dir + "_maxf{}".format(config.max_frames_per_video)

        if args.ensem_weights is not None:
            config.output_dir = config.output_dir + "_wts{}".format(args.ensem_weights.replace(',','-'))
            config.ensem_weights = list(map(int, args.ensem_weights.split(",")))
            print("Weights being used is ", config.ensem_weights)
        config.output_dir=config.output_dir + "/"

        print("printing getting the output from output dir = ", config.output_dir)
        config.project_name="{}-split{}".format(config.dataset, ele)
        config.test_split_file=config.base_dir + "splits/test.split{}.bundle".format(ele)
        # make the model, data, and optimization problem
        model, test_loader, postprocessor = make(config)
        if args.best_mof:
            model.load_state_dict(load_best_model(config))
            prefix = ''
        else:
            model.load_state_dict(load_avgbest_model(config))
            prefix = 'avg'

        if config.eval_true:
            model.eval()


        correct, correct1, total = 0, 0, 0
        postprocessor.start()

        with torch.no_grad():
            for i, item in enumerate(test_loader):
                samples = item[0][0].to(device).permute(0,2,1)
                count = item[0][1].to(device)
                labels = item[0][2].to(device)
                src_mask = torch.arange(labels.shape[1], device=labels.device)[None, :] < count[:, None]
                src_mask = src_mask.to(device)

                outplist = model(samples)
                ensembel_out = get_ensemble_out(outplist)

                pred = torch.argmax(ensembel_out, dim=1)
                correct += float(torch.sum((pred==labels)*src_mask).item())
                total += float(torch.sum(src_mask).item())

                # 7 chunk size, 8 is chunk id
                postprocessor(ensembel_out, item[0][5], labels, count, item[0][7].to(device), item[0][8], item[0][3].to(device)) 

        print(f'Accuracy: {100.0*correct/total: .2f}')
         # Add postprocessing and check the outcomes
        path = os.path.join(config.output_dir, prefix + "test_time_augmentation_split{}".format(ele))
        if not os.path.exists(path):
            os.mkdir(path)
        postprocessor.dump_to_directory(path)
    
        final_edit_score, map_v, overlap_scores = calculate_mof(config.ground_truth_files_dir, path, config.back_gd)
        acc_list.append(map_v*100)
        edit_list.append(final_edit_score)
        f1_10_list.append(overlap_scores[0])
        f1_25_list.append(overlap_scores[1])
        f1_50_list.append(overlap_scores[2])

    print("Frame accuracy = ", np.mean(np.array(acc_list)))
    print("Edit Scores = ", np.mean(np.array(edit_list)))
    print("f1@10 Scores = ", np.mean(np.array(f1_10_list)))
    print("f1@25 Scores = ", np.mean(np.array(f1_25_list)))
    print("f1@50 Scores = ", np.mean(np.array(f1_50_list)))


def load_best_model(config):
    return torch.load(config.output_dir + '/best_' + config.dataset + '_unet.wt')

def load_avgbest_model(config):
    return torch.load(config.output_dir + '/avgbest_' + config.dataset + '_unet.wt')

def make(config):
    # Make the data
    test = get_data(config, train=False)
    test_loader = make_loader(test, batch_size=config.batch_size, train=False)

    # Make the model
    model = get_model(config).to(device)
    
    num_params = sum([p.numel() for p in model.parameters()])
    print("Number of parameters = ", num_params/1e6, " million")

    # postprocessor declaration
    postprocessor = PostProcess(config, config.weights)
    postprocessor = postprocessor.to(device)
    
    return model, test_loader, postprocessor


def get_data(args, train=True):
    from unet_model.testtime_dataloader import Breakfast, collate_fn_override
    if train is True:
        fold='train'
        split_file_name = args.train_split_file
    else:
        fold='val'
        split_file_name = args.test_split_file
    dataset = Breakfast(args, fold=fold, fold_file_name=split_file_name, chunk_size=config.chunk_size)
    
    return dataset


def make_loader(dataset, batch_size, train=True):
    from unet_model.testtime_dataloader import Breakfast, collate_fn_override
    def _init_fn(worker_id):
        np.random.seed(int(seed))
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=batch_size, 
                                         shuffle=train,
                                         pin_memory=True, num_workers=6, collate_fn=collate_fn_override,
                                         worker_init_fn=_init_fn)
    return loader


def get_model(config):
    my_module = importlib.import_module(config.model_path)
    set_seed()
    return my_module.UNetSSLContrastive(config.feature_size, config.num_class)


def get_ensemble_out(outp):
    
    weights = config.ensem_weights
    ensemble_prob = F.softmax(outp[0], dim=1) * weights[0] / sum(weights)

    for i, outp_ele in enumerate(outp[1]):
        upped_logit = F.upsample(outp_ele, size=outp[0].shape[-1], mode='linear', align_corners=True)
        ensemble_prob = ensemble_prob + F.softmax(upped_logit, dim=1) * weights[i + 1] / sum(weights)
    
    return ensemble_prob

model = model_pipeline(config)
