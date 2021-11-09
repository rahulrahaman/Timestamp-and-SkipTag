import importlib
import os, sys
import warnings
import glob
import numpy as np
import torch
import pandas as pd
import random
import torch.nn as nn
import pickle
import torch.nn.functional as F
import matplotlib.pyplot as plt
import multiprocessing as mp
from time import time
from utils import get_all_scores
import argparse
import wandb
from trainutil import train, validate, final_validation
from postprocess import PostProcess

from mstcn_model import *
from utility.adaptive_data_loader import Breakfast, collate_fn_override
from utility.adaptive_data_loader import BreakfastWithWeights, collate_fn_override_wtd
from utils import calculate_mof, dotdict


os.environ["WANDB_API_KEY"] = "992b3b1371ba79f48484cfca522b3786d7fa52c2"
wandb.login()

help_text = '''This is the program to run when random selection of frames is used to and the
generalized EM framework needs to be run. Below the arguments are listed. An example command
would be: 
python runcodes/run_em_generalized.py --split 1 --select_f data/50salads_random19frame_selection.pkl 
    --init_epoch 50 --epochs 150 --sample_rate 2 --cudad 0 --base_dir /mnt/ssd/all_users/dipika/ms_tcn/data/50salads/ 
    --lr 0.0005 --postprocess
'''

my_parser = argparse.ArgumentParser(description=help_text)
my_parser.add_argument('--split', type=int, required=True)
my_parser.add_argument('--select_f', type=str, required=True)
my_parser.add_argument('--custom_name', type=str, required=False, default="")
my_parser.add_argument('--init_epoch', type=int, required=False, default=25)
my_parser.add_argument('--sample_rate', type=int, required=False, default=1)
my_parser.add_argument('--use_mse', action='store_true')
my_parser.add_argument('--use_conf', action='store_true')
my_parser.add_argument('--cudad', type=str)
my_parser.add_argument('--base_dir', type=str)
my_parser.add_argument('--lr', default=5e-4, type=float, required=False)
my_parser.add_argument('--epochs', default=80, type=int, required=False)
my_parser.add_argument('--resume_model', type=str, required=False)
my_parser.add_argument('--start_epoch', type=int, default=0, required=False)
my_parser.add_argument('--batch_size', default=8, type=int)
my_parser.add_argument('--train_batch_size', default=20, type=int)
my_parser.add_argument('--expectation_cal_gap', default=5, type=int)
my_parser.add_argument('--temp', default=1, type=float)
my_parser.add_argument('--prior_temp', default=1, type=float)
my_parser.add_argument('--feature_size', default=2048, type=int)
my_parser.add_argument('--postprocess', action='store_true')
my_parser.add_argument('--notrain', action='store_true')
seed = 42

posterior_acc_correct, posterior_acc_total, boundary_count = 0, 0, 0
posterior_boundary_total_mse = 0
results = []
boundary_frames_dict = None
selected_frames_dict = None
mat_poisson = None
args = None
label_id_to_label_name = {}
label_name_to_label_id_dict = {}


# Ensure deterministic behavior
def set_seed():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)


def get_out_dir():
    global args
    if not os.path.exists(args.base_dir + "results/em_algo/"):
        os.mkdir(args.base_dir + "results/em_algo/")

    args.output_dir = args.base_dir + f"results/em_algo/gen_{args.select_f.split('/')[-1].split('.')[0]}"
    if args.use_mse:
        args.output_dir += '_with_mse'
    if args.use_conf:
        args.output_dir += '_with_conf'

    if len(args.custom_name) > 0:
        args.output_dir += '_' + args.custom_name
    args.boundary_f = f"data/{args.dataset_name}_boundary_annotations.pkl"
    args.poisson_f = f"data/{args.dataset_name}_possion_class_dict.pkl"
    args.mean_var_f = f"data/{args.dataset_name}_meanvar_actions.pkl"

    if args.dataset_name == "50salads":
        args.num_class = 19
        args.back_gd_list = ['action_start', 'action_end']
        args.action_start = 17
        args.action_end = 18

    elif args.dataset_name == "breakfast":
        args.num_class = 48
        args.back_gd_list = ['SIL']
        args.action_start = 47
        args.action_end = 47
        
    elif args.dataset_name == "gtea":
        args.num_class = 11
        args.back_gd_list = ['background']
        args.action_start = 10
        args.action_end = 10

    args.output_dir = args.output_dir + "/"

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    args.output_dir = args.output_dir + f"split{args.split}"
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    args.output_dir = args.output_dir + "/"

    print("printing in output dir = ", args.output_dir)
    if not os.path.exists(os.path.join(args.output_dir, "posterior_weights")):
        os.mkdir(os.path.join(args.output_dir, "posterior_weights"))

    args.train_split_file = args.base_dir + "/splits/train.split{}.bundle".format(args.split)
    args.test_split_file = args.base_dir + "/splits/test.split{}.bundle".format(args.split)
    args.features_file_name = args.base_dir + "/features/"
    args.ground_truth_files_dir = args.base_dir + "/groundTruth/"
    args.label_id_csv = args.base_dir + "mapping.csv"
    return args


def get_label_idcsv():
    global label_name_to_label_id_dict, label_id_to_label_name
    df = pd.read_csv(args.label_id_csv)
    for i, ele in df.iterrows():
        label_id_to_label_name[ele.label_id] = ele.label_name
        label_name_to_label_id_dict[ele.label_name] = ele.label_id


def model_pipeline():
    global boundary_frames_dict, selected_frames_dict, mat_poisson
    global args
    args = get_out_dir()

    # Device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cudad
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = get_model(args).to(device)

    if args.resume_model is not None:
        model_file_path = os.path.join(args.output_dir, args.resume_model)
        model.load_state_dict(torch.load(model_file_path))
        print(f"Loaded model with successfully from path {model_file_path}")

    train_loader_list, test_loader, criterion_list, optimizer = make(model, device)
    print("Datasets and loaders are created")

    boundary_frames_dict = pickle.load(open(args.boundary_f, "rb"))
    selected_frames_dict = pickle.load(open(args.select_f, "rb"))
    mat_poisson = pickle.load(open(args.poisson_f, "rb"))
    get_label_idcsv()
    print("All files have been loaded")

    args.postprocess = PostProcess(args) if args.postprocess else None
    conf_loss = confidence_loss if args.use_conf else None

    if args.resume_model is not None:
        perform_expectation(model, train_loader_list[1], device)
        _ = validate(args, model, test_loader, device)
    
    if not args.notrain:
        train(device, model, train_loader_list, criterion_list, optimizer, args,
              test_loader, get_single_random, perform_expectation, conf_loss)

    final_validation(args, model, test_loader, device)
    return


def prob_vals_per_segment(selected_frames, cur_vid_feat, labels, vidid, gt_labels):
    prob_each_segment = []
    LOW_VAL = -10000000
    num_frames = len(cur_vid_feat)
    log_probs = torch.log(cur_vid_feat + 1e-8)
    cumsum_feat = torch.cumsum(log_probs, dim=0)
    prev_boundary = 0
    per_frame_weights = torch.zeros((num_frames, args.num_class))
    start_time = time()
    boundary_error = 0
    num_boundary = 0
    labels = [args.action_start] + labels if selected_frames[0] != 0 else labels
    labels = labels + [args.action_end] if selected_frames[-1] != num_frames-1 else labels
    selected_frames = [0] + selected_frames if selected_frames[0] != 0 else selected_frames
    selected_frames = selected_frames + [num_frames-1] if selected_frames[-1] != num_frames-1 else selected_frames
    
    for i, cur_ele in enumerate(selected_frames[:-1]):
        next_ele = selected_frames[i + 1]
        label_cur_ele = labels[i]
        label_next_ele = labels[i + 1]
        if cur_ele == next_ele-1:
            per_frame_weights[cur_ele, label_cur_ele] = 1.0
            if label_cur_ele != label_next_ele:
                prev_boundary = cur_ele
            continue
        
        seg_len = next_ele - cur_ele
        mat_b1_b2_c_prob = LOW_VAL * torch.ones((seg_len, seg_len, args.num_class), dtype=cumsum_feat.dtype)
        b1_prior = get_poisson_prob(cur_ele-prev_boundary, next_ele-prev_boundary, label_cur_ele)
        
        # find dummy label where we will keep the diagonal (b1=b2) probabilities, later we will distribute among
        # rest of the classes after the softmax by dividing by (num_class - 2)
        dummy_label = 0
        while True:
            if dummy_label != label_cur_ele and dummy_label != label_next_ele:
                break
            else:
                dummy_label += 1
        
        for b1 in range(cur_ele, next_ele - 1):

            cur_boundary_len = b1 - prev_boundary
            strt_index = cumsum_feat[cur_ele - 1, label_cur_ele] if cur_ele > 0 else 0
            left_sum = (cumsum_feat[b1, label_cur_ele] - strt_index)
            right_sum = cumsum_feat[next_ele-1, label_next_ele] - cumsum_feat[b1+1:next_ele, label_next_ele]  # mid_seg_len
            mid_sum = (cumsum_feat[b1+1:next_ele, :] - cumsum_feat[b1, :])  # mid_seg_len
            b2_prior = get_poisson_prob_for_all_class(1, next_ele-b1)  # mid_seg_len x num_class
            
            mat_b1_b2_c_prob[b1-cur_ele, b1+1-cur_ele:next_ele-cur_ele] = (left_sum + right_sum[:,None] + mid_sum) / args.temp \
                                                                            + (b1_prior[b1-cur_ele] + b2_prior) / args.prior_temp
            # when mid segment is absent but right and left is not the same
            # we assign the probability to a dummy label for now and then later
            # re-distribute among other classes after the softmax
            if label_cur_ele != label_next_ele:
                rightsum_wo_midseg = cumsum_feat[next_ele-1, label_next_ele] - cumsum_feat[b1, label_next_ele]
                mat_b1_b2_c_prob[b1-cur_ele, b1-cur_ele, dummy_label] = (left_sum + rightsum_wo_midseg) / args.temp \
                                                                         + b1_prior[b1-cur_ele] / args.prior_temp
        
        # when mid segment is absent b1 can also be next_ele-1
        b1 = next_ele - 1
        if label_cur_ele != label_next_ele:
            left_sum = (cumsum_feat[b1, label_cur_ele] - strt_index)
            mat_b1_b2_c_prob[b1-cur_ele, b1-cur_ele, dummy_label] = left_sum / args.temp + b1_prior[b1-cur_ele] / args.prior_temp
        else:
            # returns prob that the left class length >= seg len
            b1_prior_ = get_poisson_logcdf(next_ele - prev_boundary, label_cur_ele) 
            mat_b1_b2_c_prob[b1-cur_ele, b1-cur_ele, dummy_label] = left_sum / args.temp + b1_prior_ / args.prior_temp
        
        mat_b1_b2_c_prob[:, :, label_cur_ele] = LOW_VAL
        mat_b1_b2_c_prob[:, :, label_next_ele] = LOW_VAL
        mat_b1_b2_c_prob = torch.softmax(mat_b1_b2_c_prob.flatten(), dim=0).reshape((seg_len, seg_len, args.num_class))
        
        # re-distribute the dummy class probability among the left-over classes
        left_over_classes = args.num_class - 2 + (label_cur_ele == label_next_ele)
        for b1 in range(cur_ele, next_ele):
            assigned_prob = mat_b1_b2_c_prob[b1-cur_ele, b1-cur_ele, dummy_label]
            mat_b1_b2_c_prob[b1-cur_ele, b1-cur_ele, :] = assigned_prob/left_over_classes
            mat_b1_b2_c_prob[b1-cur_ele, b1-cur_ele, label_cur_ele] = 0
            mat_b1_b2_c_prob[b1-cur_ele, b1-cur_ele, label_next_ele] = 0
        
        marginal_b1 = torch.sum(mat_b1_b2_c_prob, axis=(1, 2))
        mean_b1 = round(torch.sum(marginal_b1.squeeze() * torch.arange(cur_ele, next_ele, 1)).item())
        cumm_b1_prob = torch.cumsum(marginal_b1, dim=0)
        cumm_b1_c_prob = torch.cumsum(torch.sum(mat_b1_b2_c_prob, dim=1), dim=0)
        cumm_b2_c_prob = torch.cumsum(torch.sum(mat_b1_b2_c_prob, dim=0), dim=0)

        per_frame_weights[cur_ele, label_cur_ele] = 1.0
        per_frame_weights[cur_ele+1:next_ele, :] = cumm_b1_c_prob[:-1] - cumm_b2_c_prob[:-1]
        per_frame_weights[cur_ele+1:next_ele, label_cur_ele] = 1 - cumm_b1_prob[:-1]
        per_frame_weights[cur_ele+1:next_ele, label_next_ele] = 0
        remaining_probability = 1 - torch.sum(per_frame_weights[cur_ele+1:next_ele, :], dim=-1)
        # we use "+=" in the next line because left and right label might be the same
        # in that case using "=" would just overwrite the previous probability
        per_frame_weights[cur_ele+1:next_ele, label_next_ele] += remaining_probability
        
        expected_boundary = round(torch.sum(torch.sum(mat_b1_b2_c_prob, axis=(0, 2)).squeeze() * \
                                  torch.arange(cur_ele, next_ele, 1)).item())
        if not (label_cur_ele == label_next_ele and expected_boundary >= next_ele-2):
            prev_boundary = expected_boundary
        if expected_boundary == 0 and i > 0:
            print(f'Estimated boundary has become zero! for {vidid} and cur_ele, next_ele {cur_ele, next_ele}')
            import pdb
            pdb.set_trace()
        # boundary_error += (boundary_frames_dict[vidid + '.txt'][current_boundary] - mean_b1)**2
        # boundary_error += (boundary_frames_dict[vidid + '.txt'][current_boundary+1] - prev_boundary)**2
        # num_boundary += 2
        # prob_each_segment.append(mat_b1_b2_c_prob)
        
    posterior_prediction = torch.argmax(per_frame_weights, dim=1)
    correct = torch.sum(posterior_prediction == gt_labels[:num_frames]).item()
    
    return vidid, per_frame_weights, [correct, num_frames, boundary_error, num_boundary]


# Step 2: Define callback function to collect the output in `results`
def collect_result(result):
    global posterior_acc_correct, posterior_acc_total, posterior_boundary_total_mse, boundary_count
    fname = os.path.join(args.output_dir, 'posterior_weights', result[0] + '.wt')
    torch.save(result[1], fname)
    correct, total, boundary_err, num_boundary = result[2]
    posterior_acc_correct += correct
    posterior_acc_total += total
    posterior_boundary_total_mse += boundary_err
    boundary_count += num_boundary
    # print(f'Dumped in file {fname} at time {time()}')
    return


def calculate_element_probb(data_feat, data_count, video_ids, gt_labels):
    global selected_frames_dict, results
    pool = mp.Pool(20)
    for iter_num in range(len(data_count)):
        cur_vidid = video_ids[iter_num]
        cur_vid_count = data_count[iter_num]
        cur_vid_feat = data_feat[iter_num][:cur_vid_count].detach().cpu()
        cur_gt_labels = gt_labels[iter_num].detach().cpu()
        
        cur_video_select_frames = selected_frames_dict[cur_vidid + ".txt"]
        selected_frames_indices_and_labels = cur_video_select_frames
        selected_frames_indices = [ele[0] for ele in selected_frames_indices_and_labels]
        selected_frames_labels = [label_name_to_label_id_dict[ele[1]] for ele in selected_frames_indices_and_labels]
        with torch.no_grad():
            # Multi-processing
            pool.apply_async(prob_vals_per_segment,
                             args=(selected_frames_indices, cur_vid_feat, selected_frames_labels, cur_vidid, cur_gt_labels),
                             callback=collect_result)
            # collect_result(prob_vals_per_segment(selected_frames_indices, cur_vid_feat, selected_frames_labels,
            #                cur_vidid, cur_gt_labels))
    # Step 4: Close Pool and let all the processes complete
    pool.close()
    pool.join()  # postpones the execution of next line of code until all processes in the queue are done.
    return results


def perform_expectation(model, dataloader, device):
    global posterior_acc_correct, posterior_acc_total, posterior_boundary_total_mse, boundary_count
    posterior_acc_correct, posterior_acc_total, posterior_boundary_total_mse, boundary_count = 0, 0, 0, 0
    model.eval()
    correct = 0.0
    total = 0.0
    curtime = time()
    print(f'Calculating expectation')

    for i, item in enumerate(dataloader):
        with torch.no_grad():
            item_0 = item[0].to(device) # features
            item_1 = item[1].to(device) # count
            item_2 = item[2].to(device) # gt frame-wise labels
            item_4 = item[4] # video-ids
            src_mask = torch.arange(item_2.shape[1], device=item_2.device)[None, :] < item_1[:, None]
            src_mask_mse = src_mask.unsqueeze(1).to(torch.float32).to(device)
            middle_pred, predictions = model(item_0, src_mask_mse)
            prob = torch.softmax(predictions[-1], dim=1)
            prob = prob.permute(0, 2, 1)
            
            calculate_element_probb(prob, item_1, item_4, item_2)
            if (i+1) % 10 == 0:
                print(f"iter {i+1} of Expectation completed in a total of {(time() - curtime)/60.: .1f} minutes")
    _ = model.train()
    posterior_acc = 100*posterior_acc_correct/posterior_acc_total
    posterior_boundary_mse = (posterior_boundary_total_mse/boundary_count)**0.5 if boundary_count > 0 else 0
    
    print(f'Expectation step finished, posterior frame-wise accuracy {posterior_acc: .2f}%, '
          f'boundary mse {posterior_boundary_mse: .2f}')
    if not args.notrain:
        wandb.log({'Posterior Accuracy': posterior_acc,
                   'Posterior Boundary MSE': posterior_boundary_mse
                  })
    return


def get_single_random(video_ids, len_frames, device):
    # Generate target for only timestamps. Do not generate pseudo labels at first 30 epochs.
    global selected_frames_dict
    boundary_target_tensor = torch.ones((len(video_ids), len_frames), dtype=torch.long, device=device) * (-100)
    for iter_num, cur_vidid in enumerate(video_ids):
        selected_frames_indices_and_labels = selected_frames_dict[cur_vidid + ".txt"]
        selected_frames_indices = [ele[0] for ele in selected_frames_indices_and_labels]
        selected_frames_labels = [label_name_to_label_id_dict[ele[1]] for ele in selected_frames_indices_and_labels]

        frame_idx_tensor = torch.from_numpy(np.array(selected_frames_indices))
        frame_labels = torch.from_numpy(np.array(selected_frames_labels)).to(device)
        boundary_target_tensor[iter_num, frame_idx_tensor] = frame_labels

    return boundary_target_tensor


def confidence_loss(probs, vid_ids, vidlens):
    global selected_frames_dict
    bsize, _, num_class = probs.shape
    loss = 0.0
    for proba, idx, vidlen in zip(probs, vid_ids, vidlens):
        logprob = torch.log(proba + 1e-8)[:vidlen]
        frame_labels = selected_frames_dict[idx + '.txt']
        mask_left = torch.zeros_like(logprob)
        mask_right = torch.zeros_like(logprob)
        for i in range(len(frame_labels)-1):
            label_id = label_name_to_label_id_dict[frame_labels[i][1]]
            next_label_id = label_name_to_label_id_dict[frame_labels[i+1][1]]
            if label_id == next_label_id:
                continue
            frame_id = frame_labels[i][0]
            next_frame_id = frame_labels[i+1][0]
            mask_left[frame_id: next_frame_id, next_label_id] = 1.0
            mask_right[frame_id: next_frame_id, label_id] = 1.0

        transition = logprob[1:] - logprob[:-1]
        right_penalty = torch.clamp(transition * mask_right[1:], min=0)
        loss += torch.sum(right_penalty)/(torch.sum(mask_right[1:]) + 1e-8)
        left_penalty = torch.clamp(-transition * mask_left[1:], min=0)
        loss += torch.sum(left_penalty)/(torch.sum(mask_left[1:]) + 1e-8)
    return loss/bsize


def load_best_model(args):
    return torch.load(args.output_dir + 'ms-tcn-emmax-best-model.wt')


def _init_fn(worker_id):
    np.random.seed(int(seed))


def get_poisson_prob(minlen, maxlen, cur_class):
    prob = mat_poisson[label_id_to_label_name[cur_class]][minlen:maxlen]
    return torch.tensor(prob)


def get_poisson_logcdf(minlen, cur_class):
    return np.log(np.sum(np.exp(mat_poisson[label_id_to_label_name[cur_class]][minlen:])) + 1e-20)


def get_poisson_prob_for_all_class(minlen, maxlen):
    global mat_poisson
    ele_list = []
    for i in range(args.num_class):
        prob = mat_poisson[label_id_to_label_name[i]][minlen:maxlen]
        ele_list.append(torch.tensor(prob))
    return torch.stack(ele_list, dim=-1)


def make(model, device):
    global args
    # Make the data
    all_train_data_files = open(args.train_split_file).read().split("\n")[0:-1]
    if len(all_train_data_files[-1]) <= 1:
        all_train_data_files = all_train_data_files[0:-1]
        # print(all_train_data_files)
    print("Length of files picked up for semi-supervised training is ", len(all_train_data_files))
    validation_data_files = open(args.test_split_file).read().split("\n")[0:-1]
    print("Length of files picked up for semi-supervised validation is ", len(validation_data_files))

    print(args)
    traindataset = BreakfastWithWeights(args, fold='train', fold_file_name=args.train_split_file,
                                        sample_rate=args.sample_rate)
    testdataset = Breakfast(args, fold='test', fold_file_name=args.test_split_file)

    trainloader = torch.utils.data.DataLoader(dataset=traindataset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              pin_memory=False, num_workers=8,
                                              collate_fn=collate_fn_override_wtd,
                                              worker_init_fn=_init_fn)
    testloader = torch.utils.data.DataLoader(dataset=testdataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             pin_memory=False, num_workers=8,
                                             collate_fn=collate_fn_override,
                                             worker_init_fn=_init_fn)

    trainloader_expectation = torch.utils.data.DataLoader(dataset=traindataset,
                                                          batch_size=args.train_batch_size,
                                                          shuffle=True,
                                                          pin_memory=False, num_workers=8,
                                                          collate_fn=collate_fn_override_wtd,
                                                          worker_init_fn=_init_fn)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion_list = [nn.CrossEntropyLoss(ignore_index=-100), nn.MSELoss(reduction='none')]

    return [trainloader, trainloader_expectation], testloader, criterion_list, optimizer


def get_model(args):
    set_seed()
    model = MultiStageModel(num_stages=4, num_layers=10, num_f_maps=64, dim=args.feature_size,
                            num_classes=args.num_class)
    return model


def main():
    global args
    args = my_parser.parse_args()
    print(args.base_dir)
    args.dataset_name = os.path.basename(os.path.normpath(args.base_dir))
    print("Dataset name ", args.dataset_name)
    if not args.notrain:
        wandb.init(project=f'EM random selection {args.dataset_name}-split{args.split}', config=args, entity='rahul-dipika')
    model = model_pipeline()


if __name__ == '__main__':
    main()
