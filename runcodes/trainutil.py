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
from utils import calculate_mof, dotdict


def validate(args, model, testloader, device, best_val_acc=None, best_combined_score=None, save=True):
    model.eval()
    print("Calculating Validation Data Accuracy")
    correct = 0.0
    total = 0.0
    vidcount = 0
    all_scores = []
    if args.postprocess is not None:
        args.postprocess.start()

    for i, item in enumerate(testloader):
        with torch.no_grad():
            item_0 = item[0].to(device)
            item_1 = item[1].to(device)
            item_2 = item[2].to(device)

            if args.sample_rate > 1:
                item_0 = item_0[:,:,::args.sample_rate]
                item_1 = torch.ceil(1.0 * item_1 / args.sample_rate).type(torch.int).to(device)
                item_2 = item_2[:,::args.sample_rate]

            src_mask = torch.arange(item_2.shape[1], device=item_2.device)[None, :] < item_1[:, None]
            src_mask_mse = src_mask.unsqueeze(1).to(torch.float32).to(device)
            middle_pred, predictions = model(item_0, src_mask_mse)
            pred = torch.argmax(predictions[-1], dim=1)
            
            if args.sample_rate > 1:
                item_1 = item[1].to(device)
                item_2 = item[2].to(device)
                src_mask = torch.arange(item_2.shape[1], device=item_2.device)[None, :] < item_1[:, None]
                pred = F.interpolate(pred.unsqueeze(1).float(), size=item_2.shape[1], mode='nearest').squeeze(1).long()

            if args.postprocess is None:
                correct += float(torch.sum((pred == item_2) * src_mask).item())
                total += float(torch.sum(src_mask).item())
                for p, l, c in zip(pred, item_2, item_1):
                    all_scores.append(get_all_scores(p[:c].detach().cpu().numpy(),
                                                     l[:c].detach().cpu().numpy(), args.back_gd_list))
            else:
                args.postprocess.forward(predictions[-1], item[4], item_2, item_1)

    if args.postprocess is None:
        final_scores = np.mean(np.array(all_scores), axis=0)
        val_acc = correct * 100.0 / total
    else:
        dumpdir = os.path.join(args.output_dir, 'predictions')
        if not os.path.exists(dumpdir):
            os.mkdir(dumpdir)
        args.postprocess.dump_to_directory(dumpdir)
        edit, acc, final_scores = calculate_mof(args.ground_truth_files_dir, dumpdir, args.back_gd_list)
        final_scores = np.concatenate([final_scores, [edit]])
        val_acc = acc
        
    if best_val_acc is not None and val_acc > best_val_acc:
        best_val_acc = val_acc
        if save:
            torch.save(model.state_dict(), args.output_dir + "ms-tcn-emmax-best-model.wt")
    combined_score = (final_scores[3] + final_scores[2] + val_acc)/3.
    if best_combined_score is not None and combined_score > best_combined_score:
        best_combined_score = combined_score
        if save:
            torch.save(model.state_dict(), args.output_dir + "ms-tcn-emmax-combbest-model.wt")
    if save:
        torch.save(model.state_dict(), args.output_dir + "ms-tcn-emmax-last-model.wt")
    print(f"Validation:: Probability Accuracy {val_acc}")
    print(f"Other scores:: Edit {final_scores[3]}, F1@[10:25:50] {final_scores[:3]}")
    _ = model.train()

    if not args.notrain:
        wandb.log({'Validation Accuracy': val_acc,
                   'Edit': final_scores[3],
                   'F1@10': final_scores[0],
                   'F1@25': final_scores[1],
                   'F1@50': final_scores[2],
                   'Best Validation Accuracy': best_val_acc,
                   'Combined best validation score': best_combined_score
                  })

    return val_acc, best_val_acc, best_combined_score, final_scores


def final_validation(args, model, testloader, device):
    best_model_path = args.output_dir + "ms-tcn-emmax-best-model.wt"
    combbest_model_path = args.output_dir + "ms-tcn-emmax-combbest-model.wt"
    last_model_path = args.output_dir + "ms-tcn-emmax-last-model.wt"
    init_model_path = args.output_dir + f"ms-tcn-initial-{args.init_epoch}-epochs.wt"

    model.load_state_dict(torch.load(init_model_path))
    print(f'============= Evaluating init model =============')
    init_val_acc, _, _, score0 = validate(args, model, testloader, device, save=False)
    print(f'Excel friendly init result: {score0[0]},{score0[1]},{score0[2]},{score0[3]},{init_val_acc}')

    model.load_state_dict(torch.load(best_model_path))
    print(f'============= Evaluating best model =============')
    best_val_acc, _, _, score1 = validate(args, model, testloader, device, save=False)
    print(f'Excel friendly best result: {score1[0]},{score1[1]},{score1[2]},{score1[3]},{best_val_acc}')

    model.load_state_dict(torch.load(combbest_model_path))
    print(f'============= Evaluating combined-best model =============')
    combbest_val_acc, _, _, score2 = validate(args, model, testloader, device, save=False)
    print(f'Excel friendly combined-best result: {score2[0]},{score2[1]},{score2[2]},{score2[3]},{combbest_val_acc}')

    model.load_state_dict(torch.load(last_model_path))
    print(f'============= Evaluating last model =============')
    last_val_acc, _, _, score3 = validate(args, model, testloader, device, save=False)
    print(f'Excel friendly last result: {score3[0]},{score3[1]},{score3[2]},{score3[3]},{last_val_acc}')
    return


def train(device, model, train_loader_list, criterion_list, optimizer, args, test_loader, get_single_random,
          perform_expectation, confidence_loss=None):
    best_val_acc, combbest_val_acc = 0, 0
    for epoch in range(args.start_epoch, args.epochs):
        print("Starting Training")
        model.train()
        for i, item in enumerate(train_loader_list[0]):
            item_0 = item[0].to(device)  # features
            item_1 = item[1].to(device)  # count
            item_2 = item[2].to(device)  # target
            weights = item[5].to(device)  # posterior weight
            video_ids = item[4]
            src_mask = torch.arange(item_2.shape[1], device=item_2.device)[None, :] < item_1[:, None]
            src_mask_mse = src_mask.unsqueeze(1).to(torch.float32).to(device)
            optimizer.zero_grad()

            middle_pred, predictions = model(item_0, src_mask_mse)
            boundary_target_tensor = get_single_random(video_ids, item_2.shape[1], item_2.device)

            loss = 0
            for p in predictions:
                if epoch <= args.init_epoch:
                    loss += criterion_list[0](p, boundary_target_tensor)
                    loss += 0.15 * torch.mean(torch.clamp(criterion_list[1](F.log_softmax(p[:, :, 1:], dim=1),
                                                                        F.log_softmax(p.detach()[:, :, :-1], dim=1)),
                                                          min=0,
                                                          max=16) * src_mask_mse[:, :, 1:])
                else:
                    prob = torch.softmax(p, dim=1)
                    prob = prob.permute(0, 2, 1)
                    total_count = torch.sum(src_mask)
                    weighted_loss_sum = -torch.sum(torch.sum(torch.log(prob + 1e-8) * weights, dim=-1) * src_mask)
                    loss += weighted_loss_sum / total_count
                    if args.use_mse:
                        loss += 0.15 * torch.mean(torch.clamp(criterion_list[1](F.log_softmax(p[:, :, 1:], dim=1),
                                                                            F.log_softmax(p.detach()[:, :, :-1], dim=1)),
                                                              min=0,
                                                              max=16) * src_mask_mse[:, :, 1:])
                    if confidence_loss is not None:
                        loss += 0.6 * confidence_loss(prob, video_ids, item_1)

            loss.backward()
            optimizer.step()
            if not args.notrain:
                wandb.log({'train loss': loss,
                           'epoch': epoch
                           })

            if (i + 1) % 20 == 0:
                print(f'Epoch {epoch + 1}: Iteration {i + 1} with loss {loss.item()}')

        if epoch == args.init_epoch:
            torch.save(model.state_dict(), args.output_dir + f"ms-tcn-initial-{epoch}-epochs.wt")

        print(f'Epoch {epoch + 1} finished, starting validation')
        val_acc, best_val_acc, combbest_val_acc, final_scores = validate(args, model, test_loader, device,
                                                                         best_val_acc, combbest_val_acc)
        
        if (epoch == args.init_epoch) or ((epoch > args.init_epoch) and (epoch % args.expectation_cal_gap == 0)):
            perform_expectation(model, train_loader_list[1], device)

        with open(args.output_dir + "run_summary.txt", "a+") as fp:
            fp.write(f"{final_scores[0]:.2f}, {final_scores[1]:.2f}, {final_scores[2]:.2f}, {final_scores[3]:.2f}, "
                     f"{val_acc:.2f}, {best_val_acc:.2f}\n")

