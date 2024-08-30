#!/usr/bin/env python

###########
# IMPORTS #
###########

import argparse
import os
import json
import time
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, ConcatDataset
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mpi
import queue
import copy
import torch.nn as nn
import torch.nn.functional as F
from utils import worker_init_fn, get_pdbs, loader_pdb, build_training_clusters, PDB_dataset, StructureDataset, StructureLoader, chiral_loader, exec_init_worker
from model_utils import featurize, get_std_opt
from new_combo_module import NewComboChiral


########
# MAIN #
########

def main(args):
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu") # Moves models and tensors to the appropriate device for training (code compatible with both)

    base_folder = time.strftime(args.path_for_outputs, time.localtime())
    if base_folder[-1] != '/':
        base_folder += '/'
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    subfolders = ['model_weights']
    for subfolder in subfolders:
        if not os.path.exists(base_folder + subfolder):
            os.makedirs(base_folder + subfolder)

    PATH = args.previous_checkpoint

    logfile = base_folder + 'log.txt'
    if not PATH:
        with open(logfile, 'w') as f:
            f.write('Epoch\tTrain\tValidation\n')

    data_path = args.path_for_training_data
    params = {
        "LIST"    : f"{data_path}/list.csv", 
        "VAL"     : f"{data_path}/valid_clusters.txt",
        "TEST"    : f"{data_path}/test_clusters.txt",
        "DIR"     : f"{data_path}",
        "DATCUT"  : "2030-Jan-01",
        "RESCUT"  : args.rescut, #resolution cutoff for PDBs
        "HOMO"    : 0.70, #min seq.id. to detect homo chains
        "CHIRAL"  : f"{data_path}/chiral_out.csv",
    }

    LOAD_PARAM = {'batch_size': 1,
                  'shuffle': True,
                  'pin_memory':False,
                  'num_workers': 4}
    if args.debug:
        args.num_examples_per_epoch = 50
        args.max_protein_length = 1000
        args.batch_size = 1000

    # Add in a chiral portion to the dataloader
    chiral_dict = chiral_loader(params)

    train, valid, test = build_training_clusters(params, args.debug)

    train_set = PDB_dataset(list(train.keys()), loader_pdb, train, params)
    train_loader = torch.utils.data.DataLoader(train_set, worker_init_fn=worker_init_fn, **LOAD_PARAM)
    valid_set = PDB_dataset(list(valid.keys()), loader_pdb, valid, params)
    valid_loader = torch.utils.data.DataLoader(valid_set, worker_init_fn=worker_init_fn, **LOAD_PARAM)
    
    model = NewComboChiral(edge_features=args.hidden_dim, 
                           hidden_dim=args.hidden_dim, 
                           num_encoder_layers=args.num_encoder_layers, 
                           num_decoder_layers=args.num_encoder_layers, 
                           dropout=args.dropout, 
                           k_neighbors=args.num_neighbors, 
                           augment_eps=args.backbone_noise, 
                           input_size=21, 
                           out1=1)      
    model.to(device)

    vanilla_train_weights = torch.load("/mnt/c/Users/thefr/Downloads/epoch200_step564.pt", map_location=device)
    dchiral_train_weights = torch.load("/mnt/c/Users/thefr/Downloads/epoch200_step606.pt", map_location=device)

    # Extract model state dictionaries
    vanilla_model_state_dict = vanilla_train_weights['model_state_dict']
    dchiral_model_state_dict = dchiral_train_weights['model_state_dict']

    # Load the state dictionaries
    model.vanilla.load_state_dict(vanilla_model_state_dict)
    model.dchiral.load_state_dict(dchiral_model_state_dict)

    # Freeze the weights in the vanilla and d-chiral models 
    for param in list(model.vanilla.parameters()) + list(model.dchiral.parameters()):
        param.requires_grad = False

    # Set models to evaluation mode
    # model.vanilla.eval()
    # model.dchiral.eval()

    if PATH:
        checkpoint = torch.load(PATH)
        total_step = checkpoint['step'] #write total_step from the checkpoint
        epoch = checkpoint['epoch'] #write epoch from the checkpoint
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        total_step = 0
        epoch = 0

    # optimizer = get_std_opt(model.chiraldetermine.parameters(), args.learning_rate, total_step)
    optimizer = torch.optim.Adam(model.chiraldetermine.parameters(), args.learning_rate, betas=(0.9, 0.999), eps=1e-08)
    criterion = nn.BCELoss(reduction='none') # Initialize binary loss function classification

    if PATH:
        optimizer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print("STARTING PROCESS POOL")
    print('----------------------')
    with ProcessPoolExecutor(max_workers=12, initializer=exec_init_worker, initargs=(chiral_dict,)) as executor:
        q = queue.Queue(maxsize=3)
        p = queue.Queue(maxsize=3)
        for _ in range(3):
            q.put_nowait(executor.submit(get_pdbs, train_loader, 1, args.max_protein_length, args.num_examples_per_epoch))
            p.put_nowait(executor.submit(get_pdbs, valid_loader, 1, args.max_protein_length, args.num_examples_per_epoch))
        pdb_dict_train = q.get().result()
        pdb_dict_valid = p.get().result()

        dataset_train = StructureDataset(pdb_dict_train, truncate=None, max_length=args.max_protein_length) 
        dataset_valid = StructureDataset(pdb_dict_valid, truncate=None, max_length=args.max_protein_length)
        
        loader_train = StructureLoader(dataset_train, batch_size=args.batch_size)
        loader_valid = StructureLoader(dataset_valid, batch_size=args.batch_size)
        
        reload_c = 0 
        for e in range(args.num_epochs):
            t0 = time.time()
            e = epoch + e
            model.train()
            
            # Intialize metrics for training
            train_sum = 0.0
            train_acc = 0
            train_total_samples = 0 
            batch_train_steps = 0

            if e % args.reload_data_every_n_epochs == 0:
                if reload_c != 0:
                    pdb_dict_train = q.get().result()
                    dataset_train = StructureDataset(pdb_dict_train, truncate=None, max_length=args.max_protein_length)
                    loader_train = StructureLoader(dataset_train, batch_size=args.batch_size)
                    pdb_dict_valid = p.get().result()
                    dataset_valid = StructureDataset(pdb_dict_valid, truncate=None, max_length=args.max_protein_length)
                    loader_valid = StructureLoader(dataset_valid, batch_size=args.batch_size)
                    q.put_nowait(executor.submit(get_pdbs, train_loader, 1, args.max_protein_length, args.num_examples_per_epoch))
                    p.put_nowait(executor.submit(get_pdbs, valid_loader, 1, args.max_protein_length, args.num_examples_per_epoch))
                reload_c += 1
            for batch in loader_train:
                X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = featurize(batch, device)
                targets = torch.zeros_like(S, dtype=torch.float32, device=device)
                mask_targets = torch.zeros_like(S, dtype=torch.int8, device=device)
                for n, n_tar in enumerate(batch):
                    chiral = n_tar["chiral"]
                    targets[n, :chiral.size(-1)] = chiral
                    mask_targets[n, :chiral.size(-1)] = 1
                optimizer.zero_grad()

                # Model forward pass 
                output = model(X, S, mask, chain_M, residue_idx, chain_encoding_all) 

                # Initialize targets for l-chiral (class 1)
                batch_size = output.size(0)
                sequence_length = output.size(1)

                # Create a tensor of zeros
                # targets = torch.zeros(batch_size, sequence_length, 2, device=output.device)

                # Set the second channel to 0 for l-chiral (class 0)
                # targets[:, :, 1] = 1.0 # Set class 1 (l-chiral) to 1
                    
                # Compute loss using BCELoss
                # un_norm_loss = criterion(output, targets) # Calculates the binary cross-entropy loss between model output and targets 
                un_norm_loss = criterion(output.squeeze(-1), targets.float()) # Calculates the binary cross-entropy loss between model output and targets 
                
                loss = (un_norm_loss * mask_targets).sum() / mask_targets.sum()
                loss.backward() # Computes the gradients of the loss
                optimizer.step() # Updates the model parameters based on the gradients

                # Get binary class predictions
                # predictions_binary = torch.argmax(output, -1)
                predictions_binary = output.squeeze(-1).round()
                # print('NOT ROUNDED:', output)
                # print('ROUNDED:', predictions_binary)

                # Convert targets to binary labels
                # targets_binary = torch.argmax(targets, -1)


                # Debugging statements
                # print(f"Output shape: {output.shape}")
                # print(f"Targets shape: {targets.shape}")
                # print(f"Targets: {targets}")
                # print(f"Predictions shape: {predictions_binary.shape}")  
                # print(f"Predictions: {predictions_binary}")
                # print(f"Target labels shape: {targets_binary.shape}") 
                # print(f"Target labels: {targets_binary}")

                # Compare predictions with target labels
                true_false = (predictions_binary == targets).float()

                # Updating training metrics
                train_sum += torch.sum(loss).cpu().data.numpy()
                train_acc += torch.sum(true_false).cpu().data.numpy()
                train_total_samples += mask_targets.sum().cpu().data.numpy()
                batch_train_steps += 1
                total_step += 1

            model.eval()
            with torch.no_grad():
            
                # Intialize metrics for validation
                validation_sum = 0.0
                validation_acc = 0
                validation_total_samples = 0 
                batch_valid_steps = 0

                for i, batch in enumerate(loader_valid):
                    X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = featurize(batch, device)
                    targets = torch.zeros_like(S, dtype=torch.float32, device=device)
                    mask_targets = torch.zeros_like(S, dtype=torch.int8, device=device)
                    for n, n_tar in enumerate(batch):
                        chiral = n_tar["chiral"]
                        targets[n, :chiral.size(-1)] = chiral
                        mask_targets[n, :chiral.size(-1)] = 1
                    
                    # Model forward pass
                    log_probs = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)

                    # Initialize targets for l-chiral (class 1)
                    batch_size = log_probs.size(0)
                    sequence_length = log_probs.size(1)

                    # Create a tensor of zeros 
                    # targets = torch.zeros(batch_size, sequence_length, 2, device=log_probs.device)

                    # Set the second channel to 1 for l-chiral (class 1)
                    # targets[:, :, 1] = 1.0 # Set class 1 (l-chiral) to 1

                    # Compute loss
                    loss = criterion(log_probs.squeeze(-1), targets.float())
                    norm_loss = (loss * mask_targets).sum() / mask_targets.sum()

                    # Get binary class predictions
                    # predictions_binary = torch.argmax(log_probs, -1)
                    predictions_binary = log_probs.squeeze(-1).round()

                    # Target labels are all ones since we are dealing with l-chiral data exclusively (FOR NOW)
                    # target_binary = torch.ones_like(predictions_binary, device=output.device)  # Shape: [batch_size, sequence_length]
                    # target_binary = torch.argmax(targets, -1)  # Shape: [batch_size, sequence_length]

                    # Compare predictions with target labels
                    true_false = (predictions_binary  == targets).float()

                    # Update validation metrics
                    validation_sum += torch.sum(norm_loss).cpu().data.numpy()
                    validation_acc += torch.sum(true_false).cpu().data.numpy()
                    # validation_total_samples += predictions_binary.numel()
                    validation_total_samples += mask_targets.sum().cpu().data.numpy()
                    batch_valid_steps += 1
            
            train_loss = train_sum / batch_train_steps
            train_accuracy = train_acc / train_total_samples
            validation_loss = validation_sum / batch_valid_steps
            validation_accuracy = validation_acc / validation_total_samples
            
            train_loss_formatted = np.format_float_positional(np.float32(train_loss), unique=False, precision=6)
            train_accuracy_formatted = np.format_float_positional(np.float32(train_accuracy), unique=False, precision=6)
            validation_loss_formatted = np.format_float_positional(np.float32(validation_loss), unique=False, precision=6)
            validation_accuracy_formatted = np.format_float_positional(np.float32(validation_accuracy), unique=False, precision=6)
    
            t1 = time.time()
            dt = np.format_float_positional(np.float32(t1-t0), unique=False, precision=1) 
            with open(logfile, 'a') as f:
                f.write(f'epoch: {e+1}, step: {total_step}, time: {dt}, train: {train_loss_formatted}, valid: {validation_loss_formatted}, train_acc: {train_accuracy_formatted}, valid_acc: {validation_accuracy_formatted}\n')
            print(f'epoch: {e+1}, step: {total_step}, time: {dt}, train: {train_loss_formatted}, valid: {validation_loss_formatted}, train_acc: {train_accuracy_formatted}, valid_acc: {validation_accuracy_formatted}')
            
            checkpoint_filename_last = base_folder+'model_weights/epoch_last.pt'.format(e+1, total_step)
            torch.save({
                        'epoch': e+1,
                        'step': total_step,
                        'num_edges' : args.num_neighbors,
                        'noise_level': args.backbone_noise,
                        'model_state_dict': model.state_dict(),
                        # 'optimizer_state_dict': optimizer.optimizer.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, checkpoint_filename_last)

            if (e+1) % args.save_model_every_n_epochs == 0:
                checkpoint_filename = base_folder+'model_weights/epoch{}_step{}.pt'.format(e+1, total_step)
                torch.save({
                        'epoch': e+1,
                        'step': total_step,
                        'num_edges' : args.num_neighbors,
                        'noise_level': args.backbone_noise, 
                        'model_state_dict': model.state_dict(),
                        # 'optimizer_state_dict': optimizer.optimizer.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, checkpoint_filename)

############
# ARGPARSE #
############

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argparser.add_argument("--path_for_training_data", type=str, default="/projects/parisahlab/lmjone/internship/ProteinMPNN-PH/training/datasets/pdb_2021aug02_sample", help="path for loading training data") 
    argparser.add_argument("--path_for_outputs", type=str, default="./exp_020", help="path for logs and model weights")
    argparser.add_argument("--previous_checkpoint", type=str, default="", help="path for previous model weights, e.g. file.pt")
    argparser.add_argument("--num_epochs", type=int, default=200, help="number of epochs to train for")
    argparser.add_argument("--save_model_every_n_epochs", type=int, default=10, help="save model weights every n epochs")
    argparser.add_argument("--reload_data_every_n_epochs", type=int, default=2, help="reload training data every n epochs")
    argparser.add_argument("--num_examples_per_epoch", type=int, default=1000000, help="number of training example to load for one epoch")
    argparser.add_argument("--batch_size", type=int, default=10000, help="number of tokens for one batch")
    argparser.add_argument("--max_protein_length", type=int, default=10000, help="maximum length of the protein complext")
    argparser.add_argument("--hidden_dim", type=int, default=128, help="hidden model dimension")
    argparser.add_argument("--num_encoder_layers", type=int, default=3, help="number of encoder layers") 
    argparser.add_argument("--num_decoder_layers", type=int, default=3, help="number of decoder layers")
    argparser.add_argument("--num_neighbors", type=int, default=48, help="number of neighbors for the sparse graph")   
    argparser.add_argument("--dropout", type=float, default=0.1, help="dropout level; 0.0 means no dropout")
    argparser.add_argument("--backbone_noise", type=float, default=0.2, help="amount of noise added to backbone during training")   
    argparser.add_argument("--rescut", type=float, default=3.5, help="PDB resolution cutoff")
    argparser.add_argument("--debug", type=bool, default=False, help="minimal data loading for debugging")
    argparser.add_argument("--gradient_norm", type=float, default=-1.0, help="clip gradient norm, set to negative to omit clipping")
    argparser.add_argument("--mixed_precision", type=bool, default=True, help="train with mixed precision")
    argparser.add_argument("--learning_rate", type=float, default=1e-3, help="learning rate alpha parameters optimizer")
 
    args = argparser.parse_args()    
    main(args)
