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
from concurrent.futures import ProcessPoolExecutor
import queue
import copy
import torch.nn as nn
import torch.nn.functional as F
from utils import worker_init_fn, get_pdbs, loader_pdb, build_training_clusters, PDB_dataset, StructureDataset, StructureLoader
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

    # Paths for l-chiral and d-chiral data
    data_path_l_chiral = args.path_for_l_chiral_data
    data_path_d_chiral = args.path_for_d_chiral_data

    # Load l-chiral dataset
    params_l_chiral = {
        "LIST": f"{data_path_l_chiral}/list.csv", 
        "VAL": f"{data_path_l_chiral}/valid_clusters.txt",
        "TEST": f"{data_path_l_chiral}/test_clusters.txt",
        "DIR": f"{data_path_l_chiral}",
        "DATCUT": "2030-Jan-01",
        "RESCUT": args.rescut,
        "HOMO": 0.70
    }

    # Load d-chiral dataset
    params_d_chiral = {
        "LIST": f"{data_path_d_chiral}/list.csv", 
        "VAL": f"{data_path_d_chiral}/valid_clusters.txt",
        "TEST": f"{data_path_d_chiral}/test_clusters.txt",
        "DIR": f"{data_path_d_chiral}",
        "DATCUT": "2030-Jan-01",
        "RESCUT": args.rescut,
        "HOMO": 0.70
    }

    LOAD_PARAM = {'batch_size': 1,
                  'shuffle': True,
                  'pin_memory':False,
                  'num_workers': 4}
    
    if args.debug:
        args.num_examples_per_epoch = 50
        args.max_protein_length = 1000
        args.batch_size = 1000

    # Build training clusters
    train_l, valid_l, _ = build_training_clusters(params_l_chiral, args.debug)
    train_d, valid_d, _ = build_training_clusters(params_d_chiral, args.debug)

    train_set_l = PDB_dataset(list(train_l.keys()), loader_pdb, train_l, params_l_chiral)
    train_set_d = PDB_dataset(list(train_d.keys()), loader_pdb, train_d, params_d_chiral)
    train_loader_l = DataLoader(train_set_l, worker_init_fn=worker_init_fn, **args.load_params)
    train_loader_d = DataLoader(train_set_d, worker_init_fn=worker_init_fn, **args.load_params)
    valid_set_l = PDB_dataset(list(valid_l.keys()), loader_pdb, valid_l, params_l_chiral)
    valid_set_d = PDB_dataset(list(valid_d.keys()), loader_pdb, valid_d, params_d_chiral)
    valid_loader_l = DataLoader(valid_set_l, worker_init_fn=worker_init_fn, **args.load_params)
    valid_loader_d = DataLoader(valid_set_d, worker_init_fn=worker_init_fn, **args.load_params)

    model = NewComboChiral(edge_features=args.hidden_dim, 
                           hidden_dim=args.hidden_dim, 
                           num_encoder_layers=args.num_encoder_layers, 
                           num_decoder_layers=args.num_encoder_layers, 
                           dropout=args.dropout, 
                           k_neighbors=args.num_neighbors, 
                           augment_eps=args.backbone_noise, 
                           input_size=21, 
                           out1=2)      
    model.to(device)

    if PATH:
        checkpoint = torch.load(PATH)
        total_step = checkpoint['step'] #write total_step from the checkpoint
        epoch = checkpoint['epoch'] #write epoch from the checkpoint
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        total_step = 0
        epoch = 0

    optimizer = get_std_opt(model.chiraldetermine.parameters(), args.hidden_dim, total_step)
    criterion = nn.BCELoss() # Initialize binary loss function classification

    if PATH:
        optimizer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    with ProcessPoolExecutor(max_workers=12) as executor:
        # Queues for l-chiral dataset
        q_l = queue.Queue(maxsize=3)
        p_l = queue.Queue(maxsize=3)
    
        # Queues for d-chiral dataset
        q_d = queue.Queue(maxsize=3)
        p_d = queue.Queue(maxsize=3)
        
    # Load l-chiral data
    for i in range(3):
        q_l.put_nowait(executor.submit(get_pdbs, train_loader_l, 1, args.max_protein_length, args.num_examples_per_epoch))
        p_l.put_nowait(executor.submit(get_pdbs, valid_loader_l, 1, args.max_protein_length, args.num_examples_per_epoch))
    
    pdb_dict_train_l = q_l.get().result()
    pdb_dict_valid_l = p_l.get().result()
    
    dataset_train_l = StructureDataset(pdb_dict_train_l, truncate=None, max_length=args.max_protein_length)
    dataset_valid_l = StructureDataset(pdb_dict_valid_l, truncate=None, max_length=args.max_protein_length)
    
    loader_train_l = StructureLoader(dataset_train_l, batch_size=args.batch_size)
    loader_valid_l = StructureLoader(dataset_valid_l, batch_size=args.batch_size)

    # Load d-chiral data
    for i in range(3):
        q_d.put_nowait(executor.submit(get_pdbs, train_loader_d, 1, args.max_protein_length, args.num_examples_per_epoch))
        p_d.put_nowait(executor.submit(get_pdbs, valid_loader_d, 1, args.max_protein_length, args.num_examples_per_epoch))
    
    pdb_dict_train_d = q_d.get().result()
    pdb_dict_valid_d = p_d.get().result()
    
    dataset_train_d = StructureDataset(pdb_dict_train_d, truncate=None, max_length=args.max_protein_length)
    dataset_valid_d = StructureDataset(pdb_dict_valid_d, truncate=None, max_length=args.max_protein_length)
    
    loader_train_d = StructureLoader(dataset_train_d, batch_size=args.batch_size)
    loader_valid_d = StructureLoader(dataset_valid_d, batch_size=args.batch_size)
        
    reload_c = 0 
    for e in range(args.num_epochs):
        t0 = time.time()
        e = epoch + e
        model.train()
        
        # Intialize metrics for training
        train_sum_l, train_sum_d = 0.0, 0.0
        train_acc_l, train_acc_d = 0, 0
        train_total_samples_l, train_total_samples_d = 0, 0
        batch_train_steps_l, batch_train_steps_d = 0, 0

        if e % args.reload_data_every_n_epochs == 0:
            if reload_c != 0:
                # Reload l-chiral dataset
                pdb_dict_train_l = q_l.get().result()
                dataset_train_l = StructureDataset(pdb_dict_train_l, truncate=None, max_length=args.max_protein_length)
                loader_train_l = StructureLoader(dataset_train_l, batch_size=args.batch_size)
                
                pdb_dict_valid_l = p_l.get().result()
                dataset_valid_l = StructureDataset(pdb_dict_valid_l, truncate=None, max_length=args.max_protein_length)
                loader_valid_l = StructureLoader(dataset_valid_l, batch_size=args.batch_size)
                
                q_l.put_nowait(executor.submit(get_pdbs, train_loader_l, 1, args.max_protein_length, args.num_examples_per_epoch))
                p_l.put_nowait(executor.submit(get_pdbs, valid_loader_l, 1, args.max_protein_length, args.num_examples_per_epoch))
                
                # Reload d-chiral dataset
                pdb_dict_train_d = q_d.get().result()
                dataset_train_d = StructureDataset(pdb_dict_train_d, truncate=None, max_length=args.max_protein_length)
                loader_train_d = StructureLoader(dataset_train_d, batch_size=args.batch_size)
                
                pdb_dict_valid_d = p_d.get().result()
                dataset_valid_d = StructureDataset(pdb_dict_valid_d, truncate=None, max_length=args.max_protein_length)
                loader_valid_d = StructureLoader(dataset_valid_d, batch_size=args.batch_size)
                
                q_d.put_nowait(executor.submit(get_pdbs, train_loader_d, 1, args.max_protein_length, args.num_examples_per_epoch))
                p_d.put_nowait(executor.submit(get_pdbs, valid_loader_d, 1, args.max_protein_length, args.num_examples_per_epoch))
                
            reload_c += 1

        # Training loop for l-chiral dataset
        for batch in loader_train_l:
            X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = featurize(batch, device)
            optimizer.zero_grad()

            # Model forward pass 
            output = model(X, S, mask, chain_M, residue_idx, chain_encoding_all) 

            # Initialize targets for l-chiral (class 1)
            batch_size = output.size(0)
            sequence_length = output.size(1)
            targets = torch.zeros(batch_size, sequence_length, 2, device=output.device)
            targets[:, :, 1] = 1.0  # l-chiral class (class 1)
            
            # Compute loss using BCELoss
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()

            # Get binary class predictions
            predictions_binary = torch.argmax(output, -1)
            targets_binary = torch.argmax(targets, -1)

            # Update metrics
            true_false = (predictions_binary == targets_binary).float()
            train_sum_l += torch.sum(loss).cpu().data.numpy()
            train_acc_l += torch.sum(true_false).cpu().data.numpy()
            train_total_samples_l += predictions_binary.numel()
            batch_train_steps_l += 1

        # Training loop for d-chiral dataset
        for batch in loader_train_d:
            X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = featurize(batch, device)
            optimizer.zero_grad()

            # Model forward pass 
            output = model(X, S, mask, chain_M, residue_idx, chain_encoding_all) 

            # Initialize targets for d-chiral (class 0)
            batch_size = output.size(0)
            sequence_length = output.size(1)
            targets = torch.zeros(batch_size, sequence_length, 2, device=output.device)
            targets[:, :, 0] = 1.0  # d-chiral class (class 0)

            # Compute loss using BCELoss
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()

            # Get binary class predictions
            predictions_binary = torch.argmax(output, -1)
            targets_binary = torch.argmax(targets, -1)

            # Update metrics
            true_false = (predictions_binary == targets_binary).float()
            train_sum_d += torch.sum(loss).cpu().data.numpy()
            train_acc_d += torch.sum(true_false).cpu().data.numpy()
            train_total_samples_d += predictions_binary.numel()
            batch_train_steps_d += 1

            model.eval()
            with torch.no_grad():
            
                # Intialize metrics for validation
                validation_sum = 0.0
                validation_acc = 0
                validation_total_samples = 0 
                batch_valid_steps = 0

                for i, batch in enumerate(loader_valid):
                    X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = featurize(batch, device)
                    
                    # Model forward pass
                    log_probs = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)

                    # Initialize targets for l-chiral (class 1)
                    batch_size = log_probs.size(0)
                    sequence_length = log_probs.size(1)

                    # Create a tensor of zeros 
                    targets = torch.zeros(batch_size, sequence_length, 2, device=log_probs.device)

                    # Set the second channel to 1 for l-chiral (class 1)
                    targets[:, :, 1] = 1.0 # Set class 1 (l-chiral) to 1

                    # Compute loss
                    loss = criterion(log_probs, targets)

                    # Get binary class predictions
                    predictions_binary = torch.argmax(log_probs, -1)

                    # Target labels are all ones since we are dealing with l-chiral data exclusively (FOR NOW)
                    # target_binary = torch.ones_like(predictions_binary, device=output.device)  # Shape: [batch_size, sequence_length]
                    target_binary = torch.argmax(targets, -1)  # Shape: [batch_size, sequence_length]

                    # Compare predictions with target labels
                    true_false = (predictions_binary  == target_binary).float()

                    # Update validation metrics
                    validation_sum += torch.sum(loss).cpu().data.numpy()
                    validation_acc += torch.sum(true_false).cpu().data.numpy()
                    validation_total_samples += predictions_binary.numel()
                    batch_valid_steps += 1
            
            train_loss = train_sum / batch_train_steps
            train_accuracy = train_acc / train_total_samples
            validation_loss = validation_sum / batch_valid_steps
            validation_accuracy = validation_acc / validation_total_samples
            
            train_loss_formatted = np.format_float_positional(np.float32(train_loss), unique=False, precision=6)
            train_accuracy_formatted = np.format_float_positional(np.float32(validation_loss), unique=False, precision=6)
            validation_loss_formatted = np.format_float_positional(np.float32(train_accuracy), unique=False, precision=6)
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
                        'optimizer_state_dict': optimizer.optimizer.state_dict(),
                        }, checkpoint_filename_last)

            if (e+1) % args.save_model_every_n_epochs == 0:
                checkpoint_filename = base_folder+'model_weights/epoch{}_step{}.pt'.format(e+1, total_step)
                torch.save({
                        'epoch': e+1,
                        'step': total_step,
                        'num_edges' : args.num_neighbors,
                        'noise_level': args.backbone_noise, 
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.optimizer.state_dict(),
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
 
    args = argparser.parse_args()    
    main(args)
