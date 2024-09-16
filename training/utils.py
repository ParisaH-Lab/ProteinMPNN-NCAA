import torch
import psutil
from torch.utils.data import DataLoader
import csv
from dateutil import parser
import numpy as np
import time
import random
import os
import re

class StructureDataset():
    def __init__(self, pdb_dict_list, verbose=True, truncate=None, max_length=100,
        alphabet='ACDEFGHIKLMNPQRSTVWYX'):
        alphabet_set = set([a for a in alphabet])
        discard_count = {
            'bad_chars': 0,
            'too_long': 0,
            'bad_seq_length': 0
        }

        self.data = []

        start = time.time()
        for i, entry in enumerate(pdb_dict_list):
            seq = entry['seq'].upper()
            name = entry['name']

            bad_chars = set([s for s in seq]).difference(alphabet_set)
            if len(bad_chars) == 0:
                if len(entry['seq']) <= max_length:
                    self.data.append(entry)
                else:
                    discard_count['too_long'] += 1
            else:
                #print(name, bad_chars, entry['seq'])
                discard_count['bad_chars'] += 1

            # Truncate early
            if truncate is not None and len(self.data) == truncate:
                return

            if verbose and (i + 1) % 1000 == 0:
                elapsed = time.time() - start
                #print('{} entries ({} loaded) in {:.1f} s'.format(len(self.data), i+1, elapsed))

            #print('Discarded', discard_count)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class StructureLoader():
    def __init__(self, dataset, batch_size=100, shuffle=True,
        collate_fn=lambda x:x, drop_last=False):
        self.dataset = dataset
        self.size = len(dataset)
        self.lengths = [len(dataset[i]['seq']) for i in range(self.size)]
        self.batch_size = batch_size
        sorted_ix = np.argsort(self.lengths)

        # Cluster into batches of similar sizes
        clusters, batch = [], []
        batch_max = 0
        for ix in sorted_ix:
            size = self.lengths[ix]
            if size * (len(batch) + 1) <= self.batch_size:
                batch.append(ix)
                batch_max = size
            else:
                clusters.append(batch)
                batch, batch_max = [], 0
        if len(batch) > 0:
            clusters.append(batch)
        self.clusters = clusters

    def __len__(self):
        return len(self.clusters)

    def __iter__(self):
        np.random.shuffle(self.clusters)
        for b_idx in self.clusters:
            batch = [self.dataset[i] for i in b_idx]
            yield batch


def worker_init_fn(worker_id):
    np.random.seed()

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer, step):
        self.optimizer = optimizer
        self._step = step
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    @property
    def param_groups(self):
        """Return param_groups."""
        return self.optimizer.param_groups

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()

def get_std_opt(parameters, d_model, step):
    return NoamOpt(
        d_model, 2, 4000, torch.optim.Adam(parameters, lr=0, betas=(0.9, 0.98), eps=1e-9), step
    )




def get_pdbs(data_loader, repeat=1, max_length=10000, num_units=1000000):
    init_alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G','H', 'I', 'J','K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T','U', 'V','W','X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g','h', 'i', 'j','k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't','u', 'v','w','x', 'y', 'z']
    extra_alphabet = [str(item) for item in list(np.arange(300))]
    chain_alphabet = init_alphabet + extra_alphabet
    c = 0
    c1 = 0
    pdb_dict_list = []
    label_list = []
    t0 = time.time()
    for _ in range(repeat):
        for step,t in enumerate(data_loader):
            # print("------------------------------------")
            # print("T PRINT FIRST")
            # print(t["seq"])
            # print(t["xyz"].shape)
            t = {k:v[0] for k,v in t.items()}
            c1 += 1
            if 'label' in list(t):
                label_list.append(t["label"])
                my_dict = {}
                s = 0
                concat_seq = ''
                concat_chiral = ''
                concat_N = []
                concat_CA = []
                concat_C = []
                concat_O = []
                concat_mask = []
                coords_dict = {}
                mask_list = []
                visible_list = []

                if 'heterochiral' in t["label"]:
                    for idx in list(np.unique(t['idx'])):
                        letter = chain_alphabet[idx]
                        res = np.argwhere(t['idx']==idx)
                        # print("RES BEFORE IF AND ELSE:", res)
                        initial_sequence= "".join(list(np.array(list(t['seq']))[res][0,]))
                        my_dict['seq_chain_'+letter]= "".join(list(np.array(list(t['seq']))[res][0,]))
                        my_dict['chiral_chain_'+letter] = "".join(list(np.array(list(t['chiral']))[res][0,]))
                        concat_seq += my_dict['seq_chain_'+letter]
                        concat_chiral += my_dict["chiral_chain_"+letter]
                        if idx in t['masked']:
                            mask_list.append(letter)
                        else:
                            visible_list.append(letter)
                        coords_dict_chain = {}
                        all_atoms = np.array(t['xyz'][res,])[0,] #[L, 14, 3]
                        coords_dict_chain['N_chain_'+letter]=all_atoms[:,0,:].tolist()
                        coords_dict_chain['CA_chain_'+letter]=all_atoms[:,1,:].tolist()
                        coords_dict_chain['C_chain_'+letter]=all_atoms[:,2,:].tolist()
                        coords_dict_chain['O_chain_'+letter]=all_atoms[:,3,:].tolist()
                        my_dict['coords_chain_'+letter]=coords_dict_chain
                    my_dict['chiral'] = convert_tensor(concat_chiral)
                    my_dict['name']= t['label']
                    my_dict['masked_list']= mask_list
                    my_dict['visible_list']= visible_list
                    my_dict['num_of_chains'] = len(mask_list) + len(visible_list)
                    my_dict['seq'] = concat_seq
                    if len(concat_seq) <= max_length:
                        pdb_dict_list.append(my_dict)
                    if len(pdb_dict_list) >= num_units:
                        break



                elif len(list(np.unique(t['idx']))) < 352:
                    # print('T UNIQUE LIST :', list(np.unique(t['idx'])))
                    for idx in list(np.unique(t['idx'])):
                        letter = chain_alphabet[idx]
                        res = np.argwhere(t['idx']==idx)
                        # print("RES BEFORE IF AND ELSE:", res)
                        initial_sequence= "".join(list(np.array(list(t['seq']))[res][0,]))
                        # print('INIT SEQ:', initial_sequence)
                        if initial_sequence[-6:] == "HHHHHH":
                            res = res[:,:-6]
                        if initial_sequence[0:6] == "HHHHHH":
                            res = res[:,6:]
                        if initial_sequence[-7:-1] == "HHHHHH":
                           res = res[:,:-7]
                        if initial_sequence[-8:-2] == "HHHHHH":
                           res = res[:,:-8]
                        if initial_sequence[-9:-3] == "HHHHHH":
                           res = res[:,:-9]
                        if initial_sequence[-10:-4] == "HHHHHH":
                           res = res[:,:-10]
                        if initial_sequence[1:7] == "HHHHHH":
                            res = res[:,7:]
                        if initial_sequence[2:8] == "HHHHHH":
                            res = res[:,8:]
                        if initial_sequence[3:9] == "HHHHHH":
                            res = res[:,9:]
                        if initial_sequence[4:10] == "HHHHHH":
                            res = res[:,10:]
                        if res.shape[1] < 4:
                            pass
                        else:
                            # print('RES IN ELSE:', res)
                            # print("SEQ CHAIN", "".join(list(np.array(list(t['seq']))[res][0,])))
                            # print("SEQ CHAIN WITHOUT last [0]", list(np.array(list(t['seq']))[res]))
                            # print("CHIRAL CHAIN", "".join(list(np.array(list(t['chiral']))[res][0,])))

                            # print("END------------------------------------")
                            
                            my_dict['seq_chain_'+letter]= "".join(list(np.array(list(t['seq']))[res][0,]))
                            my_dict['chiral_chain_'+letter] = "".join(list(np.array(list(t['chiral']))[res][0,]))
                            concat_seq += my_dict['seq_chain_'+letter]
                            concat_chiral += my_dict["chiral_chain_"+letter]
                            if idx in t['masked']:
                                mask_list.append(letter)
                            else:
                                visible_list.append(letter)
                            coords_dict_chain = {}
                            all_atoms = np.array(t['xyz'][res,])[0,] #[L, 14, 3]
                            coords_dict_chain['N_chain_'+letter]=all_atoms[:,0,:].tolist()
                            coords_dict_chain['CA_chain_'+letter]=all_atoms[:,1,:].tolist()
                            coords_dict_chain['C_chain_'+letter]=all_atoms[:,2,:].tolist()
                            coords_dict_chain['O_chain_'+letter]=all_atoms[:,3,:].tolist()
                            my_dict['coords_chain_'+letter]=coords_dict_chain
                    my_dict['chiral'] = convert_tensor(concat_chiral)
                    my_dict['name']= t['label']
                    my_dict['masked_list']= mask_list
                    my_dict['visible_list']= visible_list
                    my_dict['num_of_chains'] = len(mask_list) + len(visible_list)
                    my_dict['seq'] = concat_seq
                    if len(concat_seq) <= max_length:
                        pdb_dict_list.append(my_dict)
                    if len(pdb_dict_list) >= num_units:
                        break
    # print(pdb_dict_list)
    # print("END OF GET PDBS (LABEL LIST):", label_list)
    return pdb_dict_list



class PDB_dataset(torch.utils.data.Dataset):
    def __init__(self, IDs, loader, train_dict, params):
        self.IDs = IDs
        self.train_dict = train_dict
        self.loader = loader
        self.params = params
        # self.chiral_dict = chiral_dict

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, index):
        ID = self.IDs[index]
        sel_idx = np.random.randint(0, len(self.train_dict[ID]))
        out = self.loader(self.train_dict[ID][sel_idx], self.params)
        return out

def exec_init_worker(dict_ref):
    """This is for the PrcoessPoolExecutor to share a large dict"""
    global chiral_dict
    chiral_dict = dict_ref

def chiral_loader(params: dict):
    """Generate a dictionary of values that are associated with a residues chirality 
    for a given pdb key

    PARAMS
    -------
    params: dict
        Contains the locations of all of the information that is needed

    RETURNS
    -------
    chiral_dict: dict
        PDB_NAME: Tensor (N, 2)
    """
    # load the file that is necessary
    chiral_file = open(params["CHIRAL"], 'r')
    # init chiral dict
    chiral_dict = dict()

    # loop through all of the entries within the file
    next(chiral_file)
    for line in chiral_file:
        # Split the line on the comma and remove the newline character
        line_split = line.strip("\n").split(",")
        # Grab the key (PDB NAME)
        key = line_split[0]
        # Grab the chiral sequence
        # values = torch.tensor(
        #     [
        #         0 if x=="D" else 1 for x in line_split[1]
        #     ], dtype=torch.int8,
        # )
        chiral_dict[key] = line_split[1]
    # close file
    chiral_file.close()

    return chiral_dict

def convert_tensor(input_str: str):
    """Convert Chiral Sequence to Tensor of 1 or 0

    input_str: str
        Chiral string
    """
    return torch.tensor(
        [
            0 if x=="D" else 1 for x in input_str
        ], dtype=torch.int8,
    )

def loader_pdb(item,params):#, chiral_info: torch.Tensor): # This means PDB_dataset needs this passed as well
    """
    New addition was chiral_dict which will be used to determine the chirality of
    each residue
    """

    pdbid,chid = item[0].split('_')
    if 'heterochiral' in pdbid:
        PREFIX = os.path.join(params["DIR"], "pt_loops", "pdb", "heterochiral", pdbid)
    elif 'mirror' in pdbid:
        PREFIX = os.path.join(params["DIR"], pdbid[1:3]+"mirror", pdbid) # This is pretending first two then mirror (eg. l3mirror)
    else:
        PREFIX = os.path.join(params["DIR"], "pt_loops", "pdb", "homochiral", pdbid[1:3], pdbid)
        # PREFIX = "%s/pdb/%s/%s"%(params['DIR'],pdbid[1:3],pdbid)
    # assert "/".join(PREFIX.split('/')[:-1]) == os.path.join(params['DIR'],"pt_loops", "pdb", "heterochiral", pdbid[1:3]), f"The Path is Different Then: {os.path.join(params['DIR'],'pt_loops', 'pdb', 'heterochiral', pdbid[1:3])}"
    
    # load metadata
    if not os.path.isfile(PREFIX+".pt"):
        # print("DOESNT EXIST:", PREFIX)
        return {'seq': np.zeros(5)}
        # return {'seq': ''}
        # pass
    meta = torch.load(PREFIX+".pt")
    asmb_ids = meta['asmb_ids']
    asmb_chains = meta['asmb_chains']
    chids = np.array(meta['chains'])

    # find candidate assemblies which contain chid chain
    asmb_candidates = set([a for a,b in zip(asmb_ids,asmb_chains)
                           if chid in b.split(',')])

    # if the chains is missing is missing from all the assemblies
    # then return this chain alone
    if len(asmb_candidates)<1:
        chain_total_name = f"{pdbid}_{chid}"
        chain = torch.load("%s_%s.pt"%(PREFIX,chid))
        L = len(chain['seq'])
        # assert(L == chiral_dict[chain_total_name].size(0))
        return {'seq'    : chain['seq'].upper(),
                'xyz'    : chain['xyz'],
                'idx'    : torch.zeros(L).int(),
                'masked' : torch.Tensor([0]).int(),
                'label'  : item[0],
                'chiral' : chiral_dict[chain_total_name],
                }

    # randomly pick one assembly from candidates
    asmb_i = random.sample(list(asmb_candidates), 1)

    # indices of selected transforms
    idx = np.where(np.array(asmb_ids)==asmb_i)[0]

    # load relevant chains
    chains = {c:torch.load("%s_%s.pt"%(PREFIX,c))
              for i in idx for c in asmb_chains[i]
              if c in meta['chains']}

    # generate assembly
    asmb = {}
    chiral_chain_dict = {}
    for k in idx:

        # pick k-th xform
        xform = meta['asmb_xform%d'%k]
        u = xform[:,:3,:3]
        r = xform[:,:3,3]

        # select chains which k-th xform should be applied to
        s1 = set(meta['chains'])
        s2 = set(asmb_chains[k].split(','))
        chains_k = s1&s2

        # transform selected chains 
        for c in chains_k:
            try:
                xyz = chains[c]['xyz']
                xyz_ru = torch.einsum('bij,raj->brai', u, xyz) + r[:,None,None,:]
                asmb.update({(c,k,i):xyz_i for i,xyz_i in enumerate(xyz_ru)})
                chiral_chain_dict.update({c:chiral_dict[f"{pdbid}_{c}"]})
            except KeyError:
                print("KEY ERROR FILE:", item[0])
                return {'seq': np.zeros(5)}
                # return {'seq': ''}
                # pass

    # select chains which share considerable similarity to chid
    seqid = meta['tm'][chids==chid][0,:,1]
    homo = set([ch_j for seqid_j,ch_j in zip(seqid,chids)
                if seqid_j>params['HOMO']])
    # stack all chains in the assembly together
    seq,xyz,idx,masked,chiral_full= "",[],[],[],""
    seq_list = []
    for counter,(k,v) in enumerate(asmb.items()):
        # seq += chains[k[0]]['seq']
        # seq_list.append(chains[k[0]]['seq'])
        seq += chains[k[0]]['seq'].upper()
        seq_list.append(chains[k[0]]['seq'].upper())
        xyz.append(v)
        chiral_full += chiral_chain_dict[k[0]]
        idx.append(torch.full((v.shape[0],),counter))
        if k[0] in homo:
            masked.append(counter)

    # if len(seq) == len(chiral_full):
    # print("AMSB OF > 1")
    # print("SEQ:", len(seq))
    # print("CHIRAL:", len(chiral_full))
    return {'seq'    : seq,
            'xyz'    : torch.cat(xyz,dim=0),
            'idx'    : torch.cat(idx,dim=0),
            'masked' : torch.Tensor(masked).int(),
            'label'  : item[0],
            'chiral' : chiral_full,
            }
    # else:
    #     return {'seq': ''}

def build_training_clusters(params, debug):
    # Define a helper function for safely converting to integers
    def safe_int_conversion(value):
        # Extract numbers from a string and handle prefixes
        match = re.search(r'\d+', value)
        return int(match.group(0)) if match else None

    # Load validation and test identifiers
    with open(params['VAL'], 'r') as file:
        val_ids = set(safe_int_conversion(line.strip()) for line in file if safe_int_conversion(line.strip()) is not None)

    with open(params['TEST'], 'r') as file:
        test_ids = set(safe_int_conversion(line.strip()) for line in file if safe_int_conversion(line.strip()) is not None)

    if debug:
        val_ids = set()
        test_ids = set()

    # Read and process the data from the correct file
    with open(params['CHIRAL'], 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header if present
        rows = []
        for r in reader:
            if float(r[2]) <= params['RESCUT'] and parser.parse(r[1]) <= parser.parse(params['DATCUT']):
                cluster_id = safe_int_conversion(r[4])  # Safely convert potential prefixed numeric field
                if cluster_id:
                    rows.append([r[0], r[3], cluster_id])

    # Compile training, validation, and test sets
    train, valid, test = {}, {}, {}
    for r in rows:
        cluster_id = r[2]
        if cluster_id in val_ids:
            valid.setdefault(cluster_id, []).append(r[:2])
        elif cluster_id in test_ids:
            test.setdefault(cluster_id, []).append(r[:2])
        else:
            train.setdefault(cluster_id, []).append(r[:2])

    if debug:
        valid = train

    return train, valid, test
