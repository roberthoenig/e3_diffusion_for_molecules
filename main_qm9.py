#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Rdkit import should be first, do not move it
try:
    from rdkit import Chem
except ModuleNotFoundError:
    pass
import copy
import utils
import argparse
import wandb
from configs.datasets_config import get_dataset_info
from os.path import join
from qm9 import dataset
from qm9.models import get_optim, get_model
from qm9.sampling import sample_chain, sample, sample_sweep_conditional
from equivariant_diffusion import en_diffusion
from equivariant_diffusion.utils import assert_correctly_masked
from equivariant_diffusion import utils as flow_utils
import torch
import time
import pickle
from qm9.utils import prepare_context, compute_mean_mad
from train_test import train_epoch, test, analyze_and_save
import datetime


# ### Initialize arguments

# In[2]:


from argparse import Namespace

args = Namespace(
    n_cont = 4,
    n_onehot = 23,
    n_nodes=10,
    n_dims = 3,
    dataset_path="3dfront_bounding_boxes_for_e3_repo.pkl",
    exp_name='debug_10',
    model='egnn_dynamics',
    probabilistic_model='diffusion',
    diffusion_steps=100,
    diffusion_noise_schedule='polynomial_2',
    diffusion_noise_precision=1e-05,
    diffusion_loss_type='l2',
    n_epochs=30000,
    batch_size=32,
    lr=0.0001,
    brute_force=False,
    actnorm=True,
    break_train_epoch=False,
    dp=True,
    condition_time=True,
    clip_grad=True,
    trace='hutch',
    n_layers=7,
    inv_sublayers=1,
    nf=128,
    tanh=True,
    attention=True,
    norm_constant=1,
    sin_embedding=False,
    ode_regularization=0.001,
    dataset='3rscan',
    datadir='qm9/temp',
    filter_n_atoms=None,
    dequantization='argmax_variational',
    n_report_steps=100,
    wandb_usr=None,
    no_wandb=False,
    online=True,
    no_cuda=False,
    save_model=True,
    generate_epochs=1,
    num_workers=0,
    test_epochs=10,
    data_augmentation=False,
    conditioning=['floor_plan'],
    resume=None,
    start_epoch=0,
    ema_decay=0.999,
    augment_noise=0,
    n_stability_samples=500,
    normalize_factors=[1,
    4,
    1],
    remove_h=False,
    include_charges=True,
    visualize_every_batch=100000000.0,
    normalization_factor=1,
    aggregation_method='sum'
)


# ### Process arguments

# In[3]:


dataset_info = get_dataset_info(args.dataset, args.remove_h)

# atom_encoder = dataset_info['atom_encoder']
# atom_decoder = dataset_info['atom_decoder']

# args, unparsed_args = parser.parse_known_args()
args.wandb_usr = utils.get_wandb_username(args.wandb_usr)

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
dtype = torch.float32

if args.resume is not None:
    exp_name = args.exp_name + '_resume'
    start_epoch = args.start_epoch
    resume = args.resume
    wandb_usr = args.wandb_usr
    normalization_factor = args.normalization_factor
    aggregation_method = args.aggregation_method

    with open(join(args.resume, 'args.pickle'), 'rb') as f:
        args = pickle.load(f)

    args.resume = resume
    args.break_train_epoch = False

    args.exp_name = exp_name
    args.start_epoch = start_epoch
    args.wandb_usr = wandb_usr

    # Careful with this -->
    if not hasattr(args, 'normalization_factor'):
        args.normalization_factor = normalization_factor
    if not hasattr(args, 'aggregation_method'):
        args.aggregation_method = aggregation_method

    print(args)

utils.create_folders(args)


# ### Retrieve QM9 dataloaders

dataloaders, charge_scale = dataset.retrieve_dataloaders(args)
floor_plans = torch.cat([x['floor_plan'] for x in dataloaders['test']], dim=0)

# ### More args processing, create model

# In[11]:


if len(args.conditioning) > 0:
    print(f'Conditioning on {args.conditioning}')
    property_norms = None
    context_node_nf = 64
else:
    context_node_nf = 0
    property_norms = None

args.context_node_nf = context_node_nf


# Create EGNN flow
model, nodes_dist, prop_dist = get_model(args, device, dataset_info, dataloaders['train'])
if prop_dist is not None:
    prop_dist.set_normalizer(property_norms)
# model = torch.load("correct_model_checkpoints/model_scenes_9000.pt")
model = model.to(device)
optim = get_optim(args, model)
# print(model)

gradnorm_queue = utils.Queue()
gradnorm_queue.add(3000)  # Add large value that will be flushed.


def check_mask_correct(variables, node_mask):
    for variable in variables:
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)


# ## main()

# ###  model processing

# In[12]:


if args.resume is not None:
    flow_state_dict = torch.load(join(args.resume, 'flow.npy'))
    optim_state_dict = torch.load(join(args.resume, 'optim.npy'))
    model.load_state_dict(flow_state_dict)
    optim.load_state_dict(optim_state_dict)

# Initialize dataparallel if enabled and possible.
if args.dp and torch.cuda.device_count() > 1:
    print(f'Training using {torch.cuda.device_count()} GPUs')
    model_dp = torch.nn.DataParallel(model.cpu())
    model_dp = model_dp.cuda()
else:
    model_dp = model

# Initialize model copy for exponential moving average of params.
if args.ema_decay > 0:
    model_ema = copy.deepcopy(model)
    ema = flow_utils.EMA(args.ema_decay)

    if args.dp and torch.cuda.device_count() > 1:
        model_ema_dp = torch.nn.DataParallel(model_ema)
    else:
        model_ema_dp = model_ema
else:
    ema = None
    model_ema = model
    model_ema_dp = model_dp

best_nll_val = 1e8
best_nll_test = 1e8


# ### training

# In[13]:


def save_and_sample_chain(model, args, device, dataset_info, prop_dist,
                          epoch=0, id_from=0, batch_id='', n_nodes=None):
    rand_idx = torch.randint(low=0, high=len(floor_plans), size=(1,)).item()
    context = floor_plans[rand_idx]
    one_hot, charges, x = sample_chain(
            args=args, device=device, flow=model,
            n_tries=1, dataset_info=dataset_info, prop_dist=prop_dist,n_nodes=n_nodes,
            context=context)

    # vis.save_xyz_file(f'outputs/{args.exp_name}/epoch_{epoch}_{batch_id}/chain/',
                    #   one_hot, charges, x, dataset_info, id_from, name='chain')

    return one_hot, charges, x, context


# In[15]:


mean_losses = []
sample_norms = []
samples = []
contexts = []

import sys
chckpt_dir = sys.argv[1]
import os
os.mkdir(chckpt_dir)

# In[29]:


for epoch in range(args.start_epoch, args.n_epochs):
    start_epoch = time.time()
    mean_loss = train_epoch(args=args, loader=dataloaders['train'], epoch=epoch, model=model, model_dp=model_dp,
                model_ema=model_ema, ema=ema, device=device, dtype=dtype, property_norms=property_norms,
                nodes_dist=nodes_dist, dataset_info=dataset_info,
                gradnorm_queue=gradnorm_queue, optim=optim, prop_dist=prop_dist)
    mean_losses.append(mean_loss)
    # print(f"Epoch took {time.time() - start_epoch:.1f} seconds.")

    if epoch % 1000 == 0:
        torch.save(mean_losses, f"{chckpt_dir}/mean_losses.pkl")
        torch.save({'samples': samples, 'contexts': contexts}, f"{chckpt_dir}/samples_and_contexts.pkl")
        torch.save(model, f"{chckpt_dir}/model_chckpt_dir_{epoch}.pt")
    if epoch % 10 == 0:
        time_str = datetime.datetime.fromtimestamp(time.time()).isoformat()
        print(f"Epoch {epoch}, mean loss {mean_loss}, time {time_str}")
        s = save_and_sample_chain(model_ema, args, device, dataset_info, prop_dist, epoch=0,
                      batch_id=0, n_nodes=7)
        samples.append(s)
        # sample_different_sizes_and_save(model_ema, nodes_dist, args, device, dataset_info,
                                        # prop_dist, epoch=epoch)
        # print(f'Sampling took {time.time() - start:.2f} seconds')
        _,_,x,context = s
        sample_norm = x[-1].abs().mean().item()
        print("sample norm", sample_norm)
        contexts.append(context)
        sample_norms.append(sample_norm)
    if epoch == args.start_epoch and torch.cuda.is_available():
        print(torch.cuda.memory_summary(),flush=True)