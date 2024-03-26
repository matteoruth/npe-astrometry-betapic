import wandb
from tqdm import tqdm
from lampe.inference import NPE, NPELoss
from lampe.utils import GDStep
import zuko
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.nn as nn
from itertools import islice

def train(trainset, 
          validset,
          prior,
          epochs,
          num_obs,
          activation=nn.ELU,
          transforms=3,
          flow=zuko.flows.spline.NSF,
          NPE_hidden_features=[512] * 5,
          initial_lr=1e-3,
          weight_decay=1e-2, 
          clip=1.0,
          use_wandb=False):


    use_cuda = torch.cuda.is_available()

    if use_cuda:
        print('__CUDNN VERSION:', torch.backends.cudnn.version())
        print('__Number CUDA Devices:', torch.cuda.device_count())
        print('__CUDA Device Name:',torch.cuda.get_device_name(0))
        print('__CUDA Device Total Memory [GB]:',torch.cuda.get_device_properties(0).total_memory/1e9)
    else:
        print('Cuda not available')

    if use_wandb:
        wandb.login()
        wandb.init(project="betapic")
        config = {
            "epochs": epochs,
            "num_obs": num_obs,
            "activation": activation,
            "transforms": transforms,
            "flow": flow,
            "NPE_hidden_features": NPE_hidden_features,
            "initial_lr": initial_lr,
            "weight_decay": weight_decay,
            "clip": clip,
        }
        wandb.config.update(config)

    estimator = NPE(
        8, # The 8 parameters defining the orbit
        x_dim = num_obs,
        transforms = 3,
        build = flow,
        hidden_features = NPE_hidden_features,
        activation = activation
        ).cuda()
    
    loss = NPELoss(estimator)

    optimizer = optim.AdamW(
        estimator.parameters(), 
        lr=initial_lr, 
        weight_decay=weight_decay
        )
    
    step = GDStep(optimizer, clip=clip) 

    scheduler = sched.ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        min_lr=1e-6,
        patience=32,
        threshold=1e-2,
        threshold_mode='abs'
        )


    with tqdm(range(epochs), unit='epoch') as tq:
        for epoch in tq:

            estimator.train()
            train_loss = torch.stack([
                step(loss(prior.pre_process(theta).cuda(), x.cuda())) 
                for theta, x in islice(trainset, 1024) 
                ]).cpu().numpy()
            
            estimator.eval()
            
            with torch.no_grad():
                valid_loss = torch.stack([
                    loss(prior.pre_process(theta).cuda(), x.cuda())
                    for theta, x in islice(validset, 256) 
                    ]).cpu().numpy()
            if use_wandb:
                wandb.log({
                    "train_loss": train_loss.mean(), 
                    "valid_loss": valid_loss.mean(), 
                    "lr": optimizer.param_groups[0]['lr']
                    })
            
            scheduler.step(valid_loss.mean())

            if optimizer.param_groups[0]['lr'] <= scheduler.min_lrs[0]:
                break

            tq.set_postfix(loss=train_loss.mean(), 
                val_loss=valid_loss.mean())

    if use_wandb:
        name = wandb.run.name
        torch.save(estimator, f"models/{name}.pth")
        wandb.finish()
    else:
        torch.save(estimator, f"models/betapic.pth")