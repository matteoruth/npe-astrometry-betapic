from helpers import Prior, train
from lampe.data import H5Dataset
from orbitize import DATADIR, read_input
import torch.nn as nn
import zuko 
import argparse

def main(size):

    trainset = H5Dataset(f'datasets/betapic_{size}-train.h5', batch_size=2048, shuffle=True)  
    validset = H5Dataset(f'datasets/betapic_{size}-val.h5', batch_size=2048, shuffle=True)

    priors = Prior() # Needed to post-process the data in the training phase

    num_obs = 2*(len(read_input.read_file('{}/betaPic.csv'.format(DATADIR))) - 1) # The RV observation is discarded

    train(
        trainset=trainset,
        validset=validset,
        prior=priors,
        epochs=1024,
        NPE_hidden_features=[512] * 5,
        flow = zuko.flows.spline.NSF,
        transforms = 3,
        num_obs=num_obs,
    )   

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Train betapic model")
    parser.add_argument("--size", type=int, default=23, help="The exponent of 2 for the training dataset size")
    args = parser.parse_args()

    main(args.size)
