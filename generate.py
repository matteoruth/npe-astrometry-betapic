r"""
This module generates the datasets for the Beta Pictoris system.
The validation set is 8 times smaller than the training set and
the test set is 64 times.
"""

from helpers import Simulator, Prior
import argparse
from orbitize import read_input, DATADIR
from orbitize.system import seppa2radec, transform_errors
from lampe.data import H5Dataset, JointLoader

def main(size, name):
    data_table = read_input.read_file('{}/betaPic.csv'.format(DATADIR))

    # Discard the  RV observation, as we do not take it into account in the model
    data_table = data_table[:-1] 

    # Transform the values that are currently in the sep/pa format to the ra/dec format. 
    # Note that the function transforming the errors is not vectorisable but it is not 
    # such a problem because it is only for +- 30 values
    ra_errs = []
    dec_errs = []
    corrs = []
    ra, dec = seppa2radec(data_table["quant1"], data_table["quant2"])

    for i in range(len(data_table)):
        ra_err, dec_err, corr = transform_errors(data_table["quant1"][i], 
                                                data_table["quant2"][i], 
                                                data_table["quant1_err"][i], 
                                                data_table["quant2_err"][i], 
                                                data_table["quant12_corr"][i], 
                                                seppa2radec)
        ra_errs.append(ra_err)
        dec_errs.append(dec_err)
        corrs.append(corr)

    data_table["quant1"] = ra
    data_table["quant2"] = dec
    data_table["quant1_err"] = ra_errs
    data_table["quant2_err"] = dec_errs
    data_table["quant12_corr"] = corr
    data_table["quant_type"] = ["radec" for _ in range(len(data_table))]

    # Define the prior & simulator
    priors = Prior()
    simulator = Simulator(data_table)

    loader = JointLoader(priors, simulator, batch_size=16, vectorized=True)

    H5Dataset.store(
        loader, 
        f'datasets/{name}-test.h5', 
        size=(2**size)//64, # 130.000
        overwrite=True)
    H5Dataset.store(
        loader, 
        f'datasets/{name}-val.h5', 
        size=(2**size)//8, # 1.000.000
        overwrite=True)
    H5Dataset.store(
        loader, 
        f'datasets/{name}-train.h5', 
        size=2**size, # 8.400.000
        overwrite=True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Generate betapic datasets")
    parser.add_argument("--size", type=int, default=23, help="The exponent of 2 for the training dataset size")
    parser.add_argument("--name", type=str, default="betapic", help="Base name for datasets")
    args = parser.parse_args()

    main(args.size, args.name)