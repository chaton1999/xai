#!/usr/bin/env python3
import argparse
import numpy as np
from ase import Atoms
from ase.io import read, write, Trajectory
from ase.optimize import *
from ase.visualize import view
from NNCalculator.NNCalculator import *
import matplotlib.pyplot as plt
import seaborn as sns


'''
Script to calculate the potential energy of a structure.
Usage: python predic_mol.py -i structure.xyz
'''

#parse command line arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser._action_groups.pop()
required = parser.add_argument_group("required arguments")
required.add_argument("-i", "--input",   type=str,   help="input tra",  required=True)
optional = parser.add_argument_group("optional arguments")
optional.add_argument("--charge",  type=int, help="total charge", default=0)
optional.add_argument("--nmodules",  type=int, help="number of modules including initialization", default=6)
optional.add_argument('--type',choices=['xyz','traj'],default='xyz',help="whether to use single xyz file or a traj file with multiple geoms")


args = parser.parse_args()
print("input ", args.input)

#read input file
#an atoms object

if args.type == "xyz":
    atoms = read(args.input)

elif args.type =="traj":
    traj = Trajectory(args.input)
    atoms = traj[0]


#setup calculator object, which in this case is the NN calculator
#it is important that it is setup with the settings as used in the
#training procedure.
#setup calculator
calc = NNCalculator(
    checkpoint="model.ckpt-1", #load the model you want to used
    atoms=atoms,
    charge=args.charge,
    F=128,
    K=64,
    num_blocks=5,
    num_residual_atomic=2,
    num_residual_interaction=3,
    num_residual_output=1,
    sr_cut=10.0,
    use_electrostatic=True,
    use_dispersion=False,
    s6=1.0000,                    #s6 coefficient for d3 dispersion, by default is learned
    s8=2.3550,                    #s8 coefficient for d3 dispersion, by default is learned
    a1=0.5238,                    #a1 coefficient for d3 dispersion, by default is learned
    a2=3.5016)                   #a2 coefficient for d3 dispersion, by default is learned)


n_atoms = len(atoms)
n_modules = 5
F = 128

atoms.set_calculator(calc)
epot = atoms.get_potential_energy()
charges = atoms.get_charges()

if args.type == "xyz":
    act = np.array(calc.representation)
    Q = charges
    print(act.shape) #here (6, 9, 128), i.e. (nmodules, natoms, len of feature)
    print(act)
    np.savez("test"+ args.input +".npz", act=act, Q=Q)


quit()









if args.type == "traj":
    act = np.zeros((len(traj), n_modules, n_atoms, F))
    epot = []
    q = []
    for i in range(len(traj)):
        atoms = traj[i]
        atoms.set_calculator(calc)
        etmp = atoms.get_potential_energy()
        q.append(atoms.get_distance(6,7) - atoms.get_distance(8,7))
        epot.append(etmp)
        act[i, :, :, :] = np.array(calc.representation)




    print(act.shape)
    #print(act)
    np.savez("act_llpes_"+ args.input +".npz", act=act, epot=epot, q=q)




#the following plot part of the code was not generalized
plot=True
if plot: #show the distribution of the feature values for a particular structure
                 #as they are updated throughout the NN
    #act = act.reshape(6, -1)
    print(act.shape)

    for ii in range(n_atoms):
        sns.distplot(act[5, ii, :],label=str(ii))
        

    plt.legend()
    plt.show()
    quit()
        



