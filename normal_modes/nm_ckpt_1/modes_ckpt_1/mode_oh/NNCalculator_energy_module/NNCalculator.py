import tensorflow as tf
import numpy as np
import ase
from ase.neighborlist import neighbor_list
from .neural_network.NeuralNetwork import *
from .neural_network.activation_fn import *

'''
Calculator for the atomic simulation environment (ASE)
that evaluates energies and forces using a neural network
'''
class NNCalculator:
    #most parameters are just passed to the neural network
    def __init__(self,
                 checkpoint,                     #ckpt file from which to restore the model (can also be a list for ensembles)
                 atoms,                          #ASE atoms object
                 charge=0,                       #system charge
                 F=128,                          #dimensionality of feature vector
                 K=64,                           #number of radial basis functions
                 sr_cut=6.0,                     #short range cutoff distance 
                 lr_cut = None,                  #long range cutoff distance
                 num_blocks=5,                   #number of building blocks to be stacked
                 num_residual_atomic=2,          #number of residual layers for atomic refinements of feature vector
                 num_residual_interaction=3,     #number of residual layers for refinement of message vector
                 num_residual_output=1,          #number of residual layers for the output blocks
                 use_electrostatic=True,         #adds electrostatic contributions to atomic energy
                 use_dispersion=True,            #adds dispersion contributions to atomic energy
                 s6=None,                        #s6 coefficient for d3 dispersion, by default is learned
                 s8=None,                        #s8 coefficient for d3 dispersion, by default is learned
                 a1=None,                        #a1 coefficient for d3 dispersion, by default is learned
                 a2=None,                        #a2 coefficient for d3 dispersion, by default is learned   
                 activation_fn=shifted_softplus, #activation function
                 dtype=tf.float32):              #single or double precision

        #create neighborlist
        if lr_cut is None:
            self._sr_cutoff = sr_cut
            self._lr_cutoff = None
            self._use_neighborlist = False
        else:
            self._sr_cutoff = sr_cut
            self._lr_cutoff = lr_cut
            self._use_neighborlist = True


        #save checkpoint
        self._checkpoint = checkpoint

        #create neural network
        self._nn = NeuralNetwork(F=F,
                                 K=K,
                                 sr_cut=sr_cut,
                                 lr_cut=lr_cut,
                                 num_blocks=num_blocks,                   
                                 num_residual_atomic=num_residual_atomic,          
                                 num_residual_interaction=num_residual_interaction,     
                                 num_residual_output=num_residual_output,          
                                 use_electrostatic=use_electrostatic,         
                                 use_dispersion=use_dispersion,      
                                 s6=s6,
                                 s8=s8,
                                 a1=a1,
                                 a2=a2,      
                                 activation_fn=activation_fn, 
                                 dtype=dtype, scope="neural_network")

        #create placeholders for feeding data
        self._Q_tot      = np.array(1*[charge])
        self._Z          = tf.placeholder(tf.int32, shape=[None, ], name="Z") 
        self._R          = tf.placeholder(dtype,    shape=[None,3], name="R")        
        self._idx_i      = tf.placeholder(tf.int32, shape=[None, ], name="idx_i") 
        self._idx_j      = tf.placeholder(tf.int32, shape=[None, ], name="idx_j") 
        self._offsets    = tf.placeholder(dtype,    shape=[None,3], name="offsets") 
        self._sr_idx_i   = tf.placeholder(tf.int32, shape=[None, ], name="sr_idx_i") 
        self._sr_idx_j   = tf.placeholder(tf.int32, shape=[None, ], name="sr_idx_j") 
        self._sr_offsets = tf.placeholder(dtype,    shape=[None,3], name="sr_offsets") 
        
        #calculate atomic charges, energy and force evaluation nodes
        if self.use_neighborlist:
            (self._Ea, Qa, Dij, nhloss, self._repre,self._Ea_blocks_raw, self._Ea_blocks_scaled_no_shift, self._EshiftZ) = self.nn.atomic_properties(self.Z, self.R, self.idx_i, self.idx_j, self.offsets,self.sr_idx_i, self.sr_idx_j, self.sr_offsets)
            #--------------------------------------------------------------------------------------------------------------------------------------------------
            #Ea, Qa, Dij, nhloss = self.nn.atomic_properties(self.Z, self.R, self.idx_i, self.idx_j, self.offsets, self.sr_idx_i, self.sr_idx_j, self.sr_offsets)
            #---------------------------------------------------------------------------------------------------------------------------------------------------
        else:
            (self._Ea, Qa, Dij, nhloss, self._repre, self._Ea_blocks_raw, self._Ea_blocks_scaled_no_shift, self._EshiftZ) = self.nn.atomic_properties(self.Z, self.R, self.idx_i, self.idx_j, self.offsets)
            #--------------------------------------------------------------------------------------------------------------------------------------------------
            #self._Ea, Qa, Dij, nhloss, self._repre = self.nn.atomic_properties(self.Z, self.R, self.idx_i, self.idx_j, self.offsets)
            #--------------------------------------------------------------------------------------------------------------------------------------------------
        self._charges = self.nn.scaled_charges(self.Z, Qa, self.Q_tot)
        # Store Dij and charges for separate access
        self._Dij = Dij
        self._Qa = self._charges  # Already scaled

        #self._energy, self._forces = self.nn.energy_and_forces_from_scaled_atomic_properties(self._Ea, self.charges, Dij, self.Z, self.R, self.idx_i, self.idx_j)
        self._energy, self._forces, self._Edisp_atom = self.nn.energy_and_forces_from_scaled_atomic_properties(self._Ea, self.charges, Dij, self.Z, self.R, self.idx_i, self.idx_j)


        # Tensor for electrostatic energy (scalar per atom)
        self._elec_per_atom = self.nn.electrostatic_energy_per_atom(self._Dij, self._Qa, self.idx_i, self.idx_j)

        #create TensorFlow session and load neural network(s)
        self._sess = tf.Session()
        if(type(self.checkpoint) is not list):
            self.nn.restore(self.sess, self.checkpoint)

        #calculate properties once to initialize everything
        self._calculate_all_properties(atoms)

    def calculation_required(self, atoms, quantities=None):
        return atoms != self.last_atoms

    def _calculate_all_properties(self, atoms):
        #find neighbors and offsets
        if self.use_neighborlist or any(atoms.get_pbc()):
            idx_i, idx_j, S = neighbor_list('ijS', atoms, self.lr_cutoff)
            offsets = np.dot(S, atoms.get_cell())
            sr_idx_i, sr_idx_j, sr_S = neighbor_list('ijS', atoms, self.sr_cutoff)
            sr_offsets = np.dot(sr_S, atoms.get_cell())
            feed_dict = {self.Z: atoms.get_atomic_numbers(), self.R: atoms.get_positions(), 
                    self.idx_i: idx_i, self.idx_j: idx_j, self.offsets: offsets,
                    self.sr_idx_i: sr_idx_i, self.sr_idx_j: sr_idx_j, self.sr_offsets: sr_offsets}
        else:
            N = len(atoms)
            idx_i = np.zeros([N*(N-1)], dtype=int)

            idx_j = np.zeros([N*(N-1)], dtype=int)
            offsets = np.zeros([N*(N-1),3], dtype=float)
            count = 0
            for i in range(N):
                for j in range(N):
                    if i != j:
                        idx_i[count] = i
                        idx_j[count] = j
                        count += 1
            feed_dict = {self.Z: atoms.get_atomic_numbers(), self.R: atoms.get_positions(), 
                    self.idx_i: idx_i, self.idx_j: idx_j, self.offsets: offsets}
#"""
#self._atomic_E = self.sess.run(self._Ea, feed_dict=feed_dict)  # <-- this line enables per-atom energy access
#This line was added bt me to acces the per atom energy 
#"""
        #calculate energy and forces (in case multiple NNs are used as ensemble, this forms the average)
        if(type(self.checkpoint) is not list): #only one NN
            #self._last_energy, self._last_forces, self._last_charges = self.sess.run([self.energy, self.forces, self.charges], feed_dict=feed_dict)
            #also get representation after last interaction block
            """
            self._last_repre, self._last_energy, self._last_forces, self._last_charges = self.sess.run([self._repre, self.energy, self.forces, self.charges], feed_dict=feed_dict)
            self._atomic_E = self.sess.run(self._Ea, feed_dict=feed_dict)  # <-- this line enables per-atom energy access
            self._elec_E = self.sess.run(self._elec_per_atom, feed_dict=feed_dict)  # NEW
            """
            # NEW
            fetches = [
                       self._repre,                 # representation snapshots
                       self.energy,                 # total energy (per-structure)
                       self.forces,                 # forces
                       self.charges,                # scaled charges
                       self._Ea,                    # per-atom NN short-range energy (scaled+shift)
                       self._elec_per_atom,         # per-atom electrostatic energy
                       self._Ea_blocks_raw,         # LIST[num_blocks] of [N_atoms] (unscaled heads)
                       self._Ea_blocks_scaled_no_shift,  # LIST[num_blocks] of [N_atoms] (Escale * head)
                       self._EshiftZ]                # [N_atoms] per-atom shift (gathered )

            (self._last_repre,self._last_energy,self._last_forces,self._last_charges,self._atomic_E,self._elec_E,Ea_blocks_raw_vals,Ea_blocks_scaled_vals,EshiftZ_vals) = self.sess.run(fetches, feed_dict=feed_dict)

            # Store nicely for later access
            # shapes:
            #   Ea_blocks_raw_vals            : list length = num_blocks, each np.array [N_atoms]
            #   Ea_blocks_scaled_vals         : list length = num_blocks, each np.array [N_atoms]
            #   EshiftZ_vals                  : np.array [N_atoms]
            self._Ea_blocks_raw_vals = Ea_blocks_raw_vals
            self._Ea_blocks_scaled_vals = Ea_blocks_scaled_vals
            self._EshiftZ_vals = EshiftZ_vals


            self._disp_E   = self.sess.run(self._Edisp_atom, feed_dict=feed_dict)  #Dispersion new
            self._energy_stdev = 0
        else: #ensemble is used
            #raise NotImplementedError("Ensemble model not supported for decomposition output.")
            #TODO: adapt the code that also for ensembles I can obtain a representation, etc.
            for i in range(len(self.checkpoint)):
                self.nn.restore(self.sess, self.checkpoint[i])
                energy, forces, charges = self.sess.run([self.energy, self.forces, self.charges], feed_dict=feed_dict)
                if i == 0:
                    self._last_energy  = energy
                    self._last_forces  = forces
                    self._last_charges = charges
                    self._energy_stdev = 0
                else:
                    n = i+1
                    delta = energy-self.last_energy
                    self._last_energy += delta/n
                    self._energy_stdev += delta*(energy-self.last_energy)
                    for a in range(np.shape(charges)[0]): #loop over atoms
                        self._last_charges[a] += (charges[a]-self.last_charges[a])/n
                        for b in range(3):
                            self._last_forces[a,b] += (forces[a,b]-self.last_forces[a,b])/n 
            if(len(self.checkpoint) > 1):
                self._energy_stdev = np.sqrt(self.energy_stdev/len(self.checkpoint))

        self._last_energy = np.array(1*[self.last_energy]) #prevents some problems...

        #store copy of atoms
        self._last_atoms = atoms.copy()

    def get_potential_energy(self, atoms, force_consistent=False):
        if self.calculation_required(atoms):
            self._calculate_all_properties(atoms)
        return self.last_energy

    def get_forces(self, atoms):
        if self.calculation_required(atoms):
            self._calculate_all_properties(atoms)
        return self.last_forces

    def get_charges(self, atoms):
        if self.calculation_required(atoms):
            self._calculate_all_properties(atoms)
        return self.last_charges

    @property
    def electrostatic_energies(self):
        return self._elec_E

    @property
    def sess(self):
        return self._sess

    @property
    def last_atoms(self):
        return self._last_atoms

    @property
    def last_energy(self):
        return self._last_energy

    @property
    def last_forces(self):
        return self._last_forces

    @property
    def last_charges(self):
        return self._last_charges

    @property
    def energy_stdev(self):
        return self._energy_stdev

    @property
    def atomic_energies(self):
        return self._atomic_E
        
    @property
    def representation(self):
        return self._last_repre

    @property
    def sr_cutoff(self):
        return self._sr_cutoff

    @property
    def lr_cutoff(self):
        return self._lr_cutoff

    @property
    def use_neighborlist(self):
        return self._use_neighborlist

    @property
    def nn(self):
        return self._nn

    @property
    def checkpoint(self):
        return self._checkpoint

    @property
    def Z(self):
        return self._Z

    @property
    def Q_tot(self):
        return self._Q_tot

    @property
    def R(self):
        return self._R

    @property
    def offsets(self):
        return self._offsets

    @property
    def idx_i(self):
        return self._idx_i

    @property
    def idx_j(self):
        return self._idx_j

    @property
    def sr_offsets(self):
        return self._sr_offsets

    @property
    def sr_idx_i(self):
        return self._sr_idx_i

    @property
    def sr_idx_j(self):
        return self._sr_idx_j

    @property
    def energy(self):
        return self._energy

    @property
    def forces(self):
        return self._forces

    @property
    def charges(self):
        return self._charges

    @property
    def Ea(self):
        return self._Ea

    @property
    def dispersion_energies(self):
        return self._disp_E
     
    @property
    def block_atomic_energies_raw(self):
    """
    List of length num_blocks; each entry is a [N_atoms] array of the
    unscaled per-block energy head (out[:,0]) before Escale/Eshift.
    """
        return self._Ea_blocks_raw_vals

    @property
    def block_atomic_energies_scaled_no_shift(self):
    """
    List of length num_blocks; each entry is [N_atoms] with Escale applied
    (Escale*head). Sum over blocks equals atomic_energies - atom_energy_shift.
    """
        return self._Ea_blocks_scaled_vals

    @property
    def atom_energy_shift(self):
    """
    Per-atom energy shift array [N_atoms]. Add once to the sum of
    block_atomic_energies_scaled_no_shift to reconstruct atomic_energies.
    """
        return self._EshiftZ_vals
        


