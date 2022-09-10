"""
Tool for calculating dihedral scans with ANI.
"""

# Numpy
import numpy as np

import os

import itertools
import copy

# Neuro Chem


# RDKit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import Geometry

import ase
from ase import Atoms
from ase.optimize import BFGS, LBFGS, FIRE
from ase.constraints import FixInternals
from ase.io import read, write

import read_n_write as rnw
from torchani.units import EV_TO_KCALMOL

EV_TO_KCALMOL = 23.06054

def __convert_rdkitmol_to_nparr__(mrdk, confId=-1):
    xyz = np.zeros((mrdk.GetNumAtoms(), 3), dtype=np.float32)
    spc = []

    Na = mrdk.GetNumAtoms()
    for i in range(0, Na):
        pos = mrdk.GetConformer(confId).GetAtomPosition(i)
        sym = mrdk.GetAtomWithIdx(i).GetSymbol()

        spc.append(sym)
        xyz[i, 0] = pos.x
        xyz[i, 1] = pos.y
        xyz[i, 2] = pos.z

    return xyz, spc

class ani_tortion_scanner():
    def __init__(self, ens, relax=True, epsilon=1.e-7, fmax=0.05, alg='BFGS'):
        """
        input:
            ens: ani ensemble used for optimization
            relax: If False perform rigid scans. If True perform relaxed scans
            epsilon: Accuracy of constraint
            fmax: max force value after optimization (eV/A)
            alg: Optimization algorithom (BFGS, LBFGS, or FIRE)
        """
        self.ens = ens
        self.e_holder = []
        self.relax = relax
        self.epsilon = epsilon
        self.fmax = fmax
        self.alg = alg

    def opt(self, rdkmol, atid, f, logger='log.out'):
        """
        Perform restrained optimization at a fixed dihedral angle.
        input:
            rdkmol: rdkit mol object
            atid: atom indicies for dihedral (index from 0)
            f: file to write optimized structure to (writes xyz file)
        output:
            phi_value_list: list of dihedral angles frozen during optimization
            e: energy of optimized structure (kcal/mol)
            X: coordinates of optimized structure (Angstroms)

        """
        X, S = __convert_rdkitmol_to_nparr__(rdkmol)
        Na = len(S)
        mol = Atoms(symbols=S, positions=X)
        mol.set_calculator(self.ens)  #Set the ANI Ensemble as the calculator
        dihs = []
        for i in range(len(atid)):
            phi_restraint = mol.get_dihedral(atid[i])
            phi_fix = [phi_restraint, atid[i]]
            dihs.append(phi_fix)
        c = FixInternals(dihedrals=dihs, epsilon=self.epsilon)
        mol.set_constraint(c)
        if self.relax == True:
            if self.alg == 'LBFGS':
                dyn = LBFGS(mol, logfile=logger)  #Choose optimization algorith
            elif self.alg == 'FIRE':
                dyn = FIRE(mol, logfile=logger)
            else:
                dyn = BFGS(mol, logfile=logger)
            dyn.run(fmax=self.fmax, steps=5000)
        e = mol.get_potential_energy() * EV_TO_KCALMOL
        self.e_holder.append(e)
        phi_value_list = []
        for i in range(len(atid)):
            phi_value = mol.get_dihedral(atid[i]) * 180. / np.pi
            phi_value_list.append(phi_value)
        X = mol.get_positions()
        coor = []
        spc = mol.get_chemical_symbols()
        xyz = mol.get_positions()
        N = mol.get_number_of_atoms()
        rnw.write_xyz(f[:-4] + '_all.xyz', [xyz], [spc], cmt=str(e), aw='a')
        return phi_value_list, e, X

    def __make_combs__(self, ang, steps, inc):
        """
        For 2D scans gets pairs of diheral angles to optimize at.
        """
        TSA = []
        for j in range(len(ang)):
            SA = []
            for i in range(steps):
                SA.append(ang[j] + i * inc)
            TSA.append(SA)
        T = list(itertools.product(*TSA))
        return T

    def rot(self, f, atid, ang, steps, inc, new=True):
        """
        Perform relaxed dihedral scan (1D or 2D)
        input:
            f: mol file
            atid: atom indicies for dihedral (index from 0)
            ang: Angle to start scan from
            steps: How many steps to take in scan
            inc: how many degrees to rotate each step
        output:
            P: List of angles from scan (degrees)
            E: List of energies from scan (kcal/mol)
        """
        mol = Chem.MolFromMolFile(f, removeHs=False)
        mol_copy = copy.deepcopy(mol)
        c = mol_copy.GetConformer()
        P = []
        E = []
        T = self.__make_combs__(ang, steps, inc)
        fname = f[:-4] + '_all.xyz'
        if os.path.isfile(fname):
            if new == True:
                os.remove(fname)
        for j in range(len(T)):
            for i in range(len(atid)):
                a0 = atid[i][0]
                a1 = atid[i][1]
                a2 = atid[i][2]
                a3 = atid[i][3]
                Chem.rdMolTransforms.SetDihedralDeg(c, a0, a1, a2, a3, T[j][i])
            phi_values, e, X = self.opt(mol_copy, atid, f)
            E.append(e)
            c = mol_copy.GetConformer(-1)
            for aid in range(c.GetNumAtoms()):
                pos = Geometry.rdGeometry.Point3D(X[aid][0], X[aid][1], X[aid][2])
                c.SetAtomPosition(aid, pos)
            P.append(phi_values)
        return P, E
