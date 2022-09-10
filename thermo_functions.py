import re
import numpy as np
import pyanitools as pyt
import os
os.environ["OMP_NUM_THREADS"]="1"
from scipy import stats
import math



import pandas as pd
import io
from torchani.units import ANGSTROM_TO_BOHR

def get_orca_modes(f):
    def remove_last_line_from_string(s):
        return s[:s.rfind('\n')]
    string = open(f, 'r').read()
    modesr=re.compile("(?s)(?<=orthogonal\n\n)(.*?)(?=\n\n-*\nIR SPECTRUM)")
    ms=re.compile(" \n")
    modes=modesr.findall(string)
    blocks=re.split(ms,modes[0])
    blocks=blocks[1:]
    modes_list=[]
    for i in range(len(blocks)):
        if i!=len(blocks):
            blocks[i]=remove_last_line_from_string(blocks[i])
        data=io.StringIO(blocks[i])
        df = pd.read_csv(data,header=None,delimiter=r"\s+")
        df = df.iloc[: , 1:]
        for col in df.columns:
            mod=np.array(df[col].values)
            mod=mod.reshape(int(len(mod)/3),3)
            modes_list.append(mod)
    return modes_list

def get_orca_freqs(f):
    string = open(f, 'r').read()
    freqr=re.compile("(?s)(?<=already applied\!\)\n\n)(.*?)(?=\n\n------------\nNORMAL MO)")
    freqs=freqr.findall(string)
    data=io.StringIO(freqs[0])
    df = pd.read_csv(data,header=None,delimiter=r"\s+")
    freq_values = df.iloc[:,1].values
    return freq_values


def get_orca_mull_charges(f):
    string = open(f, 'r').read()
    chargebr=re.compile("(?s)(?<=MULLIKEN ATOMIC CHARGES\n)(.*?)(?=\nSum of atomic)")
    cr=re.compile("([-]?\d*\.\d*)")
    cblock=chargebr.findall(string)
    cblock=cblock[0].split('\n')[1:]
    cblock='\n'.join(cblock)
    charges=cr.findall(cblock)
    return np.array(charges,dtype=float)


def get_orca_dipole(f):
    string = open(f, 'r').read()
    compr=re.compile("Total Dipole Moment\s*\:\s*([-]?\d*\.\d*)\s*([-]?\d*\.\d*)\s*([-]?\d*\.\d*)")
    #totalr=re.compile("Magnitude \(Debye\)\s*\:\s*([-]?\d*\.\d*)")
    totalr=re.compile("Magnitude \(a\.u\.\)\s*\:\s*([-]?\d*\.\d*)")
    comps=np.array(compr.findall(string)[0], dtype=float)/ANGSTROM_TO_BOHR
    total=float(totalr.findall(string)[0])/ANGSTROM_TO_BOHR
    return comps, total


def get_orca_spectra(f,gnum=6):
    string = open(f, 'r').read()
    specr=re.compile("(?s)(?<=TZ\n)(.*?)(?=\nThe first freq)")
    spec=specr.findall(string)
    spec = "\n".join(spec[0].split("\n")[1:])
    reg2=re.compile("(\d+)\:\s*([-]?\d*\.\d*)\s*([-]?\d*\.\d*)")
    fi=reg2.findall(spec)
    intensities=[]
    frequencies=[]
    mode_nums=[]
    for i in range(len(fi)):
        if float(fi[i][0])>=gnum:
            intensities.append(fi[i][2])
            frequencies.append(fi[i][1])
    #data=io.StringIO(spec)
    #df = pd.read_csv(data,header=None,delimiter=r"\s+")
    #df = pd.read_csv(data,header=None,delimiter=r"\t")
    #intensities = df.iloc[:,2].values
    #frequencies = df.iloc[:,1].values
    return np.array(frequencies, dtype=float), np.array(intensities, dtype=float)


def get_orca_forces(f):
    string = open(f, 'r').read()
    forcer=re.compile("(?s)(?<=CARTESIAN GRADIENT\s------------------\s\n)(.*?)(?=Difference to)")
    force_block=forcer.findall(string)
    data=io.StringIO(force_block[0])
    df = pd.read_csv(data,header=None,delimiter=r"\s+")
    species = df.iloc[:,1].values
    xforce = np.array(df.iloc[:,3].values,dtype=float)
    yforce = np.array(df.iloc[:,4].values,dtype=float)
    zforce = np.array(df.iloc[:,5].values,dtype=float)

    forces=np.array((xforce,yforce,zforce))
    spc=[]
    for s in species:
        spc.append(s)
    return spc, forces.T



#These functions should match the results of a Gaussian freq=hpmodes calculation if given the frequencies from that calculation.
#The thermochem functions work only for non-linear structures at a minima
#calc_fc works for all molecules (probably)


def get_nm_data(f):
    string = open(f, 'r').read()
    frq=[]
    red=[]
    frc=[]
    ins=[]
    rmass = re.compile("Red. masses\s*--\s*((?:[-]?\d*\.\d*\s*){1,3})")
    masses = re.findall(rmass, string)
    rfreq = re.compile("Frequencies\s*--\s*((?:[-]?\d*\.\d*\s*){1,3})")
    matches = re.findall(rfreq, string)
    rforc = re.compile("Frc consts\s*--\s*((?:[-]?\d*\.\d*\s*){1,3})")
    forces = re.findall(rforc, string)
    rintn = re.compile("IR Inten\s*--\s*((?:[-]?\d*\.\d*\s*){1,3})")
    intens = re.findall(rintn, string)
    
    rsbox = "Symbolic ([\s\S]*?) orientation"
    sbox = re.findall(rsbox,string)
    rspec= "([A-Z][a-z]?)\s*[-]?\d*\.\d*\s*[-]?\d*\.\d*\s*[-]?\d*\.\d*"
    spc = re.findall(rspec,sbox[0])
    for m in range(len(matches)):
        matches[m]=matches[m].split()
        masses[m]=masses[m].split()
        forces[m]=forces[m].split()
        intens[m]=intens[m].split()
        for i in range(len(matches[m])):
            #print(matches[m][i].rstrip())
            #print(float(matches[m][i].rstrip()))
            frq.append(float(matches[m][i].rstrip()))
            red.append(float(masses[m][i].rstrip()))
            frc.append(float(forces[m][i].rstrip()))
            ins.append(float(intens[m][i].rstrip()))
    frq=np.array(frq, dtype=np.float32)
    red=np.array(red, dtype=np.float32)
    frc=np.array(frc, dtype=np.float32)
    ins=np.array(ins, dtype=np.float32)
    blockr=re.compile("(?<=Coord Atom Element:)(?s)(.*?)(?=requencies)")
    block=blockr.findall(string)
    n=0
    modes_all=[]
    for n in range(len(block)):
        if n==len(block)-1:
            block[n]=block[n][:block[n].rfind('\n')]
        else:
            for i in range(3):
                block[n]=block[n][:block[n].rfind('\n')]
        holder=open('hold.txt', 'w')
        holder.write(block[n])
        holder.close()
        modes=np.loadtxt('hold.txt')
        modes=modes.T[3:]
        for i in range(len(modes)):
            modes_all.append(np.reshape(modes[i], (int(len(modes[i])/3), 3)))

    return spc, modes_all, frq, red, frc, ins



def get_E(f, method="HF"):
	me=''
	for i in method:
		me+=i
		me+="\s*"
	string = open(f, 'r').read()
	regex=re.compile("\W\s*"+me+"=([\S\s]{1,15})\s*\\\\")
	Estr=regex.findall(string)[-1]
	Espl=Estr.split()
	E=''
	for i in Espl:
		E+=i
	return float(E)


#pulls thermochem data from gaussian freq=hpmodes log file
def getthermodata(fi, method='HF'):
    fil= open(fi,'r')
    data={}
    frq=[]
    mod=[]

    string = fil.read()



    S, modes, frq, red, frc, ins = get_nm_data(fi)
    data['freq']=frq
    data['species']=S
    data['Normal_modes']=modes
    data['reduced_mass']=red
    data['force_constants']=frc
    data['IR_int']=ins

    #Get the zero point energy
    zpreg="Zero-point vibrational energy.*\s*([-]?\d*\.\d*)"
    zpe=re.findall(zpreg, string)[0]
    data['zpe']=zpe

    #Get thermo properties
    therm="E\s*\(Thermal\)\s*CV\s*S\s*KCal\/Mol\s*Cal\/Mol-Kelvin\s*Cal\/Mol-Kelvin\s*Total\s*(\d*\.\d*)\s*(\d*\.\d*)\s*(\d*\.\d*)\s*.*\s*Translational\s*(\d*\.\d*)\s*(\d*\.\d*)\s*(\d*\.\d*)\s*Rotational\s*(\d*\.\d*)\s*(\d*\.\d*)\s*(\d*\.\d*)\s*Vibrational\s*(\d*\.\d*)\s*(\d*\.\d*)\s*(\d*\.\d*)"
    tprop=re.findall(therm, string)
    data['Total_E_thermal']=tprop[0][0]
    data['Total_CV']=tprop[0][1]
    data['Total_S']=tprop[0][2]
    data['trans_E_thermal']=tprop[0][3]
    data['trans_CV']=tprop[0][4]
    data['trans_S']=tprop[0][5]
    data['rot_E_thermal']=tprop[0][6]
    data['rot_CV']=tprop[0][7]
    data['rot_S']=tprop[0][8]
    data['vib_E_thermal']=tprop[0][9]
    data['vib_CV']=tprop[0][10]
    data['vib_S']=tprop[0][11]

    #Get Electronic energy in Hartrees
    e=get_E(fi, method=method)
    data['energy']=e
	
    th_coor_enth_reg="Thermal correction to Enthalpy= *([-]?\d\.\d*)"
    th_coor_enth = re.findall(th_coor_enth_reg, string)
    th_coor_enth=np.array(th_coor_enth, dtype=np.float32)
    data['th_coor_enth']=th_coor_enth[0]

    return data


#convert wavenumbers to Kelvin
def freq_to_ve(freq):
	c=29979245800                         #cm/s
	kj=1.380662e-23                        #J/K
	h=6.626176e-34                        #J*s
	kh=3.166808874e-6
	ve=freq*c*h/kj                        #K
	return ve

#Heat capacity
def get_Cv(freq, T):
	R=8.31441*0.239006                    #cal/(K*mol)
	ve=freq_to_ve(freq)
	vCv=0
	TCv=0
	Cp=0
	if T!=0:
		for i in ve:
			ex=np.exp(-i/T)
			mul=(i/T)**2
			vCv+=mul*ex/(1-ex)**2
		vCv=vCv*R
		TCv=vCv+3*R
		Cp=TCv+R
	return vCv, TCv, Cp                             #vibrational CV, Total Cv, Cp, cal/mol


#Enthalpy
def get_H(freq, T):
        R=1.987                      #cal/(K*mol)
        ve=freq_to_ve(freq)
        E=0
        if T!=0:
            for i in ve:
                E+=i*(0.5+(1/(np.exp(-(i/T))-1)))
        Ev=-E*R
        Er=1.5*R*T
        Et=1.5*R*T
        return Ev, Er, Et                 #vibrational, rotational, translational enthalpy

#Entropy
def get_S(freq, mol, T):
	R=8.31441                             #J/(K*mol)
	h=6.626176e-34                        #J*s
	k=1.380662e-23                        #J/K
	Jk=4.184                              #J to cal
	ve=freq_to_ve(freq)
	Sv=0
	S_r=0
	S_t=0
	if T!=0:
		for i in ve:
			ex=np.exp(i/T)
			nex=np.exp(-i/T)
			ln=np.log(1-nex)
			Sv+=(i/T)/(ex-1)-ln
		Sv=Sv*R/Jk
	
	
		inertias = (mol.get_moments_of_inertia())  # kg*m^2
		inertias = inertias/(10.0**10)**2
		inertias = inertias*1.66054e-27            #amu*A^2
		char = h**2/(8.0*np.pi**2*inertias*k)
		char = np.sqrt(np.product(char))
		Q_r = np.pi**(0.5)*T**(3.0 / 2.0)/char
		S_r = R * (np.log(Q_r) + 3.0 / 2.0)
		S_r=S_r/Jk
	
		mass=sum(mol.get_masses())*1.66054e-27
		Q_t=(2*np.pi*mass*k*T/h**2)**(1.5)
		Q_t*=k*T/101325
		S_t=R*(np.log(Q_t)+2.5)/Jk
	
	return Sv, S_r, S_t                    #cal/K

#Zero point energy
def get_vib_zpe(freq):
	zpe=0
	h=6.626176e-34                        #J*s
	c=29979245800                         #cm/s
	for i in freq:
		zpe+=i/2.0
	zpe=zpe*h*c/4.184
	zpe=zpe*6.022e23
	return zpe                   #cal/mol


#Heat of formation
#Uses values from Active Thermochemistry Tables for atomic enthalpies
#only supports T=298.15K and T=0K
def get_hof(f, freq, e, Hsae=0.0, Csae=0.0, Nsae=0.0, Osae=0.0, T=298.15):                        #freq in cm^-1, energies in kcal/mol
        c=29979245800                         #cm/s
        kj=1.38064852e-23                     #J/k
        kh=3.166808874e-6                     #Ha/K
        kc=kh*627.5096
        h=6.626176e-34                        #J*s
        mol=read(f)
        S=mol.get_chemical_symbols()
        enth=0
        sae=0
        corr=0
        
        for i in S:
                if i=='H':
                        sae+=Hsae
                        enth+=51.632126        #Kcal/mol
                        corr+=0.469396
#                       enth+=51.63
#                       corr+=1.01
                if i=='C':
                        sae+=Csae
                        enth+=170.0248         #Kcal/mol
                        corr+=1.310915
#                       enth+=169.98
#                       corr+=0.25
                if i=='N':
                        sae+=Nsae
                        enth+=112.4679         #Kcal/mol
                        corr+=0.445257
                if i=='O':
                        sae+=Osae
                        enth+=58.995716        #Kcal/mol
                        corr+=0.570015
	
        zpe=get_vib_zpe(freq)/1000.0
        Hv, Hr, Ht = get_H(freq,T)
        H=Hv+Hr+Ht
        zero_K_Hof=enth-(sae-(e+zpe))
        th_corr_enth=T*kc+H/1000	
        Hof=zero_K_Hof+th_corr_enth-zpe-corr
        return Hof, zero_K_Hof





#Gibbs free energy
def get_G(freq, mol, T):
	kc=1.987236538e-3                     #kcal/mol*K     Boltzman's constant
	Hv, Hr, Ht=get_H(freq, T)
	H=Hv+Hr+Ht
	Sv, Sr, St =get_S(freq, mol, T)
	S=Sv+Sr+St
	G=H/1000-T*S/1000#+kc*T
	return G  #kcal/mol



#This function works with Gaussian results but they should only be used to test and compare because if you have the modes and freqs from Guassian then you already have the reduced mass and force constants
#If the Normal modes come from ASE, leave red=[]
#frq in wavenumbers
def calc_fc(modes, frq, red=[]):
    c=3.0e10
    kgamu=1.66e-27
    N=1.0
    fcs=[]
    ms=[]
    for n in range(0,len(frq)):
        if len(red)>0:
            N=np.sqrt(red[n])
        mod=modes[n]
        U=0
        for i in range(len(mod)):
            for j in range(3):
                d=mod[i][j]/N
                U+=(d)**2
        U=1/U
        k=U*4*np.pi**2*frq[n]**2*kgamu*c**2
        k=k/100
        ms.append(U)
        fcs.append(k)
    return fcs, ms

#Call this function with with f as a gaussian freq=hpmodes log file and T=298.15 to test the above functions
def test(f, T=298.15, meth='HF'):
    c=29979245800                         #cm/s
    k=1.380662e-23                        #J/K
    kc=1.987236538e-3                     #kcal/mol*K
    h=6.626176e-34                        #J*s
    R=8.31441*0.239006                    #cal/(K*mol)
    print('Now running ', f)
    data={}
    mol=read(f)
    GAU=getthermodata(f, method=meth)
    freq=GAU['freq']              #Gaussian frequencies in cm^-1
    print("Number of atoms: ",len(GAU['species']))
    print("Number of frequencies: ", len(freq))
    vib_Cv, TCv, Cp =get_Cv(freq, T)
    print('CV: ', TCv, GAU['Total_CV'])
    Hv, Hr, Ht=get_H(freq, T)
    H=Hv+Hr+Ht
    print('Enthalpy: ', H, float(GAU['Total_E_thermal'])*1000)
    zpe=get_vib_zpe(freq)
    print('ZPE: ', zpe, float(GAU['zpe'])*1000)
    
    Sv, Sr, St =get_S(freq, mol, T)
    S=Sv+Sr+St
    print('Entropy: ', S, GAU['Total_S'])
    G=get_G(freq, mol, T)#+GAU['energy']*627.5096
    Gau_G=float(GAU['Total_E_thermal'])-T*float(GAU['Total_S'])/1000#+GAU['energy']*627.5096
    print('Gibbs free energy: ', G, Gau_G)



#test('H2.log', 298.15)
