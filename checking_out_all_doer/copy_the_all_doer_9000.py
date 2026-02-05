"""
The all doer 9000, meant for the purpose of being an all in one. I will take my individual codes and make them into functions here. Might even make another separate code that pulls from this so people don't have to see all this.
"""

import yt
from yt.units import *
import glob
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
plt.style.use('classic')
import h5py
import pickle
import subprocess
#from astropy.table import Table
import scipy
import scipy.stats
from scipy.interpolate import interp1d

mpl.rcParams['agg.path.chunksize'] = 1.0e12
mpl.rcParams['xtick.major.width']='0.75'
mpl.rcParams['xtick.minor.width']='0.75'
mpl.rcParams['ytick.major.width']='0.75'
mpl.rcParams['ytick.minor.width']='0.75'
mpl.rcParams['axes.labelweight']='bold'
mpl.rcParams['font.weight']='bold'
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['axes.labelsize'] = 14
axis_font = {'size':'20'}                                                        

#YOUR CONSTANTS
he_h = 10**10.925/(10**12)
c_h = 10**8.39/(10**12)
n_h = 10**7.86/(10**12)
o_h = 10**8.73/(10**12)
ne_h = 10**8.05/(10**12)
na_h = 10**6.29/(10**12)
mg_h = 10**7.54/(10**12)
si_h = 10**7.52/(10**12)
s_h = 10**7.16/(10**12)
ca_h = 10**6.31/(10**12)
fe_h = 10**7.46/(10**12)
meta = 1.0
ph54 = 4.171475061185105E-005
cj21 = 8.23e-3#2.4e-2#4.23e-3#515.0
mass = 1e12*1.98e33 #in grams
r_gal = 214.0
rho_0 = 2.14720235901e-28
n_rho_0 = 0.0002363848123077725
temp_0 = 1200000.0
pres_0 = 2.7137551518470134e-14
s=70 #make smaller 
s2=7
s3=10
alp = 0.01
G = 6.67e-8
r_gal_inner = 0#12*3.09e21
r_gal_outer = 100*3.09e18#214.0*3.09e21
st = "cell"
def _hspec(field,data):
    return data['h   ']
yt.add_field("hspec",function=_hspec,sampling_type=st)
def _hpspec(field,data):
    return data['hp  ']
yt.add_field("hpspec",function=_hpspec,sampling_type=st)

def _hespec(field,data):
    return data['he  ']
yt.add_field("hespec",function=_hespec,sampling_type=st)
def _hepspec(field,data):
    return data['hep ']
yt.add_field("hepspec",function=_hepspec,sampling_type=st)
def _he2pspec(field,data):
    return data['he2p']
yt.add_field("he2pspec",function=_he2pspec,sampling_type=st)

def _cspec(field,data):
    return data['c   ']
yt.add_field("cspec",function=_cspec,sampling_type=st)
def _cpspec(field,data):
    return data['cp  ']
yt.add_field("cpspec",function=_cpspec,sampling_type=st)
def _c2pspec(field,data):
    return data['c2p ']
yt.add_field("c2pspec",function=_c2pspec,sampling_type=st)
def _c3pspec(field,data):
    return data['c3p ']
yt.add_field("c3pspec",function=_c3pspec,sampling_type=st)
def _c4pspec(field,data):
    return data['c4p ']
yt.add_field("c4pspec",function=_c4pspec,sampling_type=st)
def _c5pspec(field,data):
    return data['c5p ']
yt.add_field("c5pspec",function=_c5pspec,sampling_type=st)

def _nspec(field,data):
    return data['n   ']
yt.add_field("nspec",function=_nspec,sampling_type=st)
def _npspec(field,data):
    return data['np  ']
yt.add_field("npspec",function=_npspec,sampling_type=st)
def _n2pspec(field,data):
    return data['n2p ']
yt.add_field("n2pspec",function=_n2pspec,sampling_type=st)
def _n3pspec(field,data):
    return data['n3p ']
yt.add_field("n3pspec",function=_n3pspec,sampling_type=st)
def _n4pspec(field,data):
    return data['n4p ']
yt.add_field("n4pspec",function=_n4pspec,sampling_type=st)
def _n5pspec(field,data):
    return data['n5p ']
yt.add_field("n5pspec",function=_n5pspec,sampling_type=st)
def _n6pspec(field,data):
    return data['n6p ']
yt.add_field("n6pspec",function=_n6pspec,sampling_type=st)

def _ospec(field,data):
    return data['o   ']
yt.add_field("ospec",function=_ospec,sampling_type=st)
def _opspec(field,data):
    return data['op  ']
yt.add_field("opspec",function=_opspec,sampling_type=st)
def _o2pspec(field,data):
    return data['o2p ']
yt.add_field("o2pspec",function=_o2pspec,sampling_type=st)
def _o3pspec(field,data):
    return data['o3p ']
yt.add_field("o3pspec",function=_o3pspec,sampling_type=st)
def _o4pspec(field,data):
    return data['o4p ']
yt.add_field("o4pspec",function=_o4pspec,sampling_type=st)
def _o5pspec(field,data):
    return data['o5p ']
yt.add_field("o5pspec",function=_o5pspec,sampling_type=st)
def _o6pspec(field,data):
    return data['o6p ']
yt.add_field("o6pspec",function=_o6pspec,sampling_type=st)
def _o7pspec(field,data):
    return data['o7p ']
yt.add_field("o7pspec",function=_o7pspec,sampling_type=st)

def _nespec(field,data):
    return data['ne  ']
yt.add_field("nespec",function=_nespec,sampling_type=st)
def _nepspec(field,data):
    return data['nep ']
yt.add_field("nepspec",function=_nepspec,sampling_type=st)
def _ne2pspec(field,data):
    return data['ne2p']
yt.add_field("ne2pspec",function=_ne2pspec,sampling_type=st)
def _ne3pspec(field,data):
    return data['ne3p']
yt.add_field("ne3pspec",function=_ne3pspec,sampling_type=st)
def _ne4pspec(field,data):
    return data['ne4p']
yt.add_field("ne4pspec",function=_ne4pspec,sampling_type=st)
def _ne5pspec(field,data):
    return data['ne5p']
yt.add_field("ne5pspec",function=_ne5pspec,sampling_type=st)
def _ne6pspec(field,data):
    return data['ne6p']
yt.add_field("ne6pspec",function=_ne6pspec,sampling_type=st)
def _ne7pspec(field,data):
    return data['ne7p']
yt.add_field("ne7pspec",function=_ne7pspec,sampling_type=st)
def _ne8pspec(field,data):
    return data['ne8p']
yt.add_field("ne8pspec",function=_ne8pspec,sampling_type=st)
def _ne9pspec(field,data):
    return data['ne9p']
yt.add_field("ne9pspec",function=_ne9pspec,sampling_type=st)

def _naspec(field,data):
    return data['na  ']
yt.add_field("naspec",function=_naspec,sampling_type=st)
def _napspec(field,data):
    return data['nap ']
yt.add_field("napspec",function=_napspec,sampling_type=st)
def _na2pspec(field,data):
    return data['na2p']
yt.add_field("na2pspec",function=_na2pspec,sampling_type=st)

def _mgspec(field,data):
    return data['mg  ']
yt.add_field("mgspec",function=_mgspec,sampling_type=st)
def _mgpspec(field,data):
    return data['mgp ']
yt.add_field("mgpspec",function=_mgpspec,sampling_type=st)
def _mg2pspec(field,data):
    return data['mg2p']
yt.add_field("mg2pspec",function=_mg2pspec,sampling_type=st)
def _mg3pspec(field,data):
    return data['mg3p']
yt.add_field("mg3pspec",function=_mg3pspec,sampling_type=st)

def _sispec(field,data):
    return data['si  ']
yt.add_field("sispec",function=_sispec,sampling_type=st)
def _sipspec(field,data):
    return data['sip ']
yt.add_field("sipspec",function=_sipspec,sampling_type=st)
def _si2pspec(field,data):
    return data['si2p']
yt.add_field("si2pspec",function=_si2pspec,sampling_type=st)
def _si3pspec(field,data):
    return data['si3p']
yt.add_field("si3pspec",function=_si3pspec,sampling_type=st)
def _si4pspec(field,data):
    return data['si4p']
yt.add_field("si4pspec",function=_si4pspec,sampling_type=st)
def _si5pspec(field,data):
    return data['si5p']
yt.add_field("si5pspec",function=_si5pspec,sampling_type=st)

def _sspec(field,data):
    return data['s   ']
yt.add_field("sspec",function=_sspec,sampling_type=st)
def _spspec(field,data):
    return data['sp  ']
yt.add_field("spspec",function=_spspec,sampling_type=st)
def _s2pspec(field,data):
    return data['s2p ']
yt.add_field("s2pspec",function=_s2pspec,sampling_type=st)
def _s3pspec(field,data):
    return data['s3p ']
yt.add_field("s3pspec",function=_s3pspec,sampling_type=st)
def _s4pspec(field,data):
    return data['s4p ']
yt.add_field("s4pspec",function=_s4pspec,sampling_type=st)

def _caspec(field,data):
    return data['ca  ']
yt.add_field("caspec",function=_caspec,sampling_type=st)
def _capspec(field,data):
    return data['cap ']
yt.add_field("capspec",function=_capspec,sampling_type=st)
def _ca2pspec(field,data):
    return data['ca2p']
yt.add_field("ca2pspec",function=_ca2pspec,sampling_type=st)
def _ca3pspec(field,data):
    return data['ca3p']
yt.add_field("ca3pspec",function=_ca3pspec,sampling_type=st)
def _ca4pspec(field,data):
    return data['ca4p']
yt.add_field("ca4pspec",function=_ca4pspec,sampling_type=st)

def _fespec(field,data):
    return data['fe  ']
yt.add_field("fespec",function=_fespec,sampling_type=st)
def _fepspec(field,data):
    return data['fep ']
yt.add_field("fepspec",function=_fepspec,sampling_type=st)
def _fe2pspec(field,data):
    return data['fe2p']
yt.add_field("fe2pspec",function=_fe2pspec,sampling_type=st)
def _fe3pspec(field,data):
    return data['fe3p']
yt.add_field("fe3pspec",function=_fe3pspec,sampling_type=st)
def _fe4pspec(field,data):
    return data['fe4p']
yt.add_field("fe4pspec",function=_fe4pspec,sampling_type=st)

def _elecspec(field,data):
    return data['elec']
yt.add_field("elecspec",function=_elecspec,sampling_type=st)
def _myabar(field,data):
    abar = (data['hspec'] + data['hpspec'])/1.0 + (data['hespec'] + data['hepspec'] + data['he2pspec'])/4.0 + (data['cspec'] + data['cpspec'] + data['c2pspec'] + data['c3pspec'] + data['c4pspec'] + data['c5pspec'])/12.0 + (data['nspec'] + data['npspec'] + data['n2pspec'] + data['n3pspec'] + data['n4pspec'] + data['n5pspec'] + data['n6pspec'])/14.0  + (data['ospec'] + data['opspec'] + data['o2pspec'] + data['o3pspec'] + data['o4pspec'] + data['o5pspec'] + data['o6pspec'] + data['o7pspec'])/16.0 + (data['nespec'] + data['nepspec'] + data['ne2pspec'] + data['ne3pspec'] + data['ne4pspec'] + data['ne5pspec'] + data['ne6pspec'] + data['ne7pspec'] + data['ne8pspec'] + data['ne9pspec'])/20.0 + (data['naspec'] + data['napspec'] + data['na2pspec'])/22.0 + (data['mgspec'] + data['mgpspec'] + data['mg2pspec'] + data['mg3pspec'])/24.0 + (data['sispec'] + data['sipspec'] + data['si2pspec'] + data['si3pspec'] + data['si4pspec'] + data['si5pspec'])/28.0  + (data['sspec'] + data['spspec'] + data['s2pspec'] + data['s3pspec'] + data['s4pspec'])/32.0  + (data['caspec'] + data['capspec'] + data['ca2pspec'] + data['ca3pspec'] + data['ca4pspec'])/40.0 + (data['fespec'] + data['fepspec'] + data['fe2pspec'] + data['fe3pspec'] + data['fe4pspec'])/56.0   + data['elecspec']/0.000549
    abar = 1.0/abar
    return abar
yt.add_field("myabar",function=_myabar,sampling_type=st)
def _mysoundspeed(field,data):
    cs = 1.66666667*kboltz*data['temp']/(mh*data['myabar'])
    cs = np.sqrt(cs)
    return cs
yt.add_field("mysoundspeed",function=_mysoundspeed,units="cm/s",sampling_type=st)
def _mymachnumber(field,data):
    v2 = data['velx']**2 + data['vely']**2 + data['velz']**2
    v = np.sqrt(v2)
    mn = v/data['mysoundspeed']
    return mn
yt.add_field("mymachnumber",function=_mymachnumber,sampling_type=st)
def _vinfall(field,data):
    v2 = data['velx']*data['x'] + data['vely']*data['y'] + data['velz']*data['z']
    v = v2/np.sqrt(data['x']**2+data['y']**2+data['z']**2)
    mn = v
    return mn
yt.add_field("vinfall",function=_vinfall,units="km/s",sampling_type=st)
def _sigma(field,data):
    v2 = data['velx']**2 + data['vely']**2 + data['velz']**2
    v = np.sqrt(v2-data['vinfall']**2)
    mn = v/np.sqrt(3)
    return mn
yt.add_field("sigma",function=_sigma,units="km/s",sampling_type=st)
def _velocity(field,data):
    v2 = data['velx']**2 + data['vely']**2 + data['velz']**2
    v = np.sqrt(v2)
    mn = v/np.sqrt(3)
    return mn
yt.add_field("velocity",function=_velocity,units="km/s",sampling_type=st)
def _ndens(field,data):
    return data['density'] / (data['myabar'] * mh)
yt.add_field("ndens",function=_ndens,units="cm**-3",sampling_type=st)
def _xvar(field,data):
    avgd = np.average(data['dens'])
    return np.log( data['dens'] / avgd )
yt.add_field("xvar",function=_xvar,sampling_type=st)

def _turb_kinetic(field,data):
    v2 = np.sqrt(3)*data['sigma']**2
    v = 0.5*v2
    return v
yt.add_field("turb_kinetic",function=_turb_kinetic,units="cm**2/s**2",sampling_type=st)
def _fall_kinetic(field,data):
    v2 = data['vinfall']**2
    v = 0.5*data['cell_mass']*v2
    return v
yt.add_field("fall_kinetic",function=_fall_kinetic,units="erg",sampling_type=st)
def _turb_vel(field,data):
    vol = data['dx']*data['dy']*data['dz']
    #print len(vol), type(vol), type(np.sum(vol))
    v_vel = np.sum( vol * (data['velx']*data['velx']+data['vely']*data['vely']+data['velz']*data['velz']) ) / np.sum(vol)
    v_vel = np.sqrt(v_vel)
    #d_vel = np.sum( (dd['density']) * (dd['velx']*dd['velx']+dd['vely']*dd['vely']+dd['velz']*dd['velz']) ) / np.sum(dd['density'])
    #d_vel = np.sqrt(d_vel)
    #turb_accel = (data['accx']**2 + data['accy']**2 + data['accz']**2)*centimeter**2/second**4
    #ans = np.sqrt(turb_accel)#*1.59e16*second #this is now velocity and that number is the decay timescale for turbulence
    #ans = ans/np.sqrt(3)
    return v_vel
yt.add_field("turb_vel",function=_turb_vel,units="km/s",sampling_type=st)
def _v_rad(field,data):
    v2 = data['velx']*data['x'] + data['vely']*data['y'] + data['velz']*data['z']
    v = v2/np.sqrt(data['x']**2+data['y']**2+data['z']**2)
    mn = v
    return mn
yt.add_field("v_rad",function=_v_rad,units="km/s",sampling_type=st)
    
#----HERE ARE THE NORMAL MASS FRACTIONS-------!!!!!!!!!!!!!!!

def _hfrac(field,data):
	return data['hspec']/(data['hspec']+data['hpspec'])
yt.add_field('hfrac',function=_hfrac,sampling_type=st)
def _hpfrac(field,data):
	return data['hpspec']/(data['hspec']+data['hpspec'])
yt.add_field('hpfrac',function=_hpfrac,sampling_type=st)

def _hefrac(field,data):
	return he_h*data['hespec']/(data['hespec']+data['hepspec']+data['he2pspec'])
yt.add_field('hefrac',function=_hefrac,sampling_type=st)
def _hepfrac(field,data):
	return he_h*data['hepspec']/(data['hespec']+data['hepspec']+data['he2pspec'])
yt.add_field('hepfrac',function=_hepfrac,sampling_type=st)
def _he2pfrac(field,data):
	return he_h*data['he2pspec']/(data['hespec']+data['hepspec']+data['he2pspec'])
yt.add_field('he2pfrac',function=_he2pfrac,sampling_type=st)

def _cfrac(field,data):
	return meta*c_h*data['cspec']/(data['cspec']+data['cpspec']+data['c2pspec']+data['c3pspec']+data['c4pspec']+data['c5pspec'])
yt.add_field('cfrac',function=_cfrac,sampling_type=st)
def _cpfrac(field,data):
	return meta*c_h**data['cpspec']/(data['cspec']+data['cpspec']+data['c2pspec']+data['c3pspec']+data['c4pspec']+data['c5pspec'])
yt.add_field('cpfrac',function=_cpfrac,sampling_type=st)
def _c2pfrac(field,data):
	return meta*c_h*data['c2pspec']/(data['cspec']+data['cpspec']+data['c2pspec']+data['c3pspec']+data['c4pspec']+data['c5pspec'])
yt.add_field('c2frac',function=_c2pfrac,sampling_type=st)
def _c3pfrac(field,data):
	return meta*c_h*data['c3pspec']/(data['cspec']+data['cpspec']+data['c2pspec']+data['c3pspec']+data['c4pspec']+data['c5pspec'])
yt.add_field('c3pfrac',function=_c3pfrac,sampling_type=st)
def _c4pfrac(field,data):
	return meta*c_h*data['c4pspec']/(data['cspec']+data['cpspec']+data['c2pspec']+data['c3pspec']+data['c4pspec']+data['c5pspec'])
yt.add_field('c4pfrac',function=_c4pfrac,sampling_type=st)
def _c5pfrac(field,data):
	return meta*c_h*data['c5pspec']/(data['cspec']+data['cpspec']+data['c2pspec']+data['c3pspec']+data['c4pspec']+data['c5pspec'])
yt.add_field('c5pfrac',function=_c5pfrac,sampling_type=st)

def _nfrac(field,data):
	return meta*n_h*data['nspec']/(data['nspec']+data['npspec']+data['n2pspec']+data['n3pspec']+data['n4pspec']+data['n5pspec']+data['n6pspec'])
yt.add_field('nfrac',function=_nfrac,sampling_type=st)
def _npfrac(field,data):
	return meta*n_h*data['npspec']/(data['nspec']+data['npspec']+data['n2pspec']+data['n3pspec']+data['n4pspec']+data['n5pspec']+data['n6pspec'])
yt.add_field('npfrac',function=_npfrac,sampling_type=st)
def _n2pfrac(field,data):
	return meta*n_h*data['n2pspec']/(data['nspec']+data['npspec']+data['n2pspec']+data['n3pspec']+data['n4pspec']+data['n5pspec']+data['n6pspec'])
yt.add_field('n2pfrac',function=_n2pfrac,sampling_type=st)
def _n3pfrac(field,data):
	return meta*n_h*data['n3pspec']/(data['nspec']+data['npspec']+data['n2pspec']+data['n3pspec']+data['n4pspec']+data['n5pspec']+data['n6pspec'])
yt.add_field('n3pfrac',function=_n3pfrac,sampling_type=st)
def _n4pfrac(field,data):
	return meta*n_h*data['n4pspec']/(data['nspec']+data['npspec']+data['n2pspec']+data['n3pspec']+data['n4pspec']+data['n5pspec']+data['n6pspec'])
yt.add_field('n4pfrac',function=_n4pfrac,sampling_type=st)
def _n5pfrac(field,data):
	return meta*n_h*data['n5pspec']/(data['nspec']+data['npspec']+data['n2pspec']+data['n3pspec']+data['n4pspec']+data['n5pspec']+data['n6pspec'])
yt.add_field('n5pfrac',function=_n5pfrac,sampling_type=st)
def _n6pfrac(field,data):
	return meta*n_h*data['n6pspec']/(data['nspec']+data['npspec']+data['n2pspec']+data['n3pspec']+data['n4pspec']+data['n5pspec']+data['n6pspec'])
yt.add_field('n6pfrac',function=_n6pfrac,sampling_type=st)

def _ofrac(field,data):
	return meta*o_h*data['ospec']/(data['ospec']+data['opspec']+data['o2pspec']+data['o3pspec']+data['o4pspec']+data['o5pspec']+data['o6pspec']+data['o7pspec'])
yt.add_field('ofrac',function=_ofrac,sampling_type=st)
def _opfrac(field,data):
	return meta*o_h*data['opspec']/(data['ospec']+data['opspec']+data['o2pspec']+data['o3pspec']+data['o4pspec']+data['o5pspec']+data['o6pspec']+data['o7pspec'])
yt.add_field('opfrac',function=_opfrac,sampling_type=st)
def _o2pfrac(field,data):
	return meta*o_h*data['o2pspec']/(data['ospec']+data['opspec']+data['o2pspec']+data['o3pspec']+data['o4pspec']+data['o5pspec']+data['o6pspec']+data['o7pspec'])
yt.add_field('o2pfrac',function=_o2pfrac,sampling_type=st)
def _o3pfrac(field,data):
	return meta*o_h*data['o3pspec']/(data['ospec']+data['opspec']+data['o2pspec']+data['o3pspec']+data['o4pspec']+data['o5pspec']+data['o6pspec']+data['o7pspec'])
yt.add_field('o3pfrac',function=_o3pfrac,sampling_type=st)
def _o4pfrac(field,data):
	return meta*o_h*data['o4pspec']/(data['ospec']+data['opspec']+data['o2pspec']+data['o3pspec']+data['o4pspec']+data['o5pspec']+data['o6pspec']+data['o7pspec'])
yt.add_field('o4pfrac',function=_o4pfrac,sampling_type=st)
def _o5pfrac(field,data):
	return meta*o_h*data['o5pspec']/(data['ospec']+data['opspec']+data['o2pspec']+data['o3pspec']+data['o4pspec']+data['o5pspec']+data['o6pspec']+data['o7pspec'])
yt.add_field('o5pfrac',function=_o5pfrac,sampling_type=st)
def _o6pfrac(field,data):
	return meta*o_h*data['o6pspec']/(data['ospec']+data['opspec']+data['o2pspec']+data['o3pspec']+data['o4pspec']+data['o5pspec']+data['o6pspec']+data['o7pspec'])
yt.add_field('o6pfrac',function=_o6pfrac,sampling_type=st)
def _o7pfrac(field,data):
	return meta*o_h*data['o7pspec']/(data['ospec']+data['opspec']+data['o2pspec']+data['o3pspec']+data['o4pspec']+data['o5pspec']+data['o6pspec']+data['o7pspec'])
yt.add_field('o7pfrac',function=_o7pfrac,sampling_type=st)

def _nefrac(field,data):
	return meta*ne_h*data['nespec']/(data['nespec']+data['nepspec']+data['ne2pspec']+data['ne3pspec']+data['ne4pspec']+data['ne5pspec']+data['ne6pspec']+data['ne7pspec']+data['ne8pspec']+data['ne9pspec'])
yt.add_field('nefrac',function=_nefrac,sampling_type=st)
def _nepfrac(field,data):
	return meta*ne_h*data['nepspec']/(data['nespec']+data['nepspec']+data['ne2pspec']+data['ne3pspec']+data['ne4pspec']+data['ne5pspec']+data['ne6pspec']+data['ne7pspec']+data['ne8pspec']+data['ne9pspec'])
yt.add_field('nepfrac',function=_nepfrac,sampling_type=st)
def _ne2pfrac(field,data):
	return meta*ne_h*data['ne2pspec']/(data['nespec']+data['nepspec']+data['ne2pspec']+data['ne3pspec']+data['ne4pspec']+data['ne5pspec']+data['ne6pspec']+data['ne7pspec']+data['ne8pspec']+data['ne9pspec'])
yt.add_field('ne2pfrac',function=_ne2pfrac,sampling_type=st)
def _ne3pfrac(field,data):
	return meta*ne_h*data['ne3pspec']/(data['nespec']+data['nepspec']+data['ne2pspec']+data['ne3pspec']+data['ne4pspec']+data['ne5pspec']+data['ne6pspec']+data['ne7pspec']+data['ne8pspec']+data['ne9pspec'])
yt.add_field('ne3pfrac',function=_ne3pfrac,sampling_type=st)
def _ne4pfrac(field,data):
	return meta*ne_h*data['ne4pspec']/(data['nespec']+data['nepspec']+data['ne2pspec']+data['ne3pspec']+data['ne4pspec']+data['ne5pspec']+data['ne6pspec']+data['ne7pspec']+data['ne8pspec']+data['ne9pspec'])
yt.add_field('ne4pfrac',function=_ne4pfrac,sampling_type=st)
def _ne5pfrac(field,data):
	return meta*ne_h*data['ne5pspec']/(data['nespec']+data['nepspec']+data['ne2pspec']+data['ne3pspec']+data['ne4pspec']+data['ne5pspec']+data['ne6pspec']+data['ne7pspec']+data['ne8pspec']+data['ne9pspec'])
yt.add_field('ne5pfrac',function=_ne5pfrac,sampling_type=st)
def _ne6pfrac(field,data):
	return meta*ne_h*data['ne6pspec']/(data['nespec']+data['nepspec']+data['ne2pspec']+data['ne3pspec']+data['ne4pspec']+data['ne5pspec']+data['ne6pspec']+data['ne7pspec']+data['ne8pspec']+data['ne9pspec'])
yt.add_field('ne6pfrac',function=_ne6pfrac,sampling_type=st)
def _ne7pfrac(field,data):
	return meta*ne_h*data['ne7pspec']/(data['nespec']+data['nepspec']+data['ne2pspec']+data['ne3pspec']+data['ne4pspec']+data['ne5pspec']+data['ne6pspec']+data['ne7pspec']+data['ne8pspec']+data['ne9pspec'])
yt.add_field('ne7pfrac',function=_ne7pfrac,sampling_type=st)
def _ne8pfrac(field,data):
	return meta*ne_h*data['ne8pspec']/(data['nespec']+data['nepspec']+data['ne2pspec']+data['ne3pspec']+data['ne4pspec']+data['ne5pspec']+data['ne6pspec']+data['ne7pspec']+data['ne8pspec']+data['ne9pspec'])
yt.add_field('ne8pfrac',function=_ne8pfrac,sampling_type=st)
def _ne9pfrac(field,data):
	return meta*ne_h*data['ne9pspec']/(data['nespec']+data['nepspec']+data['ne2pspec']+data['ne3pspec']+data['ne4pspec']+data['ne5pspec']+data['ne6pspec']+data['ne7pspec']+data['ne8pspec']+data['ne9pspec'])
yt.add_field('ne9pfrac',function=_nepfrac,sampling_type=st)

def _nafrac(field,data):
	return meta*na_h*data['naspec']/(data['naspec']+data['napspec']+data['na2pspec'])
yt.add_field('nafrac',function=_nafrac,sampling_type=st)
def _napfrac(field,data):
	return meta*na_h*data['napspec']/(data['naspec']+data['napspec']+data['na2pspec'])
yt.add_field('napspec',function=_napfrac,sampling_type=st)
def _na2pfrac(field,data):
	return meta*na_h*data['na2pspec']/(data['naspec']+data['napspec']+data['na2pspec'])
yt.add_field('na2pspec',function=_na2pfrac,sampling_type=st)

def _mgfrac(field,data):
	return meta*mg_h*data['mgspec']/(data['mgspec']+data['mgpspec']+data['mg2pspec']+data['mg3pspec'])
yt.add_field('mgfrac',function=_mgfrac,sampling_type=st)
def _mgpfrac(field,data):
	return meta*mg_h*data['mgpspec']/(data['mgspec']+data['mgpspec']+data['mg2pspec']+data['mg3pspec'])
yt.add_field('mgpfrac',function=_mgpfrac,sampling_type=st)
def _mg2pfrac(field,data):
	return meta*mg_h*data['mg2pspec']/(data['mgspec']+data['mgpspec']+data['mg2pspec']+data['mg3pspec'])
yt.add_field('mg2pfrac',function=_mg2pfrac,sampling_type=st)
def _mg3pfrac(field,data):
	return meta*mg_h*data['mg3pspec']/(data['mgspec']+data['mgpspec']+data['mg2pspec']+data['mg3pspec'])
yt.add_field('mg3pfrac',function=_mg3pfrac,sampling_type=st)

def _sifrac(field,data):
	return meta*si_h*data['sispec']/(data['sispec']+data['sipspec']+data['si2pspec']+data['si3pspec']+data['si4pspec']+data['si5pspec'])
yt.add_field('sifrac',function=_sifrac,sampling_type=st)
def _sipfrac(field,data):
	return meta*si_h*data['sipspec']/(data['sispec']+data['sipspec']+data['si2pspec']+data['si3pspec']+data['si4pspec']+data['si5pspec'])
yt.add_field('sipfrac',function=_sipfrac,sampling_type=st)
def _si2pfrac(field,data):
	return meta*si_h*data['si2pspec']/(data['sispec']+data['sipspec']+data['si2pspec']+data['si3pspec']+data['si4pspec']+data['si5pspec'])
yt.add_field('si2pfrac',function=_si2pfrac,sampling_type=st)
def _si3pfrac(field,data):
	return meta*si_h*data['si3pspec']/(data['sispec']+data['sipspec']+data['si2pspec']+data['si3pspec']+data['si4pspec']+data['si5pspec'])
yt.add_field('si3pfrac',function=_si3pfrac,sampling_type=st)
def _si4pfrac(field,data):
	return meta*si_h*data['si4pspec']/(data['sispec']+data['sipspec']+data['si2pspec']+data['si3pspec']+data['si4pspec']+data['si5pspec'])
yt.add_field('si4pfrac',function=_si4pfrac,sampling_type=st)
def _si5pfrac(field,data):
	return meta*si_h*data['si5pspec']/(data['sispec']+data['sipspec']+data['si2pspec']+data['si3pspec']+data['si4pspec']+data['si5pspec'])
yt.add_field('si5pfrac',function=_si5pfrac,sampling_type=st)

def _sfrac(field,data):
	return meta*s_h*data['sspec']/(data['sspec']+data['spspec']+data['s2pspec']+data['s3pspec']+data['s4pspec'])
yt.add_field('sfrac',function=_sfrac,sampling_type=st)
def _spfrac(field,data):
	return meta*s_h*data['spspec']/(data['sspec']+data['spspec']+data['s2pspec']+data['s3pspec']+data['s4pspec'])
yt.add_field('spfrac',function=_spfrac,sampling_type=st)
def _s2pfrac(field,data):
	return meta*s_h*data['s2pspec']/(data['sspec']+data['spspec']+data['s2pspec']+data['s3pspec']+data['s4pspec'])
yt.add_field('s2pfrac',function=_s2pfrac,sampling_type=st)
def _s3pfrac(field,data):
	return meta*s_h*data['s3pspec']/(data['sspec']+data['spspec']+data['s2pspec']+data['s3pspec']+data['s4pspec'])
yt.add_field('s3pfrac',function=_s3pfrac,sampling_type=st)
def _s4pfrac(field,data):
	return meta*s_h*data['s4pspec']/(data['sspec']+data['spspec']+data['s2pspec']+data['s3pspec']+data['s4pspec'])
yt.add_field('s4pfrac',function=_s4pfrac,sampling_type=st)

def _cafrac(field,data):
	return meta*ca_h*data['caspec']/(data['caspec']+data['capspec']+data['ca2pspec']+data['ca3pspec']+data['ca4pspec'])
yt.add_field('cafrac',function=_cafrac,sampling_type=st)
def _capfrac(field,data):
	return meta*ca_h*data['capspec']/(data['caspec']+data['capspec']+data['ca2pspec']+data['ca3pspec']+data['ca4pspec'])
yt.add_field('capfrac',function=_capfrac,sampling_type=st)
def _ca2pfrac(field,data):
	return meta*ca_h*data['ca2pspec']/(data['caspec']+data['capspec']+data['ca2pspec']+data['ca3pspec']+data['ca4pspec'])
yt.add_field('ca2pfrac',function=_ca2pfrac,sampling_type=st)
def _ca3pfrac(field,data):
	return meta*ca_h*data['ca3pspec']/(data['caspec']+data['capspec']+data['ca2pspec']+data['ca3pspec']+data['ca4pspec'])
yt.add_field('ca3pfrac',function=_ca3pfrac,sampling_type=st)
def _ca4pfrac(field,data):
	return meta*ca_h*data['ca4pspec']/(data['caspec']+data['capspec']+data['ca2pspec']+data['ca3pspec']+data['ca4pspec'])
yt.add_field('ca4pfrac',function=_ca4pfrac,sampling_type=st)

def _fefrac(field,data):
	return meta*fe_h*data['fespec']/(data['fespec']+data['fepspec']+data['fe2pspec']+data['fe3pspec']+data['fe4pspec'])
yt.add_field('fefrac',function=_fefrac,sampling_type=st)
def _fepfrac(field,data):
	return meta*fe_h*data['fepspec']/(data['fespec']+data['fepspec']+data['fe2pspec']+data['fe3pspec']+data['fe4pspec'])
yt.add_field('fepfrac',function=_fepfrac,sampling_type=st)
def _fe2pfrac(field,data):
	return meta*fe_h*data['fe2pspec']/(data['fespec']+data['fepspec']+data['fe2pspec']+data['fe3pspec']+data['fe4pspec'])
yt.add_field('fe2pfrac',function=_fe2pfrac,sampling_type=st)
def _fe3pfrac(field,data):
	return meta*fe_h*data['fe3pspec']/(data['fespec']+data['fepspec']+data['fe2pspec']+data['fe3pspec']+data['fe4pspec'])
yt.add_field('fe3pfrac',function=_fe3pfrac,sampling_type=st)
def _fe4pfrac(field,data):
	return meta*fe_h*data['fe4pspec']/(data['fespec']+data['fepspec']+data['fe2pspec']+data['fe3pspec']+data['fe4pspec'])
yt.add_field('fe4pfrac',function=_fe4pfrac,sampling_type=st)
#--------HERE ARE THE NUMBER DENSITIES FOR MASS FRACTIONS, FOR COLUMN DENSITIES, MAKE PROJECTION PLOTS, 10^12 - 10^15 FOR THE IONS, UP TO 10^20 FOR H ----------------!!!!!!!!!!!!!!!!!!!!	

def _hdens(field,data):
	return data['ndens']*data['hspec']/(data['hspec']+data['hpspec'])
yt.add_field('hdens',function=_hdens,units="cm**-3",sampling_type=st)
def _hpdens(field,data):
	return data['ndens']*data['hpspec']/(data['hspec']+data['hpspec'])
yt.add_field('hpdens',function=_hpdens,units="cm**-3",sampling_type=st)

def _hedens(field,data):
	return data['ndens']*he_h*data['hespec']/(data['hespec']+data['hepspec']+data['he2pspec'])
yt.add_field('hedens',function=_hedens,units="cm**-3",sampling_type=st)
def _hepdens(field,data):
	return data['ndens']*he_h*data['hepspec']/(data['hespec']+data['hepspec']+data['he2pspec'])
yt.add_field('hepdens',function=_hepdens,units="cm**-3",sampling_type=st)
def _he2pdens(field,data):
	return data['ndens']*he_h*data['he2pspec']/(data['hespec']+data['hepspec']+data['he2pspec'])
yt.add_field('he2pdens',function=_he2pdens,units="cm**-3",sampling_type=st)

def _cdens(field,data):
	return data['ndens']*meta*c_h*data['cspec']/(data['cspec']+data['cpspec']+data['c2pspec']+data['c3pspec']+data['c4pspec']+data['c5pspec'])
yt.add_field('cdens',function=_cdens,units="cm**-3",sampling_type=st)
def _cpdens(field,data):
	return data['ndens']*meta*c_h*data['cpspec']/(data['cspec']+data['cpspec']+data['c2pspec']+data['c3pspec']+data['c4pspec']+data['c5pspec'])
yt.add_field('cpdens',function=_cpdens,units="cm**-3",sampling_type=st)
def _c2pdens(field,data):
	return data['ndens']*meta*c_h*data['c2pspec']/(data['cspec']+data['cpspec']+data['c2pspec']+data['c3pspec']+data['c4pspec']+data['c5pspec'])
yt.add_field('c2pdens',function=_c2pdens,units="cm**-3",sampling_type=st)
def _c3pdens(field,data):
	return data['ndens']*meta*c_h*data['c3pspec']/(data['cspec']+data['cpspec']+data['c2pspec']+data['c3pspec']+data['c4pspec']+data['c5pspec'])
yt.add_field('c3pdens',function=_c3pdens,units="cm**-3",sampling_type=st)
def _c4pdens(field,data):
	return data['ndens']*meta*c_h*data['c4pspec']/(data['cspec']+data['cpspec']+data['c2pspec']+data['c3pspec']+data['c4pspec']+data['c5pspec'])
yt.add_field('c4pdens',function=_c4pdens,units="cm**-3",sampling_type=st)
def _c5pdens(field,data):
	return data['ndens']*meta*c_h*data['c5pspec']/(data['cspec']+data['cpspec']+data['c2pspec']+data['c3pspec']+data['c4pspec']+data['c5pspec'])
yt.add_field('c5pdens',function=_c5pdens,units="cm**-3",sampling_type=st)

def _nidens(field,data):
	return data['ndens']*meta*n_h*data['nspec']/(data['nspec']+data['npspec']+data['n2pspec']+data['n3pspec']+data['n4pspec']+data['n5pspec']+data['n6pspec'])
yt.add_field('nidens',function=_nidens,units="cm**-3",sampling_type=st)
def _npdens(field,data):
	return data['ndens']*meta*n_h*data['npspec']/(data['nspec']+data['npspec']+data['n2pspec']+data['n3pspec']+data['n4pspec']+data['n5pspec']+data['n6pspec'])
yt.add_field('npdens',function=_npdens,units="cm**-3",sampling_type=st)
def _n2pdens(field,data):
	return data['ndens']*meta*n_h*data['n2pspec']/(data['nspec']+data['npspec']+data['n2pspec']+data['n3pspec']+data['n4pspec']+data['n5pspec']+data['n6pspec'])
yt.add_field('n2pdens',function=_n2pdens,units="cm**-3",sampling_type=st)
def _n3pdens(field,data):
	return data['ndens']*meta*n_h*data['n3pspec']/(data['nspec']+data['npspec']+data['n2pspec']+data['n3pspec']+data['n4pspec']+data['n5pspec']+data['n6pspec'])
yt.add_field('n3pdens',function=_n3pdens,units="cm**-3",sampling_type=st)
def _n4pdens(field,data):
	return data['ndens']*meta*n_h*data['n4pspec']/(data['nspec']+data['npspec']+data['n2pspec']+data['n3pspec']+data['n4pspec']+data['n5pspec']+data['n6pspec'])
yt.add_field('n4pdens',function=_n4pdens,units="cm**-3",sampling_type=st)
def _n5pdens(field,data):
	return data['ndens']*meta*n_h*data['n5pspec']/(data['nspec']+data['npspec']+data['n2pspec']+data['n3pspec']+data['n4pspec']+data['n5pspec']+data['n6pspec'])
yt.add_field('n5pfrac',function=_n5pdens,units="cm**-3",sampling_type=st)
def _n6pdens(field,data):
	return data['ndens']*meta*n_h*data['n6pspec']/(data['nspec']+data['npspec']+data['n2pspec']+data['n3pspec']+data['n4pspec']+data['n5pspec']+data['n6pspec'])
yt.add_field('n6pdens',function=_n6pdens,units="cm**-3",sampling_type=st)

def _odens(field,data):
	return data['ndens']*meta*o_h*data['ospec']/(data['ospec']+data['opspec']+data['o2pspec']+data['o3pspec']+data['o4pspec']+data['o5pspec']+data['o6pspec']+data['o7pspec'])
yt.add_field('odens',function=_odens,units="cm**-3",sampling_type=st)
def _opdens(field,data):
	return data['ndens']*meta*o_h*data['opspec']/(data['ospec']+data['opspec']+data['o2pspec']+data['o3pspec']+data['o4pspec']+data['o5pspec']+data['o6pspec']+data['o7pspec'])
yt.add_field('opdens',function=_opdens,units="cm**-3",sampling_type=st)
def _o2pdens(field,data):
	return data['ndens']*meta*o_h*data['o2pspec']/(data['ospec']+data['opspec']+data['o2pspec']+data['o3pspec']+data['o4pspec']+data['o5pspec']+data['o6pspec']+data['o7pspec'])
yt.add_field('o2pdens',function=_o2pdens,units="cm**-3",sampling_type=st)
def _o3pdens(field,data):
	return data['ndens']*meta*o_h*data['o3pspec']/(data['ospec']+data['opspec']+data['o2pspec']+data['o3pspec']+data['o4pspec']+data['o5pspec']+data['o6pspec']+data['o7pspec'])
yt.add_field('o3pdens',function=_o3pdens,units="cm**-3",sampling_type=st)
def _o4pdens(field,data):
	return data['ndens']*meta*o_h*data['o4pspec']/(data['ospec']+data['opspec']+data['o2pspec']+data['o3pspec']+data['o4pspec']+data['o5pspec']+data['o6pspec']+data['o7pspec'])
yt.add_field('o4pdens',function=_o4pdens,units="cm**-3",sampling_type=st)
def _o5pdens(field,data):
	return data['ndens']*meta*o_h*data['o5pspec']/(data['ospec']+data['opspec']+data['o2pspec']+data['o3pspec']+data['o4pspec']+data['o5pspec']+data['o6pspec']+data['o7pspec'])
yt.add_field('o5pdens',function=_o5pdens,units="cm**-3",sampling_type=st)
def _o6pdens(field,data):
	return data['ndens']*meta*o_h*data['o6pspec']/(data['ospec']+data['opspec']+data['o2pspec']+data['o3pspec']+data['o4pspec']+data['o5pspec']+data['o6pspec']+data['o7pspec'])
yt.add_field('o6pdens',function=_o6pdens,units="cm**-3",sampling_type=st)
def _o7pdens(field,data):
	return data['ndens']*meta*o_h*data['o7pspec']/(data['ospec']+data['opspec']+data['o2pspec']+data['o3pspec']+data['o4pspec']+data['o5pspec']+data['o6pspec']+data['o7pspec'])
yt.add_field('o7pdens',function=_o7pdens,units="cm**-3",sampling_type=st)

def _nedens(field,data):
	return data['ndens']*meta*ne_h*data['nespec']/(data['nespec']+data['nepspec']+data['ne2pspec']+data['ne3pspec']+data['ne4pspec']+data['ne5pspec']+data['ne6pspec']+data['ne7pspec']+data['ne8pspec']+data['ne9pspec'])
yt.add_field('nedens',function=_nedens,units="cm**-3",sampling_type=st)
def _nepdens(field,data):
	return data['ndens']*meta*ne_h*data['nepspec']/(data['nespec']+data['nepspec']+data['ne2pspec']+data['ne3pspec']+data['ne4pspec']+data['ne5pspec']+data['ne6pspec']+data['ne7pspec']+data['ne8pspec']+data['ne9pspec'])
yt.add_field('nepdens',function=_nepdens,units="cm**-3",sampling_type=st)
def _ne2pdens(field,data):
	return data['ndens']*meta*ne_h*data['ne2pspec']/(data['nespec']+data['nepspec']+data['ne2pspec']+data['ne3pspec']+data['ne4pspec']+data['ne5pspec']+data['ne6pspec']+data['ne7pspec']+data['ne8pspec']+data['ne9pspec'])
yt.add_field('ne2pdens',function=_ne2pdens,units="cm**-3",sampling_type=st)
def _ne3pdens(field,data):
	return data['ndens']*meta*ne_h*data['ne3pspec']/(data['nespec']+data['nepspec']+data['ne2pspec']+data['ne3pspec']+data['ne4pspec']+data['ne5pspec']+data['ne6pspec']+data['ne7pspec']+data['ne8pspec']+data['ne9pspec'])
yt.add_field('ne3pdens',function=_ne3pfrac,units="cm**-3",sampling_type=st)
def _ne4pdens(field,data):
	return data['ndens']*meta*ne_h*data['ne4pspec']/(data['nespec']+data['nepspec']+data['ne2pspec']+data['ne3pspec']+data['ne4pspec']+data['ne5pspec']+data['ne6pspec']+data['ne7pspec']+data['ne8pspec']+data['ne9pspec'])
yt.add_field('ne4pdens',function=_ne4pdens,units="cm**-3",sampling_type=st)
def _ne5pdens(field,data):
	return data['ndens']*meta*ne_h*data['ne5pspec']/(data['nespec']+data['nepspec']+data['ne2pspec']+data['ne3pspec']+data['ne4pspec']+data['ne5pspec']+data['ne6pspec']+data['ne7pspec']+data['ne8pspec']+data['ne9pspec'])
yt.add_field('ne5pdens',function=_ne5pdens,units="cm**-3",sampling_type=st)
def _ne6pdens(field,data):
	return data['ndens']*meta*ne_h*data['ne6pspec']/(data['nespec']+data['nepspec']+data['ne2pspec']+data['ne3pspec']+data['ne4pspec']+data['ne5pspec']+data['ne6pspec']+data['ne7pspec']+data['ne8pspec']+data['ne9pspec'])
yt.add_field('ne6pdens',function=_ne6pdens,units="cm**-3",sampling_type=st)
def _ne7pdens(field,data):
	return data['ndens']*meta*ne_h*data['ne7pspec']/(data['nespec']+data['nepspec']+data['ne2pspec']+data['ne3pspec']+data['ne4pspec']+data['ne5pspec']+data['ne6pspec']+data['ne7pspec']+data['ne8pspec']+data['ne9pspec'])
yt.add_field('ne7pdens',function=_ne7pdens,units="cm**-3",sampling_type=st)
def _ne8pdens(field,data):
	return data['ndens']*meta*ne_h*data['ne8pspec']/(data['nespec']+data['nepspec']+data['ne2pspec']+data['ne3pspec']+data['ne4pspec']+data['ne5pspec']+data['ne6pspec']+data['ne7pspec']+data['ne8pspec']+data['ne9pspec'])
yt.add_field('ne8pdens',function=_ne8pdens,units="cm**-3",sampling_type=st)
def _ne9pdens(field,data):
	return data['ndens']*meta*ne_h*data['ne9pspec']/(data['nespec']+data['nepspec']+data['ne2pspec']+data['ne3pspec']+data['ne4pspec']+data['ne5pspec']+data['ne6pspec']+data['ne7pspec']+data['ne8pspec']+data['ne9pspec'])
yt.add_field('ne9pdens',function=_ne9pdens,units="cm**-3",sampling_type=st)

def _nadens(field,data):
	return data['ndens']*meta*na_h*data['naspec']/(data['naspec']+data['napspec']+data['na2pspec'])
yt.add_field('nadens',function=_nadens,units="cm**-3",sampling_type=st)
def _napdens(field,data):
	return data['ndens']*meta*na_h*data['napspec']/(data['naspec']+data['napspec']+data['na2pspec'])
yt.add_field('napdens',function=_napdens,units="cm**-3",sampling_type=st)
def _na2pdens(field,data):
	return data['ndens']*meta*na_h*data['na2pspec']/(data['naspec']+data['napspec']+data['na2pspec'])
yt.add_field('na2pdens',function=_na2pdens,units="cm**-3",sampling_type=st)

def _mgdens(field,data):
	return data['ndens']*meta*mg_h*data['mgspec']/(data['mgspec']+data['mgpspec']+data['mg2pspec']+data['mg3pspec'])
yt.add_field('mgdens',function=_mgdens,units="cm**-3",sampling_type=st)
def _mgpdens(field,data):
	return data['ndens']*meta*mg_h*data['mgpspec']/(data['mgspec']+data['mgpspec']+data['mg2pspec']+data['mg3pspec'])
yt.add_field('mgpdens',function=_mgpdens,units="cm**-3",sampling_type=st)
def _mg2pdens(field,data):
	return data['ndens']*meta*mg_h*data['mg2pspec']/(data['mgspec']+data['mgpspec']+data['mg2pspec']+data['mg3pspec'])
yt.add_field('mg2pdens',function=_mg2pdens,units="cm**-3",sampling_type=st)
def _mg3pdens(field,data):
	return data['ndens']*meta*mg_h*data['mg3pspec']/(data['mgspec']+data['mgpspec']+data['mg2pspec']+data['mg3pspec'])
yt.add_field('mg3pdens',function=_mg3pdens,units="cm**-3",sampling_type=st)

def _sidens(field,data):
	return data['ndens']*meta*si_h*data['sispec']/(data['sispec']+data['sipspec']+data['si2pspec']+data['si3pspec']+data['si4pspec']+data['si5pspec'])
yt.add_field('sidens',function=_sidens,units="cm**-3",sampling_type=st)
def _sipdens(field,data):
	return data['ndens']*meta*si_h*data['sipspec']/(data['sispec']+data['sipspec']+data['si2pspec']+data['si3pspec']+data['si4pspec']+data['si5pspec'])
yt.add_field('sipdens',function=_sipdens,units="cm**-3",sampling_type=st)
def _si2pdens(field,data):
	return data['ndens']*meta*si_h*data['si2pspec']/(data['sispec']+data['sipspec']+data['si2pspec']+data['si3pspec']+data['si4pspec']+data['si5pspec'])
yt.add_field('si2pdens',function=_si2pdens,units="cm**-3",sampling_type=st)
def _si3pdens(field,data):
	return data['ndens']*meta*si_h*data['si3pspec']/(data['sispec']+data['sipspec']+data['si2pspec']+data['si3pspec']+data['si4pspec']+data['si5pspec'])
yt.add_field('si3pdens',function=_si3pdens,units="cm**-3",sampling_type=st)
def _si4pdens(field,data):
	return data['ndens']*meta*si_h*data['si4pspec']/(data['sispec']+data['sipspec']+data['si2pspec']+data['si3pspec']+data['si4pspec']+data['si5pspec'])
yt.add_field('si4pdens',function=_si4pdens,units="cm**-3",sampling_type=st)
def _si5pdens(field,data):
	return data['ndens']*meta*si_h*data['si5pspec']/(data['sispec']+data['sipspec']+data['si2pspec']+data['si3pspec']+data['si4pspec']+data['si5pspec'])
yt.add_field('si5pdens',function=_si5pdens,units="cm**-3",sampling_type=st)

def _sdens(field,data):
	return data['ndens']*meta*s_h*data['sspec']/(data['sspec']+data['spspec']+data['s2pspec']+data['s3pspec']+data['s4pspec'])
yt.add_field('sdens',function=_sdens,units="cm**-3",sampling_type=st)
def _spdens(field,data):
	return data['ndens']*meta*s_h*data['spspec']/(data['sspec']+data['spspec']+data['s2pspec']+data['s3pspec']+data['s4pspec'])
yt.add_field('spdens',function=_spdens,units="cm**-3",sampling_type=st)
def _s2pdens(field,data):
	return data['ndens']*meta*s_h*data['s2pspec']/(data['sspec']+data['spspec']+data['s2pspec']+data['s3pspec']+data['s4pspec'])
yt.add_field('s2pdens',function=_s2pdens,units="cm**-3",sampling_type=st)
def _s3pdens(field,data):
	return data['ndens']*meta*s_h*data['s3pspec']/(data['sspec']+data['spspec']+data['s2pspec']+data['s3pspec']+data['s4pspec'])
yt.add_field('s3pdens',function=_s3pdens,units="cm**-3",sampling_type=st)
def _s4pdens(field,data):
	return data['ndens']*meta*s_h*data['s4pspec']/(data['sspec']+data['spspec']+data['s2pspec']+data['s3pspec']+data['s4pspec'])
yt.add_field('s4pdens',function=_s4pdens,units="cm**-3",sampling_type=st)

def _cadens(field,data):
	return data['ndens']*meta*ca_h*data['caspec']/(data['caspec']+data['capspec']+data['ca2pspec']+data['ca3pspec']+data['ca4pspec'])
yt.add_field('cadens',function=_cadens,units="cm**-3",sampling_type=st)
def _capdens(field,data):
	return data['ndens']*meta*ca_h*data['capspec']/(data['caspec']+data['capspec']+data['ca2pspec']+data['ca3pspec']+data['ca4pspec'])
yt.add_field('capdens',function=_capdens,units="cm**-3",sampling_type=st)
def _ca2pdens(field,data):
	return data['ndens']*meta*ca_h*data['ca2pspec']/(data['caspec']+data['capspec']+data['ca2pspec']+data['ca3pspec']+data['ca4pspec'])
yt.add_field('ca2pdens',function=_ca2pdens,units="cm**-3",sampling_type=st)
def _ca3pdens(field,data):
	return data['ndens']*meta*ca_h*data['ca3pspec']/(data['caspec']+data['capspec']+data['ca2pspec']+data['ca3pspec']+data['ca4pspec'])
yt.add_field('ca3pdens',function=_ca3pdens,units="cm**-3",sampling_type=st)
def _ca4pdens(field,data):
	return data['ndens']*meta*ca_h*data['ca4pspec']/(data['caspec']+data['capspec']+data['ca2pspec']+data['ca3pspec']+data['ca4pspec'])
yt.add_field('ca4pdens',function=_ca4pdens,units="cm**-3",sampling_type=st)

def _fedens(field,data):
	return data['ndens']*meta*fe_h*data['fespec']/(data['fespec']+data['fepspec']+data['fe2pspec']+data['fe3pspec']+data['fe4pspec'])
yt.add_field('fedens',function=_fedens,units="cm**-3",sampling_type=st)
def _fepdens(field,data):
	return data['ndens']*meta*fe_h*data['fepspec']/(data['fespec']+data['fepspec']+data['fe2pspec']+data['fe3pspec']+data['fe4pspec'])
yt.add_field('fepdens',function=_fepdens,units="cm**-3",sampling_type=st)
def _fe2pdens(field,data):
	return data['ndens']*meta*fe_h*data['fe2pspec']/(data['fespec']+data['fepspec']+data['fe2pspec']+data['fe3pspec']+data['fe4pspec'])
yt.add_field('fe2pdens',function=_fe2pdens,units="cm**-3",sampling_type=st)
def _fe3pdens(field,data):
	return data['ndens']*meta*fe_h*data['fe3pspec']/(data['fespec']+data['fepspec']+data['fe2pspec']+data['fe3pspec']+data['fe4pspec'])
yt.add_field('fe3pdens',function=_fe3pdens,units="cm**-3",sampling_type=st)
def _fe4pdens(field,data):
	return data['ndens']*meta*fe_h*data['fe4pspec']/(data['fespec']+data['fepspec']+data['fe2pspec']+data['fe3pspec']+data['fe4pspec'])
yt.add_field('fe4pdens',function=_fe4pdens,units="cm**-3",sampling_type=st)
def _elecdens(field,data):
	return data['ndens']*data['elec']
yt.add_field('elecdens',function=_elecdens,units="cm**-3",sampling_type=st)
def _Cool_energy(field,data):
    return 9.4e16*cm**2*gram/second**2*(data['lh  ']+data['lhe ']+data['lc  ']+data['ln  ']+data['lo  ']+data['lne ']+data['lna ']+data['lmg ']+data['lsi ']+data['ls  ']+data['lca ']+data['lfe ']+data['lfl ']+data['lph '])#*erg/second
yt.add_field('Cool_energy',function=_Cool_energy,units="erg",sampling_type=st)

def _Upar(field,data):
    ph54 = 4.171475061185105E-005
    cj21 = 8.23e-3#2.4e-2#4.23e-3#515.0
    return cj21*ph54/data['ndens']
yt.add_field('Upar',function=_Upar,units="cm**3",sampling_type=st)

def _CoolH(field,data):
    return data['lh  ']*data['cell_mass']*erg/(second*gram)
yt.add_field('CoolH',function=_CoolH,units="erg/s",sampling_type=st)

def _CoolHe(field,data):
    return data['lhe ']*data['cell_mass']*erg/(second*gram)
yt.add_field('CoolHe',function=_CoolHe,units="erg/s",sampling_type=st)

def _CoolC(field,data):
    return data['lc  ']*data['cell_mass']*erg/(second*gram)
yt.add_field('CoolC',function=_CoolC,units="erg/s",sampling_type=st)

def _CoolN(field,data):
    return data['ln  ']*data['cell_mass']*erg/(second*gram)
yt.add_field('CoolN',function=_CoolN,units="erg/s",sampling_type=st)

def _CoolO(field,data):
    return data['lo  ']*data['cell_mass']*erg/(second*gram)
yt.add_field('CoolO',function=_CoolO,units="erg/s",sampling_type=st)

def _CoolNe(field,data):
    return data['lne ']*data['cell_mass']*erg/(second*gram)
yt.add_field('CoolNe',function=_CoolNe,units="erg/s",sampling_type=st)

def _CoolNa(field,data):
    return data['lna ']*data['cell_mass']*erg/(second*gram)
yt.add_field('CoolNa',function=_CoolNa,units="erg/s",sampling_type=st)

def _CoolMg(field,data):
    return data['lmg ']*data['cell_mass']*erg/(second*gram)
yt.add_field('CoolMg',function=_CoolMg,units="erg/s",sampling_type=st)

def _CoolSi(field,data):
    return data['lsi ']*data['cell_mass']*erg/(second*gram)
yt.add_field('CoolSi',function=_CoolSi,units="erg/s",sampling_type=st)

def _CoolS(field,data):
    return data['ls  ']*data['cell_mass']*erg/(second*gram)
yt.add_field('CoolS',function=_CoolS,units="erg/s",sampling_type=st)

def _CoolCa(field,data):
    return data['lca ']*data['cell_mass']*erg/(second*gram)
yt.add_field('CoolCa',function=_CoolCa,units="erg/s",sampling_type=st)

def _CoolFe(field,data):
    return data['lfe ']*data['cell_mass']*erg/(second*gram)
yt.add_field('CoolFe',function=_CoolFe,units="erg/s",sampling_type=st)

def _CoolFl(field,data):
    return data['lfl ']*data['cell_mass']*erg/(second*gram)
yt.add_field('CoolFl',function=_CoolFl,units="erg/s",sampling_type=st)

def _CoolPh(field,data):
    return data['lph ']*data['cell_mass']*erg/(second*gram)
yt.add_field('CoolPh',function=_CoolPh,units="erg/s",sampling_type=st)

def _CoolCt(field,data):
    return data['lct ']*data['cell_mass']*erg/(second*gram)
yt.add_field('CoolCt',function=_CoolCt,units="erg/s",sampling_type=st)

def _Cool(field,data):
    return (data['lh  ']+data['lhe ']+data['lc  ']+data['ln  ']+data['lo  ']+data['lne ']+data['lna ']+data['lmg ']+data['lsi ']+data['ls  ']+data['lca ']+data['lfe ']+data['lfl ']+data['lph ']+data['lct '])*data['cell_mass']*erg/(second*gram)
yt.add_field('Cool',function=_Cool,units="erg/s",sampling_type=st)

def _Cool_metal(field,data):
    return (data['lc  ']+data['ln  ']+data['lo  ']+data['lne ']+data['lna ']+data['lmg ']+data['lsi ']+data['ls  ']+data['lca ']+data['lfe ']+data['lfl ']+data['lph ']+data['lct '])*data['cell_mass']*erg/(second*gram)
yt.add_field('Cool_metal',function=_Cool_metal,units="erg/s",sampling_type=st)

def _E_tot(field,data): #the E_tot in the sim is specific, needs to be multiplied by cell mass -> erg
    return data['ener']*data['cell_mass']
yt.add_field('E_tot',function=_E_tot,units="erg",sampling_type=st)

def _kinetic(field,data):
    v2 = data['velx']**2 + data['vely']**2 + data['velz']**2
    v = 0.5*data['cell_mass']*v2
    return v
yt.add_field("kinetic",function=_kinetic,units="erg",sampling_type=st)

def _internal(field,data): #the internal in the sim is specific, needs to be multiplied by cell mass -> erg
    return data['eint']*data['cell_mass']
yt.add_field('internal',function=_internal,units="erg",sampling_type=st)

#result = -G*integrate.quad(lambda x: mass**2/x**2,0,r200(mass))[0]
#result2 = -(G*mass/r200(mass)+4*np.pi*G*integrate.quad(lambda x: x*rho(mass,x),0,r200(mass))[0])

def _potential(field,data):
    rad = np.sqrt((data['x'].in_units('cm'))**2+(data['y'].in_units('cm'))**2+(data['z'].in_units('cm'))**2)
    R_s = 6.608e23*cm/10.0
    rho_0 = mass*g/(4*np.pi*R_s**3*(np.log(11)-10.0/11.0)) #3.667827169075327e-25 DM central density
    ans = 4*np.pi*G*rho_0*(R_s)**3/rad*np.log(1+rad/R_s)*data['density']*data['dx']*data['dy']*data['dz']
    return ans
yt.add_field('potential',function=_potential,units="erg",sampling_type=st)
#"""
def _magnetic(field,data): 
    return np.sqrt((data['magx'].in_units('G'))**2+(data['magy'].in_units('G'))**2+(data['magz'].in_units('G'))**2)
yt.add_field('magnetic',function=_magnetic,units="uG",sampling_type=st)

def _mag_energy(field,data): 
    return ((data['magx'].in_units('G'))**2+(data['magy'].in_units('G'))**2+(data['magz'].in_units('G'))**2)*data['dx'].in_units('cm')*data['dy'].in_units('cm')*data['dz'].in_units('cm')/(8.0*np.pi)
yt.add_field('mag_energy',function=_mag_energy,units="erg",sampling_type=st)

def _mag_density(field,data): 
    return ((data['magx'].in_units('G'))**2+(data['magy'].in_units('G'))**2+(data['magz'].in_units('G'))**2)/(8.0*np.pi)*erg/(gauss**2*cm**3)
yt.add_field('mag_density',function=_mag_density,units="erg/cm**3",sampling_type=st)

def _plasma_beta(field,data): 
    return 8.0*np.pi*data['pressure']/(data['magx']**2+data['magy']**2+data['magz']**2)
yt.add_field('plasma_beta',function=_plasma_beta,sampling_type=st)

def _mag_z(field,data): #getting the magnitude
    return np.sqrt((data['magz'].in_units('G'))**2)
yt.add_field('mag_z',function=_mag_z,units="uG",sampling_type=st)

#"""
def _mass_rate(field,data):
    return data['v_rad']*data['density']*4*np.pi*(r_gal_outer**2-r_gal_inner**2)*3.086e21**2*cm**2 
yt.add_field('mass_rate',function=_mass_rate,units="Msun/yr",sampling_type=st)

def _mass_rate2(field,data):
    return data['v_rad']*data['density']*(2*data['dx']*data['dz']+2*data['dx']*data['dy']+2*data['dz']*data['dy'])
yt.add_field('mass_rate2',function=_mass_rate2,units="Msun/yr",sampling_type=st)

def _angular_y(field,data):
    return abs(data['cell_mass']*(data['z'].in_units('cm')*data['velx'].in_units('cm/s')-data['x'].in_units('cm')*data['velz'].in_units('cm/s')))
yt.add_field('angular_y',function=_angular_y,units="Msun*kpc*km/s",sampling_type=st)

def _angular_mom(field,data):
    return abs(data['z'].in_units('cm')*data['velx'].in_units('cm/s')-data['x'].in_units('cm')*data['velz'].in_units('cm/s'))
yt.add_field('angular_mom',function=_angular_mom,units="kpc*km/s",sampling_type=st)

def _angular_y_rate(field,data):
    return data['v_rad']*data['density']*(data['z'].in_units('cm')*data['velx'].in_units('cm/s')-data['x'].in_units('cm')*data['velz'].in_units('cm/s'))*4*np.pi*(r_gal_outer**2-r_gal_inner**2)*3.086e21**2*cm**2 
yt.add_field('angular_y_rate',function=_angular_y_rate,units="g*cm**2/s**2",sampling_type=st)

def _x_z_dist(field,data):
    return np.sqrt(data['x']**2 + data['z']**2)
yt.add_field('x_z_dist',function=_x_z_dist,units='kpc',sampling_type=st)
#"""

#def _rm_z(field,data):
#    return data['mag_z']*data['elecdens']*812*cm**3/(3.086e21*m**3)
#yt.add_field('rm_z',function=_rm_z,units="uG/m**3",sampling_type=st)

#def _tot_pres(field,data): 
#    return data['pressure'] + (data['magx']**2+data['magy']**2+data['magz']**2)/(8.0*np.pi)
#yt.add_field('tot_pres',function=_tot_pres,units="dyn/cm**2",sampling_type=st)

#"""
def _free_time(field,data):
	C = 10.0
	Rs = 6.608e23*cm/C
	rad = np.sqrt((data['x'].in_units('cm'))**2+(data['y'].in_units('cm'))**2+(data['z'].in_units('cm'))**2)
	third = 1.0+C
	forth = (rad/(rad+Rs)-np.log(1+rad/Rs))*3.086e21*cm/rad**3
	second = (np.log(third)-C/(third))
	first = -G*mass*gram*forth/(second)
	return np.sqrt(2.0*rad/first)
yt.add_field('free_time',function=_free_time,units="Gyr",sampling_type=st)

def _cool_time(field,data):
    ans = data['internal']/data['Cool']
    return ans
yt.add_field('cool_time',function=_cool_time,units="Gyr",sampling_type=st)

def _ratio(field,data):
    return data['cool_time']/data["free_time"]
yt.add_field('ratio',function=_ratio,sampling_type=st)

def _Si4_O6(field,data):
    return data['si3pfrac']/data['o5pfrac']
yt.add_field('Si4_O6',function=_Si4_O6,sampling_type=st)

def _N5_O6(field,data):
    return data['n4pfrac']/data['o5pfrac']
yt.add_field('N5_O6',function=_N5_O6,sampling_type=st)

def _C4_O6(field,data):
    return data['c3pfrac']/data['o5pfrac']
yt.add_field('C4_O6',function=_C4_O6,sampling_type=st)

"""
def kepler(names,colors,file_name):
    shells = 25
    r = np.arange(0,230,9)
    x = np.arange(4.5,229,9)
    for p in range(2):
	    p = p + 2#items[p]
	    if p==0 :
            ii = [37]#np.arange(0,38,1)
            path = pathh[1]
	    if p==1 :
           ii =[38]#np.arange(0,77,1)
           path = pathh[2]
	    if p==2 :
           ii = [38]#np.arange(38,39,1)
           path = pathh[4]
	    if p==3 :
           ii = [38]#np.arange(38,39,1)
           path = pathh[4]
	    for i in range(len(ii)):
	       i=i+0
	       if ii[i] < 100:
		       stuff = yt.load(path+names[p]+"/ISM_hdf5_chk_00"+str(ii[i]))
		       sim = stuff.sphere([0.5,0.5,0.5],(225,"kpc"))
	       elif 100 <= ii[i]:
		       stuff = yt.load(path+names[p]+"/ISM_hdf5_chk_0"+str(ii[i]))
		       sim = stuff.sphere([0.5,0.5,0.5],(225,"kpc"))   
	       mass = sim['cell_mass'].v.flatten()
	       rad = sim['radius'].v.flatten()
	       sim_vel = sim['velocity'].v.flatten()
	       dens = sim['density'].v.flatten()
	       binned_mass = 0
	       kep = []
	       turb = []
	       turb_en = []
	       
	       for j in range(shells):
		   a = np.sum([mass[b] for b, y in enumerate(rad) if r[j] <= abs(y) < r[j+1]])
		   #d = [dens[b] for b, w in enumerate(rad) if r[j] <= abs(w) < r[j+1]]
		   binned_mass += a
		   kep.append(G*binned_mass/(x[j]*3.086e21))#/1.e5
		   c = [(sim_vel[b]**2 - kep[j]) for b, v in enumerate(rad) if r[j] <= abs(v) < r[j+1]]
		   t = np.array(c) - kep[j]
		   t = np.sqrt(t)
		   turb.append(np.average(t))
		   c = np.sqrt(c)
		   turb_en.append(np.average(c))
		   #e = np.array(d)/2.*(t*1.e5)**2
		   #turb_en.append(e)
               #turb = [m for n in turb for m in n]
               #turb_en = [m for n in turb_en for m in n]
               #print len(turb), len(turb_en)
               #everything = []
               #cave = np.average(turb_en)
               #caves = np.average(turb)
               #everything.append((stuff.current_time.v, cave, caves))
               #print('HERE', everything)
               #if i==0:
               #    fout = open("turbulent"+file_name[p]+".dat","w")
               #    fout.write("Time(s) thing1(erg/cm^-3) turb(km/s) \n")
               #    for k in energies:    
               #        fout.write(" ".join(map(str, k)) + "\n") 
               #else:
               #    for k in energies:    
               #        fout.write(" ".join(map(str, k)) + "\n") 
            #fout.close()
               #plt.plot(x,turb_en,color=colors[p],linestyle='-')
               #plt.yscale('log')
               plt.plot(x,np.sqrt(kep)/1.e5,color=colors[p],linestyle='-.',label='kepler')
               plt.plot(x,np.array(turb_en)/1.e5,color=colors[p],linestyle='-',label='velocity')
               plt.plot(x,np.array(turb)/1.e5,color=colors[p],linestyle='--',label='turb')
               plt.legend(loc='upper right',frameon=False,ncol=2,prop={'size': 14})
    plt.savefig('test2.png')
    return 
"""       
def pickle_dump(names,file_names): #takes a input list of names and checkpoint files
        """
        Makes pickle files from the column density projections of various ions, good for comparing agains t data. axis = 0 for x, axis = 1 for y, axis = 2 for z projections
	"""
        axis = 2
        for p in range(1):
            p = 0
            if p==0 :
                ii = [63,96]#np.arange(0,38,1)
                path = "/Volumes/research1/MAIHEM_runs/Z1/High_rot_z1"
            if p==1 :
                ii =[38]#np.arange(0,77,1)
                path = pathh[2]
            if p==2 :
                ii = [99]#np.arange(38,39,1)
                path = pathh[0]
            if p==3 :
                ii =np.arange(38,39,1)
                path = pathh[4]
            for i in range(len(ii)):
               i=i+0
               stuff = yt.load(path+"/ISM_hdf5_chk_00"+str(ii[i]))
#    prj1 = yt.SlicePlot(stuff,'z',('gas','radius'),center=[0.5,0.5,0.5],width=(400,'kpc'))#,data_source=my_sphere)
#    rad = prj1.frb['gas','radius']
#    prj1.save()
               prj1 = yt.ProjectionPlot(stuff,axis,('gas','o5pdens'),weight_field=None,center=[0.5,0.5,0.5],width=(400,'kpc'))#data_source=my_sphere)
               o5pden = prj1.frb['gas','o5pdens']
               prj2 = yt.ProjectionPlot(stuff,axis,('gas','n4pdens'),weight_field=None,center=[0.5,0.5,0.5],width=(400,'kpc'))#data_source=my_sphere)
               n4pden = prj2.frb['gas','n4pdens']
               prj3 = yt.ProjectionPlot(stuff,axis,('gas','si3pdens'),weight_field=None,center=[0.5,0.5,0.5],width=(400,'kpc'))#data_source=my_sphere)
               si3pden = prj3.frb['gas','si3pdens']
               prj4 = yt.ProjectionPlot(stuff,axis,('gas','si2pdens'),weight_field=None,center=[0.5,0.5,0.5],width=(400,'kpc'))#data_source=my_sphere)
               si2pden = prj4.frb['gas','si2pdens']
               prj5 = yt.ProjectionPlot(stuff,axis,('gas','sipdens'),weight_field=None,center=[0.5,0.5,0.5],width=(400,'kpc'))#,data_source=my_sphere)
               sipden = prj5.frb['gas','sipdens']
               prj6 = yt.ProjectionPlot(stuff,axis,('gas','hdens'),weight_field=None,center=[0.5,0.5,0.5],width=(400,'kpc'))#,data_source=my_sphere)
               hden = prj6.frb['gas','hdens']
               prj8 = yt.ProjectionPlot(stuff,axis,('gas','c3pdens'),weight_field=None,center=[0.5,0.5,0.5],width=(400,'kpc'))#,data_source=my_sphere)
               c3pden = prj8.frb['gas','c3pdens']
               prj9 = yt.ProjectionPlot(stuff,axis,('gas','mgpdens'),weight_field=None,center=[0.5,0.5,0.5],width=(400,'kpc'))#data_source=my_sphere)
               mgpden = prj9.frb['gas','mgpdens']
               prj10 = yt.ProjectionPlot(stuff,axis,('gas','temperature'),weight_field='density',center=[0.5,0.5,0.5],width=(400,'kpc'))#data_source=my_sphere)
               temp = prj10.frb['gas','temperature']
               prj7 = yt.ProjectionPlot(stuff,axis,('gas','ndens'),weight_field='ndens',center=[0.5,0.5,0.5],width=(400,'kpc'))#data_source=my_sphere)
               ndens = prj7.frb['gas','ndens']
               prj11 = yt.ProjectionPlot(stuff,axis,('gas','cpdens'),weight_field=None,center=[0.5,0.5,0.5],width=(400,'kpc'))#data_source=my_sphere)
               cpden = prj11.frb['gas','cpdens']
               prj12 = yt.ProjectionPlot(stuff,axis,('gas','c2pdens'),weight_field=None,center=[0.5,0.5,0.5],width=(400,'kpc'))#data_source=my_sphere)
               c2pden = prj12.frb['gas','c2pdens']
               prj13 = yt.ProjectionPlot(stuff,axis,('gas','fepdens'),weight_field=None,center=[0.5,0.5,0.5],width=(400,'kpc'))#data_source=my_sphere)
               fepden = prj13.frb['gas','fepdens']
               prj14 = yt.ProjectionPlot(stuff,axis,('gas','fe2pdens'),weight_field=None,center=[0.5,0.5,0.5],width=(400,'kpc'))#data_source=my_sphere)
               fe2pden = prj14.frb['gas','fe2pdens']
               prj15 = yt.ProjectionPlot(stuff,axis,('gas','npdens'),weight_field=None,center=[0.5,0.5,0.5],width=(400,'kpc'))#data_source=my_sphere)
               npden = prj15.frb['gas','npdens']
               prj16 = yt.ProjectionPlot(stuff,axis,('gas','odens'),weight_field=None,center=[0.5,0.5,0.5],width=(400,'kpc'))#data_source=my_sphere)
               oden = prj16.frb['gas','odens']
               prj17 = yt.ProjectionPlot(stuff,axis,('gas','ne7pdens'),weight_field=None,center=[0.5,0.5,0.5],width=(400,'kpc'))#data_source=my_sphere)
               ne8den = prj16.frb['gas','ne7pdens']
               print('picklin')
               out = open("stuff4High_rot_z1"+str(axis)+".pkl",'wb') 
               pickle.dump([o5pden,n4pden,si3pden,si2pden,sipden,hden,c3pden,mgpden,temp,ndens,cpden,c2pden,fepden,fe2pden,npden,oden,ne8den],out)
               out.close()
               print('pickling 4 this is done')
        return
	
def pickle_dump2(names,file_names): #takes a input list of names and checkpoint files
        """
        Makes pickle files from the column density projections of various ions, good for comparing against data. axis = 0 for x, axis = 1 for y, axis = 2 for z projections
	"""
        thing = ['x','y','z','v_rad','angular_y','mymachnumber','cell_mass','ratio','cool_time','magnetic','temperature','density','pressure']
        items = [2,3]
        for p in range(1):
            p = 2#items[p]
            if p==0 :
                ii = [37]#np.arange(0,38,1)
                path = pathh[1]
            if p==1 :
                ii =[38]#np.arange(0,77,1)
                path = pathh[2]
            if p==2 :
                ii = [79]#np.arange(38,39,1)
                path = pathh[2]
            if p==3 :
                ii = [38,74,119]#np.arange(38,39,1)
                path = pathh[0]
            for i in range(len(ii)):
               i=i+0
               if ii[i] < 100:
	               stuff = yt.load(path+names[p]+"/ISM_hdf5_chk_00"+str(ii[i]))
        	       sim = stuff.sphere([0.5,0.5,0.5],(214,"kpc"))
               elif 100 <= ii[i]:
	               stuff = yt.load(path+names[p]+"/ISM_hdf5_chk_0"+str(ii[i]))
        	       sim = stuff.sphere([0.5,0.5,0.5],(214,"kpc"))     	
               a = sim[thing[0]].v.flatten()
               b = sim[thing[1]].v.flatten()
               c = sim[thing[2]].v.flatten()
               d = sim[thing[3]].v.flatten()
               e = sim[thing[4]].v.flatten()
               f = sim[thing[5]].v.flatten()
               g = sim[thing[6]].v.flatten()
               h = sim[thing[7]].v.flatten()
               j = sim[thing[8]].v.flatten()
               k = sim[thing[9]].v.flatten()
               l = sim[thing[10]].v.flatten()
               m = sim[thing[11]].v.flatten()
               n = sim[thing[12]].v.flatten()
               print('picklin')
               out = open("stuff4"+file_names[p]+str(ii[i])+".pkl",'wb') 
               pickle.dump([a,b,c,d,e,f,g,h,j,k,l,m,n],out)
               out.close()
               print('pickling 4 '+str(ii[i])+' '+file_names[p]+' is done')
        return
        
def read_pickle(file_names,colors,legend):
  """
  Reads in pickle files to plot things like velocity vs radii
  """  
  interest = ["x","y","z","V_{radial} \ km\ s^{-1}","| L_{y} | \ g\ cm^{2}\ s^{-1}","3D\  Mach\ Number","Mass\ (g)","t_{cool}/t_{ff}","Cooling\ Time\ (Gyr)","Magnetic\ Field\ (\mu G)","Temperature\ (K)", "Density\ (g\ cm^{-3)","Pressure\ (cgs\ units)"]
  names = ['x','y','z','','angular_y','mymachnumber','cell_mass','ratio','cool_time','magnetic','temperature','density','pressure']
  items = [2,3,4]
  bins = 21
  a = 214./bins
  r = np.arange(0,251,12)
  sub = ['3gyr','6gyr','9gyr']
  styles = ['-','--','-.']
  thing = 3
  for i in range(1):
    i = 1 #controls the year
    for p in range(2):
        p = items[p]
        if p==0 :
                #legend = ["High"]
                ii = [37]
                path = pathh[4]
        if p==1 :
                #legend = ["MHD"]
                path = pathh[2]
        if p==2 :
                #legend = ["Rot\ 3\ Gyr","Rot\ 6\ Gyr","Rot\ 9\ Gyr"]
                ii = [79,120]
                path = pathh[4]
        if p==3 :
                #legend = ["MHD\ Rot\ 3\ Gyr","MHD\ Rot\ 6\ Gyr","MHD\ Rot\ 9\ Gyr"]
                ii = [76,119]
                path = pathh[4]
        if p==4 :
                ii = [0,0,125]
                path = pathh[4]
        print(file_names[p], ii, i)
        fil = open(path+"pkls/stuff4"+file_names[p]+str(ii[i])+".pkl",'r') 
        model = pickle.load(fil)
        fil.close()
        everything = []
        avg = []
        sums = []
        for jj in range(len(model)):
                   ans = np.array(model[jj])
                   everything.append(abs(ans))
        rad = np.sqrt(everything[0]**2+everything[1]**2+everything[2]**2)/3.086e21
        #print 'hERE', interest[thing], everything[thing][0:10], 'rads', max(rad), rad[0:10]
        for kk in range(bins):
                       works = 0
                       works = [everything[thing][b] for b, x in enumerate(rad) if a*kk <= x < a*(kk+1)]
                       works = [works[b] for b, x in enumerate(works) if np.isfinite(x)] #np.array(works)#*value[thing]
                       avg.append(works)
                       #sums.append(abs(sum(works)))
        avg = np.array(avg)#; sums = np.array(sums)
        #print avg
        stuff = np.zeros((3,len(avg)))
        percentile=[0,50,100]
        for o in range(len(avg)):
               ans = np.percentile(avg[o],percentile)
               stuff[0,o],stuff[1,o],stuff[2,o]= abs(ans[0]-ans[1]),ans[1],abs(ans[2]-ans[1])
               #print 'HERE', ans
        plt.plot(r,stuff[1,:],color=colors[p],label=r'$\rm \bf '+legend[p]+'$')
        #plt.plot(r,sums,color=colors[p],label=r'$\rm \bf '+legend[p]+'$')
        plt.fill_between(r,stuff[1,:]-stuff[0,:],stuff[2,:]+stuff[1,:],color=colors[p],alpha=0.2)
    plt.xlabel(r'$\rm \bf r \ (kpc)$',**axis_font)
    plt.ylabel(r"$\rm \bf "+interest[thing]+"$",**axis_font)
    plt.xlim([0,250]) #these limits are for impact parameter
    plt.ylim([-350,350]) 
    #plt.ylim([1e-4,1e2])
    #plt.ylim([1e60,1e71])
    plt.axhline(y=0,linestyle='--',lw=2,color='grey')
    #plt.yscale('log')
    plt.savefig(names[thing]+'_'+sub[i]+'_mhd_rot_compare.png',bbox_inches='tight')#plt.show()
    plt.close()
    print('1 down')
    return

def make_slices():
    """
    Makes slices through the simulation domain

    """
    path = "/scratch/ebuie/ISO_Turb/midway/new_mhd/"
    ylabels = ['N_{O VI} \ (cm^{-2})','N_{N V} \ (cm^{-2})','N_{Si IV} \ (cm^{-2})','N_{Si III} \ (cm^{-2})','N_{Si II} \ (cm^{-2})','N_{H I} \ (cm^{-2})','N_{C IV} \ (cm^{-2})','N_{Mg II} \ (cm^{-2})','N_{C II} \ (cm^{-2})','N_{C III} \ (cm^{-2})','N_{Fe II} \ (cm^{-2})','N_{Fe III} \ (cm^{-2})','N_{N II} \ (cm^{-2})','N_{O I} \ (cm^{-2})','Temperature\ (K)','Number\ Density\ (cm^{-3})','Entropy \ (cm^{-2})','Pressure_{turb} (dyn\ cm^{-2})','B_{||}\ (\mu G)','Internal','Plasma\ Beta','N_{O I} \ (cm^{-2})','N_{Ne VIII} \ (cm^{-2})']
    field = ['Upar','n4pdens','si3pdens','si2pdens','sipdens','hdens','c3pdens','mgpdens','cpdens','c2pdens','fepdens','fe2pdens','npdens','odens','temperature','ndens','entropy','pres_turb','magnetic','internal','plasma_beta','velocity','v_rad','sigma'] #
    lim1 = [2e16,2e16,2e16,2e16,2e16,2e20,2e16,2e16,2e16,2e16,2e16,2e16,2e16,2e16,2e7,1e0,2e2,5e-13,2e1,1e55,200,2e39,500,200]
    lim2 = [8e12,8e12,8e12,8e12,8e12,8e12,8e12,8e12,8e12,8e12,8e12,8e12,8e12,8e12,8e3,8e-6,8e-2,5e-15,5e-3,1e51,8e-3,8e34,2e-1,-200]  
    #1 is face on corresponding to x-z plane, 2 is edgeon corresponding to x-y plane
    axis = 1
    for p in range(1):
        p = p + 8
        avg_dens = []
        if p==0 :
            titles = ['NF Ideal MHD 3 Gyr','NF Ideal MHD 4 Gyr','NF Ideal MHD 5 Gyr','NF Ideal MHD 6 Gyr','NF Ideal MHD 7 Gyr' ]
            ii = [32,43,53,65,76]
        if p==1 :
            titles = ['NF Non-Ideal MHD 3 Gyr','NF Non-Ideal MHD 4 Gyr','NF Non-Ideal MHD 5 Gyr','NF Non-Ideal MHD 6 Gyr','NF Non-Ideal MHD 7 Gyr' ]
            ii = [32,41,51,62,73]
            path = "research1/MAIHEM_Z1/pretty_gal_nonideal"
        if p==2:
            titles = ['NF Hydro 3 Gyr','NF Hydro 4 Gyr','NF Hydro 5 Gyr','NF Hydro 6 Gyr','NF Hydro 7 Gyr' ]
            ii = [31,40,50,61,71]#np.arange(38,39,1)
            path = "research1/MAIHEM_Z1/pretty_gal_hydro"
        if p==3 :
            titles = ['Z_{\odot}\ 4 Gyr','Z_{\odot}\ 5 Gyr','Z_{\odot}\ 6 Gyr','Z_{\odot}\ 7 Gyr', 'Z_{\odot}\ 7.6 Gyr']#,'Z1 8 Gyr','Z1 9 Gyr' ]
            ii = [42,53,63,74,80]#,85,96]
            name = "MHD_Feedback_Z1"
        if p==4 :
            titles = ['0.3Z_{\odot}\ 3 Gyr','0.3Z_{\odot}\ 4 Gyr','0.3Z_{\odot}\ 5 Gyr','0.3Z_{\odot}\ 6 Gyr','0.3Z_{\odot}\ 7 Gyr', '0.3Z_{\odot}\ 7.6 Gyr']
            ii = [38,50,63,76,89,98]#103,117]
            path="/Volumes/research2/MAIHEM_Z03/"
            name = "MHD_Feedback_Z03"
        if p==5 :
            titles = ['0.7Z_{\odot}\ 3 Gyr','0.7Z_{\odot}\ 4 Gyr','0.7Z_{\odot}\ 5 Gyr','0.7Z_{\odot}\ 6 Gyr','0.7Z_{\odot}\ 7 Gyr', '0.7Z_{\odot}\ 7.6 Gyr']
            ii = [41,52,63,75,83]
            path="/Volumes/research1/MAIHEM_Z07/"
            name = "MHD_Feedback_Z07"  
        if p==6 :
            titles = ['Z_{\odot}\ low\ 4 Gyr','Z_{\odot}\ low\ 5 Gyr','Z_{\odot}\ low\ 6 Gyr','Z_{\odot}\ low\ 7 Gyr', 'Z_{\odot}\ low\ 7.6 Gyr']
            ii = [42,52,63,73,80]
            path="/Volumes/research1/MAIHEM_Z1/"
            name = "MHD_low_Feedback_Z1"
        if p==7 :
            titles = ['0.3Z_{\odot}\ low\ 4 Gyr','0.3Z_{\odot}\ low\ 5 Gyr','0.3Z_{\odot}\ low\ 6 Gyr','0.3Z_{\odot}\ low\ 7 Gyr', '0.3Z_{\odot}\ low\ 7.6 Gyr']
            ii = [41,52,62,73,80]
            path="/Volumes/research2/MAIHEM_Z03/"
            name = "MHD_low_Feedback_Z03"
        if p==8:
            titles=['stuff']
            ii = np.arange(0,13,1)
            name = "1E23_S100_z1_mhd"
        for i in range(len(ii)):
            i=i+0
            if ii[i]< 10:
                stuff = yt.load(path+name+"/ISM_hdf5_chk_000"+str(ii[i]))
            elif 10 < ii[i] < 100:
                stuff = yt.load(path+name+"/ISM_hdf5_chk_00"+str(ii[i]))
            elif 100 <= ii[i]:
                stuff = yt.load(path+name+"/ISM_hdf5_chk_0"+str(ii[i]))
            #print(titles[i])
            for j in range(1): 
#OVI(0), NV(1), SiIV(2), SiIII(3), SiII(4), HI(5), CIV(6), MgII(7), CII(8), CIII(9), FeII(10), FeIII(11), NII(12), OI(13), temp(14), ndens(15), entropy(16), pressure(17), magnetic(18), internal(19), plasma_beta(20), velocity(21), v_radial(22), sigma(23) 
               j = j + 18
               print(field[j])
               slc = yt.SlicePlot(stuff,axis,field[j],center=[0.5,0.5,0.5],width=(100,'pc'))
               #slc.set_units(field[j],'uG')
               #slc.set_log(field[j], False)
               slc.set_zlim(field[j],lim1[j],lim2[j])
               #slc.annotate_sphere([0.5, 0.5, 0.5], radius=(214, 'kpc'), circle_args={'color':'black'})
               #slc.set_cmap(field=field[j],cmap='RdBu')
               #slc.set_cmap(field=field[j],cmap='RED-PURPLE')
               #slc.set_cmap(field=field[j],cmap='RED TEMPERATURE_r')
               #slc.annotate_timestamp(draw_inset_box=True)
               #slc.annotate_quiver('velx','velz')
               #slc.annotate_streamlines('velx', 'velz')
               #slc.annotate_line_integral_convolution(('magx'),('magz'), lim=(0.45, 0.62))
               #slc.annotate_streamlines('magx', 'magz')
               #slc.annotate_quiver('magx','magz',factor=15)
               slc.set_font({'style':'normal','weight':'bold','size':'32'})
               #slc.annotate_text((-220, 200), r"$\rm "+titles[i]+"$", coord_system="plot")
               slc.annotate_timestamp(draw_inset_box=True)
               #slc.set_colorbar_label(field[j],r"$\rm "+ylabels[j]+"$")
               #slc.set_log(field[j], True, linthresh=1.e1)
               #slc.annotate_scale(coeff=30,unit='kpc')
               slc.save("Slice_"+name+"_"+field[j]+"_"+str(ii[i])+"axis_"+str(axis)+".png")
               #ln = 'mv ISM*png '+path+name[p]+'/'
               #subprocess.check_output(ln,shell=True)
	
    return
    	
def make_projs():
    """
    Makes projections through the simulation domain
    
    """
    ylabels = ['N_{O VI} \ (cm^{-2})','N_{N V} \ (cm^{-2})','N_{Si IV} \ (cm^{-2})','N_{Si III} \ (cm^{-2})','N_{Si II} \ (cm^{-2})','N_{H I} \ (cm^{-2})','N_{C IV} \ (cm^{-2})','N_{Mg II} \ (cm^{-2})','N_{C II} \ (cm^{-2})','N_{C III} \ (cm^{-2})','N_{Fe II} \ (cm^{-2})','N_{Fe III} \ (cm^{-2})','N_{N II} \ (cm^{-2})','N_{O I} \ (cm^{-2})','Temperature\ (K)','Number\ Density\ (cm^{-3})','Entropy\ (cm^{2} \dot keV)','Pressure\ (dyn \dot cm^{-2})','Magnetic\ (\mu G)','Internal','P_{Th}/P_{B}']
    #1 is face on corresponding to x-z plane, 2 is edgeon corresponding to x-y plane 
    axis = 1
    for p in range(2):
        p = p + 6
        avg_dens = []
        if p==0 :
            titles = ['NF Ideal MHD 3 Gyr','NF Ideal MHD 4 Gyr','NF Ideal MHD 5 Gyr','NF Ideal MHD 6 Gyr','NF Ideal MHD 7 Gyr' ]
            ii = [32,43,53,65,76]
            path = "research1/MAIHEM_Z1/pretty_gal"
        if p==1 :
            titles = ['NF Non-Ideal MHD 3 Gyr','NF Non-Ideal MHD 4 Gyr','NF Non-Ideal MHD 5 Gyr','NF Non-Ideal MHD 6 Gyr','NF Non-Ideal MHD 7 Gyr' ]
            ii = [32,41,51,62,73]
            path = "research1/MAIHEM_Z1/pretty_gal_nonideal"
        if p==2:
            titles = ['NF Hydro 3 Gyr','NF Hydro 4 Gyr','NF Hydro 5 Gyr','NF Hydro 6 Gyr','NF Hydro 7 Gyr' ]
            ii = [31,40,50,61,71]#np.arange(38,39,1)
            path = "research1/MAIHEM_Z1/pretty_gal_hydro"
        if p==3 :
            titles = ['Z_{\odot}\ 4 Gyr','Z_{\odot}\ 5 Gyr','Z_{\odot}\ 6 Gyr','Z_{\odot}\ 7 Gyr', 'Z_{\odot}\ 7.6 Gyr']#,'Z1 8 Gyr','Z1 9 Gyr' ]
            ii = [42,53,63,74,80]#,85,96]
            path="/Volumes/research1/MAIHEM_Z1/"
            name = "MHD_Feedback_Z1"
        if p==4 :
            titles = ['0.3Z_{\odot}\ 3 Gyr','0.3Z_{\odot}\ 4 Gyr','0.3Z_{\odot}\ 5 Gyr','0.3Z_{\odot}\ 6 Gyr','0.3Z_{\odot}\ 7 Gyr', '0.3Z_{\odot}\ 7.6 Gyr']
            ii = [38,50,63,76,89,98]#103,117]
            path="/Volumes/research2/MAIHEM_Z03/"
            name = "MHD_Feedback_Z03"
        if p==5 :
            titles = ['0.7Z_{\odot}\ 3 Gyr','0.7Z_{\odot}\ 4 Gyr','0.7Z_{\odot}\ 5 Gyr','0.7Z_{\odot}\ 6 Gyr','0.7Z_{\odot}\ 7 Gyr', '0.7Z_{\odot}\ 7.6 Gyr']
            ii = [41,52,63,75,83]
            path="/Volumes/research1/MAIHEM_Z07/"
            name = "MHD_Feedback_Z07"      
        if p==6 :
            titles = ['Z_{\odot}\ low\ 4 Gyr','Z_{\odot}\ low\ 5 Gyr','Z_{\odot}\ low\ 6 Gyr','Z_{\odot}\ low\ 7 Gyr', 'Z_{\odot}\ low\ 7.6 Gyr']
            ii = [42,52,63,73,80]
            path="/Volumes/research1/MAIHEM_Z1/"
            name = "MHD_low_Feedback_Z1"
        if p==7 :
            titles = ['0.3Z_{\odot}\ low\ 4 Gyr','0.3Z_{\odot}\ low\ 5 Gyr','0.3Z_{\odot}\ low\ 6 Gyr','0.3Z_{\odot}\ low\ 7 Gyr', '0.3Z_{\odot}\ low\ 7.6 Gyr']
            ii = [41,52,62,73,80]
            path="/Volumes/research2/MAIHEM_Z03/"
            name = "MHD_low_Feedback_Z03"
        for i in range(len(ii)):
            i=i + 0
            if ii[i]< 10:
                stuff = yt.load(""+path+name+"/ISM_hdf5_chk_000"+str(ii[i]))
            elif 10 < ii[i] < 100:
                stuff = yt.load(""+path+name+"/ISM_hdf5_chk_00"+str(ii[i]))
            elif 100 <= ii[i]:
                stuff = yt.load(""+path+name+"/ISM_hdf5_chk_0"+str(ii[i]))
            print(titles[i])
            sim = stuff.sphere([0.5,0.5,0.5],(225,"kpc"))
            #sim = sim.cut_region(["obj['temperature'] < 5.0e4"])
            #Various fields
            #OVI(0), NV(1), SiIV(2), SiIII(3), SiII(4), HI(5), CIV(6), MgII(7), CII(8), CIII(9), FeII(10), FeIII(11), NII(12), OI(13), temp(14), ndens(15), entropy(16), pressure(17), magnetic(18), internal(19), plasma beta(20), mag energy(21), mag density(22), 1D velocity(23)
            field = ['o5pdens','n4pdens','si3pdens','si2pdens','sipdens','hdens','c3pdens','mgpdens','cpdens','c2pdens','fepdens','fe2pdens','npdens','odens','temperature','ndens','entropy','pressure','magnetic','internal','plasma_beta','mag_energy','mag_density','v_rad'] #
            #Limits for the aformentioned fields. If plotting elements, these limits are for projected column densities (ProjectionPlot)
            lim1 = [2e16,2e16,2e16,2e16,2e16,2e20,2e16,2e16,2e16,2e16,2e16,2e16,2e16,2e16,2e7,1e0,2e2,5e-13,2e1,1e55,1.5e2] #
            lim2 = [8e12,8e12,8e12,8e12,8e12,8e12,8e12,8e12,8e12,8e12,8e12,8e12,8e12,8e12,8e3,8e-6,8e-2,5e-15,5e-3,1e51,8e-3]  #]
            for j in range(1): 
                j = j+20
                slc = yt.ProjectionPlot(stuff,axis,field[j],center=[0.5,0.5,0.5],width=(450,'kpc'),weight_field='density')
                slc.set_zlim(field[j],lim1[j],lim2[j])
                #slc.set_unit("mass","Msun")
                
                #slc.set_zlim("mymachnumber",1e0,1e2)
                #slc.set_xlim(1e2, 1e6)
                #slc.set_ylim(1e-29, 1e-23)
                #slc.annotate_sphere([0.5, 0.5, 0.5], radius=(214, 'kpc'), circle_args={'color':'black'})
                #slc.annotate_timestamp(draw_inset_box=True)
                slc.set_cmap(field=field[j],cmap='GnBu') #for projections
                slc.set_font({'style':'normal','weight':'bold','size':'32'})
                #slc.set_xlabel(r"$\bf x\ (kpc)$") #if axis=2, x-x,y-y; if axis=1, x-z,y-x; if axis=0, x-y, y-z
                #slc.set_ylabel(r"$\bf z\ (kpc)$")
                slc.set_colorbar_label(field[j],r"$\rm \bf "+ylabels[j]+"$")
                slc.annotate_text((-220, 200), r"$\rm "+titles[i]+"$", coord_system="plot")
                #slc.annotate_scale(coeff=30,unit='kpc')
                #slc.annotate_text((-200, 180), r"$"+titles+"$", coord_system="plot",text_args={"color": "black"})
                #slc.annotate_quiver(("flash", "magx"),("flash", "magy"),factor=16,color="purple",)
                
                slc.save("Projection_"+name+"_"+field[j]+"_"+str(ii[i])+"axis_"+str(axis)+".png")
    return
  

def phase():
    """
    Function that makes 2D phase plots of temperature and density along with pressure. 
    """
    field = ["ndens","temperature","Cool","plasma_beta","cell_mass","Si4_O6","C4_O6","N5_O6","radius","magnetic","ratio","entropy","v_rad","angular_mom","cdens","cpdens","c3pdens"]
    axis_font = {'size':'22'}
    for p in range(1):
        p = p + 8
        avg_dens = []
        if p==0 :
            titles = ['NF Ideal MHD 3 Gyr','NF Ideal MHD 4 Gyr','NF Ideal MHD 5 Gyr','NF Ideal MHD 6 Gyr','NF Ideal MHD 7 Gyr' ]
            ii = [32,43,53,65,76]
            path = "pretty_gal"
        if p==1 :
            titles = ['NF Non-Ideal MHD 3 Gyr','NF Non-Ideal MHD 4 Gyr','NF Non-Ideal MHD 5 Gyr','NF Non-Ideal MHD 6 Gyr','NF Non-Ideal MHD 7 Gyr' ]
            ii = [32,41,51,62,73]
            path = "pretty_gal_nonideal"
        if p==2:
            titles = ['NF Hydro 3 Gyr','NF Hydro 4 Gyr','NF Hydro 5 Gyr','NF Hydro 6 Gyr','NF Hydro 7 Gyr' ]
            ii = [31,40,50,61,71]#np.arange(38,39,1)
            path = "pretty_gal_hydro"
        if p==3 :
            titles = ['Z_{\odot}\ 3\ Gyr','Z_{\odot}\ 4\ Gyr','Z_{\odot}\ 5\ Gyr','Z_{\odot}\ 6\ Gyr','Z_{\odot}\ 7\ Gyr', 'Z_{\odot}\ 7.6\ Gyr']#,'Z1 8 Gyr','Z1 9 Gyr' ]
            ii = [32,42,53,63,74,80]#,85,96]
            path="/Volumes/research1/MAIHEM_Z1/"
            name = "MHD_Feedback_Z1"
        if p==4 :
            titles = ['0.3Z_{\odot}\ 4\ Gyr','0.3Z_{\odot}\ 5\ Gyr','0.3Z_{\odot}\ 6\ Gyr','0.3Z_{\odot}\ 7\ Gyr', '0.3Z_{\odot}\ 7.6\ Gyr']
            ii = [50,63,76,89,98]#103,117]
            path="/Volumes/research2/MAIHEM_Z03/"
            name = "MHD_Feedback_Z03"
        if p==5 :
            titles = ['0.7Z_{\odot}\ 4\ Gyr','0.7Z_{\odot}\ 5\ Gyr','0.7Z_{\odot}\ 6\ Gyr','0.7Z_{\odot}\ 7\ Gyr', '0.7Z_{\odot}\ 7.6\ Gyr']
            ii = [41,52,63,75,83]
            path="/Volumes/research1/MAIHEM_Z07/"
            name = "MHD_Feedback_Z07"        
        if p==6 :
            titles = ['Z_{\odot}\ low\ 4 Gyr','Z_{\odot}\ low\ 5 Gyr','Z_{\odot}\ low\ 6 Gyr','Z_{\odot}\ low\ 7 Gyr', 'Z_{\odot}\ low\ 7.6 Gyr']
            ii = [42,52,63,73,80]
            path="/Volumes/research1/MAIHEM_Z1/"
            name = "MHD_low_Feedback_Z1"
        if p==7 :
            titles = ['0.3Z_{\odot}\ low\ 4 Gyr','0.3Z_{\odot}\ low\ 5 Gyr','0.3Z_{\odot}\ low\ 6 Gyr','0.3Z_{\odot}\ low\ 7 Gyr', '0.3Z_{\odot}\ low\ 7.6 Gyr']
            ii = [41,52,62,73,80]
            path="/Volumes/research2/MAIHEM_Z03/"
            name = "MHD_low_Feedback_Z03"
        if p==8:
            ii = [1,2,3,4]
            path="/scratch/ebuie/ISO_Turb/midway/1E23_S100_z1"
            name="_mhd"
        for i in range(len(ii)):
            i=i + 0
            if ii[i]< 10:
                stuff = yt.load(path+name+"/ISM_hdf5_chk_000"+str(ii[i]))
            elif 10 < ii[i] < 100:
                stuff = yt.load(path+name+"/ISM_hdf5_chk_00"+str(ii[i]))
            elif 100 <= ii[i]:
                stuff = yt.load(path+name+"/ISM_hdf5_chk_0"+str(ii[i]))
            sim = stuff.all_data()#.sphere([0.5,0.5,0.5],(225,"kpc"))#all_data()
            #print(titles[i])
            #sim = simulation.cut_region(["obj['ratio'] != np.inf"])
            #ndens [0], temp[1], Cool[2], plasma beta[3], cell mass[4], Si4/O6[5], C4/O6[6], N5/O6[7], radius[8], magnetic[9], ratio[10], entropy[11], v rad[12], ang y[13]
            x = 1; y = 9; k = 4
            plot = yt.PhasePlot(sim,field[x],field[y],field[k],weight_field=None)
            #plot.set_xlim(10**(-3),5e2)#(4.e-3,1.e1)
            #plot.set_log(field[y],False)
            #plot.set_unit(field[k],"erg/yr")
            #plot.set_unit(field[k],"Msun")
            #plot.set_xlim(1e-3,1e1)#(1e-3,1e3) entropy
            #plot.set_log(field[x],False)
            #plot.set_xlim(0,225) #(2e-5,500)#(8.e-30,2e-23) density
            #plot.set_ylim(8e-1,8e4)#(2e-18,4e-11)#
            #plot.set_cmap(field=field[k],cmap='Blue-Red')
            #plot.set_zlim(field[k],8e3,1e8)#temp,8e3,2e7)#pressure,5e-18,4e-11)#mass,8e3,1e8)#plasma_beta,8e-2,1e2)
            #plot.set_colorbar_label(field[k],r"$\rm Plasma\ \beta$")
            #plot.set_xlabel(r"$\rm NV\ / \ OVI$")
            #plot.set_ylabel(r"$\rm SiIV\ / \ OVI$")
            plot.set_font({'style':'normal','weight':'bold','size':'28'})
            #plot.annotate_text(150,1e39,r"$\rm "+titles[i]+"$")#radii, cooling
            #plot.annotate_text(2e1,1.5e32,r"$\rm "+titles[i]+"$")#plasma beta, cooling
            #plot.annotate_text(1.5e-3,4e4,r"$\rm "+titles[i]+"$")
            plot.save("Phase_"+name+"_"+field[x]+"_"+field[y]+"_"+field[k]+"_"+str(ii[i])+".png")
    return
    
def adder():
    """
    Can be used to add things up from checkpoint files in some specific region determined by r_gal_inner and outer (e.g. mass rate in some region, average velocity of that material, average cooling time)
    """
    sims = [0,2,3,4,5,6]
    for p in range(1):
        p = p+7
        energies = []
        field = ['mass_rate','radius','temperature','cell_mass','Cool','CoolH','Cool_metal','CoolCt','magnetic']
        thing = -1
        if p==0 : #pretty_gal
            ii = np.arange(0,76,1)
            path = "/scratch/ebuie/Galaxy_Turb/"
            name = "MHD_NoFeedback_Z1"
        if p == 1: #high_rot_no_mag_z1
            ii = np.arange(0,101,1)
            path = "/scratch/ebuie/Galaxy_Turb/"
            name = "HYDRO_Feedback_Z1"
        if p==2: #high_rot_z1
            ii = np.arange(0,100,1)
            path = "/scratch/ebuie/Galaxy_Turb/"
            name = "MHD_Feedback_Z1"
        if p==3: #Z03 high_rot_z03
            ii = np.arange(0,118,1)
            path="/scratch/ebuie/Galaxy_Turb/"
            name = "MHD_Feedback_Z03"
        if p==4: #Z03 high_rot_z07
            ii = np.arange(0,84,1)
            path="/scratch/ebuie/Galaxy_Turb/"
            name = "MHD_Feedback_Z07" 
        if p==5 :
            ii = np.arange(0,105,1) 
            path="/scratch/ebuie/Galaxy_Turb/"
            name = "MHD_low_Feedback_Z1"
        if p==6 :
            ii = np.arange(0,103,1)
            path="/scratch/ebuie/Galaxy_Turb/"
            name = "MHD_low_Feedback_Z03"
        if p==7:
            ii = np.arange(0,13,1)
            path="/scratch/ebuie/ISO_Turb/midway/new_mhd/"
            name="1E25_S100_z1_mhd"
        for i in range(len(ii)):
            energies = []
            i=i+0
            if ii[i]< 10:
                stuff = yt.load(path+name+"/ISM_hdf5_chk_000"+str(ii[i]))
            elif 10 < ii[i] < 100:
                stuff = yt.load(path+name+"/ISM_hdf5_chk_00"+str(ii[i]))
            elif 100 <= ii[i]:
                stuff = yt.load(path+name+"/ISM_hdf5_chk_0"+str(ii[i]))
            my_sphere = stuff.all_data()#stuff.sphere([0.5,0.5,0.5],(214,"kpc"))#all_data() 
            #my_sphere = my_sphere.cut_region(["obj['temperature'] < 1.0e5"])
            if i==0:
                fout = open(field[thing]+"_"+name+".dat","w")
                print("Working on "+field[thing]+" "+name+str(ii[i])+" calulations")
            else:
                print("Working on "+field[thing]+" "+name+str(ii[i])+" calulations")
            rad_values = my_sphere['index','radius'].v.flatten()
            #temp_values = my_sphere['gas','temperature'].v.flatten()
            interest = my_sphere[field[thing]].v.flatten()
            mass = my_sphere[field[3]].v.flatten()
            #h_temp = my_sphere['hfrac'].v.flatten()
            #ovi_temp = my_sphere['o5pfrac'].v.flatten()
            thing2 = 0
            #thing3 = 0
            #if weighted do [interest[b]*mass[b] and np.sum(thing2)/weight instead
            thing2 = [interest[b] for b, x in enumerate(rad_values) if r_gal_inner < abs(x) < r_gal_outer]
            #print(thing2[0:10])
            #thing3 = [thing2[a] for a, y in enumerate(temp_values) if 1.0e4 < y < 1.0e5]
            #print(thing3)
            #rad_values = [rad_values[b]/3.09e21 for b, x in enumerate(rad_values) if r_gal_inner < abs(x) < r_gal_outer]
            #plt.plot(rad_values, thing2,'ko')
            #plt.savefig('magnetic_pretty_gal.png')
            
            #weight = float(np.sum([mass[b] for b, x in enumerate(rad_values) if r_gal_inner < abs(x) < r_gal_outer]))
            thing2 = [x for x, x in enumerate(thing2) if np.isfinite(x)]
            thing2 = np.average(thing2)
            #thing2 = np.sum(thing2)/weight#/1.99e33
            #thing3 = 0
            #thing3 = [interest[b]*ovi_temp[b] for b, x in enumerate(rad_values) if r_gal_inner < abs(x) < r_gal_outer]
            #thing3 = [x for x, x in enumerate(thing3) if np.isfinite(x)]
            #thing3 = np.sum(thing3)/float(np.sum([ovi_temp[b] for b, x in enumerate(rad_values) if r_gal_inner < abs(x) < r_gal_outer]))
            energies.append((stuff.current_time.v,thing2))
            print('THING HERE', energies)
            
            if i==0:
                fout.write("Time(s) magnetic(uG) \n")#fout.write("Time(s) Cooling(erg/s) Potential(erg) Kinetic(erg) Internal(erg) Total \n")
                for j in energies:    
                    fout.write(" ".join(map(str, j)) + "\n") 
            else:
                for j in energies:    
                    fout.write(" ".join(map(str, j)) + "\n")  
        fout.close()
    return

def plot_things(names,legend):
	"""
	Meant to import .dat files and plot those quantities, (if you don't have all the checkpoint files in one place)
	"""
	interest = ["CoolH_","Cool_metal_","temperature_vol_","magnetic_","cell_mass_rad_cut_","angular_y_rate_","plasma_beta_","pressure_","mymachnumber_","Cool_"]
	label = ['H\ Cooling\ (10^{35}\ erg\ yr^{-1})','Metal\ Cooling\ (10^{35}\ erg\ yr^{-1})','log\ T\ (K)','Magnetic\ Field\ (\mu G)','Halo\ Mass\ (10^{11}\ M_{\odot})',"Average\ \dot{M}\ (M_{\odot}\ yr^{-1})","L_{y}\ Rate\ (10^{57}\ g\ cm^{2}\ s^{-2})","Plasma\ Beta","Pressure\ (10^{-14}\ dyn\ cm^{-2})","Mach\ Number","Cooling\ (10^{35}\ erg\ yr^{-1})"]
	factor = [1.0e35*3.15e7,1.0e35*3.15e7,1.0,1.0,1.0e11,1.0e57,1.0,1.e-14,1.0,1.0e35*3.15e7]
    #YOUR VARIOUS NAMES OF FOLDERS, pretty gal[0], HYDRO Z1[1], MHD Z1[2], MHD Z03[3], HYDRO Z03[4], MHD Z07[5], HYDRO Z07[6], MHD low Z1[7], MHD low Z03[8]
	ii = [2,3,5,7,8]
	colors = ['r','r','k','orange','b','b','b','g','purple']
	styles = ['-','--','-.']
	for i in range(len(ii)):
            i = ii[i]
            thing = 2
            path = "/Volumes/research1/data_files/"#pathh[4]
            #print factor[thing]#path+interest[thing]+names[i]+".dat"
            time = np.loadtxt(path+interest[thing]+names[i]+".dat",skiprows=1,usecols=(0,))
            cooling = np.loadtxt(path+interest[thing]+names[i]+".dat",skiprows=1,usecols=(1,))
            #coolingh = np.loadtxt(path+interest[0]+names[i]+".dat",skiprows=1,usecols=(1,))
            #coolingme = np.loadtxt(path+interest[1]+names[i]+".dat",skiprows=1,usecols=(1,))
            print(names[i])
            #cooling2 = np.loadtxt(path+interest[1]+names[i]+".dat",skiprows=1,usecols=(1,))
            #cooling = np.array([(cooling[b+1] - cooling[b])/(time[b+1] - time[b]) for b in range(len(cooling)-1)])
            time = [time[b+1]/3.156e16 for b in range(len(time)-1)]
            cooling = [cooling[b+1]/factor[thing] for b in range(len(cooling)-1)]
            #coolingh = [coolingh[b+1]/factor[thing] for b in range(len(cooling)-1)]
            #coolingme = [coolingme[b+1]/factor[thing] for b in range(len(cooling)-1)]
            plt.plot(time,np.log10(cooling),color=colors[i],lw=3,linestyle=styles[0],label=r'$\rm \bf '+legend[i]+'$')
            #plt.plot(time,coolingme,color=colors[i],lw=3,linestyle=styles[2],label=r'$\rm \bf '+legend[i]+'$')
	plt.ylabel(r'$\rm \bf '+label[thing]+'$',**axis_font)
	plt.tick_params(axis='both', which='major', labelsize=17)
	plt.legend(loc='best',frameon=False,ncol=2,prop={'size': 14})
	plt.xlim([0,7.6])
	plt.xticks(np.arange(0,8,1))
	#plt.yscale('log')
	#plt.ylim([1e34,2e35])
	#plt.axhline(y=1.3,linestyle='--',lw=3,color='grey')
	#plt.ylim([0.5,1.2])
	#plt.title(r"$\rm \bf 70\ -\ 100\ kpc$",**axis_font)
	plt.xlabel(r'$\rm \bf Time\ (Gyr)$',**axis_font)
	plt.savefig(interest[thing]+"mhd_compare.png",bbox_inches='tight')
	plt.close()
	print('Done', interest[thing])
    
def plt_pickle_cols(plot_names,ylabels):
    #Also must get radius points
    path = pathh[4]
    rad = open(path+'pkls/radius_box.pkl','rb')#highres
    radius = pickle.load(rad)
    rad.close()
    radius = np.array(radius)
    radius = radius.flatten()
    #stuff4[blank] has OVI(0), NV(1), SiIV(2), SiIII(3), SiII(4), HI(5), CIV(6), MgII(7), temp(8), ndens(9), CII(10), CIII(11), FeII(12), FeIII(13), NII(14), OI(15), NeVIII(16) for some
    rad = np.arange(10,251,10)
    file_names = ["High_turb","High_turb_y","High_rot_no_mag_6p0","High_rot_6p0","High_rot_no_mag_3p0_y","High_rot_3p0_y"]
    legend = ["Hydro\ High\ (edge-on)","Hydro\ High\ (face-on)","Hydro\ Rot\ (edge-on)","MHD\ Rot\ (edge-on)","Hydro\ Rot\ (face-on)","MHD\ Rot\ (face-on)"]
    colors = ['g','g','b','orange','b','orange']
    for m in range(2):
        m = m+8
        items = [2,3]
        for i in range(2):
            i=items[i]
            fil = open(path+'pkls/stuff4'+file_names[i]+'.pkl','r') 
            model = pickle.load(fil)
            fil.close()
            everything = []
            for mm in range(len(model)):
                ans = np.array(model[mm])
                everything.append((ans.flatten()))
            everything.append(radius) #this is for appendin the radius values to end of list, after 4 loop stuff
            everything = np.array(everything)
            bin1 = [];bin2 = [];bin3 = [];bin4 = [];bin5 = [];bin6 = [];bin7 = [];bin8 = [];bin9 = [];bin10 = [];bin11 = [];bin12 = [];bin13 = [];bin14 = [];bin15 = [];bin16 = [];bin17 = [];bin18 = [];bin19 = [];bin20 = [];bin21 = [];bin22 = [];bin23 = [];bin24 = [];bin25 = [];avg=[]
            print('just gotta plot fam',file_names[i],m)
            for ii in range(len(radius)):
                if 0.1 <= everything[-1][ii] < 20/2.:  
                    bin1.append((everything[m][ii]))
                elif 20/2. <= everything[-1][ii] < 40/2.:
                    bin2.append((everything[m][ii]))
                elif 40/2. <= everything[-1][ii] < 60/2.:
                    bin3.append((everything[m][ii]))
                elif 60/2. <= everything[-1][ii] < 80/2.:
                    bin4.append((everything[m][ii]))
                elif 80/2. <= everything[-1][ii] < 100/2.:
                    bin5.append((everything[m][ii]))
                elif 100/2. <= everything[-1][ii] < 120/2.:
                    bin6.append((everything[m][ii]))
                elif 120/2. <= everything[-1][ii] < 140/2.:
                    bin7.append((everything[m][ii]))
                elif 140/2. <= everything[-1][ii] < 160/2.:
                    bin8.append((everything[m][ii]))
                elif 160/2. <= everything[-1][ii] < 180/2.:
                    bin9.append((everything[m][ii]))
                elif 180/2. <= everything[-1][ii] < 200/2.:
                    bin10.append((everything[m][ii]))
                elif 200/2. <= everything[-1][ii] < 220/2.:
                    bin11.append((everything[m][ii]))
                elif 220/2. <= everything[-1][ii] < 240/2.:
                    bin12.append((everything[m][ii]))
                elif 240/2. <= everything[-1][ii] < 260/2.:
                    bin13.append((everything[m][ii]))
                elif 260/2. <= everything[-1][ii] < 280/2.:
                    bin14.append((everything[m][ii]))
                elif 280/2. <= everything[-1][ii] < 300/2.:
                    bin15.append((everything[m][ii]))
                elif 300/2. <= everything[-1][ii] < 320/2.:
                    bin16.append((everything[m][ii]))
                elif 320/2. <= everything[-1][ii] < 340/2.:
                    bin17.append((everything[m][ii]))
                elif 340/2. <= everything[-1][ii] < 360/2.:
                    bin18.append((everything[m][ii]))
                elif 360/2. <= everything[-1][ii] < 380/2.:
                    bin19.append((everything[m][ii]))
                elif 380/2. <= everything[-1][ii] < 400/2.:
                    bin20.append((everything[m][ii]))
                elif 400/2. <= everything[-1][ii] < 420/2.:
                    bin21.append((everything[m][ii]))
                elif 420/2. <= everything[-1][ii] < 440/2.:
                    bin22.append((everything[m][ii]))
                elif 440/2. <= everything[-1][ii] < 460/2.:
                    bin23.append((everything[m][ii]))
                elif 460/2. <= everything[-1][ii] < 480/2.:
                    bin24.append((everything[m][ii]))
                elif 480/2. <= everything[-1][ii] < 500/2.:
                    bin25.append((everything[m][ii]))
            avg.append((bin1,bin2,bin3,bin4,bin5,bin6,bin7,bin8,bin9,bin10,bin11,bin12,bin13, bin14,bin15,bin16,bin17,bin18,bin19,bin20,bin21,bin22,bin23,bin24,bin25))
            avg = np.array(avg)
            stuff = np.zeros((5,len(avg[0])))
            percentile=[0,1,50,99,100]
            for o in range(len(avg[0])):
                ans = np.log10(np.percentile(avg[0][o],percentile))
                #sna = np.sqrt(np.sum(abs(np.log10(avg[0][o]) - ans)**2)/len(avg[0][o]))
                stuff[0,o],stuff[1,o],stuff[2,o],stuff[3,o],stuff[4,o] = abs(ans[0]-ans[2]),abs(ans[1]-ans[2]),ans[2],abs(ans[3]-ans[2]),abs(ans[4]-ans[2])
            #if i == 1 or i == 4 or i == 5:
            #    plt.plot(rad,stuff[2,:],color=colors[i],linestyle='--')
            #    plt.fill_between(rad,stuff[2,:]-stuff[1,:],stuff[2,:]+stuff[3,:],color=colors[i],hatch='x',label=r'$\rm \bf '+legend[i]+'$',alpha=0.2)
            #else:
            plt.fill_between(rad,stuff[2,:]-stuff[1,:],stuff[2,:]+stuff[3,:],color=colors[i],label=r'$\rm \bf '+legend[i]+'$',alpha=0.2)
            plt.plot(rad,stuff[2,:],color=colors[i],linestyle='-')
            plt.ylabel(r'$\rm \bf log\ '+ylabels[m]+'$',**axis_font)
            plt.xlabel(r'$\rm \bf r \ (kpc)$',**axis_font)
            plt.xlim([0,250]) #these limits are for all things vs impact parameter
	#"""
        if m == 8:
            plt.ylim([3.5,6.5]) #these limits are for all ion plots except HI
            plt.axhline(y=np.log10(temp_0),linestyle='--',lw=2,color='grey')
            plt.tick_params(axis='both', which='major', labelsize=14)
            plt.legend(loc='lower right',frameon=False,prop={'size': 14},ncol=2)
            plt.savefig(file_names[i]+'_'+plot_names[m]+'.png',bbox_inches='tight')#plt.show()
            plt.close()
            print('Saved '+file_names[i]+'_'+plot_names[m]+'.png')
        elif m == 9:
            plt.ylim([-5,2]) #these limits are for all ion plots except HI
            plt.tick_params(axis='both', which='major', labelsize=14)
            plt.axhline(y=np.log10(n_rho_0),linestyle='--',lw=2,color='grey')
            plt.legend(loc='upper right',frameon=False,prop={'size': 14},ncol=2)
            plt.savefig(file_names[i]+'_'+plot_names[m]+'.png',bbox_inches='tight')#plt.show()
            plt.close()
            print('Saved '+file_names[i]+'_'+plot_names[m]+'.png')
        elif m == 6:
            plt.ylim([11.0,17.0]) 
            plt.tick_params(axis='both', which='major', labelsize=14)
            plt.legend(loc='upper right',frameon=False,prop={'size': 14},ncol=2)
            plt.savefig(file_names[i]+'_'+plot_names[m]+'.png',bbox_inches='tight')#plt.show()
            plt.close()
            print('Saved '+file_names[i]+'_'+plot_names[m]+'.png')
        elif m == 0 or 1 or 2 or 3 or 4 or 5 or 7 or 10 or 11 or 12 or 13 or 14 or 15:
        #COS-Halos data
    	    data = Table.read(path+'COS_Halos_MTL_for_miki.txt',format='ascii.basic')
    	    #print path+'COS_Halos_MTL_for_miki.txt'
    	    info = Table.read(path+'COS_Halos_MTL_for_miki_info.txt',format='ascii.basic')
    	    H1_meas = np.loadtxt(path+'COS_data_table',usecols=(2,3,)) #first entry is column, second is error
    	    radii = np.array(np.loadtxt(path+'COS_data_table',usecols=(21,)))
    	    data_ions = [] #holds ion[0], column[1], error[2], flag[3], name[4], best fit column[5]
    	    for t in range(len(info['name'])):
            #Gets COS-Halos stuff
                sy = info['name'][t]
                sets=np.where(data['name'] == sy.strip())
                #print sy, data['ion'][sets]
                fil = open(path+'COS_newtest/all_pickle_files/'+sy.strip()+'_emcee.pkl','rb')
                model = pickle.load(fil)
                fil.close()
                sets=sets[0] #gets number of entries for a system
                for ii in range(len(sets)):
                    aa = radii[t]
                    if(data['ion'][sets[ii]] == 'HI'):
                        continue
                    else: 
                #append ion value to list
                        data_ions.append((data['ion'][sets[ii]],data['logn'][sets[ii]],data['elogn'][sets[ii]],data['flag'][sets[ii]],sy,aa))
            #else: 
                #appends the average error to limit systems
            #    data_ions.append((data['ion'][sets[ii]],data['logn'][sets[ii]],data['flag'][sets[ii]],sy,model['best_fit'][ii-1],aa))
    	    ions = ["CII","CIII","FeII","FeIII","MgII","NII","OI","OVI","SiII","SiIII","SiIV","NV"]
#stuff4[blank] has OVI(0), NV(1), SiIV(2), SiIII(3), SiII(4), HI(5), CIV(6), MgII(7), temp(8), ndens(9), CII(10), CIII(11), FeII(12), FeIII(13), NII(14), OI(15)
    	    diff = [] #holds the differences between model and obs column, originally diff was here
    	    for jj in range(1): 
                if m == 0:
                    jj = 7 #this picks the COS-Halos ion of interest
                if m == 1:
                    jj = 11
                if m == 2:
                    jj = 10
                if m == 3:
                    jj = 9
                if m == 4:
                    jj = 8
                if m == 5:
                    jj = 10
                if m == 7:
                    jj = 4
                if m == 10:
                    jj = 0
                if m == 11:
                    jj = 1
                if m == 12:
                    jj = 2
                if m == 13:
                    jj = 3
                if m == 14:
                    jj = 5
                if m == 15:
                    jj = 6
                ion_num = np.where(ions[jj]==np.array(data_ions))[0]
                for w in range(len(ion_num)):
                    kk = ion_num[w]
                    diff.append((data_ions[kk][1],data_ions[kk][2],data_ions[kk][3],data_ions[kk][-1])) #this appends columns for certain ions
            	
    	    if m == 5:
        #This part is for plottin COS-Halos stuff overlaid on modeled data
                plt.errorbar(radii[:],H1_meas[:,0],c='k',yerr=H1_meas[:,1],fmt='s',ms=s2)
                plt.ylim([11.0,22.0]) #these limits are for all ion plots except HI
                plt.tick_params(axis='both', which='major', labelsize=14)
                plt.legend(loc='upper right',frameon=False,prop={'size': 14},ncol=2)
                plt.savefig(file_names[i]+'_'+plot_names[m]+'.png',bbox_inches='tight')#plt.show()
                plt.close()
                print('Saved '+file_names[i]+'_'+plot_names[m]+'.png')
    	    elif m == 0 or 1 or 2 or 3 or 4 or 7 or 10 or 11 or 12 or 13 or 14 or 15:
                diff = np.array(diff)  
                for cc in range(len(diff)):
                    if diff[cc,2]==-1: #check if upper limit  
                        plt.errorbar(diff[cc,-1],diff[cc,0],c='k',fmt='v',ms=s2)
                    elif diff[cc,2]==-2: #check if lower limit
                        plt.errorbar(diff[cc,-1],diff[cc,0],c='k',fmt='^',ms=s2)
                    else: 
                        plt.errorbar(diff[cc,-1],diff[cc,0],c='k',yerr=diff[cc,1],fmt='s',ms=s2)
                plt.ylim([11.0,17.0]) #these limits are for all ion plots except HI
                plt.tick_params(axis='both', which='major', labelsize=14)
                plt.legend(loc='upper right',frameon=False,prop={'size': 14},ncol=2)
                print(i,'here')
                plt.savefig(file_names[i]+'_'+plot_names[m]+'.png',bbox_inches='tight')#plt.show()
                plt.close()
                print('Saved '+file_names[i]+'_'+plot_names[m]+'.png')
	#"""
        if m == 16:
            plt.ylim([11.0,17.0]) 
            plt.tick_params(axis='both', which='major', labelsize=14)
            plt.legend(loc='upper right',frameon=False,prop={'size': 14},ncol=2)
            plt.savefig(file_names[i]+'_'+plot_names[m]+'.png',bbox_inches='tight')#plt.show()
            plt.close()
            print('Saved '+file_names[i]+'_'+plot_names[m]+'.png')
    return

def radial_bins(names):
    """
    This function gets things in radial bins for more detailed analysis, e.g. entropy
    """
    #YOUR VARIOUS NAMES OF FOLDERS, pretty gal[0], HYDRO Z1[1], MHD Z1[2], MHD Z03[3], HYDRO Z03[4], MHD Z07[5], HYDRO Z07[6], MHD low Z1[7], MHD low Z03[8]
    runs = [2,3,5,7,8]
    colors = ['grey','r','k','orange','b','b','b','k','orange']
    field = ['ndens','mag_energy']
    label = ['Number\ Density\ (cm^{-3})','Magnetic\ Energy\ (erg)']
    thing = 1
    path = "/scratch/ebuie/Galaxy_Turb/"
    for p in runs:
        if p==0 : #pretty_gal
            ii = np.arange(0,76,1)
            name = "MHD_NoFeedback_Z1"
        if p == 1: #high_rot_no_mag_z1
            ii = np.arange(0,101,1)
            name = "HYDRO_Feedback_Z1"
        if p == 2 :
            titles = ['Z_{\odot}\ 3\ Gyr','Z_{\odot}\ 4\ Gyr','Z_{\odot}\ 5\ Gyr','Z_{\odot}\ 6\ Gyr','Z_{\odot}\ 7\ Gyr', 'Z_{\odot}\ 7.6\ Gyr']#,'Z1 8 Gyr','Z1 9 Gyr' ]
            ii = [32,42,53,63,74,80]#,85,96]
            name = "MHD_Feedback_Z1"
        if p == 3 :
            titles = ['0.3Z_{\odot}\ 4\ Gyr','0.3Z_{\odot}\ 5\ Gyr','0.3Z_{\odot}\ 6\ Gyr','0.3Z_{\odot}\ 7\ Gyr', '0.3Z_{\odot}\ 7.6\ Gyr']
            ii = [50,63,76,89,98]#103,117]
            name = "MHD_Feedback_Z03"
        if p == 4:
            ii = np.arange(0,101,1)
            name = "HYDRO_Feedback_Z03"
        if p == 5 :
            titles = ['0.7Z_{\odot}\ 4\ Gyr','0.7Z_{\odot}\ 5\ Gyr','0.7Z_{\odot}\ 6\ Gyr','0.7Z_{\odot}\ 7\ Gyr', '0.7Z_{\odot}\ 7.6\ Gyr']
            ii = [41,52,63,75,83]
            name = "MHD_Feedback_Z07"
        if p == 6: #high_rot_no_mag_z1
            ii = np.arange(0,101,1)
            name = "HYDRO_Feedback_Z07"
        if p==7 :
            titles = ['Z_{\odot}\ low\ 4 Gyr','Z_{\odot}\ low\ 5 Gyr','Z_{\odot}\ low\ 6 Gyr','Z_{\odot}\ low\ 7 Gyr', 'Z_{\odot}\ low\ 7.6 Gyr']
            ii = [42,52,63,73,80]
            name = "MHD_low_Feedback_Z1"
        if p==8 :
            titles = ['0.3Z_{\odot}\ low\ 4 Gyr','0.3Z_{\odot}\ low\ 5 Gyr','0.3Z_{\odot}\ low\ 6 Gyr','0.3Z_{\odot}\ low\ 7 Gyr', '0.3Z_{\odot}\ low\ 7.6 Gyr']
            ii = [41,52,62,73,80]
            name = "MHD_low_Feedback_Z03"
        for i in range(1):
            i= -1
            if ii[i]< 10:
                stuff = yt.load(path+name+"/ISM_hdf5_chk_000"+str(ii[i]))
            elif 10 < ii[i] < 100:
                stuff = yt.load(path+name+"/ISM_hdf5_chk_00"+str(ii[i]))
            elif ii[i] > 100:
                stuff = yt.load(path+name+"/ISM_hdf5_chk_0"+str(ii[i]))
            sim = stuff.sphere([0.5,0.5,0.5],(225,"kpc"))
            rad_values = sim['index','radius'].in_units('kpc').value
            interest = sim[field[thing]].value
            values, r = [], []
            for j in range(30):
                lim1, lim2 = j*7.5, (j+1)*7.5
                values.append(np.average([interest[b] for b, x in enumerate(rad_values) if lim1 < abs(x) < lim2]))
                r.append((lim1+lim2)/2.)
                #print('VALUES: ',values)
            if p < 7:
                plt.plot(r,values,color=colors[p],lw=3,linestyle='-',label=r'$\rm \bf '+legend[p]+'$')
            else:
                plt.plot(r,values,color=colors[p],lw=3,linestyle='--',label=r'$\rm \bf '+legend[p]+'$')
            print(name+' done')
        plt.ylabel(r'$\rm \bf '+label[thing]+'$',**axis_font)
        #plt.axvspan(0, 0.99, facecolor='purple', alpha=0.2)
        plt.xlabel(r'$\rm \bf r\ (kpc)$',**axis_font)
        plt.yscale('log')
        #plt.xscale('log')
        plt.legend(loc='best',frameon=False,ncol=2,prop={'size': 14})
        #plt.tick_params(labeltop=False,direction='in')
        #plt.tick_params(axis='both', which='major', labelsize=17)
    plt.savefig('Plot_'+field[thing]+'_7p6gyr.png')
    return
    
def plotter(names):
	"""
	meant to make simple profile plots with yt
	"""
	profiles = []; labels = []
	for p in range(1):
               p = p+3
        
               field = ['density','radius','plasma_beta','v_rad','Cool','angular_y_rate']
               if p==0 :
                       ii = [37]
                       path = pathh[1]
               if p == 1:
                       ii = [38]
                       path = pathh[0]
               if p==2:
                       ii = np.arange(38,77,1)
                       path = pathh[2]
               if p==3:
                       ii = np.arange(7,15,1)
                       path = pathh[2]
               for i in range(len(ii)):
                       thing = 0
                       if ii[i]< 10:
                            stuff = yt.load(path+names[p]+"/ISM_hdf5_chk_000"+str(ii[i]))
                       elif 10 < ii[i] < 100:
                             stuff = yt.load(path+names[p]+"/ISM_hdf5_chk_00"+str(ii[i]))
                       elif 100 <= ii[i]:
                           stuff = yt.load(path+names[p]+"/ISM_hdf5_chk_0"+str(ii[i]))
                       my_sphere = ([0.5,0.5,0.5],(214,"kpc"))#all_data() 
                       profiles.append(yt.create_profile(my_sphere, "radius", [field[thing]], extrema={'radius': (0,250)}))
                       labels.append("T = %.1f" % my_sphere.current_time)
               plot = yt.ProfilePlot.from_profiles(profiles,labels=label)
               plot.save()
	return        
#VARIOUS PATHS paper 3 runs[0], paper 2 runs[1], stampede2 paper 3 runs[2], stampede2 paper 2 runs[3], ed's laptop new runs[4], ed laptop old runs[5], tessie pickle files[6]
pathh = ["/Volumes/research1/MAIHEM_runs/Z1/"]

    
#YOUR VARIOUS NAMES OF FOLDERS, pretty gal[0], HYDRO Z1[1], MHD Z1[2], MHD Z03[3], HYDRO Z03[4], MHD Z07[5], HYDRO Z07[6], MHD low Z1[7], MHD low Z03[8]
names = ["pretty_gal_nonideal","High_rot_no_mag_z1","High_rot","High_rot","High_rot_no_mag","High_rot","High_rot_no_mag"]
file_names = ["pretty_gal","High_rot_no_mag_z1","MHD_Feedback_Z1","MHD_Feedback_Z03","High_rot_no_mag_z03","MHD_Feedback_Z07","High_rot_no_mag_z07","MHD_low_Feedback_Z1","MHD_low_Feedback_Z03"]
colors = ['g','r','k','purple','b','b']
legend = ["MHD\ +\ NF\ Z1","Hydro\ Z1","Z_{\odot}","0.3Z_{\odot}","Hydro\ Z03","0.7Z_{\odot}","Hydro\ Z07","Z_{\odot}\ low","0.3Z_{\odot}\ low"]

#stuff4[blank] has OVI(0), NV(1), SiIV(2), SiIII(3), SiII(4), HI(5), CIV(6), MgII(7), temp(8), ndens(9), CII(10), CIII(11), FeII(12), FeIII(13), NII(14), OI(15)
#ylabels = ['N_{O VI} \ (cm^{-2})','N_{N V} \ (cm^{-2})','N_{Si IV} \ (cm^{-2})','N_{Si III} \ (cm^{-2})','N_{Si II} \ (cm^{-2})','N_{H I} \ (cm^{-2})','N_{C IV} \ (cm^{-2})','N_{Mg II} \ (cm^{-2})','Temperature\ (K)','Number\ Density\ (cm^{-3})','N_{C II} \ (cm^{-2})','N_{C III} \ (cm^{-2})','N_{Fe II} \ (cm^{-2})','N_{Fe III} \ (cm^{-2})','N_{N II} \ (cm^{-2})','N_{O I} \ (cm^{-2})','N_{Ne VIII} \ (cm^{-2})']

plots=['N_OVI_column_avg_box','N_NV_column_avg_box','N_SiIV_column_avg_box','N_SiIII_column_avg_box','N_SiII_column_avg_box','N_HI_column_avg_box','N_CIV_column_avg_box','N_MgII_column_avg_box','Temp_avg_box','ndens_column_avg_box','N_CII_column_avg_box','N_CIII_column_avg_box','N_FeII_column_avg_box','N_FeIII_column_avg_box','N_NII_column_avg_box','N_OI_column_avg_box','N_NeVIII_column_avg_box']

#WHAT DO YOU WANT IT TO DO?
def main():
	#make_slices()
	#plt_HI_ndens(plots,ylabels)
	#plt_pickle_cols(plots,ylabels)
	#adder()
	#plot_things(file_names,legend)
	#make_projs()
	#radial_bins(names)
	#emission(names,file_names)
	phase()
	#plotter(names)
    #pickle_dump(names,file_names)
	#pickle_dump2(names,file_names)
	#read_pickle(file_names,colors,legend)
	#python_phase(file_names,colors,legend)
	#kepler(names,colors,file_names)
	
if __name__ == '__main__':
	main()

		
	   



