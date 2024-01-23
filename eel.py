#!/usr/bin/env python
import numpy as np
import os, sys
from glob import glob
from os.path import join as pjoin
import numpy as np
import scipy.optimize
import netCDF4
import nemo_rho
from argparse import ArgumentParser

rundir0 ='.'

parser = ArgumentParser(description='Create initial T,S (and possibly bathymetry) for ELVER')
parser.add_argument('-i','--rundir', dest='rundir',help='directory to read files from', default=rundir0)
parser.add_argument('-b','--bshape', dest='bathy_shape',help='bathymetric shape'
                    ,default=None,choices=['tophat','flat',None,'promontory','roll','xroll'])
parser.add_argument('--btype', dest='bathy_type',help='bathymetric shape'
                    ,default=None,choices=['tophat','flat',None,'promontory','roll','xroll'])
parser.add_argument('-T','--Tshape', dest='Tshape',help='T-field shape, default %(default)s',default='flat_ML',choices=['flat_ML','flat','FK08','sloping_ML','step'])
parser.add_argument('--maxdepth', dest='maxdepth',help='maximum depth, default taken from mesh_zgr',default=None)
parser.add_argument('-S','--Sshape', dest='Sshape',help='S-field shape, default %(default)s',default='point',choices=['point','step'])
parser.add_argument('--CMLshape', dest='CMLshape',help='CML-field shape, default %(default)s',default='point1',choices=['point1'])
parser.add_argument('--CIntshape', dest='CIntshape',help='CInt-field shape, default %(default)s',default='point2',choices=['point2'])
parser.add_argument('--MLD', type=float,dest='MLD',help='Mixed-layer depth, default %(default)s',default=300.)
parser.add_argument('--noTS', dest='noTS',help='no T & S file', action='store_true',default=False)
parser.add_argument('-C', dest='C',help='output tracer fields', action='store_true',default=False)
parser.add_argument('--eos', dest='eos',help='equation of state, default %(default)s', default='linearT',choices=['linearT','linearTS','JM94'])
args = parser.parse_args()

maskpath = pjoin(args.rundir,'mask.nc')
fmask = netCDF4.Dataset(maskpath)
tmask = fmask.variables['tmask'][0,...].astype(np.float64)
tmaskb = tmask<1.e-8
tmaskval = 1.e10
nzp1,nyp2,nxp2 = tmask.shape
nz = nzp1-1
fmask.close()

meshpath = pjoin(args.rundir,'mesh.nc')
if not os.path.exists(meshpath):
    zmeshpath = pjoin(args.rundir,'mesh_zgr.nc')
    hmeshpath = pjoin(args.rundir,'mesh_hgr.nc')
    fzmesh = netCDF4.Dataset(zmeshpath)
    fhmesh = netCDF4.Dataset(hmeshpath)
    meshfiles = fzmesh,fhmesh
else:
    fzmesh = netCDF4.Dataset(meshpath)
    fhmesh = fzmesh
    meshfiles = fzmesh,

meshnames = fzmesh.variables.keys()
if 'e3t' in meshnames:
    # gdept_0 includes dummy point below ocean floor
    zt = fzmesh.variables['gdept'][0,:,:,:]
    if 'hbatt' in meshnames:
        coord = 'sco'
    else:
        coord = 'zps'
        
    if args.maxdepth is None:
        zbotxy = fzmesh.variables['gdepw'][0,-1,...]
        zbot = zbotxy.max()
    else:
        zbot = args.maxdepth

else:
    zt = np.tile(zt0[:,None,None],(nyp2,nxp2))
    coord = 'zco'
    if args.maxdepth is None:
        zbot = fzmesh.variables['gdepw_0'][0,-1]
    else:
        zbot = args.maxdepth

    zt0 = fzmesh.variables['gdept_0'][0,:]

phit = fhmesh.variables['gphit'][0,...]
phiv = fhmesh.variables['gphiv'][0,...]
for fmesh in meshfiles: fmesh.close()



def get_TS(Tshape='flat_ML',MLD=300.,Sshape='point',dtdz0 = .01):

    T,S = np.zeros([2,nzp1,nyp2,nxp2],np.float64)
    ny = nyp2 - 2


    if Tshape=='flat_ML':
        for j in range(nyp2):
            T[:,j,:] = tmask[:,j,:]*(20. - (1./3.)*(j + 1.5 + zt[:,j,:]*(3.*dtdz0)))

        for k in range(nz-1,-1,-1):
            for j in range(nyp2):
                for i in range(nxp2):
                    if zt[k,j,i] < MLD: T[k,j,i] = T[k+1,j,i]
    elif Tshape=='sloping_ML':
        for j in range(nyp2):
            T[:,j,:] = tmask[:,j,:]*(20. - (1./3.)*(j + 1.5 + zt[:,j,:]*(3.*dtdz0)))

        MLDy = np.linspace(0.5*MLD,1.5*MLD,nyp2)
        for k in range(nz-1,-1,-1):
            for j in range(nyp2):
                for i in range(nxp2):
                    if zt[k,j,i] < MLDy[j]: T[k,j,i] = T[k+1,j,i]
    elif Tshape=='flat':
        for j in range(nyp2):
            T[:,j,:] = tmask[:,j,:]*(20. - (1./3.)*(15. + 1.5 + zt[:,j,:]*(3.*dtdz0)))
    elif Tshape=='step':
        for k in range(nz-1,-1,-1):
            for j in range(nyp2):
                for i in range(nxp2):
                    if zt[k,j,i] < 600:
                        T[k,j,i] = 20.
                    else:
                        T[k,j,i] = 10.
    elif Tshape=='FK08':
        f = 7.29e-5
        grav  = 9.80665
        rn_alpha = 2.e-4

        H0 = 200.; M2 = -(2.*f)**2; Lf = 18.; N20,N2ML = (64.*f)**2,0.
        inML = (zt<H0).astype(np.float64)
        inInterior = 1.0 - inML
        if ny % 2 == 0:
            y0 = phiv[ny/2,3]
        else:
            y0 = phit[ny/2,3]
        for j in range(nyp2):
            T[:,j,:] = tmask[:,j,:]*(10. + ((inML[:,j,:]*N2ML+ inInterior[:,j,:]*N20)*(H0-zt[:,j,:]) + .5*Lf*M2*1.e3*np.tanh(2.*(phit[None,j,:]-y0)/Lf)
                                      )/(grav * rn_alpha) )

    dS = 0.2
    if Sshape=='point':
        k,j = 11,(2*ny)/3 + 2
        S[...] = 35.
        if args.eos=='linearTS':
            alpha,beta = 0.2,0.77
            drho = beta*dS
            T[k,j,:] += drho/alpha
            print 'compensation with %s dt=%f' % (args.eos,drho/alpha)
        elif args.eos=='JM94':
            alpbet0 = 0.05
            T0,S0,depth0 = T[k,j,1],S[k,j,1],zt[k,j,1]
            rho0 =  nemo_rho.eos.rho(T0,S0,depth0)
            nemo_rho.eos.initialize(S0+dS,depth0,rho0)
            Thi = T0 + dS/alpbet0
            Tnew = scipy.optimize.brentq(nemo_rho.eos.drho,T0,Thi)
            T[k,j,:] = Tnew
            print 'compensation with %s dt=%f' % (args.eos,Tnew-T0)

        S[k,j,:] += dS
    elif Sshape=='step':
        S[...] = 35.
        k= 25
        S[k:,...] = 34.

    T[tmaskb] = tmaskval
    S[tmaskb] = tmaskval

    return T,S

def get_C(CMLshape='point1',CIntshape='point2'):

    CML,CInt = np.zeros([2,nzp1,nyp2,nxp2],np.float64)
    ny = nyp2 - 2

    if CMLshape=='point1':
        k,j = 11,(2*ny)/3 + 2
        CML[k,j,:] = 1.0

    if CIntshape=='point2':
        k,j = 24,(1*ny)/3 + 2
        CInt[k,j,:] = 1.0

    return CML,CInt

def get_bathy(zbot,shape='tophat'):
    bathy = np.zeros([nyp2,nxp2],np.float64)
    eps = 1.e-20
    if shape=='tophat':
        mid_bump = int(round(nyp2*(5./12.),0))
        mid_bumpC = mid_bump - 1

        bathy[1:-1,:] = zbot + eps # C indexing
        bathy[mid_bumpC-3:mid_bumpC+3,:] = 500.+eps
        bathy[0,:] = -1.0 -eps
    elif shape=='flat':
        bathy[1:-1,:] = zbot+eps # C indexing
        bathy[0,:] = -1.0 -eps
    elif shape=='promontory':
        mid_bump = int(round(nyp2*(5./12.),0))
        mid_bumpC = mid_bump - 1

        bathy[1:-1,:] = zbot+eps # C indexing
        bathy[mid_bumpC-3:mid_bumpC+3,:] = 500.+eps
        bathy[mid_bumpC,2:] = 0.
        bathy[0,:] = -1.0 -eps
    elif shape=='roll':
        mid_bump = int(round(nyp2*0.5,0))
        mid_bumpC = mid_bump - 1
        halfwidth = 5
        height = 500.
        rhalfwidth = float(halfwidth)
        phase = np.linspace(-1.,1.,2*halfwidth+1)*np.pi*.5
        dh = height*np.cos(phase)

        bathy[1:-1,:] = zbot+eps # C indexing
        bathy[0,:] = -1.0 -eps
        bathy[mid_bumpC-halfwidth:mid_bumpC+halfwidth+1,:] -= dh[:,None]
    elif shape=='xroll':
        mid_bump = int(round(nyp2*0.5,0))
        mid_bumpC = mid_bump - 1
        halfwidth = 5
        height = 500.
        rhalfwidth = float(halfwidth)
        phase = np.linspace(-1.,1.,2*halfwidth+1)*np.pi*.5
        xi = np.arange(nxp2,dtype=np.float64)
        xmid = .5*xi[-1]
        xphase = (xi-xmid)*np.pi/xi[-2]
        dh = height*np.outer(np.cos(phase),np.cos(xphase)**2)

        bathy[1:-1,:] = zbot+eps # C indexing
        bathy[0,:] = -1.0 -eps
        bathy[mid_bumpC-halfwidth:mid_bumpC+halfwidth+1,:] -= dh
    else:
        sys.exit('%s not implemented' % shape)

    return bathy

def get_mbathy(bathy,zt0):
    nyp2,nxpy = bathy.shape
    mbathy = np.zeros([nyp2,nxp2],np.int32)
    mbathy[1:-1,:] = zt0.searchsorted(bathy[1:-1,:])
    mbathy[0,:] = -1
    return mbathy


def make_initCDF(outfile,**kwargs):

    vnames = kwargs.keys()
    nzp1,nyp2,nxp2 = kwargs[vnames[0]].shape

    #=============================================================================
    # create netCDF file......
    #=============================================================================
    fout = netCDF4.Dataset(outfile, 'w', format='NETCDF3_CLASSIC')
    fdim = fout.createDimension
    fout.createDimension('Z', nzp1)
    fout.createDimension('Y', nyp2)
    fout.createDimension('X', nxp2)

    unitDicts ={'T':'deg C','S':'psu'}

    for vname in vnames:
        Nd = fout.createVariable('init_'+vname,'f8',('Z','Y','X'))
        Nd.units = unitDicts.get(vname,'#')
        Nd[...] = kwargs[vname]
    fout.close()


def make_TS_cdf(Theta,S,outfile):
    #=============================================================================
    # create netCDF file......
    #=============================================================================
    fout = netCDF4.Dataset(outfile, 'w', format='NETCDF3_CLASSIC')
    nzp1,nyp2,nxp2 = Theta.shape
    fdim = fout.createDimension
    fout.createDimension('Z', nzp1)
    fout.createDimension('Y', nyp2)
    fout.createDimension('X', nxp2)

    ThetaNd = fout.createVariable('init_T','f8',('Z','Y','X'))
    SNd = fout.createVariable('init_S','f8',('Z','Y','X'))

    # Z.units,Y.units,X.units = '#','#','#'
    ThetaNd.units = 'deg C'
    SNd.units = 'psu'

    ThetaNd[...] = Theta
    SNd[...] = S
    fout.close()
    # print 1/0

def make_bathy_cdf(bathy=None,mbathy=None,bathyfile='bathy_meter.nc',mbathyfile='bathy_level.nc'):
    #=============================================================================
    # create netCDF files......
    #=============================================================================
    if bathy is not None:
        nyp2,nxp2 = bathy.shape
        fout = netCDF4.Dataset(bathyfile, 'w', format='NETCDF3_CLASSIC')
        fdim = fout.createDimension
        fout.createDimension('Y', nyp2)
        fout.createDimension('X', nxp2)

        bathyNd = fout.createVariable('Bathymetry','f8',('Y','X'))
        bathyNd.units = 'm'
        bathyNd[...] = bathy
        fout.close()

    if mbathy is not None:
        nyp2,nxp2 = mbathy.shape
        fout = netCDF4.Dataset(mbathyfile, 'w', format='NETCDF3_CLASSIC')
        fdim = fout.createDimension
        fout.createDimension('Y', nyp2)
        fout.createDimension('X', nxp2)

        mbathyNd = fout.createVariable('Bathy_level','f8',('Y','X'))
        mbathyNd.units = '#'
        bathy[...] = mbathy # change to real; use bathy in function as workspace
        mbathyNd[...] = bathy
        fout.close()

if not args.noTS:
    T,S = get_TS(Tshape=args.Tshape,Sshape=args.Sshape,MLD=args.MLD)
    if 'ML' in args.Tshape:
        TSstring = '%s%.0fm%s' %(args.Tshape,args.MLD,args.eos)
    else:
        TSstring = '%s%s' %(args.Tshape,args.eos)
    flink = 'elver.init_TS.nc'
    fname = 'elver.init_TS%s_%s.nc' % (TSstring,coord)
    make_initCDF(fname,T=T,S=S)
    if os.path.islink(flink): os.remove(flink)
    os.symlink(fname,flink)

if args.C:
    CML,CInt = get_C(CMLshape=args.CMLshape,CIntshape=args.CIntshape)
    flink = 'elver.init_trc.nc'
    fname = 'elver.init_trc%s_%s_%s.nc' % ('CML','CInt',coord)
    make_initCDF(fname,CML=CML,CInt=CInt)
    if os.path.islink(flink): os.remove(flink)
    os.symlink(fname,flink)

if args.bathy_shape is not None:
    bathy = get_bathy(zbot, shape=args.bathy_shape)
    if coord == 'zco':
        mbathy = get_mbathy(bathy,zt0)
        make_bathy_cdf(mbathy=mbathy,bathyfile='bathy_meter.nc',mbathyfile='bathy_level.nc')
    else:
        make_bathy_cdf(bathy=bathy,bathyfile='bathy_meter.nc',mbathyfile='bathy_level.nc')
