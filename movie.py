#!/usr/bin/env python
import os,sys,shutil
import os.path
from os.path import join as pjoin,splitext,split,expanduser
import subprocess

import numpy as np
import numpy.ma as ma
from argparse import ArgumentParser
from glob import glob

parser = ArgumentParser(description='Move files to store directory')
rundir0 = '.'#'/Users/agn/Programming/NEMO/nemo_v3_3_beta/NEMOGCM/CONFIG/ORCA2_LIM/EXP00'
parser.add_argument('-i','--rundir', dest='rundir',help='directory to read files from', default=rundir0)
parser.add_argument('-m','--movie', dest='movie',action='store_true',help='make movie of tracer evolution', default=False)
parser.add_argument('-c','--clean', dest='clean',action='store_true',help='clean output directory files', default=False)
parser.add_argument('--FK', dest='FK',action='store_true',help='Plot psiv_mle', default=False)
parser.add_argument('-s','--sourcedir', dest='sourcedir',help='VC directory of source files', default=None)
parser.add_argument(dest='run_name',help='run name')
parser.add_argument('--restarts', dest='restarts',help='copy restart files',action='store_true', default=False)
parser.add_argument('--eos', dest='eos',help='equation of state', default='linearT',choices=['linearT','linearTS','JM94'])
parser.add_argument('--filetype', dest='filetype',help='type of data', default='average',choices=['average','instant','restart'])
outbase0 = '/Users/agn/Programming/NEMO/RUNS/'
parser.add_argument('-o','--outbase', dest='outbase',help='base directory to put new directory in', default=outbase0)
args = parser.parse_args()
print args.run_name

if args.sourcedir is None:
    args.sourcedir = os.environ['V34']
if args.FK:
    args.FK = '--FK'
else:
    args.FK = ''

#Create spatially merged files
subprocess.call(['mp.py'])
restartfiles = glob('*restart*')
for restartfile in restartfiles:
    if 'all' not in restartfile:
        runtype = restartfile.split('_')[0]
        break

#Create directory to place merged files
outbasedir = pjoin(args.outbase,runtype)
if not os.path.exists(outbasedir):
   os.mkdir(outbasedir)

storedir = pjoin(outbasedir,args.run_name)
if os.path.exists(storedir):
    if args.clean:
        oldfiles = os.listdir(storedir)
        [os.remove(pjoin(storedir,file)) for file in oldfiles]
else:
   os.mkdir(storedir)

# Write source and namelists gitSHA as filenames in storedir
gitfiles = glob('gitSHA1*') + glob('namelistGIT*')
for gitfile in gitfiles:
    shutil.copy2(gitfile,storedir)

if args.restarts:
    if runtype == 'ELVER' or runtype == 'ELVERW':
        if not os.access('all_restarts.nc',os.R_OK):
            os.system('ncrcat -4 *restart.nc -o all_restarts.nc')
        restarts2move = ['all_restarts.nc']
        restarts = glob('%s*restart.nc' % runtype)
        restarts2copy = [restarts[-1]]
    else:
        restarts = glob('%s*restart.nc' % runtype)
        ice_restarts = glob('%s*restart_ice.nc' % runtype)
        restarts2move = restarts[:-1]
        restarts2copy = [restarts[-1]]
        if ice_restarts:
            restarts2move += ice_restarts[:-1]
            restarts2copy.append(ice_restarts[-1])

    links = ['restart.nc','restart_ice_in.nc']
    if restarts2copy:
        nrestarts = len(restarts2copy)
        for filename,linkname in zip(restarts2copy,links[:nrestarts]):
            if os.path.lexists(linkname):
                os.remove(linkname)
            os.symlink(filename,linkname)
else:
    restarts2move, restarts2copy = [],[]

if  os.access('out',os.R_OK): os.system('head -2000 out > start_out')
movelist1 =  glob('*grid_?.nc') + glob('*grid_?i.nc') + \
           glob('*icemod.nc') + glob('*Eq.nc') + restarts2move
movelist2 = ['output.init.nc','out','start_out']
copylist1 = ['namelist'] + ['ocean.output'] + ['timing.output'] + ['iodef.xml'] + glob('../*cpp*') + restarts2copy
copylist2 =  [x+'.nc' for x in
              ['mask','mesh_zgr','mesh_hgr','elver.init_TS','bathy_level','bathy_meter']]
missinglist = []

for fname in movelist1+copylist1:
    if not os.access(fname,os.R_OK):
        missinglist.append(fname)
if len(missinglist)>0:
    sys.exit("can't find: \n%s" % ('\n'.join(missinglist)))

print ('moving/copying files to %s' % storedir)
newnames = {'output.init_0000.nc':'output.init.nc'}
for fname in movelist1+movelist2:
    newname = newnames.get(fname,fname)
    newpath = pjoin(storedir,newname)
    if os.access(fname,os.R_OK):os.rename(fname,newpath)
    print '%s moved' % fname
for fname in copylist1+copylist2:
    if os.access(fname,os.R_OK):
        if os.path.islink(fname):
            shutil.copy2(os.readlink(fname),storedir)
            print '%s copied' % fname
        else:
            shutil.copy2(fname,storedir)
            print '%s copied' % fname

if args.movie:
    os.chdir(storedir)
    command = 'PlotG11_mp.py -n --dtime 50 --salinity --processes 6 --eos %s %s --filetype %s' % (args.eos,args.FK,args.filetype)
    print command
    os.system(command)

