#!/usr/bin/env python
#%run /Users/agn/VC_code/KNcode/PlotG11.py
import numpy as np
import os, sys
from os.path import join as pjoin
import nemo_rho

class NEMOGM:
    def __init__(self,nstepmax=None,nstep=0,dn=1,skew=False,restart=True,dirtry=None,salinity=False,ix=1,eos='linearT'):
        import netCDF4
        import glob

        if dirtry is None:
            dirtry0 = pjoin(os.environ['SCRSATA']
                                 ,'NEMO_GRI/OUTPUT/EEL')
    #         dirtry = '/Volumes/AGN/omf/scratch/lsm/NEMOBENCH/AGN/'
            if skew:
                dirtry = pjoin(dirtry0,'NEWISO_aeiv1000_ts1day_aht0.0_avt0.0/')
                dirtry = pjoin(dirtry0,'NEWISO_aeiv1000_ts1day_aht0.0_avt0_conv.0/')
            else:
    #             dirtry +='OLDISO_aeiv1000_ts1hr_aht0.1/'
    #             dirtry +='OLDISO_aeiv1000_ts1hr_aht0.1_explicit/'
                dirtry = pjoin(dirtry0,'OLDISO_aeiv1000_ts1hr_aht0.1_avt0.0_tvd/')

        print ' Using files in directory ',dirtry
        meshpath = pjoin(dirtry,'mesh_zgr.nc')
        fmesh = netCDF4.Dataset(meshpath)
        
        names = fmesh.variables.keys()
        if 'e3t' in names:
            # gdept_0 includes dummy point below ocean floor
            zt = fmesh.variables['gdept'][0,:-1,1:-1,1:-1]
            zz = fmesh.variables['gdepw'][0,:,1:-1,ix]
            # zt = fmesh.variables['gdept_0'][0,:-1]
            # zz = fmesh.variables['gdepw_0'][0,:]
            dz = fmesh.variables['e3t'][0,:-1,1:-1,1:-1]
        else:    
            # gdept_0 includes dummy point below ocean floor
            zt = fmesh.variables['gdept_0'][0,:-1]
            zz = fmesh.variables['gdepw_0'][0,:]
            dz = fmesh.variables['e3t_0'][0,:-1]
        # print 'zt= \n',zt,'\nzz= \n',zz,'\ndz= \n',dz
        fmesh.close()

        meshpath = pjoin(dirtry,'mesh_hgr.nc')
        fmesh = netCDF4.Dataset(meshpath)
        # yt includes dummy points to S and N of actual domain.
        yt = fmesh.variables['gphit'][0,1:-1,-1]
        yy = fmesh.variables['gphiv'][0,:-1,-1]
        xt = fmesh.variables['glamt'][0,-1,1:-1]
        xx = fmesh.variables['glamu'][0,-1,-1:-1]
        # print 'xt= \n',xt,'\n xx= \n',xx
        # print 'yt= \n',yt,'\n yy= \n',yy
        dx = fmesh.variables['e1t'][0,1:-1,1:-1]
        dy = fmesh.variables['e2t'][0,-1,-1]
        fmesh.close()
        
        self.area = dy*dx.sum()
        
        nzz = zz.shape[0]
        nyy = yy.shape[0]

        gravity  = 9.80665
        self.eos=eos
        if 'e3t' in names:
            self.dV = dy*dz*dx[None,:,:]
            self.zg_r1000 = zt*gravity*1.e-3 # multiply by 1.e-3 to give PE in kJ
            self.zt = zt[:,:,ix]
            self.zz = np.zeros([nzz,nyy])
            self.zz[:,1:-1] = .5*(zz[:,1:] + zz[:,:-1])
            self.zz[:,0] = zz[:,0]
            self.zz[:,-1] = zz[:,-1]
        else:
            self.dV = dy*dz[:,None,None]*dx[None,:,:]
            self.zg_r1000 = zt[:,None,None]*gravity*1.e-3
            self.zt = np.tile(zt[:,None],nyy-1)
            self.zz = np.tile(zz[:,None],nyy)
        self.dV_of_yz = self.dV[:,:,ix]
        # print 'dx= \n',dx,'\ndy= \n',dy,'\ndV= \n',self.dV_of_yz
        self.yy = np.tile(yy,(nzz,1))
        self.yt = yt

        if restart:
            frag = 'all*.nc'
            Temperature = 'tn'
            Salinity = 'sn'
            psiname = 'psiy_eiv'
        else:
            frag = '*grid_T.nc'
            Temperature = 'votemper'
            Salinity = 'vosaline'
        filepath = glob.glob(pjoin(dirtry,frag))[0]
        print filepath
        f = netCDF4.Dataset(filepath)

        TNd = f.variables[Temperature]
        SNd = f.variables[Salinity]
        nt = TNd.shape[0]
        if psiname in f.variables.keys():
            self.psiyNd = f.variables[psiname]
            self.psi = np.zeros([nzz,nyy],dtype=self.psiyNd.dtype)
        else:
            self.psiyNd = None
            self.psi = None
        if nstepmax is None:
            nstepmax = nt

        if nstep==0:
            initfile = 'output.init.nc'
            tryfile = pjoin(dirtry,initfile)
            if os.path.isfile(tryfile):
                initpath = tryfile
            else:
                tryfile = pjoin(dirtry0,initfile)
                if os.path.isfile(tryfile):
                    initpath = tryfile
                else:
                    sys.exit("can't find initfile")
            f0 =  netCDF4.Dataset(initpath)
            T0Nd = f0.variables['votemper']
            T0 = T0Nd[0,:-1,1:-1,1:-1]
            S0Nd = f0.variables['vosaline']
            S0 = S0Nd[0,:-1,1:-1,1:-1]
            f0.close()
        else:
            T0 = TNd[nstep-1,:-1,1:-1,1:-1]
            S0 = SNd[nstep-1,:-1,1:-1,1:-1]
        self.salinity = salinity
        if self.psi is None:
            slimit = None
            self.psiLevels = None#linspace(-slimit,slimit,9)
            self.psiAnn = None#linspace(-slimit,slimit,5)
        else:
            slimit = 0.01
            contmax = slimit/10.
            low_levels = np.linspace(-contmax,contmax,9)
            low_anns = np.linspace(-contmax,contmax,5)
            hi_levels = np.linspace(-slimit,slimit,5)
            hi_anns = np.linspace(-slimit,slimit,5)
            
            self.psiLevels = 2000.*np.union1d(low_levels,hi_levels)
            self.psiAnn = 2000.*np.union1d(low_anns,hi_anns)
              

        self.time = f.variables['time_counter'][:]
        if restart: self.time *= f.variables['rdttra1'][:]
        self.nstepmax=nstepmax
        self.nstep=nstep
        self.dn = dn
        self.ix = ix
        tmskval = 0.
        self.rhmsk = T0==tmskval
        self.rho0 = self.find_sigma(T0,S0)
        if self.salinity:
            self.S0 = S0
        else:
            self.S0 = None
        self.Drho0Dz = 2.e-3
        self.TNd =  TNd
        self.SNd =  SNd
        self.dvar =  None
        self.var =  None
        self.f =f
        # print 1/0
        

    def find_sigma(self,T,S):
        if self.eos=='linearT':
            alpha = -2.e-4
            beta = 1.e-3
            # Require sigma = 26. at S=35.,T=15, so
            #   26= sigma0 +.207*15.+1.035*35.=> sigma0=-13.33
            rho = -7.12 +1035.*(alpha*T + beta*35.)
            # rho = 0.2*(15.-self.TNd[nnc,:-1,1:-1,1:-1]) + 26.
        elif self.eos=='linearTS':
            alpha = -2.e-4
            beta = 1.e-3
           # Require sigma = 26. at S=35.,T=15, so
            #   26= sigma0 +.207*15.+1.035*35.=> sigma0=-13.33
            rho = -7.12 +1035.*(alpha*T + beta*S)
            # rho = 0.2*(15.-self.TNd[nnc,:-1,1:-1,1:-1]) + 26.
        elif self.eos=='JM94':
            refdepth = 500.
            drho = 26. - nemo_rho.eos.rho(15.,35.,refdepth)
            refdepth_km = refdepth/1000.
            rho = drho + nemo_rho.eos.sigma_n(T.ravel(),S.ravel(),refdepth_km).reshape(T.shape)
        else:
            rho=None
        return rho
    
    def output(self):
        nnc = self.nstep - 1
        dvar = None
        var = None
        
        if nnc < 0:
            rho = self.rho0
            time = 0.
            S = self.S0
        else:
            if self.psiyNd is not None:
                self.psi[:,:] = self.psiyNd[nnc,:,:-1,self.ix]
            T,S = self.TNd[nnc,:-1,1:-1,1:-1], self.SNd[nnc,:-1,1:-1,1:-1]
            rho = self.find_sigma(T,S)
            time = self.time[nnc]
            if not self.salinity:
                S = None
        psi = self.psi
        return rho,psi,dvar,var,S,time

    def step_time(self,dn):
        self.nstep += dn
        return self.nstep

    def finish(self):
        self.f.close()

class toyGM:
    def __init__(self,nstepmax=None,maxtime=None,nstep=0,dn=None,dtime=None,power=2, skew=True,leapfrog=False,asselin=False,odd_asselin=False,test=False,f_asselin=0.,euler=False,testprint=False,dt00=0.0001,k_test=1.e3):

        mm1,nm1 = 30,30 # no of active T-points in y, z
        refine = 1
        mm1 = mm1*refine
        nm1 = nm1*refine
#         mm1,nm1 = 90,90 # no of active T-points in y, z
#         mm1,nm1 = 600,600 # no of active T-points in y, z
#         mm1,nm1 = 10,10 # no of active T-points in y, z
        refine = mm1/30
#         dt00 = 100.e-6
        dt0 = dt00/(refine**2)#0.2e-6#8.3333e-6#25.e-6#4.e-6#50.e-6#
        print 'mm1,nm1= ',mm1,nm1,' refine= ',refine,' dt0= ',dt0
        m,n = mm1+1,nm1+1 # no of  psi-points in y, z
        mp1,np1 = mm1+2,nm1+2 # total no of T-points in y, z

        dy,dz = 1./mm1,1./nm1
        yy = np.linspace(0.,1.,m)
        yt = .5*(yy[:-1]+yy[1:])
        zz = np.linspace(0.,1.,n)
        self.zt = .5*(zz[:-1]+zz[1:])
        self.zz,self.yy = yy,zz

        Eps = 1.e-4
        if nstepmax is None:
            if maxtime is not None:
                nstepmax = np.int32(maxtime/dt0 + Eps)
            else:
                sys.exit('must specify nstepmax or maxtime')
        if dn is None:
            if dtime is not None:
                dn = np.int32(dtime/dt0 + Eps)
            else:
                sys.exit('must specify dn or dtime')

        import griffies
        GMT = griffies.gmtoy
        self.GMT = GMT

        def get_bot(m,mp1,n):
            Eps = 1.e-4
            start_frac,stop_frac = 0.33333,0.5
            jstart,jstop = [int((m-1)*frac+Eps)+1 for frac in [start_frac,stop_frac]]
            print jstart,jstop
# for consistency with NEMO.....
            jstart-=3
            jstop-=2
            nbot = np.empty(mp1)
            nbot[:jstart] = n
            nbot[jstop:] = n
            nbot[jstart:jstop] = 1+(n-1)/2
#             nbot[jstart:jstop] = n
            return nbot

        if GMT.length_t > 0:
            GMT.dealloc_vars()

        GMT.kh = 0.0
        GMT.kv = 0.#0.05
        GMT.agm = 1.#2.#e-3
        GMT.slimit = 2.#7.78#10.#2.
        print 'slope limit= ',GMT.slimit
        GMT.power = power
        GMT.convect = 0
        GMT.simple = 0
        GMT.refine = refine#20#3

        GMT.skew=skew
        GMT.leapfrog = leapfrog
        GMT.asselin = asselin
        GMT.odd_asselin = odd_asselin
        GMT.f_asselin = f_asselin
        GMT.convect=0
        GMT.test = test
        GMT.k_test = k_test
        if test and testprint:
            GMT.testprint = True
        else:
            GMT.testprint = False

        GMT.mm1,GMT.nm1,GMT.m,GMT.n,GMT.mp1,GMT.np1 = mm1,nm1,m,n,mp1,np1
        GMT.dy,GMT.dz,GMT.dt0 = dy,dz,dt0

        GMT.do_allocation()
        GMT.nbot[...] = get_bot(m,mp1,n)
        GMT.initialize()
        rhmsk = (GMT.tmask[1:-1,1:-1]==0.).T

        GMT.ttt[...] =np.array([[1.,1.,1.],[0.,0.,0.]]).T#ones([3])
        GMT.max_energy = 100.

        initial = 'random'
        initial = 'rslope'
#         initial = 'mslope'
        initial = 'slope'

        if initial =='random':
            rho = np.random.uniform(low=-1.0,high=1.0,size=mm1*nm1)\
                       .reshape(nm1,mm1)
        elif initial =='slope' or  initial =='rslope' or  initial =='mslope':
            rho = np.zeros([mm1,nm1])
            for j in range (mm1):
                y = yt[j] #- .5*dy
                nb = GMT.nbot[j]
                for k in range(nm1):
                    z = -self.zt[k] #+ .5*dz
                    rho[k,j] = y - z -1.
#                     rho[k,j] = 0.5 - z -1.
            if initial =='rslope':
               rho *= -1.
            elif initial =='mslope':
                for j in range(mm1):
                    rho[:10,j] = rho[10,j]
 #        print 'set rho'

        rho[rhmsk] = -1.e5
        fullrho = np.zeros([GMT.length_t,np1,mp1]) - 1.e5
        for i in range(GMT.length_t):
            fullrho[i,1:-1,1:-1]=rho[:,:]
        GMT.rho[...] = fullrho.T
 #        print 'set rho to fortran'

        self.rhmsk = rhmsk

        psimax = min(GMT.slimit,2.)*GMT.agm
        self.psiLevels = np.linspace(-psimax,psimax,17)
        self.psiAnn = np.linspace(-psimax,psimax,9)

        self.dt0 = dt0
        self.nstepmax=nstepmax
        self.nstep=nstep
        self.dn = dn
        self.nnc = GMT.nnc
        self.dV_of_yz = dy*dz*np.outer(np.ones_like(self.zt),np.ones_like(yt))
        self.area = 1.
        self.rho0 = rho
        self.Drho0Dz = 1.
        self.S0 = None

        self.rho = GMT.rho[1:-1,1:-1,:].T
        self.psi = GMT.psi.T
        self.psimask = GMT.psimask.T
        self.dvar =  GMT.dvar[1:-1,1:-1].T
        self.var =  GMT.var[1:-1,1:-1,:].T
        self.step = GMT.step

    def step_time(self,dn):
        #         nnm,nnc,nnp = self.GMT.nnm, self.GMT.nnc, self.GMT.nnp
        #         print self.rho0[4,3],self.nstep,self.rho[(nnm-1,nnc-1,nnp-1),4,3]-self.rho0[4,3]
        #         print nnm,nnc,nnp
        #         rho = self.rho[self.GMT.nnp-1,...]
        #         rhmsk = rho==-1.e5
#         if self.nstep>4: print 1/0
        #         if abs(rho[~rhmsk]).max() > 1.: print 1/0
        #         self.step(dn)
        #         self.nstep += dn
        #         nnm,nnc,nnp = self.GMT.nnm, self.GMT.nnc, self.GMT.nnp
        #         print self.rho0[4,3],self.nstep,self.rho[(nnm-1,nnc-1,nnp-1),4,3]-self.rho0[4,3]
        #         print nnm,nnc,nnp
        #         return self.nstep
        if self.nstep==0:
            mixts=True
            time_series =self.step(dn,mixts).T
            self.time_series = [time_series]
        else:
            mixts=euler
            time_series = self.step(dn,mixts)[1:].T
            self.time_series += [time_series]

        self.nstep += dn
#         if self.nstep>3: print 1/0
        return self.nstep

    def output(self):
        nnc = self.nnc - 1
        rho = self.rho[nnc,...]
        psi = self.psi[...]
        dvar = self.dvar[...]
        var = self.var[nnc,...]
        S = None
        time = self.dt0*self.nstep
        return rho,psi,dvar,var,S,time

    def finish(self):
        pass

if  __name__=='__main__':
    import shutil,glob
    from optparse import OptionParser

    usage = "usage: %prog  [options]"
    parser = OptionParser(usage)
    parser.add_option('-o','--odd',dest='odd_asselin',
              help='asselin factor, only use for odd timesteps',
              default=None)
    parser.add_option('-t','--test',dest='test',
              help='test time stepping with SHM..set k, suggest = 1.e3',
              default=None)
    parser.add_option('-i','--indir',dest='indir',
              help='input directory',
              default=None)
    parser.add_option('-l','--leapfrog',dest='leapfrog',action='store_true',
              help='use leapfrog',
              default=False)
    parser.add_option('-s','--salinity',dest='salinity',action='store_true',
              help='plot out salinity field',
              default=False)
    parser.add_option('-a','--advective',dest='advective',action='store_true',
              help='do simple advection',
              default=False)
    parser.add_option('-p','--pictures',dest='pictures',
              help='''Either: movie, publish (pdf), nothing (nothing),
              screen (display figures), noninteractive (produce figures)''',
              default='movie')
    parser.add_option('-f','--filter',dest='asselin',
              help='asselin factor filter; default is False',
              default=None)
    parser.add_option('-e','--euler',dest='euler',action='store_true',
              help='mix every dtime; default is False',
              default=False)
    parser.add_option('--eos', dest='eos',help='equation of state: choices are linearT,linearTS', default='linearT')
    parser.add_option('--dtime',dest='dtime',
              help='Time between plots (and possible Euler timesteps). Default is 1/20th maxtime',
              default=None)
    parser.add_option('--dt00',dest='dt00',
              help='Time step. Default is 0.0001',
              default='0.0001')
    parser.add_option('-n','--nemo',dest='nemo',action='store_true',
              help='Default is False',
              default=False)
    parser.add_option('--power',dest='power',
              help='variance order that is monitored. Defaults to 2 (simple variance)',
              default='2')

    options,args = parser.parse_args(sys.argv[1:])

    leapfrog = options.leapfrog
    salinity = options.salinity
    euler=options.euler
    asselin=False
    odd_asselin=False
    if options.asselin is not None:
        asselin=True
        f_asselin = float(options.asselin)
    elif options.odd_asselin is not None:
        odd_asselin=True
        f_asselin = float(options.odd_asselin)
    else:
        f_asselin = 0.

    if options.test is None:
        test = False
        k_test = 1.e3
    else:
        test = True
        k_test = float(options.test)
        if k_test == 0.:k_test = 1.e3

    pictures = options.pictures
    if pictures=='nothing': pictures=None
    power = int(options.power)
    skew = not options.advective
    nemo = options.nemo

    if not nemo:
        maxtime = 0.12
        dt00 = float(options.dt00)
        if options.dtime is None:
            dtime = maxtime/20.
        else:
            dtime = float(options.dtime)

        runflag=''
        nsteps=int(dtime/dt00+0.001)
        se = 'e%i'%nsteps

        pure = leapfrog and not asselin and not odd_asselin and not euler

        sa = ('%6.3f'%f_asselin).split('.')[1]
        sk = '%05d' % (int(k_test+.1))
        for l,s in zip([test,leapfrog,pure,euler,odd_asselin,asselin],['t'+sk,'l','p',se,'o'+sa,'f'+sa]):
            if l:runflag += s
            print l,s,runflag

        if runflag=='':
            runflag='d'

    currdir = os.getcwd()


    if nemo:
        GM = NEMOGM(nstepmax=None,nstep=0,dn=1,skew=True,dirtry=currdir,salinity=salinity,eos=options.eos)
    else:
        GM = toyGM(maxtime=maxtime,nstep=0,dtime = dtime,power=power,skew=skew,
                   leapfrog=leapfrog,asselin=asselin,odd_asselin=odd_asselin,test=test,
                   f_asselin=f_asselin,euler=euler,testprint=False,dt00=dt00,k_test=k_test)
        #     GM = toyGM(nstepmax=30000,dn=100....

    dn = GM.dn
    nstep = GM.nstep
    nstepmax = GM.nstepmax

    if pictures=='movie':
        tempdir = 'temp'
        if os.path.exists(tempdir):
            oldfiles = os.listdir(tempdir)
            [os.remove(tempdir+'/'+file) for file in oldfiles]
        else:
           os.mkdir(tempdir)

        os.chdir(tempdir)

    import plot_picts
    vmin = None
    vmax = None
    psiLevels = GM.psiLevels
    psiAnn = GM.psiAnn

    if nemo:
        vmin = 25.0
        vmax = 28.
        # vmin = None
        # vmax = None
    else:
        vmin = -0.25
        vmax = 0.075
        # vmin = -1.0
        # vmax = 1.0

    picture = plot_picts.GMplot(GM,nbins=30,pictures=pictures,vmin=vmin,vmax=vmax,
                 psiLevels=psiLevels,psiAnn=psiAnn,power=power,tex=False,Smin=None,Smax=None)

    while 1:
        rho,psi,dvar,var,S,time = GM.output()
        picture(rho,time,psi=psi,dvar=dvar,var=var,S=S)
        picture.savefile(nstep,dpi=None)
        if nstep>=nstepmax: break
        nstep = GM.step_time(dn)

    GM.finish()
    if not nemo and GM.time_series is not None:
        T = np.hstack(GM.time_series)
        import netCDF4
        ncfile=pjoin(currdir,'time_series_dt00_%6.4f.nc' %dt00)
        if os.path.exists(ncfile):
            cdf_flag='a'
        else:
            cdf_flag='w'
        f = netCDF4.Dataset(ncfile, cdf_flag, format='NETCDF4')
        if not 'time' in f.dimensions: f.createDimension('time', None)
        print 'runflag= ',runflag
        if not runflag in f.groups.keys():
            print 'creating group ',runflag
            group = f.createGroup(runflag)
#             u=group.createVariable('u','f8',('time',),fill_value = GM.GMT.mask_val)
            u=group.createVariable('u','f8',['time'])
            print GM.GMT.mask_val
            u.mask_value = GM.GMT.mask_val
            u[:] = T[0,:]
            v=group.createVariable('v','f8',['time'])
            v[:] = T[1,:]
            v.mask_value = GM.GMT.mask_val
        f.close()

    if pictures=='noninteractive' or pictures is None:
        pass
    elif pictures=='movie':
        print 'Making movie animation.mpg - this make take a while'
        os.system('mencoder -nosound -ovc lavc \
        -lavcopts vbitrate=5000:vcodec=mjpeg \
        -mf type=png:fps=30 -o '+currdir+'/WGM%05d_%07d.avi \
        mf://\*.png -v'%(dn,nstepmax))
        # cleanup
        for fname in picture.files: os.remove(fname)
        os.chdir(currdir)
        os.rmdir(tempdir)
    else:
        picture.show()
