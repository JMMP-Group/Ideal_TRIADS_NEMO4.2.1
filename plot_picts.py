import os, sys
import numpy as np
import numpy.ma as ma
import bin_all as B
from sentinels import CenteredNorm

def smooth(x,window_len=10,window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=np.r_[2*x[0]-x[window_len:1:-1],x,2*x[-1]-x[-1:-window_len:-1]]
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval(window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='same')
    return y[window_len-1:-window_len+1]


class GMplot:
    def __init__(self,GM,nbins=40,pictures='screen',
                 vmin=None,vmax=None,binmin=None,binmax=None,binVolmax =None,
                 psiLevels=None,psiAnn=None,
                 diffMin=-0.05,diffMax=0.05,
                 dvarbinVolmin=-0.05,dvarbinVolmax=0.03,dvarmin=-20.,dvarmax=20.,
                 SbinVolmin=0.,SbinVolmax=None,Smin=34.7,Smax=35.3,
                 bleck=True,power=2,tex=False,S00=35.):

        self.S00 = S00
        ########################################################################
        # check if variables are dimensional
        ########################################################################
        if GM.zz.max() > 10.:
            self.dimensional = True
        else:
            self.dimensional = False

        # if variables are dimensional express time in days
        if self.dimensional:
            self.timeUnits = 86400.
        else:
            self.timeUnits = 1.

        ########################################################################
        # calculate starting pe and average variance..
        ########################################################################
        rho = GM.rho0
        rhmsk = GM.rhmsk
        self.rhmsk = rhmsk
        self.power = power

        self.zt = GM.zt
        self.zg_r1000 = GM.zg_r1000
        dV =  self.masked_ravel(GM.dV)
        rV =  1./dV.sum()
        self.rA =  1./GM.area
        self.rV,self.dV = rV,dV
        rho_ravelled = self.masked_ravel(rho)
        self.rhmean0 = np.dot(dV,rho_ravelled)*rV
        self.rhvmean0 = np.dot(dV,self.masked_ravel(rho**power))*rV
        self.pe0 =  -np.dot(dV,self.masked_ravel(self.zg_r1000*rho))*self.rA
        print 'pe0= ',self.pe0,' kJ/m2'

        self.pictures = pictures
        if pictures is None: return
        ########################################################################
        # picture will be drawn. First sort out rho and psi
        ########################################################################

        # save grid
        yy = GM.yy
        zz = GM.zz
        self.yy = yy
        self.zz = zz
        self.zv = GM.zv

        # set domain size
        self.ymin,self.ymax = yy.min(),yy.max()
        self.zmin,self.zmax = zz.min(),zz.max()

        # set rho minimum and maximum
        vmin0 = ma.masked_where(rhmsk,rho).min()
        vmax0 = ma.masked_where(rhmsk,rho).max()

        if vmin is None:
            vmin = vmin0
        if vmax is None:
            vmax = vmax0
        self.vmin = vmin
        self.vmax = vmax
        print 'vmin0, vmax0',vmin0,vmax0
        print 'vmin, vmax',vmin,vmax

        # set fields on i=ix
        self.ix = GM.ix
        self.rhmsk_ix = self.rhmsk[...,self.ix]


        # set contour and annotation levels for psi
        if psiLevels is not None:
           self.psiLevels =  psiLevels
        if psiAnn is not None:
           self.psiAnn =  psiAnn

        print '\n\n psiLevels = ', psiLevels
        print 'psiAnn = ', psiAnn,'\n'

        ########################################################################
        # Setup bins and maximum watermass 'density' for watermass census
        ########################################################################

        # set minimum, maximum  bin edge and bin width
        if binmin is None:
            self.binmin = vmin0
        else:
            self.binmin = binmin
        if binmax is None:
            self.binmax = vmax0
        else:
            self.binmax = binmax

        self.drho = (self.binmax -self.binmin)/nbins

        # if we do Bleck remapping, add extra bins
        if bleck:
            self.bleck = True
            nbins += 2
            self.binmin -= self.drho
            self.binmax += self.drho

        # set all bin boundaries
        self.rho_bin_edges = np.linspace(self.binmin,self.binmax,nbins+1)

        # set max of bin plot to be 6 x even spread over all densities
        if binVolmax is None:
            self.binVolmax = 6./(rV*(vmax0-vmin0))#/nbins
        else:
            self.binVolmax = binVolmax
        # if variables are dimensional bin volumes in 10^12 m^3/[drho in bins]
        if self.dimensional:
            self.binVolmax *= 1.e-12

        ########################################################################
        # Setup variables and axes limits for calculation of effective diffusivity
        ########################################################################

        # need horizontal area & typical drhodz to relate change in mass to diffusivity
        self.area = GM.area
        self.Drho0Dz = GM.Drho0Dz

        # will later calculate vol change of water denser than given density ...
        #  first calculate vol water denser than given density at start, G0
        #  initialise arry for vol water denser than given density later, G
        self.G,self.G0 = np.zeros([2,nbins+1])
        # G,G0 = [np.zeros([nbins+1]) for x in range(2)]
        self.bin_widths = np.diff(self.rho_bin_edges)
        if self.bleck:
            rho_bin_centres = self.rho_bin_edges[:-1] + .5*self.drho
            hist = B.bin_bleck_wts(rho_bin_centres,rho_ravelled,dV)
        else:
            hist = B.bin_all_wts(self.rho_bin_edges,rho_ravelled,dV)
#         hist_smth = smooth(hist,window_len=5,window='bartlett')
        self.G0[1:]=hist.cumsum()

        self.diffMin,self.diffMax = diffMin,diffMax

        ########################################################################
        # If d/dt variance field [dvar] is used, set its limits
        ########################################################################
        dvar = GM.dvar
        if dvar is not None:
            self.dvarmax= dvarmax
            self.dvarmin= dvarmin
            self.dvarbinVolmin = dvarbinVolmin
            self.dvarbinVolmax = dvarbinVolmax

        ########################################################################
        # If salinity [S] is used, set its limits
        ########################################################################
        S = GM.S0
        if S is not None:
            self.Smax= Smax
            self.Smin= Smin
            self.SbinVolmin = SbinVolmin
            self.SbinVolmax = SbinVolmax

        ########################################################################
        # and its sum and variamce..
        ########################################################################
            S_ravelled = self.masked_ravel(S)
            self.Smean0 = np.dot(dV,S_ravelled)*rV
            self.Svmean0 = np.dot(dV,self.masked_ravel(S**power))*rV
        ########################################################################
        # Initialize empty list of plotfiles
        ########################################################################
        self.files = []


        ########################################################################
        # Setup figsize and import matplotlib stuff
        ########################################################################
        if dvar is not None or S is not None:
            self.figsize=(15,7.5)
        else:
            self.figsize=(8,7.5)
        self.fig = None

        # if displaying figures produce each in new figure window, otherwise save and delete.
        # if displaying figures use default interactive backend,
        #                        otherwise use noninteractive (Agg or pdf)
        if pictures=='screen':
            self.deleteFig = False
        else:
            self.deleteFig = True
            import matplotlib
            if pictures=='movie' or pictures=='noninteractive':
                matplotlib.use('Agg')
            elif pictures=='publish':
                 matplotlib.use('pdf')

        # need transforms to move from axes-relative to figure relative coordinates..
        import matplotlib.transforms as mt
        self.mt = mt

        import matplotlib.pyplot as PL
        import add_cmaps
        cm = matplotlib.cm
        # import matplotlib.cm as cm
        self.PL = PL
        self.cm = cm

        self.tex = tex
        self.show = PL.show

        from matplotlib import rc
        rc('axes', titlesize=13)
        # if using tex use mathtime with sistyle to format numbers
        if tex:
            rc('ps',usedistiller='xpdf')
            rc('text', usetex=True)
            rc('text', dvipnghack=True)
            rc('font', family='serif')
            rc('text.latex', preview=True)
            rc('text.latex',no_font_preamble=True)
            rc('text.latex',preamble=r"""
            %\documentclass{article}
            %\usepackage{mathptmx}
            \usepackage[scaled=0.92]{helvet}
            \usepackage{type1cm}
            %\usepackage[T1]{fontenc}
            \usepackage{mathtime}
            \usepackage{sistyle}
            """)
        else:
#             rc('font', sans-serif='stixsans')
            rc('mathtext', default='regular')
            rc('mathtext', fontset='stixsans')

    def masked_ravel(self,rho):
        return rho.ravel().compress(~self.rhmsk.ravel())

    def __call__(self,rho,time,psi=None,dvar=None,var=None,S=None):
        ########################################################################
        # calculate pe change and variance changes
        ########################################################################
        rhmsk = self.rhmsk
        rV,dV = self.rV, self.dV
        power = self.power
        rho_ravelled = self.masked_ravel(rho)
        rhmean = np.dot(dV,rho_ravelled)*rV
        drhmean = rhmean - self.rhmean0
        rhvmean = np.dot(dV,self.masked_ravel(rho**power))*rV
        varmean = rhvmean - self.rhvmean0

        pe = -np.dot(dV,self.masked_ravel(self.zg_r1000*rho))*self.rA
        dpe = pe - self.pe0
        print '\n at time ',time/self.timeUnits,' pe= ',pe,' kJ/m2  dpe= ',dpe,' kJ/m2  dvar mean= ',varmean

        # if we step forward variance
        if dvar is not None:
            dvarmskd = ma.masked_where(rhmsk,dvar)
            dvarmean2 = dvarmskd.mean()
            print 'dvardt= ',dvarmean2#'dvardt mean %8.3g'% dvarmean2
            if var is not None:
                varmean2 = ma.masked_where(rhmsk,var).mean()
                print 'dvar mean from variance timestepping ',varmean2#'dvar mean from variance timestepping %8.3g'%varmean2


        if self.pictures is None: return
        ########################################################################
        # picture will be drawn. First plot rho and psi
        ########################################################################

        PL = self.PL
        mt = self.mt
        ix = self.ix
        rhmsk_ix = self.rhmsk_ix

        # either clear old figure instance or create a new one
        if self.fig is not None and self.deleteFig:
            PL.clf()
        else:
            self.fig = PL.figure(figsize=self.figsize)
        fig = self.fig

        # if plotting dvar as well want two plots side-by-side
        if dvar is not None or S is not None:
            fig.subplots_adjust(left=0.05, bottom=None, right=1., top=None,
                    wspace=None, hspace=None)
            ax = fig.add_subplot(121)
        else:
            ax = fig.add_subplot(111)

        # ensure background is black for (masked) topography
        ax.set_axis_bgcolor('k')

        # plot rho and (possibly) psi
        ymin,ymax = self.ymin,self.ymax
        zmin,zmax = self.zmin,self.zmax

        rhmskd = ma.masked_where(rhmsk_ix,rho[...,ix])
        pcolor = ax.pcolorfast
        rhplot = pcolor(self.yy,self.zv,rhmskd,vmax=self.vmax,vmin=self.vmin,
                           cmap=self.cm.PinkSST_r)
        if psi is not None:
            psicont = ax.contour(self.yy,self.zv,psi,self.psiLevels,colors='w')
            ax.clabel(psicont,self.psiAnn)

        ax.set_xlim(ymin,ymax)
        ax.set_ylim(zmax,zmin)

        if self.tex:
            ax.set_title(r'$\Delta\rho= \num{ %8.3g }$ $\Delta\mathrm{var }=\num{%8.3g}$ $\Delta\mathrm{PE }= \num{%.3g}$' \
                 %(drhmean,varmean,dpe))
            ax.text(0.05,0.95,r'$\rho\textrm{ at }\num{ %8.6g}$'
                    % (time/self.timeUnits),transform=ax.transAxes,fontsize=21,color='white',va='top')
        else:
            ax.set_title(r'$\Delta\rho=$ %8.3g $\Delta\mathrm{var }=$ %8.3g $\Delta\mathrm{PE }=$ %8.3g $kJ\,m^{-2}$' \
                 %(drhmean,varmean,dpe))
            ax.text(0.05,0.95,r'$\rho\,\mathrm{at}\, %8.6g$'
                    % (time/self.timeUnits),transform=ax.transAxes,fontsize=21,color='white',va='top')
        fig.colorbar(shrink=0.5,ax=ax,mappable=rhplot)

        ########################################################################
        # now plot histogram of watermass distribution
        ########################################################################

        #    first find position for axes and draw them
        lbwh_axes = 0.4,0.1,0.45,0.3
        Bbox = self.mt.Bbox.from_bounds(*lbwh_axes)
        trans = ax.transAxes + fig.transFigure.inverted()
        lbhw_fig = self.mt.TransformedBbox(Bbox, trans).bounds
        a2=fig.add_axes(lbhw_fig,axisbg='b')
        a2.patch.set_alpha(0.2)

        #    bin densities, weight by volume of box
        if self.bleck:
            rho_bin_centres = self.rho_bin_edges[:-1] + .5*self.drho
            hist = B.bin_bleck_wts(rho_bin_centres,rho_ravelled,dV)
        else:
            hist = B.bin_all_wts(self.rho_bin_edges,rho_ravelled,dV)

        #   divide by width of box to give watermass 'density' [volume/unit of rho]
        #   if dimensional, express volumes in 10^6 m^3/drho
        if self.dimensional:
            hist_plot = (1.e-12/self.drho)*hist
        else:
            hist_plot = hist/self.drho
        # a2.tick_params(labelsize='small')
        #   plot bar chart in white
        a2.bar(self.rho_bin_edges[:-1],hist_plot,width=self.bin_widths,
               color=(1.,1.,1.0),linewidth=0.)

        a2.set_xlim(self.binmin,self.binmax)
        a2.set_ylim(0.,self.binVolmax)

        for label in a2.xaxis.get_ticklabels():
            label.set_color('white')
            label.set_fontsize(8)
        for label in a2.yaxis.get_ticklabels():
            label.set_color('white')
            label.set_fontsize(8)

        # if dimensional, density nos too large to have all ticks as major ticks,
        #   so alternate with minor ticks
        if self.dimensional:
            xticks = a2.get_xticks()
            nxticks = len(xticks)
            major_ticks = [xticks[i] for i in range(0,nxticks,2)]
            minor_ticks = list(set(xticks)-set(major_ticks))
            minor_ticks.sort()
            if major_ticks[-1]<major_ticks[0]:
                minor_ticks.reverse()

            a2.set_xticks(major_ticks)
            a2.set_xticks(minor_ticks,minor=True)

        ########################################################################
        # now plot equivalent diapycnal diffusivity
        ########################################################################

        #   first calculate time-integrated diapycnal diffusive density transport D
        #   as difference of mass denser than given density from at start
        self.G[1:]=hist.cumsum()
        dG = self.G - self.G0
        D = self.drho*( dG.cumsum() - .5*(dG + dG[0]))

        #   then calculate diffusivity.....divide by time elapsed, area and drhodz
        #   dividing by total time gives time-mean diffusivity
        #   Where variables are dimensional,express in cm^2/s

        if time> 0.:
            K_diap = -D/(self.area*self.Drho0Dz*time)
            if self.dimensional:
                K_diap *= 1.e4  # express in cm^2/s
        else:
            K_diap = np.zeros_like(D)

        # use twinaxis, plot curve in red
        a2r = a2.twinx()
        a2r.patch.set_alpha(0.)
        a2r.plot(self.rho_bin_edges,K_diap,color='r')
        self.diffMin = K_diap.min()
        self.diffMax = K_diap.max()
        if self.diffMax==self.diffMin:self.diffMax += 1.e-9

        for label in a2r.yaxis.get_ticklabels():
            label.set_color('white')
            label.set_fontsize(8)
        a2r.set_ylim(self.diffMin,self.diffMax)
        a2r.set_xlim(self.binmin,self.binmax)
        # a2r.tick_params(labelsize='small')

        ########################################################################
        # plot second figure showing dvar and binning of that into rho classes ..
        ########################################################################
        if dvar is not None:
            ax = fig.add_subplot(122)
            # ensure background is black for (masked) topography
            ax.set_axis_bgcolor('k')

            # plot dvar/dt [dvar]
            pcolor = ax.pcolorfast
            dvarmskd =  ma.masked_where(rhmsk_ix,dvar[...,ix])
            varplot=pcolor(self.yy,self.zz,dvarmskd,vmin=self.dvarmin,vmax=self.dvarmax
                           ,cmap=PL.get_cmap('greem'))
            ax.set_xlim(ymin,ymax)
            ax.set_ylim(zmax,zmin)

            if var is not None:
                if self.tex:
                    ax.set_title(r'$\frac{\partial}{\partial t}\langle\mathrm{var}\rangle= \num{ %8.3g}$ $\Delta\mathrm{var }=\num{%8.3g}$'
                 %(dvarmean2,varmean2))
                    ax.text(0.1,0.2,r'$\frac{\partial}{\partial t}\textrm{var at time }\num{ %8.6g}$'
                            % (time/self.timeUnits),fontsize=21,color='black')
                else:
                    ax.set_title(r'$\frac{\partial}{\partial t}\langle\mathrm{var}\rangle= %8.3g$  $\Delta\mathrm{var }=%8.3g$'
                 %(dvarmean2,varmean2))
                    ax.text(0.1,0.2,r'$\frac{\partial\,\mathrm{var}}{\partial t}\;\mathrm{at}\;%8.6g$'
                            % (time/self.timeUnits),fontsize=21,color='black')
            else:
                ax.set_title('dvar at time %8.6g dvarmean %8.6g'
                      %(time/self.timeUnits,dvarmean2))
            fig.colorbar(shrink=0.5,ax=ax,mappable=varplot)

            ########################################################################
            # bin dvar/dt into rho classes ..
            ########################################################################
            # setup axes
            lbwh_axes = 0.55,0.1,0.43,0.3
            Bbox = self.mt.Bbox.from_bounds(*lbwh_axes)
            trans = ax.transAxes + fig.transFigure.inverted()
            lbhw_fig = self.mt.TransformedBbox(Bbox, trans).bounds
            a2=fig.add_axes(lbhw_fig,axisbg='white')
            a2.patch.set_alpha(0.2)

            # calculate weights
            dvarbin = self.masked_ravel(dvar)*dV

            if self.bleck:
                hist = B.bin_bleck_wts(rho_bin_centres,rho_ravelled,dvarbin)
            else:
                hist = B.bin_all_wts(self.rho_bin_edges,rho_ravelled,dvarbin)

            #   divide by width of box to give dvar/dt 'density' [rate of change of variance/unit of rho]
            #   if dimensional, express volumes in 10^6 m^3/drho
            if self.dimensional:
                hist_plot = (1.e-12/self.drho)*hist
            else:
                hist_plot = hist/self.drho

            #   plot bar chart in black
            a2.bar(self.rho_bin_edges[:-1],hist,width=self.bin_widths,color=(0.,0.,.0),linewidth=0.)

            a2.set_xlim(self.binmin,self.binmax)
            a2.set_ylim(self.dvarbinVolmin,self.dvarbinVolmax)

            for label in a2.xaxis.get_ticklabels():
                label.set_color('black')
                label.set_fontsize(8)
            for label in a2.yaxis.get_ticklabels():
                label.set_color('black')
                label.set_fontsize(8)
            #             PL.setp(a2.get_xticklabels(), color='k')
            #             PL.setp(a2.get_yticklabels(), color='k')

        ########################################################################
        # plot second figure showing S and binning of that into rho classes ..
        ########################################################################
        if S is not None:
            S_ravelled = self.masked_ravel(S)
            Smean = np.dot(dV,S_ravelled)*rV
            dSmean = Smean - self.Smean0
            Svmean = np.dot(dV,self.masked_ravel(S**power))*rV
            dSvmean = Svmean - self.Svmean0

            ax = fig.add_subplot(122)
            # ensure background is black for (masked) topography
            ax.set_axis_bgcolor('k')

            # plot salinity [S]
            pcolor = ax.pcolorfast
            Smskd = ma.masked_where(rhmsk_ix,S[...,ix]-self.S00)
            norm = CenteredNorm()
            Splot=pcolor(self.yy,self.zv,Smskd,vmin=self.Smin,vmax=self.Smax
                           ,cmap=PL.get_cmap('greem'),norm=norm)
            ax.set_xlim(ymin,ymax)
            ax.set_ylim(zmax,zmin)

            if self.tex:
                ax.set_title(r'$\langle\mathrm{S}\rangle= \num{ %8.4g}$ $\Delta\mathrm{S}^2=\num{%8.4g}$'
             %(dSmean,dSvmean))
                ax.text(0.05,0.95,r'$\frac{\partial}{\partial t}\textrm{S at time }\num{ %8.6g}$'
                        % (time/self.timeUnits),transform=ax.transAxes,fontsize=21,color='red',va='top')
            else:
                ax.set_title(r'$\Delta\langle S\rangle= %8.4g$  $\Delta\langle S^2\rangle=%8.4g$'
             %(dSmean,dSvmean))
                ax.text(0.05,0.95,r'$S\;\mathrm{at}\;%8.6g$'
                        % (time/self.timeUnits),transform=ax.transAxes,fontsize=21,color='red',va='top')
            fig.colorbar(shrink=0.5,ax=ax,mappable=Splot)

            ########################################################################
            # bin S into rho classes ..
            ########################################################################
            # setup axes
            lbwh_axes = 0.55,0.1,0.43,0.3
            Bbox = self.mt.Bbox.from_bounds(*lbwh_axes)
            trans = ax.transAxes + fig.transFigure.inverted()
            lbhw_fig = self.mt.TransformedBbox(Bbox, trans).bounds
            a2=fig.add_axes(lbhw_fig,axisbg='white')
            a2.patch.set_alpha(0.2)

            # calculate weights
            Sbin = self.masked_ravel(S-self.S00)*dV

            if self.bleck:
                hist = B.bin_bleck_wts(rho_bin_centres,rho_ravelled,Sbin)
            else:
                hist = B.bin_all_wts(self.rho_bin_edges,rho_ravelled,Sbin)

            #   divide by width of box to give S/dt 'density' [rate of change of variance/unit of rho]
            #   if dimensional, express volumes in 10^6 m^3/drho
            if self.dimensional:
                hist_plot = (1.e-12/self.drho)*hist
            else:
                hist_plot = hist/self.drho

            #   plot bar chart in black
            a2.bar(self.rho_bin_edges[:-1],hist,width=self.bin_widths,color=(0.,0.,.0),linewidth=0.)

            a2.set_xlim(self.binmin,self.binmax)
            a2.set_ylim(self.SbinVolmin,self.SbinVolmax)

            for label in a2.xaxis.get_ticklabels():
                label.set_color('black')
                label.set_fontsize(8)
            for label in a2.yaxis.get_ticklabels():
                label.set_color('black')
                label.set_fontsize(8)
            #             PL.setp(a2.get_xticklabels(), color='k')
            #             PL.setp(a2.get_yticklabels(), color='k')

    def savefile(self,nstep,dpi=None):
        pictures= self.pictures
        if pictures is not None:
            if pictures=='publish':
                suffix = 'pdf'
            else:
                suffix = 'png'
            if dpi==None:
                if pictures=='movie':
                    dpi=100
                else:
                    dpi=300
            fname = '_tmp%07d.'%nstep + suffix
            print 'Saving frame', fname
            self.fig.savefig(fname,dpi=dpi)
            self.files.append(fname)
