import time
import os
import numpy as np
import pyccl as ccl
import chaospy as cp
import warnings
from velocileptors.LPT.cleft_fftw import CLEFT
from velocileptors.EPT.cleft_kexpanded_resummed_fftw import RKECLEFT

from scipy.interpolate import interp1d
from scipy.signal import savgol_filter


def norm(x, x_mean=None, x_mult=None):

    if x_mean is None:
        x_mean = np.mean(x, axis=0)

    if x_mult is None:
        x_mult = 2 / (np.max(x, axis=0) - np.min(x, axis=0))

    x_normed = (x - x_mean[np.newaxis, ...]) * x_mult[np.newaxis, ...]

    return x_normed, x_mean, x_mult


def unnorm(x_normed, x_mean, x_mult):

    x = x_normed / x_mult[np.newaxis, ...] + x_mean[np.newaxis, ...]

    return x


class LPTEmulator(object):

    """ Main emulator object """

    def __init__(self, nbody_training_data_file='spectra_aem_compensated.npy',
                 lpt_training_data_file='cleft_spectra_twores.npy',
                 kbin_file='kbins.npy',
                 zs=None,
                 training_cosmo_file='cosmos.txt',
                 surrogate_type='PCE',
                 smooth_spectra=True, window=11, savgol_order=3,
                 kmin=0.1, kmax=1.0, extrap=True, kmin_pl=0.5, kmax_pl=0.6,
                 use_physical_densities=True, usez=False, zmax=2.0,
                 use_sigma_8=True, forceLPT=True, offset=False, tanh=True, kecleft=False):
        """
        Initialize the emulator object. Default values for all kwargs were
        used for fiducial results in 2101.11014, so don't change these unless
        you have a good reason!

        Kwargs:
            nbody_training_data_file : string
                File name containing the spectra that will be used to train the emulator.
            lpt_training_data_file : string
                File name containing the LPT spectra at the same cosmologies as the spectra
                in the nbody_training_data_file.
            kbin_file : string
                File containing the array of k values that spectra are measured at.
            zs : array like
                Array containing the redshifts that spectra are measured at.
            training_cosmo_file : string
                File name containing the cosmologies that the training spectra are measured at.
            surrogate_type: string
                Type of surrogate model to use. Only "PCE" is currently supported.
            smooth_spectra : bool
                Whether to apply a Savitsky-Golay smoothing to the training spectra
            window : int
                If smooth_spectra==True, then window specifies the window size to use
                when smoothing.
            savgol_order : int
                If smooth_spectra==True, then window specifies the order to use
                when smoothing.
            kmin : float
                Minimum k value that we will build the emulator for. For k<kmin
                pure LPT will be used.
            kmax : float
                Maximum k value to build the emulator for. The model will not make predictions
                for k>kmax.
            extrap : bool
                Whether to apply a power law extrapolation to the 1-1, 1-delta, and delta-delta
                LPT spectra at high k before constructing n-body / LPT ratios.
            kmin_pl : float
                Minimum k value to fit the power law extrapolation to.
            kmax_pl : float
                Maximum k value to fit the power law extrapolation to.
            use_physical_densities : bool
                Whether or not to use ombh^2 and omch^2 instead of om and oc to train emulator.
            usez : bool
                Whether or not to use redshift (as opposed to scale factor) to train the emulator.
            zmax : bool
                Maximum redshift value that will be used in training.
            use_sigma_8 : bool
                Whether to use sigma_8 instead of A_s when training the emulator.
            forceLPT : bool
            offset : bool
            tanh : bool
            kecleft: bool
                Sets whether to use "full" CLEFT or "k-expanded" CLEFT to make LPT predictions. 
                KECLEFT mode allows you to quickly compute spectra at fixed cosmology as a function
                of redshift.


        """

        self.nbody_training_data_file = nbody_training_data_file
        self.kbin_file = kbin_file
        self.lpt_training_data_file = lpt_training_data_file
        self.training_cosmo_file = training_cosmo_file
        self.surrogate_type = surrogate_type
        self.use_physical_densities = use_physical_densities
        self.use_sigma_8 = use_sigma_8

        if not zs:
            self.zs = np.array(
                [3.0, 2.0, 1.0, 0.85, 0.7, 0.55, 0.4, 0.25, 0.1, 0.0])
        else:
            self.zs = zs

        self.smooth_spectra = smooth_spectra
        self.window = window
        self.savgol_order = savgol_order
        self.usez = usez
        self.zmax = zmax
        self.kmin = kmin
        self.kmax = kmax
        self.extrap = extrap
        self.kmin_pl = kmin_pl
        self.kmax_pl = kmax_pl
        self.forceLPT = forceLPT

        self.param_mean = None
        self.param_mult = None
        self.offset = offset
        self.tanh = tanh

        #KECLEFT attributes
        self.kecleft = kecleft
        self.last_LPTcosmo = None
        self.last_cleftobj = None
        if self.kecleft and self.extrap:
            warnings.warn("kecleft and extrap are both set. Setting extrap to False.")
        if self.kecleft:
            self.lpt_training_data_file = 'kecleft_spectra.npy'
            self.extrap = False

        self._load_data(None)

        self._build_emulator()

    def _cleft_pk(self, cosmovec, snapscale):
        '''
        Returns a spline object which computes the cleft component spectra. Computed either in
        "full" CLEFT or in "k-expanded" CLEFT which allows for faster redshift dependence.
        Args:
            cosmovec : array-like
                Vector containing cosmology in the order (ombh2, omch2, w0, ns, sigma8, H0, Neff).
                If self.use_sigma_8 != True, then ln(A_s/10^{-10}) should be provided instead of sigma8.
            snapscale : float
                scale factor
            kecleft: bool
                Bool to check if the calculation is being made with 
        Returns:
            cleft_aem : InterpolatedUnivariateSpline 
                Spline that computes basis spectra as a function of k
        '''







        if self.use_physical_densities:
            if self.use_sigma_8:
                cosmo = ccl.Cosmology(Omega_b=cosmovec[0] / (cosmovec[5] / 100)**2,
                                      Omega_c=cosmovec[1] /
                                      (cosmovec[5] / 100)**2,
                                      h=cosmovec[5] / 100, n_s=cosmovec[3],
                                      w0=cosmovec[2], Neff=cosmovec[6],
                                      sigma8=cosmovec[4])
            else:
                cosmo = ccl.Cosmology(Omega_b=cosmovec[0] / (cosmovec[5] / 100)**2,
                                      Omega_c=cosmovec[1] /
                                      (cosmovec[5] / 100)**2,
                                      h=cosmovec[5] / 100, n_s=cosmovec[3],
                                      w0=cosmovec[2], Neff=cosmovec[6],
                                      A_s=np.exp(cosmovec[4]) * 1e-10)
        else:
            if self.use_sigma_8:
                cosmo = ccl.Cosmology(Omega_b=cosmovec[0],
                                      Omega_c=cosmovec[1] - cosmovec[0],
                                      h=cosmovec[5] / 100, n_s=cosmovec[3],
                                      w0=cosmovec[2], Neff=cosmovec[6],
                                      sigma8=cosmovec[4])
            else:
                cosmo = ccl.Cosmology(Omega_b=cosmovec[0],
                                      Omega_c=cosmovec[1] - cosmovec[0],
                                      h=cosmovec[5] / 100, n_s=cosmovec[3],
                                      w0=cosmovec[2], Neff=cosmovec[6],
                                      A_s=np.exp(cosmovec[4]) * 1e-10)





        k = np.logspace(-3, 1, 1000)

        if self.kecleft:
            #If using kecleft, check that we're only varying the redshift

            if (cosmovec == self.last_LPTcosmo).all():
                #Take the last kecleft object used
                cleftobj = self.last_cleftobj

            else:
                #Do the full calculation again, as the cosmology changed.
                pk = ccl.linear_matter_power(
                    cosmo, k * cosmo['h'], 1) * (cosmo['h'])**3 

                #Function to obtain the no-wiggle spectrum.
                # Not implemented yet, maybe Wallisch maybe B-Splines?               
                # pnw = p_nwify(pk)
                #For now just use Stephen's standard savgol implementation.
                cleftobj = RKECLEFT(k, pk)

                self.last_cleftobj = cleftobj

            #Adjust growth factors
            D = ccl.background.growth_factor(cosmo, snapscale)
            cleftobj.make_ptable(D=D,kmin=k[0],kmax=k[-1],nk=1000)
            cleftpk = cleftobj.pktable.T

        else:
            #Using "full" CLEFT, have to always do calculation from scratch
            pk = ccl.linear_matter_power(
                cosmo, k * cosmo['h'], snapscale) * (cosmo['h'])**3
            cleftobj = CLEFT(k, pk, N=2700, jn=10, cutoff=1)
            cleftobj.make_ptable()

            cleftpk = cleftobj.pktable.T

            # Different cutoff for other spectra, because otherwise different
            # large scale asymptote

            cleftobj = CLEFT(k, pk, N=2700, jn=5, cutoff=10)
            cleftobj.make_ptable()

        cleftpk[3:, :] = cleftobj.pktable.T[3:, :]
        cleftpk[2, :] /= 2
        cleftpk[6, :] /= 0.25
        cleftpk[7, :] /= 2
        cleftpk[8, :] /= 2

        cleftspline = interp1d(cleftpk[0], cleftpk, fill_value='extrapolate')

        #Store last cosmology used
        self.last_LPTcosmo = cosmovec


        return cleftspline

    def _powerlaw_extrapolation(self, spectra, k=None):
        '''    

        fit power law indices to all the 1,1 and 1,delta spectra for high k extrapolation
        kmin, kmax are the values used for this extrapolation.

        optionally, feed in a 'k' parameter which sets kbins to re-compute the extrapolated spectra
        '''
        k_idx = np.where((self.kmin_pl < self.k) & (self.k < self.kmax_pl))[0]

        # this assumes that 1,1 and 1,delta are the first two spectra and
        # that these are the only ones that need power law extrapolation
        alpha = (np.log(spectra[..., :2, k_idx[0]] / spectra[..., :2, k_idx[-1]]) /
                 (np.log(self.k[k_idx[0]]) - np.log(self.k[k_idx[-1]])))
        p0 = spectra[..., :2, k_idx[-1]] / (self.k[k_idx[-1]]**alpha)
        k_idx = self.k > self.kmax_pl

        spectra[..., :2, k_idx] = p0[..., np.newaxis] *\
            self.k[k_idx]**alpha[..., np.newaxis]
        if k is not None:
            specspline = interp1d(self.k, spectra, axis=-1,
                                  fill_value='extrapolate')
            spectra = specspline(k)
        return spectra

    def _load_data(self, filename):

        aem_file = '/'.join([os.path.dirname(os.path.realpath(__file__)),
                             'data',
                             self.nbody_training_data_file])
        lpt_file = '/'.join([os.path.dirname(os.path.realpath(__file__)),
                             'data',
                             self.lpt_training_data_file])
        k_file = '/'.join([os.path.dirname(os.path.realpath(__file__)),
                           'data',
                           self.kbin_file])
        self.spectra_aem = np.load(aem_file)
        self.spectra_lpt = np.load(lpt_file)
        self.k = np.load(k_file)

    def _get_pcs(self, evec_spec, spectra, npc):

        nout = np.prod(spectra.shape[:2])
        pcs_spec = np.zeros((nout, 10, self.npc))

        for si in range(10):
            pcs_spec[:, si, :] = np.dot(spectra[:, :, si, :].reshape(-1, self.nk),
                                        evec_spec[si, :, :npc])

        return pcs_spec

    def _ratio_and_smooth(self, spectra_aem, spectra_lpt):

        simoverlpt = spectra_aem / spectra_lpt

        # smooth the ratios before taking log
        if self.smooth_spectra:
            simoverlpt = savgol_filter(simoverlpt, self.window,
                                       self.savgol_order, axis=-1)

        simoverlpt = np.log10(simoverlpt)
        simoverlpt[~np.isfinite(simoverlpt)] = 0

        self.zidx = np.min(np.where(self.zs <= self.zmax))
        self.nz = len(self.zs[self.zidx:])

        self.kmax_idx = np.searchsorted(self.k, self.kmax)
        self.kmin_idx = np.searchsorted(self.k, self.kmin)
        self.nk = self.kmax_idx - self.kmin_idx

        simoverlpt = simoverlpt[:, self.zidx:, :, self.kmin_idx:self.kmax_idx]

        return simoverlpt

    def _smooth_transition(self, simoverlpt):
        '''
        Additional post-processing on ratios so they're smooth near the transition between LPT
        and the emulator. 

        Does two things:
        1) Computes the offset for log(Nbody/LPT) in the IR range from the mean at every redshift bin
        2) Applies a "high-pass" type filter at low-k so PCs don't go insane.

        Notes:
        Could also try to do a Savgol pass for (2) at low-k instead of the current filter
        '''
        nsim, nz, nspec, nk = simoverlpt.shape
        if not self.offset and not self.tanh:
            return simoverlpt

        # kstar = 0.125 where we broadly want the transition to be final
        kstar = 0.125

        # Hard-coded offset, just use window from kmin_idx to kmin_idx+4 for now

        kvals = self.k[self.kmin_idx:self.kmax_idx]

        offidx = ((kvals > self.kmin) & (kvals < kstar))

        filter_tanh = 0.5*(1 + np.tanh(2.5*(kvals - kstar)/kstar))
        if not self.tanh:
            filter_tanh = np.ones_like(filter_tanh)

        newsimoverlpt = 1.*simoverlpt
        for i in range(nz):
            for j in range(nspec):
                meanratio = np.mean(simoverlpt, axis=0)[i, j]

                offset = np.mean(meanratio[offidx])
                if not self.offset:
                    offset = 0

                newsimoverlpt[:, i, j] -= offset
                # Filter only the cubic spectra
                if j in range(nspec):
                    newsimoverlpt[:, i, j] *= filter_tanh

        return newsimoverlpt

    def _setup_training_data(self, spectra_lpt, spectra_aem):

        # apply power law extrapolation to LPT spectra where they diverge at high k
        if self.extrap:

            spectra_lpt = self._powerlaw_extrapolation(spectra_lpt)

        simoverlpt = self._ratio_and_smooth(spectra_aem, spectra_lpt)

        # Smooth the ratios even more/calibrate them to LPT to remove kink
        simoverlpt = self._smooth_transition(simoverlpt)

        self.simoverlpt = simoverlpt

        nsim = len(simoverlpt)

        # Non mean-subtracted PCs
        Xs = np.zeros((10, self.nk, self.nk))
        for i in range(10):
            Xs[i, :, :] = np.dot(simoverlpt[:, :, i, :].reshape(
                self.nz * (nsim - self.degree_cv), -1).T,
                simoverlpt[:, :, i, :].reshape(self.nz * (nsim - self.degree_cv), -1))

        # PC basis for each type of spectrum, independent of z and cosmo
        evec_spec = np.zeros((10, self.nk, self.nk))

        # variance per PC
        vars_spec = np.zeros((10, self.nk))

        # computing PCs
        for si in range(10):
            var, pcs = np.linalg.eig(Xs[si, ...])

            evec_spec[si, :, :] = pcs
            vars_spec[si, :] = var

        self.evec_spec = evec_spec
        self.evec_spline = interp1d(self.k[self.kmin_idx:self.kmax_idx],
                                    self.evec_spec[..., :self.npc], axis=1,
                                    fill_value='extrapolate')

        self.pcs_spec = self._get_pcs(self.evec_spec,
                                      simoverlpt, self.npc)
        self.pcs_spec_normed, \
            self.pcs_mean, self.pcs_mult = norm(self.pcs_spec)

    def _setup_design(self, cosmofile, param_mean=None, param_mult=None):

        cosmo_file = '/'.join([os.path.dirname(os.path.realpath(__file__)),
                               'data',
                               cosmofile])

        cosmos = np.genfromtxt(cosmo_file, names=True)
        ncosmos = len(cosmos)
        self.training_cosmos = cosmos

        if not self.use_physical_densities:
            if not self.use_sigma_8:
                dt = np.dtype([('omegab', np.float), ('omegam', np.float),
                               ('w0', np.float), ('ns', np.float),
                               ('ln10As', np.float), ('H0', np.float),
                               ('Neff', np.float)])
                cosmos_temp = np.zeros(ncosmos, dtype=dt)
                cosmos_temp['omegab'] = cosmos['ombh2'] / \
                    (cosmos['H0'] / 100)**2
                cosmos_temp['omegam'] = (cosmos['omch2'] + cosmos['ombh2']) /\
                    (cosmos['H0'] / 100)**2
                cosmos_temp['w0'] = cosmos['w0']
                cosmos_temp['ns'] = cosmos['ns']
                cosmos_temp['ln10As'] = cosmos['ln10As']
                cosmos_temp['H0'] = cosmos['H0']
                cosmos_temp['Neff'] = cosmos['Neff']
                cosmos = cosmos_temp
            else:
                dt = np.dtype([('omegab', np.float), ('omegam', np.float),
                               ('w0', np.float), ('ns', np.float),
                               ('sigma8', np.float), ('H0', np.float),
                               ('Neff', np.float)])
                cosmos_temp = np.zeros(ncosmos, dtype=dt)
                cosmos_temp['omegab'] = cosmos['ombh2'] / \
                    (cosmos['H0'] / 100)**2
                cosmos_temp['omegam'] = (cosmos['omch2'] + cosmos['ombh2']) /\
                    (cosmos['H0'] / 100)**2
                cosmos_temp['w0'] = cosmos['w0']
                cosmos_temp['ns'] = cosmos['ns']
                cosmos_temp['sigma8'] = cosmos['sigma8']
                cosmos_temp['H0'] = cosmos['H0']
                cosmos_temp['Neff'] = cosmos['Neff']
                cosmos = cosmos_temp
        else:
            if not self.use_sigma_8:
                dt = np.dtype([('ombh2', np.float), ('omch2', np.float),
                               ('w0', np.float), ('ns', np.float),
                               ('ln10As', np.float), ('H0', np.float),
                               ('Neff', np.float)])
                cosmos_temp = np.zeros(ncosmos, dtype=dt)
                cosmos_temp['ombh2'] = cosmos['ombh2']
                cosmos_temp['omch2'] = cosmos['omch2']
                cosmos_temp['w0'] = cosmos['w0']
                cosmos_temp['ns'] = cosmos['ns']
                cosmos_temp['ln10As'] = cosmos['ln10As']
                cosmos_temp['H0'] = cosmos['H0']
                cosmos_temp['Neff'] = cosmos['Neff']
                cosmos = cosmos_temp

            else:
                dt = np.dtype([('ombh2', np.float), ('omch2', np.float),
                               ('w0', np.float), ('ns', np.float),
                               ('sigma8', np.float), ('H0', np.float),
                               ('Neff', np.float)])
                cosmos_temp = np.zeros(ncosmos, dtype=dt)
                cosmos_temp['ombh2'] = cosmos['ombh2']
                cosmos_temp['omch2'] = cosmos['omch2']
                cosmos_temp['w0'] = cosmos['w0']
                cosmos_temp['ns'] = cosmos['ns']
                cosmos_temp['sigma8'] = cosmos['sigma8']
                cosmos_temp['H0'] = cosmos['H0']
                cosmos_temp['Neff'] = cosmos['Neff']
                cosmos = cosmos_temp

        param_ranges = np.array(
            [[np.min(cosmos[k]), np.max(cosmos[k])] for k in cosmos.dtype.names])
        if self.usez:
            param_ranges = np.vstack([param_ranges, [0, self.zmax]])
        else:
            param_ranges = np.vstack([param_ranges, [1 / (self.zmax + 1), 1]])

        # if self.param_mean and self.param_mult are already defined
        # we will use those. This is mostly useful for our test suites
        param_ranges_scaled, self.param_mean,\
            self.param_mult = norm(
                param_ranges.T, self.param_mean, self.param_mult)

        self.param_ranges_scaled = param_ranges_scaled.T

        # Design matrix for the PCs, with 7 parameters (wCDM + z)
        zidx = np.min(np.where(self.zs <= self.zmax))
        z = self.zs[zidx:]
        a = 1 / (1 + z)

        if self.usez:
            design = np.hstack([np.tile(cosmos.view(('<f8', 7)), self.nz)[np.arange(ncosmos) != self.ncv].reshape(
                self.nz * (ncosmos - self.degree_cv), 7), np.tile(z, ncosmos - self.degree_cv)[:, np.newaxis]])
        else:
            design = np.hstack([np.tile(cosmos.view(('<f8', 7)), self.nz)[np.arange(ncosmos) != self.ncv].reshape(
                self.nz * (ncosmos - self.degree_cv), 7), np.tile(a, ncosmos - self.degree_cv)[:, np.newaxis]])

        design_scaled = (
            design - self.param_mean[np.newaxis, :]) * self.param_mult[np.newaxis, :]

        return design, design_scaled

    def _train_surrogates(self):

        if self.surrogate_type == 'PCE':

            distribution = cp.J(*[cp.Uniform(self.param_ranges_scaled[i][0],
                                             self.param_ranges_scaled[i][1]) for i in range(8)])

            self.surrogates = []

            # PCE coefficient regression
            for i in range(10):
                pce = cp.orth_ttr(self.npoly[i], distribution,
                                  cross_truncation=self.qtrunc)
                surrogate = cp.fit_regression(
                    pce, self.design_scaled.T, np.real(self.pcs_spec_normed[:, i, :]))

                self.surrogates.append(surrogate)

        else:
            raise(ValueError(
                'Surrogate type {} not implemented!'.format(self.surrogate_type)))

    def _build_emulator(self, hyperparams=None):
        '''
        Trains the emulator with polynomial chaos expansion regression.
        Default values for all kwargs were used for fiducial results in
        2101.11014, so don't change these unless you have a good reason!

        Kwargs:
            npc : int
                number of principal components. This must always be defined,
                because all models we consider build surrogates for principal components.
            npoly : int/array like
                Polynomial order for the PCE. If of int type, the same order
                is used for all parameters. If array like, then needs to be of size
                (n_spec, n_param), where n_spec is the number of bias basis spectra,
                i.e. 10, and n_param is the number of cosmological/redshift
                parameters that the emulator is a function of, i.e. 8. The order is
                the same as in the training cosmology file, redshift/scale factor is last.
            qtrunc : float
                hyperbolic truncation parameter for PCE regression.
            ncv: int
                number of cosmologies to leave out when training, for cross-validation.

        '''

        if hyperparams is None:
            self.npc = 2
            ncv = None
            self.ncv = ncv
        else:
            self.npc = hyperparams['npc']
            ncv = hyperparams['ncv']
            self.ncv = ncv

        if self.surrogate_type == 'PCE':
            if hyperparams is None:
                npoly = np.array([1, 2, 1, 1, 3, 2, 1, 3])
                npoly = np.tile(npoly, [10, 1])
                qtrunc = 1
            else:
                if 'npoly' not in hyperparams.keys():
                    npoly = np.array([1, 2, 1, 1, 3, 2, 1, 3])
                    npoly = np.tile(npoly, [10, 1])

                else:
                    npoly = hyperparams['npoly']
                    if len(npoly.shape) == 1:
                        npoly = np.tile(npoly, [10, 1])

                if 'qtrunc' not in hyperparams.keys():
                    qtrunc = 1
                else:
                    qtrunc = hyperparams['qtrunc']

            self.npoly = npoly
            self.qtrunc = qtrunc

        # Pulling all of the measured P(k) into a file
        spectra_aem = np.copy(self.spectra_aem)
        spectra_lpt = np.copy(self.spectra_lpt)

        if ncv is None:
            self.degree_cv = 0
        else:
            self.degree_cv = 1
            spectra_aem = spectra_aem[np.arange(len(spectra_aem)) != ncv]
            spectra_lpt = spectra_lpt[np.arange(len(spectra_lpt)) != ncv]

        self._setup_training_data(spectra_lpt, spectra_aem)
        self.design, self.design_scaled = self._setup_design(
            self.training_cosmo_file)
        self._train_surrogates()

        self.trained = True

    def predict(self, k, cosmo, **kwargs):
        """
        Make predictions from a trained emulator given a vector of wavenumbers and
        a cosmology.

        Args:
            k : array-like
                1d vector of wave-numbers. Maximum k cannot be larger than 
                self.kmax. For k < self.kmin, predictions will be made using
                velocileptors, for self.kmin <= k < self.kmax predictions
                use the emulator.
            cosmo : array-like
                Vector containing cosmology/scale factor in the order
                (ombh2, omch2, w0, ns, sigma8, H0, Neff, a).
                If self.use_sigma_8 != True, then ln(A_s/10^{-10})
                should be provided instead of sigma8. If self.usez==True 
                then a should be replaced with redshift.
        Kwargs:
            Kwargs to be passed to _pce_predict.

        Output: 
            Emulator predictions for the 10 basis spectra of the 2nd order lagrangian bias expansion.
            Order of spectra is 1-1, delta-1, delta-delta, delta2-1, delta2-delta, delta2-delta2
            s2-1, s2-delta, s2-delta2, s2-s2.


        """

        if not self.trained:
            raise(ValueError('Need to call build_emulator before making predictions'))

        if self.surrogate_type == 'PCE':
            pk_emu, lambda_pce = self._pce_predict(k, cosmo, **kwargs)

        else:
            raise(ValueError(
                'Surrogate type {} not implemented!'.format(self.surrogate_type)))

        return pk_emu

    def basis_to_full(self, k, btheta, emu_spec, halomatter=True):
        """
        Take an LPTemulator.predict() array and combine with bias parameters to obtain predictions for P_hh and P_hm. 


        Inputs:
        -k: set of wavenumbers used to generate emu_spec.
        -btheta: vector of bias + shot noise. See notes below for structure of terms
        -emu_spec: output of LPTemu.predict() at a cosmology / set of k values
        -halomatter: whether we compute only P_hh or also P_hm

        Outputs:
        -pfull: P_hh (k) or a flattened [P_hh (k),P_hm (k)] for given spectrum + bias params.


        Notes:
        Bias parameters can either be

        btheta = [b1, b2, bs2, SN]

        or

        btheta = [b1, b2, bs2, bnabla2, SN]

        Where SN is a constant term, and the bnabla2 terms follow the approximation

        <X, nabla^2 delta> ~ -k^2 <X, 1>. 

        Note the term <nabla^2, nabla^2> isn't included in the prediction since it's degenerate with even higher deriv
        terms such as <nabla^4, 1> which in principle have different parameters. 


        To-do:
        Include actual measured nabla^2 correlators once the normalization issue has been properly worked out.

        """
        if len(btheta) == 4:
            b1, b2, bs, sn = btheta
            #Cross-component-spectra are multiplied by 2, b_2 is 2x larger than in velocileptors
            bterms_hh = [1, 
                         2*b1 , b1**2   , 
                         b2   , b2*b1   , 0.25*b2**2, 
                         2*bs , 2*bs*b1 , bs*b2     , bs**2]
        
            #hm correlations only have one kind of <1,delta_i> correlation
            bterms_hm = [1   , 
                         b1  , 0,
                         b2/2, 0, 0,
                         bs  , 0, 0, 0]
            
            pkvec = emu_spec

        else:
            b1, b2, bs, bk2, sn = btheta
            #Cross-component-spectra are multiplied by 2, b_2 is 2x larger than in velocileptors
            bterms_hh = [1, 
                         2*b1 , b1**2   , 
                         b2   , b2*b1   , 0.25*b2**2, 
                         2*bs , 2*bs*b1 , bs*b2     , bs**2, 
                         2*bk2, 2*bk2*b1, bk2*b2    , 2*bk2*bs]
        
            #hm correlations only have one kind of <1,delta_i> correlation
            bterms_hm = [1   , 
                         b1  , 0,
                         b2/2, 0, 0,
                         bs  , 0, 0, 0,
                         bk2 , 0, 0, 0]
            
            pkvec = np.zeros(shape=(14, len(k)))
            pkvec[:10] = emu_spec
            
            #IDs for the <nabla^2, X> ~ -k^2 <1, X> approximation.
            nabla_idx = [0, 1, 3, 6]
            
            #Higher derivative terms
            pkvec[10:] = -k**2 * pkvec[nabla_idx]      
            
        bterms_hh = np.array(bterms_hh)
        
        
        p_hh = np.einsum('b, bk->k', bterms_hh, pkvec) + sn
        pfull = p_hh
        
        if halomatter:
            bterms_hm = np.array(bterms_hm)
            p_hm = np.einsum('b, bk->k', bterms_hm,pkvec)
            pfull = np.hstack([p_hh, p_hm])   
            
        return pfull

    def _pce_predict(self, k, cosmo, lambda_pce=None, spectra_lpt=None,
                     evec_spec=None, simoverlpt=None, timing=False):
        '''
        Args:
            k : array-like
                1d vector of wave-numbers.
            cosmo : array-like
                Vector containing cosmology/scale factor in the order
                (ombh2, omch2, w0, ns, sigma8, H0, Neff, a).
                If self.use_sigma_8 != True, then ln(A_s/10^{-10})
                should be provided instead of sigma8. If self.usez==True 
                then a should be replaced with redshift.
        Kwargs:
            lambda_pce : array-like
                Array of shape (n_spec, n_pc) of PC coefficients to use
                to make predictions. Mostly used for validation of PCE procedure.
            spectra_lpt : array-like
                LPT predictions for spectra to be used in lieu of a velocileptors
                call. 
            evec_spec : array-like
                Array of PC spectra. For use when validating PCA procedure.
            simoverlpt : array-like
                Array of n-body/lpt ratios. For use when validating PCA procedure.
            timing : bool
                If True, then print timing info.

        Output:
            pk_emu : array-like
                Emulator predictions for the 10 basis spectra of the 2nd order lagrangian bias expansion. 


        '''

        if np.max(k) > self.k[self.kmax_idx]:
            raise(ValueError(
                "Trying to compute spectra beyond the maximum value of the emulator!"))
        evecs = self.evec_spline(k)
        cosmo_scaled = (
            cosmo - self.param_mean[np.newaxis, :]) * self.param_mult[np.newaxis, :]

        # if we already have PCs, just make prediction using them
        if lambda_pce is None:

            # otherwise, check to see if we have PC vecs and spectra, in which case
            # compute PCs with them
            if evec_spec is not None:
                if simoverlpt is None:
                    raise(ValueError(
                        "need to provide non-linear ratios if want PCA only resids"))

                lambda_pce = self._get_pcs(evec_spec, simoverlpt, self.npc)
                lambda_pce_normed = None

            # otherwise just use the surrogates to compute PCs
            else:
                lambda_pce_normed = np.zeros((len(cosmo), 10, self.npc))

                for i in range(10):
                    start = time.time()
                    lambda_pce_normed[:, i, ...] = self.surrogates[i](
                        *cosmo_scaled.T).T
                    end = time.time()

                    if timing:
                        print('took {}s'.format(end - start))

                lambda_pce = unnorm(lambda_pce_normed, self.pcs_mean,
                                    self.pcs_mult)

        simoverlpt_emu = np.einsum('bkp, cbp->cbk', evecs, lambda_pce)

        if spectra_lpt is None:
            ncosmos = len(cosmo)
            spectra_lpt = np.zeros((ncosmos, 10, len(k)))
            spectra_lpt = np.zeros((ncosmos, 10, len(k)))

            for i in range(ncosmos):
                spectra_lpt[i, :, :] = self._cleft_pk(cosmo[i, :-1],
                                                      cosmo[i, -1])(k)[1:11, :]
        if self.extrap:
            spectra_lpt = self._powerlaw_extrapolation(spectra_lpt, k)

        pk_emu = np.zeros_like(spectra_lpt)
        pk_emu[:] = spectra_lpt
        # Enforce agreement with LPT
        if self.forceLPT:

            pk_emu[..., k > self.kmin] = (
                10**(simoverlpt_emu) * pk_emu)[..., k > self.kmin]
        else:
            pk_emu[...] = (
                10**(simoverlpt_emu) * pk_emu[...])
        return pk_emu, lambda_pce
        # pk_emu = pk_emu[...]
