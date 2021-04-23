# anzu

Measurements and emulation of Lagrangian bias models for clustering and lensing cross-correlations.

![Anzu wyliei](http://stanford.edu/~kokron/anzu_2.png)
<sub>Adapted from work by Fred Wierum - Own work, CC BY-SA 4.0, https://commons.wikimedia.org/w/index.php?curid=49448119 <sub>


`anzu` is two independent codes related to hybrid Lagrangian bias models in large-scale structure combined into one repository.

The first part of the code, in the `fields` directory, measures the hybrid "basis functions" defined in the model of [Modi et al. 2020](https://arxiv.org/abs/1910.07097) and [Kokron et al. 2021.](https://arxiv.org/abs/2101.11014) The second part, in the `anzu` directory, takes measurements of these basis functions and constructs an emulator to obtain predictions from them at any cosmology (within the bounds of the training set). 

The code is self-contained such that given a set of N-body simulations used to build emulators you should be able to measure basis functions. Alternatively, if you have your own measurements of the basis functions, the code here should in principle be useful for construction of your own emulator. 


# Installation

The code has the following dependencies, which need to be installed before installing `anzu`:

`numpy`, `scipy`, `CCL`, `velocileptors` and `chaospy`.

`CCL` is the [Core Cosmology Library](https://github.com/LSSTDESC/CCL) developed by LSST DESC, and can be installed via conda (from the conda-forge channel). This is used to generate linear power spectra.

[`velocileptors`](https://github.com/sfschen/velocileptors) is used for pure LPT predictions for the spectra required for the 1-loop bias expansion employed by `anzu`. 

[`chaospy`](https://github.com/jonathf/chaospy) is used for polynomial chaos expansions, and is conda installable (again from conda-forge).

After you have installed these, `anzu` can be installed via

`python3 -m pip install -v git+https://github.com/kokron/anzu`


# Basic Usage

Making predictions with `anzu` is as simple as 

```python
from anzu.emu_funcs import LPTEmulator
import numpy as np

emu = LPTEmulator()

k = np.logspace(-2,0,100)
cosmo_vec = np.atleast_2d([0.023, 0.108, -0.73, 0.98, 0.69,  63.2,  2.95,  1.0])
emu_spec = emu.predict(k, cosmo_vec)

```

Folding in your favorite set of bias parameters to generate predictions for halo-halo and halo-matter power spectra is also simple!

```python

#UNIT-redmagic bias parameters from 2101.11014
#         b1,    b2,    bs2,   bnabla2, SN
bvec = [0.786, 0.583, -0.406, -0.512, 1755]

biased_spec = emu.basis_to_full(k, bvec, emu_spec[0])

```

The default emulator makes predictions as a function of (\Omega_b h^2, \Omega_c h^2, w, n_s, \sigma_8, H_0, N_{\rm eff}, a) in that order, which
is what is specified by `cosmo_vec`. This needs to be a two dimensional array, whose number of rows is the number of cosmologies you would like to make predictions
at.
You will also need to provide an array of wavenumbers that you want to compute spectra at, which should be in units of Mpc/h.

See `notebooks/train_and_predict.ipynb` for more worked examples for how to use the code.



