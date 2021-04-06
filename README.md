# anzu

Measurements and emulation of Lagrangian bias models for clustering and lensing cross-correlations.

![Anzu wyliei](http://stanford.edu/~kokron/anzu_2.png)
<sub>Adapted from work by Fred Wierum - Own work, CC BY-SA 4.0, https://commons.wikimedia.org/w/index.php?curid=49448119 <sub>


anzu is two independent codes related to hybrid Lagrangian bias models in large-scale structure combined into one repository.

The first part of the code measures the hybrid "basis functions" defined in the model of Modi et al. 2020 and Kokron et al. 2021. The second part takes measurements of these basis functions and constructs an emulator to obtain predictions from them at any cosmology (within the bounds of the training set). 

It's self-contained such that given a set of N-body simulations used to build emulators you should be able to construct a model. Alternatively, if you have your own measurements of the basis functions, the code here should in principle be useful for construction of your own emulator. 


# Installation

Anzu can be installed via

`python3 -m pip install -v git+https://github.com/kokron/anzu`

The code has the following dependencies:

`numpy`, `scipy`, `CCL`, `velocileptors` and `chaospy`.

`CCL` is the [Core Cosmology Library](https://github.com/LSSTDESC/CCL) developed by LSST DESC, and can be installed via conda (from the conda-forge channel). This is used to generate linear power spectra.

`velocileptors` is used for pure LPT predictions for the spectra required for the 1-loop bias expansion employed by `anzu`. 

`chaospy` is used for polynomial chaos expansions, and is conda installable (again from conda-forge).

# Basic Usage

Making predictions with `anzu` is as simple as 

```python
from anzu.emu_funcs import LPTEmulator
emu = LPTEmulator()

cosmo_vec = np.atleast_2d([0.023, 0.108, -0.73, 0.98, 0.69,  63.2,  2.95,  1.0])
emu_spec = emu.predict(k, np.atleast_2d(cosmo_vec))

```

See `notebooks/train_and_predict.ipynb` for more worked examples for how to use the code.



