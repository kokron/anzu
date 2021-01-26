# anzu

Measurements and emulation of Lagrangian bias models for clustering and lensing cross-correlations.

![Anzu wyliei](http://stanford.edu/~kokron/anzu_2.png)
<sub>Adapted from work by Fred Wierum - Own work, CC BY-SA 4.0, https://commons.wikimedia.org/w/index.php?curid=49448119 <sub>


anzu is two independent codes related to hybrid Lagrangian bias models in large-scale structure combined into one repository.

The first part of the code measures the hybrid "basis functions" defined in the model of Modi et al. 2020 and Kokron et al. 2021. The second part takes measurements of these basis functions and constructs an emulator to obtain predictions from them at any cosmology (within the bounds of the training set). 

It's self-contained such that given a set of N-body simulations used to build emulators you should be able to construct a model. Alternatively, if you have your own measurements of the basis functions, the code here should in principle be useful for construction of your own emulator. 

The full code will be publicly released when the paper has been accepted.



