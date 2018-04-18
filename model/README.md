# Model

This directory contains simulations using the Compressive Spatial
Summation model developed by Kendrick Kay et al. [1, 2]. We reimplemented
the model in Python, and the original MATLAB version can be found here:
[http://kendrickkay.net/socmodel/](http://kendrickkay.net/socmodel/)

# Description of files

- [cssmodel](cssmodel): this directory contains the python
  implementation of the CSS model and a module containing helper
  functions for the simulation
- [cssmodel_example.ipynb](cssmodel_example.ipynb): this jupyter
  notebook contains an example of the CSS model reproducing the example
  in [http://kendrickkay.net/socmodel/html/cssmodel_example.html](http://kendrickkay.net/socmodel/html/cssmodel_example.html)
- [run_simulation.ipynb](run_simulation.ipynb): this jupyter notebook
  contains the code used to run the simulations reported in the paper.
- [plot_simdata.Rmd](plot_simdata.Rmd): this RMarkdown notebook is used
  to plot the results of the simulation
- [plot_simdata.nb.html](plot_simdata.nb.html): HTML rendering of the
  RMarkdown notebook
- [inputs](inputs): directory containing various data used in the
  simulation; mostly values exported from figures in [2]
- [outputs](outputs): directory containing the results of the
  simulations


# References

1. [Kay, K.N., Winawer, J., Mezer, A., & Wandell, B.A. Compressive spatial summation in human visual cortex. *Journal of Neurophysiology* (2013).](https://www.ncbi.nlm.nih.gov/pubmed/23615546)
2. [Kay, K. N., Weiner, K. S., & Grill-Spector, K. Attention reduces spatial uncertainty in human ventral temporal cortex. *Current Biology* (2015).](https://www.ncbi.nlm.nih.gov/pubmed/25702580)
