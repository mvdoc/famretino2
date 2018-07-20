# Data and code to replicate Kay et al., 2015

This folder contains the data of the experiment by Kay et al., 2015
available from http://kendrickkay.net/vtcdata. 

The code I wrote to replicate the analyses can be found under `code`.
The MATLAB function `cssmodel_vtcdata.m` runs the model fitting for
specific subjects and ROIs. A Condor submit file is available to run the
computation in parallel on a cluster.

The derivative data (model parameters) are saved in the `output`
directory. 

Additional analyses are run in Python. 

Please check the README file under `code` for more information.

You can manually download the data from the website (first you'll have
to remove the symlinks in this repository), or alternatively (preferred
way) you can get the data using [DataLad](http://datalad.org/) or [git annex](https://git-annex.branchable.com/). 
Once you have datalad or git annex installed, you can get the data with
either

```terminal
datalad get *
```

or 

```terminal
git annex get *
```


# References

Kay, K.N., Weiner, K.S., & Grill-Spector, K. Attention reduces spatial
uncertainty in human ventral temporal cortex. Current Biology (2015).
