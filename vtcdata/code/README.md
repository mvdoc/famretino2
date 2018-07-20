# Running CSS model on vtcdata

This directory contains MATLAB scripts to run the CSS model on the
vtcdata available from http://kendrickkay.net/vtcdata/. It requires the
`knkutils` repository available from
https://github.com/kendrickkay/knkutils. It is provided in this
directory as a git submodule, so when you clone this repository, you
should init it with

```terminal
git submodule update --init knkutils
```

Example usage can be seen in the condor submit file
`cssmodel_vtcdata_all.submit`. 

There are two scripts

- `cssmodel_vtcdata.m` to run the model on the pRF-estimation experiment
- `cssmodel_vtcdata_facetask.m` to run the model on the face task experiment

Outputs are generated in the `../output` directory. They can take a
while, so we provide the results in the [OSF repository](https://osf.io/wdaxs/?view_only=28741ad9b640480a9af6b593ade1ebcf).

From those results, we generate the final parameters used in the
simulation using the script `make_parameter_files.py` and
`make_parameter_files_facetask.py`. We provide the output of these
scripts under `../output`.
