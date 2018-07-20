# Idiosyncratic, retinotopic bias in face identification modulated by familiarity

This repository contains the analysis scripts, raw data, and simulations  for
*Idiosyncratic, retinotopic bias in face identification modulated by
personal familiarity* by Matteo Visconti di Oleggio Castello, Morgan
Taylor, Patrick Cavanagh, and M. Ida Gobbini. A preprint is available
on [bioRxiv](https://www.biorxiv.org/content/early/2018/01/26/253468).

All the analysis are provided as RMarkdown notebooks; a converted HTML
version is already present in the directory that can be inspected using
a regular browser.

Simulations are provided as Jupyter notebook. Please read the README.md
file in the model directory for more information. The simulations were
run using the singularity image for which we provide the definition file
in the directory `singularity`.

This repository uses [packrat](https://rstudio.github.io/packrat/) to
track dependencies. If you open RStudio to the top directory of this
project, and restart R ('Session' --> 'Restart R' in Rstudio), packrat
should automatically start downloading the required packages from CRAN.

For a description of the columns of the raw data, please refer to the
README.md files in the `data` directories.
