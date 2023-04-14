ses-ling
==============================
[![figshare](https://img.shields.io/badge/doi-10.6084%2Fm9.figshare.22633417-blue.svg?style=flat-square)](https://doi.org/10.6084/m9.figshare.22633417)

Analyse geotagged tweets with LanguageTool to explore the relation between socioeconomic
status and the use of standard language.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modelling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── scripts            <- Scripts to send to a cluster e.g.
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── environment.yml   <- The conda environment file for reproducing the analysis environment, e.g.
    │                         generated with `conda env export -f environment.yml`
    |
    ├── ses_ling                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes the project a Python module
    │   │
    │   ├── data           <- Data processing code
    │   │   └── make_dataset.py
    │   │
    │   ├── utils       <- Code for any utilities
    │   │
    │   └── visualization  <- Code to create exploratory and results oriented visualizations
    │       └── visualize.py
    |
    └── setup.py           <- makes project pip installable (pip install -e .) so ses_ling can be imported



--------

<p><small>Project based on <a target="_blank" href="https://github.com/drivendata/cookiecutter-data-science">a fork of the cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
