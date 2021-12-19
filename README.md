# GGPONC 2.0

This repository contains the code to reproduce results from the paper *Clinical Entity Extraction and Coordination Ellipses in GGPONC 2.0*

## Preparation

1. Get access to GGPONC following the instructions on the [project homepage](https://www.leitlinienprogramm-onkologie.de/projekte/ggponc-english/) and place the 2.0 release in the `data` folder
2. Initialize [INCEpTION Analytics](https://github.com/zesch/inception-analytics) `git submodule init; git submodule update`
3. Install Python dependencies `pip install -r requirements.txt -r inception-analytics/requirements.txt` 

## Preprocessing (optional)

The prepocessed, e.g. sentence-splitted and tokenized text portions, are already included in the release (`data/raw_text`) 

If you want to do the preprocessing yourself, build the jar file from [https://github.com/hpi-dhc/ggponc_preprocessing](https://github.com/hpi-dhc/ggponc_preprocessing) and use it with `data/xml/cpg-corpus-cms.xml`

## Notebooks

In `notebooks`, we provide the following Jupyter Notebooks to reproduce the results from the paper:

- [01_Statistics.ipynb](notebooks/01_Statistics.ipynb)
    - Corpus Statistics and IAA Calculcation with INCEpTION Analytics
- [02_NER_Baselines.ipynb](notebooks/02_NER_Baselines.ipynb)
    - NER Baselines using BERT / HuggingFace Transformers
- [03_NER_Analysis.ipynb](notebooks/03_NER_Analysis.ipynb)
    - Analysis of NER errors and cooridation ellipses

## Running NER Experiments with Hydra

In `experiments`, we provide [Hydra](https://github.com/facebookresearch/hydra) configurations for the different NER experiments with the best hyperparameters found through grid search.
To run such an experiment, do:
- `cd experiments`
- `python run_experiment.py -cn <experiment>.yaml cuda=<cuda devices>`
    - for instance: `python run_experiment.py -cn 01_ggponc_coarse_short.yaml cuda=0`

If you have installed and configured [Weights & Biases](https://wandb.ai/), it will automatically sync your runs.

To run a hyperparameter sweep, pass the optiom `-m` to Hydra:
- `python run_experiment.py -m -cn=01_ggponc_coarse_short.yaml cuda=0 learning_rate=1e-6,5e-6,1e-5,5e-5,1e-4 label_smoothing_factor=0.0,0.05,0.1,0.2 weight_decay=0.0,0.05,0.1` 
- Note: change the launcher in `ggponc_base_config.yaml` to `submitit_local` if you cannot run the hyperparameter sweep on a Slurm cluster (see also [here](https://hydra.cc/docs/plugins/submitit_launcher/))

## Annotation Guide

Please refer to the [annotation guide](annotation_guide/anno_guide.pdf) for a detailed description of the entity classes and rules.

## Citing GGPONC

- TODO
