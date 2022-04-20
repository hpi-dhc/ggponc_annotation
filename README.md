[![DOI](https://zenodo.org/badge/411603699.svg)](https://zenodo.org/badge/latestdoi/411603699)


# GGPONC 2.0 — The German Clinical Guideline Corpus for Oncology

This repository contains the code to reproduce results from the paper:

*GGPONC 2.0 — The German Clinical Guideline Corpus for Oncology: Curation Workflow, Annotation Policy, Baseline NER Taggers*
(To appear at LREC '22)

## Preparation

1. Get access to GGPONC following the instructions on the [project homepage](https://www.leitlinienprogramm-onkologie.de/projekte/ggponc-english/) and place the the contents of the 2.0 release (v2.0_2022_03_24) in the `data` folder
2. Initialize [INCEpTION Analytics](https://github.com/zesch/inception-analytics) `git submodule init; git submodule update`
3. Install Python dependencies `pip install -r requirements.txt -r inception-analytics/requirements.txt` 

## Preprocessing (optional)

The prepocessed, i.e., sentence-splitted and tokenized text portions, are already included in the release (`data/plain_text`) 

If you want to do the preprocessing yourself, build the jar file from [https://github.com/hpi-dhc/ggponc_preprocessing](https://github.com/hpi-dhc/ggponc_preprocessing) and use it with `data/xml/cpg-corpus-cms.xml`

## Notebooks

In `notebooks`, we provide the following Jupyter Notebooks to reproduce the results from the paper:

- [01_Statistics.ipynb](notebooks/01_Statistics.ipynb)
    - Corpus Statistics and IAA Calculcation with INCEpTION Analytics
- [02_NER_Baselines.ipynb](notebooks/02_NER_Baselines.ipynb)
    - NER Baselines using BERT / HuggingFace Transformers
- [03_NER_Analysis.ipynb](notebooks/03_NER_Analysis.ipynb)
    - Analysis of NER errors and coordination ellipses
- **new**: [04_spaCy_Spancat.ipynb](notebooks/04_spaCy_Spancat.ipynb) 
    - Alternative NER implementation using spaCy's [SpanCategorizer](https://spacy.io/api/spancategorizer) feature, which can handle overlapping and nested mentions

## Running NER Experiments with HuggingFace and Hydra

In `experiments`, we provide [Hydra](https://github.com/facebookresearch/hydra) configurations for the different NER experiments with the best hyperparameters found through grid search.
To run such an experiment, do:
- `cd experiments`
- `python run_experiment.py -cn <experiment>.yaml cuda=<cuda devices>`
    - for instance: `python run_experiment.py -cn 01_ggponc_coarse_short.yaml cuda=0`

If you have installed and configured [Weights & Biases](https://wandb.ai/), it will automatically sync your runs.

To run a hyperparameter sweep, pass the optiom `-m` to Hydra, e.g.:
- `python run_experiment.py -m -cn=01_ggponc_coarse_short.yaml cuda=0 learning_rate=1e-6,5e-6,1e-5,5e-5,1e-4 label_smoothing_factor=0.0,0.05,0.1,0.2 weight_decay=0.0,0.05,0.1` 

## Annotation Guide

Please refer to the [annotation guide](annotation_guide/anno_guide.pdf) for a detailed description of the entity classes and rules.

## Citing GGPONC

According to the [terms of use of GGPONC](https://www.leitlinienprogramm-onkologie.de/projekte/ggponc-english/), contributions which are based on the corpus must cite the following publication: 

* Florian Borchert, Christina Lohr, Luise Modersohn, Thomas Langer, Markus Follmann, Jan Philipp Sachs, Udo Hahn, and Matthieu-P. Schapranow. **GGPONC: A Corpus of German Medical Text with Rich Metadata Based on Clinical Practice Guidelines**. In Proceedings of the 11th International Workshop on Health Text Mining and Information Analysis, 38–48. Online: Association for Computational Linguistics, 2020.

BibTeX:
```
@inproceedings{borchert-etal-2020-ggponc,
    title = "{GGPONC}: A Corpus of {G}erman Medical Text with Rich Metadata Based on Clinical Practice Guidelines",
    author = "Borchert, Florian  and
      Lohr, Christina  and
      Modersohn, Luise  and
      Langer, Thomas  and
      Follmann, Markus  and
      Sachs, Jan Philipp  and
      Hahn, Udo  and
      Schapranow, Matthieu-P.",
    booktitle = "Proceedings of the 11th International Workshop on Health Text Mining and Information Analysis",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.louhi-1.5",
    doi = "10.18653/v1/2020.louhi-1.5",
    pages = "38--48",
  }
```

