[![DOI](https://zenodo.org/badge/411603699.svg)](https://zenodo.org/badge/latestdoi/411603699)


# GGPONC 2.0 — Annotation and Experiments

This repository contains the code to reproduce the results from the paper:

Florian Borchert, Christina Lohr, Luise Modersohn, Jonas Witt, Thomas Langer, Markus Follmann, Matthias Gietzelt, Bert Arnrich, Udo Hahn, and Matthieu-P. Schapranow. 2022. [GGPONC 2.0 - The German Clinical Guideline Corpus for Oncology: Curation Workflow, Annotation Policy, Baseline NER Taggers](https://aclanthology.org/2022.lrec-1.389/). In Proceedings of the Thirteenth Language Resources and Evaluation Conference, pages 3650–3660, Marseille, France. European Language Resources Association.

## Preparation

1. Get access to GGPONC following the instructions on the [project homepage](https://www.leitlinienprogramm-onkologie.de/projekte/ggponc-english/) and place the the contents of the 2.0 release (v2.0_2022_03_24) in the `data` folder
2. Install Python dependencies `pip install -r requirements.txt` 

## Preprocessing (optional)

The prepocessed, i.e., sentence-splitted and tokenized text portions, are already included in the release (`data/plain_text`) 

If you want to do the preprocessing yourself, build the jar file from [https://github.com/hpi-dhc/ggponc_preprocessing](https://github.com/hpi-dhc/ggponc_preprocessing) and use it with `data/xml/cpg-corpus-cms.xml`

## Notebooks

In `notebooks`, we provide the following Jupyter Notebooks to reproduce the results from the paper:

- [01_Statistics.ipynb](notebooks/01_Statistics.ipynb)
    - Corpus Statistics and IAA Calculcation with INCEpTALYTICS
- [02_NER_Baselines.ipynb](notebooks/02_NER_Baselines.ipynb)
    - NER Baselines using BERT / HuggingFace Transformers
- [03_NER_Analysis.ipynb](notebooks/03_NER_Analysis.ipynb)
    - Analysis of NER errors and coordination ellipses
- [04_spaCy_Spancat.ipynb](notebooks/04_spaCy_Spancat.ipynb) 
    - Alternative NER implementation using spaCy's [SpanCategorizer](https://spacy.io/api/spancategorizer) feature, which can handle overlapping and nested mentions
- [05_Ellipses.ipynb](notebooks/05_Ellipses.ipynb)
    - Extraction of elliptical coordinated compound noun phrases from GGPONC 2.0 annotations

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

Florian Borchert, Christina Lohr, Luise Modersohn, Jonas Witt, Thomas Langer, Markus Follmann, Matthias Gietzelt, Bert Arnrich, Udo Hahn, and Matthieu-P. Schapranow. 2022. **GGPONC 2.0 - The German Clinical Guideline Corpus for Oncology: Curation Workflow, Annotation Policy, Baseline NER Taggers**. In Proceedings of the Thirteenth Language Resources and Evaluation Conference, pages 3650–3660, Marseille, France. European Language Resources Association.

BibTeX:

```
@inproceedings{borchert-etal-2022-ggponc,
    title = "{GGPONC} 2.0 - The {G}erman Clinical Guideline Corpus for Oncology: Curation Workflow, Annotation Policy, Baseline {NER} Taggers",
    author = "Borchert, Florian  and
      Lohr, Christina  and
      Modersohn, Luise  and
      Witt, Jonas  and
      Langer, Thomas  and
      Follmann, Markus  and
      Gietzelt, Matthias  and
      Arnrich, Bert  and
      Hahn, Udo  and
      Schapranow, Matthieu-P.",
    booktitle = "Proceedings of the Thirteenth Language Resources and Evaluation Conference",
    month = jun,
    year = "2022",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    pages = "3650--3660"
}

```

