import sys

import pandas as pd
from inceptalytics.utils import construct_feature_path
from inceptalytics import Project
from time import time
from datetime import datetime
import argparse
from pathlib import Path

layer = "webanno.custom.SNOMEDEntity"
features = ['detail', 'value']

# Skip sentences for which gamma computation did not finish after 24 hrs
skip_sentences = [
    '18_hodgkin-lymphom_0051.txt_1979-2462',
    '22_prostatakarzinom_0153.txt_0-458'
]


def calculate_agreement(file):
    
    results = []

    print('> File:', file)

    total_start_time = time()
    
    project = Project.from_zipped_xmi(str(file))

    selected_annotators = [a for a in project.annotators if a != 'annotator_jw']

    for f in features:
        print('>> Feature: ', f)
        view = project.select(
            annotation=construct_feature_path(layer, f), 
            annotators=selected_annotators, 
            source_files=project.source_file_names
        )
        
        view._annotation_dataframe = view._annotation_dataframe.query("not sentence in @skip_sentences")
        
        n_docs = len(view._annotation_dataframe.index.get_level_values(0).unique())
        n_sentences = len(view._annotation_dataframe.index.get_level_values(1).unique())

        labels = [l for l in view.labels if l != 'None']
        print(labels)

        if not labels:
            print('>>> Skipping empty layer')
            continue
        for l in (['all'] + labels):
            print('>>> Label:', l)
            start_time = time()
            gamma = (view if l == 'all' else view.filter_labels(l)).iaa(measure='gamma')
            results.append({
                'file' : file.name,
                'n_docs' : n_docs,
                'n_sentences' : n_sentences,
                'feature' : f,
                'n_anno' : len(selected_annotators),
                'label' : l,
                'gamma' : gamma
            })
            end_time = time()
            print(">>> %.2f seconds" % (end_time - start_time))

    total_end_time = time()
        
    print("> Total time: %.2f seconds" % (total_end_time - total_start_time))

    gamma_agreements = pd.DataFrame(results)
    gamma_agreements.to_csv(f'{file.name}_{datetime.now().strftime("%Y-%m-%d_%H_%M_%S")}_gamma_agreement_.csv')

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    args = parser.parse_args()    
    calculate_agreement(Path(args.filename))