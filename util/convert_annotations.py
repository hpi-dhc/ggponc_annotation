from io import StringIO
import pandas as pd
import re
from typing import Dict, Tuple, List
import math
import logging
from tqdm import tqdm
import numpy as np
import json
from pathlib import Path
import argparse
import csv
import spacy
from spacy.tokens import DocBin, Doc, Span
import itertools

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

fh = logging.FileHandler('error.log')
fh.setLevel(logging.DEBUG)

log.addHandler(fh)

entity_values = {
    'value' : ['Substance', 'Procedure', 'Finding'],
    'detail' : ['Nutrient or Body Substance', 'External Substance', 'Clinical Drug', 'Other Finding', 'Diagnosis or Pathology', 'Therapeutic', 'Diagnostic']
}

entity_mapping = {
    'Substance' : ['Nutrient or Body Substance', 'External Substance', 'Clinical Drug'],
    'Finding' : ['Other Finding', 'Diagnosis or Pathology'],
    'Procedure' : ['Therapeutic', 'Diagnostic']
}

SPEC = 'Specification'

EMPTY_REGEX = r'([\*_](\[[\d_]+\])?\|?)+' # Any kind of empty field in WebAnno TSV
VALUE_REGEX = r'([^\[\[]+)(\[[\d_]+\])?$' # Any kind of non-empty field in WebAnno TSV

def read_webanno(tsv_files) -> Tuple[pd.DataFrame, List]:
    """Turns a bunch of WebAnno 3.3 TSV files into a DataFrame"""    
    dfs = []
    sentences = []
    for tsv_file in tsv_files:
        try:
            filtered_input = ""
            with open(tsv_file, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    if line.startswith('#Text='):
                        sentences.append(line[6:].strip().replace('\\\\', '\\'))
                        continue
                    if line.startswith('#') or not(line.strip()):
                        continue
                    filtered_input += line
            webanno_df = pd.read_csv(StringIO(filtered_input), sep='\t', header=None, na_filter=False, quoting=csv.QUOTE_NONE)
            cols = ['token_id', 'span', 'token', 'prefix', 'detail', '_unknown2', 'fragment', 'suffix', 'value', 'specified_by']
            if len(webanno_df.columns) >= len(cols):
                webanno_df = webanno_df[webanno_df.columns[0:len(cols)]]
                webanno_df.columns = cols
            elif len(webanno_df.columns) <= 4:
                webanno_df = webanno_df[webanno_df.columns[0:3]]
                webanno_df.columns = cols[0:3]
                webanno_df = webanno_df.reindex(columns=cols).fillna('_')
            else:
                raise Exception(f"Unknown number of columns in {tsv_file}")
            
            webanno_df['ts_id'] = webanno_df['token_id']
            webanno_df[['sentence_id', 'token_id']] = webanno_df.ts_id.str.split('-', expand=True).astype(int)
            webanno_df['file'] = tsv_file.name
            dfs.append(webanno_df[['file', 'sentence_id', 'token_id', 'ts_id', 'span', 'token', 'value', 'detail', 'specified_by', 'prefix', 'suffix', 'fragment']])
        except Exception as e:
            log.error(filtered_input)
            log.error(tsv_file)
            raise e
    return pd.concat(dfs).set_index(['file', 'sentence_id']), sentences

def webanno_to_iob_df(webanno_df, level, long_spans, debug=False, collect_errors=False, skip_errors=False, all_columns=False, select_type=None, select_level=None):
    error = False
    out = []
    for file, sentence_id in tqdm(webanno_df.index.unique()):
        try:
            d = webanno_df.loc[(file, sentence_id)]
            iob_df, success = _webanno_sentence_to_iob(d, level, long_spans, debug, all_columns=all_columns, select_type=select_type, select_level=select_level)
            out.append(iob_df)
            if not success:
                log.error(f"Error processing: {file}, {sentence_id}, aborting.")
                return iob_df
        except Exception as e:
            error = True
            message = f'{file}, {sentence_id}, {e.args}'
            log.debug(message)
            log.error(message)
            if not collect_errors and not skip_errors:
                raise e
    if error and not skip_errors:
        raise Exception("Errors during conversion, check error.log")
    return pd.concat(out)

def webanno_to_spans(webanno_df, sentences, level, long_spans=True, character_offsets=False, debug=False, collect_errors=False, skip_errors=False):
    error = False
    out = []
    for s, sentence in tqdm(zip(webanno_df.index.unique(), sentences)):
        file, sentence_id = s
        spans = []
        for entity_type in (entity_values['value'] if long_spans else [None]):
            try:
                d = webanno_df.loc[(file, sentence_id)]
                sdf, success = _webanno_sentence_to_dataframe(
                    d, level, long_spans=long_spans, debug=debug, all_columns=False, select_type=entity_type, select_level=('value' if entity_type else None)
                )
                spans += _to_spans(sdf, character_offsets)    
                assert success
            except Exception as e:
                error = True
                message = f'{file}, {sentence_id}, {e.args}'
                log.error(message)
                if not collect_errors and not skip_errors:
                    raise e
        spans.sort(key = lambda t: t[0])
        if not success:
            log.error(f"Error processing: {file}, {sentence_id}, aborting.")
            return out
        if not character_offsets:
            sentence = list(sdf.token)
        s_start = int(sdf.iloc[0].span.split('-')[0])
        s_end = int(sdf.iloc[-1].span.split('-')[1])
        out.append({'sentence' : sentence, 's_start' : s_start, 's_end' : s_end, 'file': file, 'sentence_id' : sentence_id, 'spans' : spans})
    if error and not skip_errors:
        raise Exception("Errors during conversion, check error.log")
    return out

def write_conll(conll_df, to_file):
    with open(to_file, 'w', encoding='utf-8') as out:
        for file, sentence_id in conll_df.index.unique():            
            d = conll_df.loc[(file, sentence_id)]
            for _, row in d.iterrows():
                out.write(f"{row.token}\t{row.output}\n")
            out.write("\n")

def write_huggingface(conll_df, to_file):
    with open(to_file, 'w', encoding='utf-8') as out_json:
        for file, sentence_id in conll_df.index.unique():
            d = conll_df.loc[(file, sentence_id)]
            huggingface_json = {
                'fname' : file,
                'sentence_id' : sentence_id,
                'tokens' : list(d.token),
                'tags' : list(d.output)
            }
            out_json.write((json.dumps(huggingface_json, ensure_ascii=False) + '\n'))

def write_spacy(spans, to_file):    
    nlp = spacy.blank('de')
    db = DocBin()
    
    for s in spans:
        doc = Doc(nlp.vocab, s['sentence'])
        spans = []
        for start, end, _, label in s['spans']:
            spans.append(Span(doc, start, end + 1, label=label))
        doc.spans['snomed'] = spans
        db.add(doc)
    
    db.to_disk(to_file)

def write_json(spans, to_file):
    res = []
    for f, s in itertools.groupby(spans, key=lambda s: s['file']):
        passages = []
        entities = []
        for sentence in sorted(list(s), key=lambda s: int(s['sentence_id'])):            
            sentence_text = sentence['sentence'].replace('\\t', ' ')
            s_start = sentence['s_start']
            s_end = sentence['s_end']
            passages.append({
                'id' : int(sentence['sentence_id']),
                'type' : 'sentence',
                'text' : sentence_text,
                'offsets': [[s_start, s_end]]
            })
            for start, end, tokens, label in sentence['spans']:
                text = sentence_text[start - s_start:end - s_start]
                for t in tokens:
                    if not t in text:
                        print(f"{t}, {text}, {f}, {sentence['sentence_id']}")
                        print(sentence)
                        assert False
                entities.append({
                    'text' : [text],
                    'type' : label,                    
                    'offsets' : [[start, end]]
                })
        res.append(
            {
                'document_id' : f,
                'passages' : passages,
                'entities' : entities
            }
        )
    with open(to_file, 'w', encoding='utf-8') as fh:
        json.dump(res, fh, ensure_ascii=False)

def resolve_ellipses(ellipses, iob_df):    
    results = []
    for _, r in ellipses.id.reset_index().drop_duplicates().iterrows():
        has_fragment = False
        idx = r.file, r.sentence_id
        _id = r.id
        sentence = ellipses.loc[idx]
        span = sentence[sentence.id == _id]
        full_sentence = iob_df.loc[idx]
        full_span = full_sentence[full_sentence.id == _id]
        for fragment in span.fragment: # Expand span to fragments
            if fragment and not re.match(EMPTY_REGEX, fragment):
                has_fragment = True
                fragment = re.split('[|;]', fragment)[0]
                m = re.match(VALUE_REGEX, fragment)
                if not m:
                    print(span.fragment)
                fragment_index = int(m.group(1).split('-')[1])
                fragment_start = min(fragment_index, full_span.token_id.min())
                fragment_end = max(fragment_index, full_span.token_id.max())
                full_span = full_sentence[(full_sentence.token_id >= fragment_start) & (full_sentence.token_id <= fragment_end)]
                expanded_context = full_sentence[full_sentence.id.isin(full_span.id.dropna().unique())].token_id
                fragment_start = min(fragment_start, expanded_context.min())
                fragment_end = max(fragment_end, expanded_context.max())
                full_span = full_sentence[(full_sentence.token_id >= fragment_start) & (full_sentence.token_id <= fragment_end)]
        fragment, resolution, missing_prefix, missing_suffix = join_and_resolve(full_span)
        sentence_start_offset = int(full_sentence.span.values[0].split('-')[0])
        results.append({
            'file' : idx[0],
            'sentence_id' : idx[1],
            'full_sentence' : list(full_sentence.token),
            'span_index_start': full_span.token_id.min() - 1,
            'span_index_end' : full_span.token_id.max() - 1,
            'full_span' : fragment,
            'offsets' : [[int(t) - sentence_start_offset for t in tuple(v.split('-'))] for v in full_span.span.values],
            'resolution' : resolution,
            'fragment' : has_fragment,
            'missing_prefix' : missing_prefix,
            'missing_suffix' : missing_suffix,
        })
    return pd.DataFrame(results)
    
def join_and_resolve(span):
    """
    Expands prefix / suffix / fragment annotations in a span
    """
    raw = []
    solution = []
    suffix = None
    suffix_id = None
    prefix_id = None
    has_prefix = False
    has_suffix = False
    cur_prefix = None
    cur_suffix = None 
    for _, i in span.iterrows():
        token = i.token
        if cur_suffix and i.token == '-':
            token = ''
        cur_prefix = None
        cur_suffix = None 
        if i.prefix:
            if re.match(EMPTY_REGEX, i.prefix):
                new_prefix_id = None
            else:
                has_prefix = True
                p = i.prefix.split('|')[0]
                m = re.match(VALUE_REGEX, p)
                new_prefix_id = m.group(2)
                if not new_prefix_id or (new_prefix_id != prefix_id):
                    cur_prefix = m.group(1).replace(r'\_', ' ')
            prefix_id = new_prefix_id
        if i.suffix:
            if re.match(EMPTY_REGEX, i.suffix):
                new_suffix_id = None
                if suffix:
                    cur_suffix = suffix
                suffix = None
            else:
                has_suffix = True
                s = i.suffix.split('|')[0]
                m = re.match(VALUE_REGEX, s)
                new_suffix_id = m.group(2)
                if (not new_suffix_id or (new_suffix_id != suffix_id)) and suffix:
                    cur_suffix = suffix
                suffix = m.group(1).replace(r'\_', ' ')
            suffix_id = new_suffix_id
        raw.append(token)
        if cur_suffix:
            solution[-1] = solution[-1] + cur_suffix.rstrip()
        if cur_suffix and token in ['-', '–']:
            solution.append('')
            continue
        if cur_prefix and len(solution) > 0 and solution[-1] in ['-', '–']:
            solution[-1] = ''
        solution.append((f'{cur_prefix.lstrip() if cur_prefix else ""}{token}'))
    if suffix:
        solution[-1] += suffix.rstrip()
    return raw, solution, has_prefix, has_suffix
            
def _resolve_conflict(p1 : Dict, p2 : Dict, level : str):
    """Resolves conflicts between overlapping entity classes"""
    if p1[f'{level}_entity_class'] == p2[f'{level}_entity_class']:
        return p2
    if p1[f'{level}_entity_class'] == 'Diagnostic' and p2[f'{level}_entity_class'] == 'Therapeutic':
        return p2
    if p1[f'{level}_entity_class'] == 'Therapeutic' and p2[f'{level}_entity_class'] == 'Diagnostic':
        return p1
    return None

def _split_multi_annotations(v_string, level):
    """ Splits an annotation of a token of the form Annotation1[Number1]|Annotation2[Number2] into its parts and resolves conclicts, when possible"""
    parts = v_string.split('|')
    regex = re.compile(r'([A-Za-z ]+)\[(\d+)\]')
    def convert_part(part):
        match = regex.match(part)
        if match:
            value = match.group(1)
            counter = int(match.group(2))
        else:
            value = part
            counter = -1
        if value in entity_values[level]: # Single token
            return {f'{level}_entity_id' : counter, f'{level}_entity_class' : value}
        elif value == SPEC:
            return {f'{level}_specification_id' : counter}
        elif value == '_':
            return {}
        elif level == 'detail' and re.match(r'\*(\[\d+\])?', value):
            return {f'{level}_specification_id' : counter}
        else:
            raise Exception(v_string, value) 
    def merge_parts(p1, p2):
        if f'{level}_specification_id' in p1 and f'{level}_specification_id' in p2:
            return {f'{level}_specification_id' : (p1[f'{level}_specification_id'], p2[f'{level}_specification_id'])} 
        if not len(set(p1.keys()).intersection(set(p2.keys()))) == 0:
            if f'{level}_entity_class' in p1 and f'{level}_entity_class':
                log.warning('Duplicate annotation: ' + v_string)
                res = _resolve_conflict(p1, p2, level)
                if not res:
                    raise Exception(v_string)
            else:
                raise Exception(v_string)
        return {**p1, **p2}
    if len(parts) == 1:
        return convert_part(parts[0])
    elif len(parts) == 2:
        return merge_parts(convert_part(parts[0]), convert_part(parts[1]))
    elif len(parts) == 3:
        return merge_parts(merge_parts(convert_part(parts[0]), convert_part(parts[1])), convert_part(parts[2]))
    else:
        raise Exception('Multi-stacking not implemented: ' + v_string)

def _expand_specs(sentence_df, level, select_type, select_level):
    """ Propagate token labels to the corresponding specification sections """
    sdf = sentence_df.copy()
    #from IPython.display import display
    #display(sdf)
    for i, row in sdf[(sdf.specified_by != '_') & (sdf[f'value_specification_id'].isna())].iterrows():
        specs = row.specified_by.split('|')
        for s in specs:
            m = re.match(r'(\d+-\d+)\[(\d+)_\d+\]', s)
            if m: # Specification with number -> point to ID
                ts_id = m.group(1)
                spec_id = int(m.group(2)) 
                idx = (sdf.ts_id == ts_id) | sdf[f'value_specification_id'].apply(
                    lambda sid: sid == spec_id or (type(sid) is tuple and spec_id in sid))
                if (sdf.loc[idx, 'value_specification_id'] == spec_id).sum() == 0:
                    #assert sum(idx) == 1, sdf[idx]
                    spec_id = sdf[idx].iloc[0].value_specification_id
            else: # Specification without number -> point to Token-ID
                idx = (sdf.ts_id == s)
                assert sum(idx) == 1, s
                spec_id = sdf[idx].iloc[0].value_specification_id
            if select_type and row[f'{select_level}_entity_class'] != select_type: # Drop the specification
                def get_spec_id(cur_spec_id): #Remove unused specs
                    if not spec_id:
                        return cur_spec_id
                    if cur_spec_id == spec_id:
                        return math.nan
                    if type(cur_spec_id) == tuple:
                        res = [t for t in cur_spec_id if t != spec_id]
                        if len(res) == 1:
                            return res[0]
                        return math.nan if not res else tuple(res)
                            
                sdf.loc[idx & (sdf[f'{select_level}_entity_id'] != select_type), 'value_specification_id'] = sdf.loc[idx & (sdf[f'{select_level}_entity_id'] != select_type), 'value_specification_id'].map(get_spec_id)
                sdf.loc[idx & (sdf.specified_by != '_'), 'value_specification_id'] = math.nan
                sdf.loc[sdf.ts_id == row.ts_id, 'specified_by'] = '_'
            else:                
                sdf.loc[idx, f'{level}_entity_id'] = row[f'{level}_entity_id']
                sdf.loc[idx, f'{level}_entity_class'] = row[f'{level}_entity_class']
                sdf.loc[idx, f'value_specification_id'] = math.nan
    return sdf

def _to_iob(sentence_df, all_columns):
    """ creates IOB tags from resolved WebAnno entities"""
    sdf = sentence_df.copy()
    entity_counter = -1
    out_col = sdf.columns.get_loc("output")
    for i, row_tuple in enumerate(sdf.iterrows()):
        row = row_tuple[1]
        out = row['output'].replace(' ', '_')
        entity_id = row['entity_id']
        if out != 'O' and not(math.isnan(entity_id)) and entity_id != entity_counter:
            entity_counter = entity_id
            sdf.iloc[i, out_col] = 'B-' + out
        elif out != 'O' and entity_id == entity_counter:
            sdf.iloc[i, out_col] = 'I-' + out
        else:
            entity_counter = -1
            assert out == 'O', sentence_df
    return sdf[['token', 'output']] if not all_columns else sdf

def _to_spans(sentence_df, character_offsets : bool):
    """ creates spans from resolved WebAnno entities"""
    sdf = sentence_df.copy()
    
    def merge_entities(tokens):
        s_start = tokens[0][0]
        s_end = tokens[-1][0]
        span_start = int(tokens[0][2].split('-')[0])
        span_end = int(tokens[-1][2].split('-')[1])
        if character_offsets:
            return (span_start, span_end, [t[1] for t in tokens], tokens[0][3])
        else:
            return (s_start, s_end, [t[1] for t in tokens], tokens[0][3])
    
    entity_counter = -1
    entities = []
    cur_entity = None
    for i, row_tuple in enumerate(sdf.iterrows()):
        row = row_tuple[1]
        out = row['output'].replace(' ', '_')
        entity_id = row['entity_id']
        if not(math.isnan(entity_id)) and entity_id != entity_counter and out != 'O': #new entity
            if cur_entity:
                entities.append(merge_entities(cur_entity))
            entity_counter = entity_id
            cur_entity = [(i, row.token, row.span, out)]
        elif entity_id == entity_counter:
            cur_entity += [(i, row.token, row.span, out)]
        else:
            entity_counter = -1
            assert out == 'O', sentence_df
            if cur_entity:
                entities.append(merge_entities(cur_entity))
                cur_entity = None
    if cur_entity:
        entities.append(merge_entities(cur_entity))
    return entities

def _close_gaps(sentence_df):
    sdf = sentence_df.copy()
    entity_ids = sdf.entity_id.unique()
    for eid in entity_ids:
        if math.isnan(eid):
            continue
        entity_range = np.nonzero((sdf.entity_id == eid).values)[0]
        if len(entity_range) == 0:
            continue
        e_min, e_max = entity_range[0], entity_range[-1]
        out_col = sdf.columns.get_loc('output')
        e_class = sdf.iloc[e_min, out_col]
        if sdf.iloc[e_min:e_max, out_col].isin([e_class, pd.NA]).all(): # Nothing else inbetween
            # Close gap
            sdf.iloc[e_min:e_max, out_col] = e_class
            sdf.iloc[e_min:e_max, sdf.columns.get_loc('entity_id')] = eid
    return sdf

def _webanno_sentence_to_dataframe(sentence_df, level, long_spans, debug, all_columns, select_type=None, select_level=None):
    assert bool(select_type) == bool(select_level)
    """ Turns a Webanno sentence DataFrame to a DataFrame with token labels """
    assert not select_type or select_type in entity_values['value' if select_level != level else select_level]
    value_split_anno = pd.DataFrame(list(sentence_df['value'].apply(lambda r: _split_multi_annotations(r, 'value'))), 
            index=sentence_df.index, columns=['value_entity_id', 'value_entity_class', 'value_specification_id'])
    detail_split_anno = pd.DataFrame(list(sentence_df['detail'].apply(lambda r: _split_multi_annotations(r, 'detail'))), 
            index=sentence_df.index, columns=['detail_entity_id', 'detail_entity_class', 'detail_specification_id'])
    sdf = pd.concat([sentence_df, value_split_anno, detail_split_anno], axis=1)
    sdf['spec_id'] = sdf['value_specification_id'] # Store Spec ID for further processing
    max_id = sdf[f'{level}_entity_id'].max()
    sdf.loc[sdf[f'{level}_entity_id'] == -1, [f'{level}_entity_id']] = sdf[sdf[f'{level}_entity_id'] == -1].token_id + max_id
    if not long_spans: # Ignore specifications and just take the value
        sdf['entity_id'] = sdf[f'{level}_entity_id']
        sdf['output'] = pd.NA
        mask = sdf[f'{level}_entity_class'].isin(entity_values[level])
        sdf.loc[mask, 'output'] = sdf.loc[mask, f'{level}_entity_class']
        mask = sdf['output'].isna() & ((sdf[level] == '_') | (sdf[level] == '*') | (sdf[level] == SPEC) | (~sdf.value_specification_id.isna()))
        sdf.loc[mask, 'output'] = 'O'
    else:
        j = 0
        while (~(sdf['value_specification_id'].isna())).sum() > 0: # Get rid of all specifications
            sdf = _expand_specs(sdf, level, select_type, select_level)
            j += 1
            if j > 10:
                raise Exception("Stuck expanding sections")
        sdf['entity_id'] = sdf[f'{level}_entity_id']
        sdf['output'] = pd.NA
        if not select_type:
            mask = sdf[f'{level}_entity_class'].isin(entity_values[level])
        else:
            if level == select_level:
                mask = sdf[f'{level}_entity_class'] == select_type
            else:
                assert select_level == 'value' and level == 'detail', (select_level, level)
                mask = sdf[f'{level}_entity_class'].isin(entity_mapping[select_type])
        sdf.loc[mask, 'output'] = sdf.loc[mask, f'{level}_entity_class']
        sdf = _close_gaps(sdf)
        if not select_type:
            mask = sdf['output'].isna() & ((sdf[level] == '_') | (sdf[level] == '*'))
        else:
            mask = sdf['output'].isna()
        sdf.loc[mask, 'output'] = 'O'
        
    if sdf.output.isna().sum() != 0 and debug:
        return sdf, False
    assert sdf.output.isna().sum() == 0

    if not long_spans:
        for _, anno in sdf.iterrows():
            _check_annos(anno)
    return sdf, True



def _webanno_sentence_to_iob(sentence_df, level, long_spans, debug, all_columns=False, select_type=None, select_level=None):
    """ Turns a Webanno sentence DataFrame to a DataFrame with IOB tags """
    sdf, success = _webanno_sentence_to_dataframe(sentence_df, level, long_spans, debug, all_columns, select_type=select_type, select_level=select_level)
    return _to_iob(sdf, all_columns), success

def _check_annos(anno):
    def is_empty(v):
        return not v or (type(v) == float and math.isnan(v))
    if not(is_empty(anno.value_entity_class)) and is_empty(anno.detail_entity_class):
        raise Exception("Missing detail")
    if anno.detail_entity_class in ['Other Finding', 'Diagnosis or Pathology'] and anno.value_entity_class != 'Finding':
        raise Exception(f'Wrong value {anno.value_entity_class}')
    if anno.detail_entity_class in ['Nutrient or Body Substance', 'External Substance', 'Clinical Drug'] and anno.value_entity_class != 'Substance':
        raise Exception(f'Wrong value {anno.value_entity_class}')
    if anno.detail_entity_class in ['Diagnostic', 'Therapeutic'] and anno.value_entity_class != 'Procedure':
        raise Exception(f'Wrong value {anno.value_entity_class}')
    if anno.value_entity_class == 'Finding' and not anno.detail_entity_class in ['Other Finding', 'Diagnosis or Pathology']:
        raise Exception(f'Wrong detail {anno.detail_entity_class}')
    if anno.value_entity_class == 'Substance' and not anno.detail_entity_class in ['Nutrient or Body Substance', 'External Substance', 'Clinical Drug']:
        raise Exception(f'Wrong detail {anno.detail_entity_class}')
    if anno.value_entity_class == 'Procedure' and not anno.detail_entity_class in ['Diagnostic', 'Therapeutic']:
        raise Exception(f'Wrong detail {anno.detail_entity_class}')


def main():
    parser = argparse.ArgumentParser(description='Convert WebAnno Files')
    parser.add_argument('input_folder')
    parser.add_argument('file_prefix')
    parser.add_argument('output_folder')
    parser.add_argument('--collect_errors', action="store_true")
    parser.add_argument('--skip_errors', action="store_true")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--iob', action="store_true")
    group.add_argument('--spacy', action="store_true")
    group.add_argument('--bigbio', action="store_true")
    parser.add_argument('entity_type', nargs='?', default=None)
    
    args = parser.parse_args()
        
    output_folder = Path(args.output_folder)
    
    formats = ['conll', 'huggingface']
    levels = ['detail', 'value']
    extend = ['short', 'long']
    
    webanno_df, sentences = read_webanno(Path(args.input_folder).glob('*.tsv'))
    
    output_folder.mkdir(exist_ok=True)
    
    if args.skip_errors:
        log.warning("SKIPPING ERRORS, USE FOR DEVELOPMENT ONLY")

    if args.iob:
        log.info("Converting to HuggingFace and ConLL (IOB-encoded, one label per token)")
        for l in levels:
            granularity = 'coarse' if l == 'value' else 'fine'
            for e in extend:
                if args.entity_type and (granularity == 'coarse' or e == 'short'):
                    continue
                iob_df = webanno_to_iob_df(webanno_df, l, e == 'long', 
                    collect_errors=args.collect_errors, skip_errors=args.skip_errors, 
                    select_type=args.entity_type, select_level='value' if args.entity_type else None)
                for f in formats:
                    (output_folder / f / granularity / e).mkdir(exist_ok=True, parents=True)
                    if f == 'conll':
                        out_file = f'{args.file_prefix}_{granularity}_{e}{"." + args.entity_type if args.entity_type else ""}.conll'
                        log.info(f'Writing {out_file}')
                        write_conll(iob_df, output_folder / f / granularity / e / out_file)
                    elif f == 'huggingface':
                        out_file = f'{args.file_prefix}_{granularity}_{e}{"." + args.entity_type if args.entity_type else ""}.json'
                        log.info(f'Writing {out_file}')
                        write_huggingface(iob_df, output_folder / f / granularity / e / out_file)
    elif args.spacy:
        log.info("Converting to spaCy spans (potentially overlapping spans)")
        (output_folder / 'spacy').mkdir(exist_ok=True, parents=True)
        out_file = f'{args.file_prefix}.spacy'
        spans = webanno_to_spans(webanno_df, sentences, 'detail', character_offsets=False, collect_errors=args.collect_errors, skip_errors=args.skip_errors)
        log.info(f'Writing {out_file}')
        write_spacy(spans, output_folder / 'spacy' / out_file)
    elif args.bigbio:
        log.info("BigBIO")
        for l in levels:
            granularity = 'coarse' if l == 'value' else 'fine'
            for e in extend:
                (output_folder / 'json' / granularity / e).mkdir(exist_ok=True, parents=True)
                log.info(f"{l}, {e}")
                spans = webanno_to_spans(webanno_df, sentences, l, long_spans=(e == 'long'), character_offsets=True, collect_errors=args.collect_errors, skip_errors=args.skip_errors)                
                out_file = output_folder / 'json' / granularity / e / f'{args.file_prefix}.json'
                log.info(f'Writing {out_file}')
                write_json(spans, out_file)

if __name__ == "__main__":
    main()
    
