import spacy
from spacy.pipeline import spancat

from spacy.training import Example
from typing import Optional, Iterable, Dict, Set, List, Any, Callable, Tuple
#from spacy.tokens import Token, Doc, Span

def flat_score(examples: Iterable[Example], **kwargs) -> Dict[str, Any]:
    scores = spancat.spancat_score(examples=examples, **kwargs)
    res = {}
    for k,v in scores.items():
        if type(v) is dict:
            for k1, v1 in v.items():
                if type(v1) is dict:
                    for k2, v2 in v1.items():
                        res[k1 + '_' + k2] = v2
                else:
                    res[k1] = v1
        else:
            res[k] = v
    return res

@spacy.registry.scorers("phlobo.flat_scorer")
def make_spancat_flatscorer():
    return flat_score