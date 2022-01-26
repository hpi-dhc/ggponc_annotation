import spacy
from spacy.pipeline import spancat
from spacy.pipeline.spancat import Suggester

from spacy.training import Example
from typing import Optional, Iterable, Dict, Set, List, Any, Callable, Tuple, cast
from thinc.api import Optimizer, Ops, get_current_ops
from thinc.types import Ragged, Ints2d, Floats2d, Ints1d
from spacy.tokens import Doc

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

@spacy.registry.misc("phlobo.chunk_and_ngram_suggester")
def build_chunk_and_ngram_suggester(sizes: List[int], max_depth: int) -> Suggester:

    def chunk_and_ngram_suggester(docs: Iterable[Doc], *, ops: Optional[Ops] = None) -> Ragged:
        if ops is None:
            ops = get_current_ops()
        spans = []
        lengths = []
        for i, doc in enumerate(docs):            
            length = 0
            if doc.has_annotation("DEP"):
                chunks = []
                nc = list(doc.noun_chunks)
                
                for chunk in nc:
                    chunks.append([chunk.start, chunk.end])                            

                    def extend_chunk(head, the_chunk): 
                        for other_chunk in nc:
                            if head in other_chunk:
                                start = min(the_chunk.start, other_chunk.start)
                                end = max(the_chunk.end, other_chunk.end)
                                return other_chunk, doc[start:end]
                        return None, None

                    extended = True
                    extension_chunk = chunk
                    
                    depth = 0
                    
                    while extended and depth < max_depth:
                        depth += 1
                        extension_chunk, extended = extend_chunk(extension_chunk.root.head.head, chunk)

                        if extended:
                            chunks.append([extended.start, extended.end])                            
                            
                chunks = ops.asarray(chunks, dtype="i")
                if chunks.shape[0] > 0:
                    spans.append(chunks)
                    length += chunks.shape[0]
            
            # Add n-grams
            starts = ops.xp.arange(len(doc), dtype="i")
            starts = starts.reshape((-1, 1))
            for size in sizes:
                if size <= len(doc):
                    starts_size = starts[: len(doc) - (size - 1)]
                    spans.append(ops.xp.hstack((starts_size, starts_size + size)))
                    length += spans[-1].shape[0]
                if spans:
                    assert spans[-1].ndim == 2, spans[-1].shape            
            
            lengths.append(length)
                    
        lengths_array = cast(Ints1d, ops.asarray(lengths, dtype="i"))
        if len(spans) > 0:
            output = Ragged(ops.xp.vstack(spans), lengths_array)
        else:
            output = Ragged(ops.xp.zeros((0, 0), dtype="i"), lengths_array)
 
        assert output.dataXd.ndim == 2
    
        return output
 
    return chunk_and_ngram_suggester
