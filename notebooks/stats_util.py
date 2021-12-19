sort_order = {
    'O' : 1000,
    'Coarse' : 0,
    'Fine' : 1,
    'all' : 0,
    'Finding': 1,
    'Diagnosis or Pathology' : 2,
    'Other Finding' : 3,
    'Substance': 4,
    'Clinical Drug' : 5,
    'Nutrient or Body Substance': 6,
    'External Substance' : 7,
    'Procedure': 8,
    'Therapeutic': 9,
    'Diagnostic' : 10,
    'Specification': 11,
    'Train' : 0,
    'Dev' : 1,
    'Test' : 2,
    'Tokens / Mention' : -1,
    'Short': 0,
    'Long': 1,
    'count': 0,
    'Total': 999,
    'n_sentences': 1,
    'n_files' : 2,
    'token_count': 3
}

def sort_fn(x): 
    return [sort_order[i] if i in sort_order else sort_order[i.capitalize()] for i in x]
