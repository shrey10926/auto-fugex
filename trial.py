import re, spacy
import pandas as pd
from re import search

nlp = spacy.load("en_core_web_md")

def base_regex_fn(disc):

    clean_text = re.sub(r'[^a-zA-Z0-9,.$%]', ' ', disc.lower()).strip()
    regex_inter = re.sub(r'[$]', '\$', clean_text)

    doc = nlp(regex_inter)

    filtered_tokens = [token.text for token in doc if not token.is_stop]

    seen = set()
    tok_list = [x for x in filtered_tokens if not (x in seen or seen.add(x))]

    final = ' '.join((' '.join(tok_list)).split())
    inter_step = re.sub(r'\$\s+', '$', final)
    inter_step1 = re.sub(r'\s+%+', '%', inter_step)
    split = inter_step1.split()

    return split

def base_fuzzy(disc):

    clean_text = re.sub(r'[^a-zA-Z0-9,.$%]', ' ', disc.lower()).strip()

    doc = nlp(clean_text)

    filtered_tokens = [token.text for token in doc if not token.is_stop]

    seen = set()
    tok_list = [x for x in filtered_tokens if not (x in seen or seen.add(x))]

    final = ' '.join((' '.join(tok_list)).split())
    inter_step = re.sub(r'\$\s+', '$', final)
    inter_step1 = re.sub(r'\s+%+', '%', inter_step)
    split = inter_step1.split()

    return split


def create_sublist(ip_list, win_size = 5, overlap = 1):

    result = []
    if not isinstance(ip_list, list) or not all(isinstance(item, str) for item in ip_list):
        raise ValueError(f'input list must be a list of strings')

    if win_size <= 0:
        raise ValueError(f'window size ({win_size}) must be grater than 0!')

    if overlap < 0:
        raise ValueError(f'overlap ({overlap}) must be non negative!')
    
    if overlap >= win_size:
        raise ValueError(f'overlap ({overlap}) must be less than window size ({win_size})!')

    if len(ip_list) < win_size:
        raise ValueError(f'disclosure length ({len(ip_list)}) must be greater than the window size ({win_size})!')

    if not ip_list:
        return result

    i = 0
    while i < len(ip_list):
        sublist = ip_list[i : i + win_size]
        result.append(sublist)
        i += win_size - overlap

    if len(result[-1]) == 1:
        return result[:-1]
    else:
        return result


def regex_process(ip_list, pattern = '(\s[A-Za-z0-9!@#$&?:()-.%+,\/|]*\s+){0,5}'):

    inter_dict, final_dict = {}, {}
    for idx, value in enumerate(ip_list):
        inter_dict[f'regex{idx + 1}'] = value

    for k, v in inter_dict.items():
        final_dict[k] = '(?i)' + '(' + pattern.join(v) + ')'

    return final_dict

# disclosure = f"""You can earn fuck shit"""
disclosure = f"""You can earn 25,000 membership reward points after you make eligible purchases with your new blue business plus card totalling $3,000 or more in the first 3 months of your card membership starting from the date your account was opened. You won't be eligible for the membership reward points if you already have existing american express card. You also won't be eligible for the welcome bonus if you cancel or downgrade any american express cards."""
op1 = create_sublist(base_regex_fn(disclosure))
op2 = regex_process(create_sublist(base_regex_fn(disclosure)))
