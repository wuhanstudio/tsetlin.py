import sys

USE_TQDM   = True

if sys.implementation.name == 'micropython':
    USE_TQDM = False

if USE_TQDM:
    from tqdm import tqdm

def m_tqdm(iterable, desc=""):
    return tqdm(iterable, desc=desc) if USE_TQDM else iterable
