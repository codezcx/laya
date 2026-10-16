"""Common montage and manifest helpers; historical classifier removed."""
import json
ALIASES={'T3':'T7','T4':'T8','T5':'P7','T6':'P8'}
CHANNELS=['Fp1','Fp2','Fz','F3','F4','F7','F8','FC1','FC2','FC5','FC6','Cz','C3','C4','T7','T8','CP1','CP2','CP5','CP6','Pz','P3','P4','P7','P8','PO3','PO4','Oz','O1','O2']

def canonical(name):return ALIASES.get(name.upper(),name.upper()).upper()


def picks(names):
 names=[canonical(n) for n in names];assert len(names)==len(set(names));return [names.index(canonical(c)) for c in CHANNELS]


def readrows(p):return [json.loads(s) for s in p.read_text().splitlines()]
