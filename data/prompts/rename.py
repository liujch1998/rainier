import os, sys
import glob
import copy

mapping = {
    'openbookqa': 'obqa',
    'arc_easy': 'arc_e',
    'arc_hard': 'arc_h',
    'ai2_science_elementary': 'ai2sci_e',
    'ai2_science_middle': 'ai2sci_m',
    'commonsenseqa': 'csqa',
    'qasc': 'qasc',
    'physical_iqa': 'piqa',
    'social_iqa': 'siqa',
    'winogrande_xl': 'wg',
    'numersense': 'numersense',
    'riddlesense': 'riddlesense',
    'quartz': 'quartz',
    'hellaswag': 'hellaswag',
}

for f in glob.glob('*.txt'):
    g = copy.deepcopy(f)
    for k, v in mapping.items():
        g = g.replace(k, v)
    os.rename(f, g)

