DATASETS = [
    "openbookqa",
    "arc_easy",
    "arc_hard",
    "ai2_science_elementary",
    "ai2_science_middle",
    "commonsenseqa",
    "commonsenseqa_test",
    "qasc",
    "qasc_test",
    "social_iqa",
    "social_iqa_test",
    "physical_iqa",
    "physical_iqa_test",
    "winogrande_xl",
]

import os, sys

os.makedirs('uqa', exist_ok=True)
for ds in DATASETS:
    os.system(f'gsutil -m cp -r gs://unifiedqa/data/{ds} ./uqa/')

