"""Repository paths.

Datasets are read from data/ unless CBIS_DDSM_DIR or INBREAST_DIR point elsewhere,
and every script writes its results under outputs/.
"""
import os

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(REPO_DIR, 'data')
OUTPUT_DIR = os.path.join(REPO_DIR, 'outputs')

# CBIS-DDSM (JPEG version): folder that contains csv/ and jpeg/
CBIS_DDSM_DIR = os.environ.get('CBIS_DDSM_DIR', os.path.join(DATA_DIR, 'CBIS-DDSM'))

# INbreast: folder that contains AllDICOMs/ and INbreast.csv (or .xls)
INBREAST_DIR = os.environ.get('INBREAST_DIR', os.path.join(DATA_DIR, 'INbreast'))
