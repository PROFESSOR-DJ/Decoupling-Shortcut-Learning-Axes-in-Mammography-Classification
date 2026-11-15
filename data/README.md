# Data

The datasets are **not included** in this repository, because their licences do not allow redistribution. Download them from the sources below and place them in this folder.

## 1. CBIS-DDSM (main dataset)

Curated Breast Imaging Subset of the Digital Database for Screening Mammography: scanned film mammograms with biopsy-confirmed labels and BI-RADS density.

| | Link |
|---|---|
| Official collection page (The Cancer Imaging Archive) | https://www.cancerimagingarchive.net/collection/cbis-ddsm/ |
| Dataset DOI | https://doi.org/10.7937/K9/TCIA.2016.7O02S9CY |
| **JPEG version used by this code** (Kaggle) | https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset |

The code reads the **JPEG version**, which has the `csv/` and `jpeg/` folders. The official TCIA release is in DICOM format. To use it, convert it to the same layout first.

**Download steps (Kaggle):**

1. Sign in to Kaggle and open the dataset link above.
2. Click **Download**, and unzip the archive.
3. Rename or move the unzipped folder to `data/CBIS-DDSM/`, so that it contains `csv/` and `jpeg/`.

Or, with the Kaggle command-line tool:

```bash
pip install kaggle
kaggle datasets download -d awsaf49/cbis-ddsm-breast-cancer-image-dataset -p data/CBIS-DDSM --unzip
```

## 2. INbreast (external validation)

Full-field digital mammograms from Centro Hospitalar de S. João, Porto, Portugal.

| | Link |
|---|---|
| Dataset paper (Moreira et al., *Academic Radiology*, 2012) | https://pubmed.ncbi.nlm.nih.gov/22078258/ |

INbreast has no public download page. Access is granted by the dataset authors on request: contact them through the paper above and agree to their terms of use. Keep the images private; do not commit them to this repository.

Use **INbreast Release 1.0** and place it in `data/INbreast/`, so that it contains `AllDICOMs/` and `INbreast.csv` (or `INbreast.xls`).

## Expected layout

```
data/
├── CBIS-DDSM/
│   ├── csv/          case-description CSV files (mass and calcification, train and test)
│   └── jpeg/         one folder of JPEG images per DICOM series
└── INbreast/
    ├── AllDICOMs/    INbreast DICOM images
    └── INbreast.csv  (or INbreast.xls) image-level metadata
```

## Data stored elsewhere

If the datasets are kept in another location, point to them with environment variables instead of copying them here (see `src/paths.py`):

```bash
# Windows (PowerShell)
$env:CBIS_DDSM_DIR = "D:\datasets\CBIS-DDSM"
$env:INBREAST_DIR  = "D:\datasets\INbreast"

# Linux / macOS
export CBIS_DDSM_DIR=/data/CBIS-DDSM
export INBREAST_DIR=/data/INbreast
```

## Citing the datasets

If you use these datasets, cite them as their providers ask:

- Sawyer-Lee, R., F. Gimenez, A. Hoogi, and D. Rubin (2016). Curated Breast Imaging Subset of Digital Database for Screening Mammography (CBIS-DDSM) [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/K9/TCIA.2016.7O02S9CY
- Lee, R.S., F. Gimenez, A. Hoogi, K.K. Miyake, M. Gorovoy, and D.L. Rubin (2017). A curated mammography data set for use in computer-aided detection and diagnosis research. *Scientific Data* 4: 170177.
- Clark, K., B. Vendt, K. Smith, et al. (2013). The Cancer Imaging Archive (TCIA): maintaining and operating a public information repository. *Journal of Digital Imaging* 26 (6): 1045–1057.
- Moreira, I.C., I. Amaral, I. Domingues, A. Cardoso, M.J. Cardoso, and J.S. Cardoso (2012). INbreast: toward a full-field digital mammographic database. *Academic Radiology* 19 (2): 236–248.
