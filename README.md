# Tesseract-Uchen-Training

A customized [tesstrain](https://github.com/tesseract-ocr/tesstrain) pipeline for training Tesseract OCR on Tibetan Uchen script.

## Overview

This repository extends the upstream `tesstrain` pipeline with data preparation scripts and a Makefile configured for Tibetan Uchen training. It produces a `tib.traineddata` model trained from scratch on OpenPecha Uchen datasets.

## Repository Structure

```text
.
├── Scripts/              # Custom data preparation scripts
├── Makefile              # Customized tesstrain Makefile
├── model/                # Trained model checkpoint
├── plots/                # Training plots
└── (upstream tesstrain files)
```

## Customizations

### Data Preparation Scripts (`Scripts/`)

- **`lmdbdownload.py`** — Extracts samples from LMDB-format datasets. Validates labels against a Tibetan Uchen charset, filters out invalid or oversized samples, and writes paired image and label files. Uses multiprocessing.
- **`preprocess.py`** — Converts extracted image/label pairs into tesstrain format with `.gt.txt` ground truth files.

Train/test/eval splits were generated using [OpenPecha/Tibetan_stacks_and_frequency](https://github.com/OpenPecha/Tibetan_stacks_and_frequency).

### Makefile Configuration

The Makefile is pre-configured for Tibetan training:

| Parameter | Value |
|---|---|
| `MODEL_NAME` | `tib` |
| `LANG_TYPE` | `Indic` |
| `PSM` | `13` |
| `EPOCHS` | `3` |
| `LEARNING_RATE` | `0.002` |
| `RATIO_TRAIN` | `0.9` |
| `TARGET_ERROR_RATE` | `0.001` |
| `START_MODEL` | (blank — trained from scratch) |

The network specification is a custom CNN + LSTM architecture tuned for line images around 1551×100 pixels:
[1,36,0,1 Ct3,3,32 Mp2,2 Ct3,3,48 Mp2,2 Lfys96 Lfx128 Lrx128 Lfx192 O1c###]

`LANG_TYPE=Indic` enables the pass-through recoder, which is needed for Tibetan's stacked consonants and complex graphemes.

## Workflow

1. **Prepare data** — convert your dataset into tesstrain format:
```bash
   python Scripts/preprocess.py
```

2. **Download Tesseract language data**:
```bash
   make tesseract-langdata
```

3. **Train**:
```bash
   make training
```

4. **Evaluate**:
```bash
   make evaluation
```

5. **Plot results**:
```bash
   make plot
```

## Training Data

Training data is sourced from six OpenPecha Uchen datasets, totaling approximately 3.7M images.

| Dataset | Share |
|---|---|
| NorbuKetaka | 48.60% |
| Derge | 18.95% |
| Google Books | 16.42% |
| Lithang | 11.90% |
| Lhasa | 3.53% |
| Karmapa | 0.60% |

Proportional subsets used for staged training:

| Subset | Total Images |
|---|---|
| Small | 37,000 |
| Medium | 370,000 |
| Full | 3,688,354 |

## Results

The model checkpoint in `model/tib.traineddata` is from the `tib_small` training run on the 37K subset.

**Best BCER: 5.03%**

## Technical Details

- Tesseract version: 5.4.1
- Engine: LSTM (Tesseract 4+ neural OCR)
- Tools: `make`, `lstmtraining`, `lstmeval`

## Upstream

Based on [tesseract-ocr/tesstrain](https://github.com/tesseract-ocr/tesstrain). See the upstream repository for general Tesseract training documentation.
