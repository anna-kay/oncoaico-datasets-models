## 1. Overview
This repository contains datasets, code, and experimental setups for machine learning tasks focused on segmentation and detection in colonoscopy data.

## 2. Data
**Pre-existing datasets used in this repo:**

| Dataset | Paper | Github | Download link | License |
| :------------: | :------------: | :------------: | :------------: | :------------: |
| CVC-300 | [paper](https://pubmed.ncbi.nlm.nih.gov/29065595/) | - | [Kaggle link](https://www.kaggle.com/datasets/nourabentaher/cvc-300)| Public for research; no clear license |
| CVC-ColonDB | [paper](https://pubmed.ncbi.nlm.nih.gov/26462083/) | - | [Kaggle link](https://www.kaggle.com/datasets/longvil/cvc-colondb)| Public for research; no clear license |
| CVC-ClinicDB |[paper](https://www.sciencedirect.com/science/article/abs/pii/S0895611115000567) | [github](https://github.com/openmedlab/Awesome-Medical-Dataset/blob/main/resources/CVC-ClinicDB.md) | [link](https://polyp.grand-challenge.org/CVCClinicDB) | Public for research; no clear license |
| ETIS-LaribPolypDB |[paper](https://pubmed.ncbi.nlm.nih.gov/24037504/)| - | - | Public for research; no clear license |
| Kvasir SEG (HyperKvasir)| [paper](https://www.nature.com/articles/s41597-020-00622-y) | [github](https://github.com/DebeshJha/Kvasir-SEG)| [link](https://datasets.simula.no/hyper-kvasir/) | Public for research; no clear license |
| PolypGen | [paper](https://www.nature.com/articles/s41597-023-01981-y) | [github](https://github.com/DebeshJha/PolypGen) | [link](https://www.synapse.org/Synapse:syn26376615/wiki/) | CC BY 4.0 |
| LDPolypVideo | - | x | x | CC BY 4.0 |
| REAL-COLON | [paper](https://www.nature.com/articles/s41597-024-03359-0) | - | [link](https://plus.figshare.com/articles/media/REAL-colon_dataset/22202866) | CC BY 4.0 |

**How the datasets are used & Data processing:**

The development of ONCO-AICO faced several data-related challenges. Although multiple colonoscopy datasets are available, many lack segmentation annotations, while those that include them are often small, limiting deep learning performance. Most datasets also do not provide predefined train–validation–test splits, increasing the risk of data leakage and reducing experimental consistency. Additionally, data is typically sourced from a limited number of clinical environments, constraining model generalization.

Given that ONCO-AICO is designed for training on colonoscopy videos, it must handle realistic conditions where most frames contain no findings (negative samples). However, existing datasets primarily consist of positive samples, which can lead to false positives during inference.

Attempts to incorporate additional datasets (e.g., CVC-300, CVC-ColonDB, ETIS-LaribPolypDB) were hindered by the absence of official data splits, making their use less reliable. Moreover, no available dataset explicitly included both positive and negative samples.

To address these limitations, custom datasets were constructed:

- **Segmentation Dataset**: Combines multiple sources to improve generalization.
- **Mixed Dataset**: Extends the segmentation dataset with negative (healthy) samples, enabling more realistic training and reducing false positives.

<br>

| Dataset | Train | Validation | Test | Total |
| :------------: | :------------: | :------------: | :------------: | :------------: |
| Kvasir-SEG| 900 images | 100 | - | |
| CVC-ClinicDB | 600 | 62 | - | |
| CVC-300 | - | - | 60 | |
| CVC-ColonDB | - | - | 380 | |
| ETIS-LaribPolypDB | - | -| 196 | |
| PolypGen | 1122 | 289 | 342 |
| Segmentation Dataset | 2622 | 451 | 978 | |
<br>

<br>

| | Dataset | Train | Validation | Test | Total |
| :------------: | :------------: | :------------: | :------------: | :------------: | :------------: | 
| Positive | Segmentation Dataset| 2622 | 451 | 978 | 4051 |
| Negative | REAL-Colon | 1423 | 203 | 408 | 2034 | 
| Negative |  LDPolypVideo | 1429 | 204 | 409 | 2042 | 
| | Mixed Dataset | 5474 | 858 | 1795 | 8127 |
<br>

**Comments/Limitations:**

Lorem ipsum

## 3. Models

This repository uses the following models:

1. **CaraNet**, [paper](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12032/120320D/CaraNet--context-axial-reverse-attention-network-for-segmentation-of/10.1117/12.2611802.short), [github](https://github.com/AngeLouCN/CaraNet)
2. **Polyp-PVT**, [paper](https://arxiv.org/abs/2108.06932), [github](https://github.com/dengpingfan/polyp-pvt)
3. **MSRF-Net**, [paper](https://ieeexplore.ieee.org/abstract/document/9662196/), [github](https://github.com/NoviceMAn-prog/MSRF-Net)

We are grateful for the foundation these works have provided.

## 4. Project Structure
```
|- data/
|- - SEGMENTATION_DATASET/
|- - train/
|- - - images/
|- - - masks/
|- - val/
|- - - images/
|- - - masks/
|- - test/
|- - - CVC-300/
|- - - - images/
|- - - - masks/
|- - - CVC-ColonDB/
|- - - - images/
|- - - - masks/
|- - - ETIS-LaribPolypDB/
|- - - - images/
|- - - - masks/
|- - - PolypGen/
|- - - - images/
|- - - - masks/
|- - - all_testsets/
|- - - - images/
|- - - - masks/
|- - MIXED_DATASET/
|- - train/
|- - - images/
|- - - masks/
|- - val/
|- - - images/
|- - - masks/
|- - test/
|- - - positives/
|- - - - images/
|- - - - masks/
|- - - negatives/
|- - - - images/
|- - - - masks/
|- - - mixed_testsets/
|- - - - images/
|- - - - masks/
|- models/
|- - CaraNet/
|- - - TODO
|- - PolypPVT/
|- - - TODO
|- - MSRFNet/
|- - - TODO
|- training/
|- - train.py
|- - utils/
|- - - dataloader.py
|- - - format_conversion.py
|- - - utils.py
|- - lib/ 
|- evaluation/
|- - evaluate_predictions.py
|- - run_eval.py
|- inference/
|- - generate_predictions.py
|- outputs/
|- - predictions/
|- - - CaraNet/
|- - - PolypPVT/
|- - - MSRFNet/
```

## 5. How to run
**1. To train using train & val splits of the MIXED_DATASET:**
```
python training/train.py \
    --dataset TODO \
    --model TODO \
    --
```

**2. To run evaluation of the test sets for the MIXED_DATASET:**
```
python run_eval.py \
    --pred_base ./inference_test_datasets \ TODO: Review
    --gt_base ./MIXED_DATASET/test \
    --datasets negatives positives mixed_testsets
    --model polyp_pvt
```
Logic:

**inference/generate_predictions.py:** 1)loads model & data, 2)generates predictions (Can be used just for inference)

**evaluation/evaluate_predictions.py:** 1)loads predictions & labels, 2)calculates metrics

**run_eval.py:** 1)loads test set, 2)calls generate_predictions, 3) calls evaluate_predictions, 3) prints metrics

**3. To run inference on real-world predictions:**
```
python generate_predictions.py \
    --new_data ... \ TODO: Review
    --model polyp_pvt
```

## 6. Acknowledgement

This work is related to ONCO-AICO is an AI-assisted training platform for junior endoscopists to improve polyp detection skills. The platform utilizes annotated colonoscopy videos, along with explainable AI (xAI) feedback and performance scoring.

Note: This repo focuses solely on the underlying datasets and machine learning experiments.

This work was conducted as part of the EU-funded [ONCOSCREEN](https://oncoscreen.health/) project (Horizon Europe grant agreement No. 101097036).
