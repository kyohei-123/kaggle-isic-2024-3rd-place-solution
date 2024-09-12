# Kaggle ISIC2024 3rd place solution
3rd place solution  for [ISIC 2024 - Skin Cancer Detection Challenge](https://www.kaggle.com/competitions/isic-2024-challenge/overview)
## Environment
- WSL2 Ubuntu 20.04.6 LTS
- GPU RTX3090 x 1

## Setup python environment
```bash
python -m venv .venv
source ./.venv/bin/activate
pip install -r requirements.txt
```
## Data
Place the competiion data into input directory
```bash
cd ./input
kaggle competitions download -c isic-2024-challenge
unzip isic-2024-challenge.zip -d isic-2024-challenge
cd ../
```

## Prepare fold
```bash
python define_fold.py
```
## Train image models
### model1
convnextv2_nano.fcmae_ft_in22k_in1k (CV 0.1599389)
```bash
python train_20240827234748.py
```
Model weights are save to `./output_20240827234748`
### model2
vit_tiny_patch16_224.augreg_in21k_ft_in1k (CV 0.1612504)
```bash
python train_20240830205516.py
```
Model weights are save to `./output_20240830205516`
### model3
vit_tiny_patch16_224.augreg_in21k_ft_in1k (CV 0.1460650)
```bash
python train_20240831025049.py
```
Model weights are save to `./output_20240831025049`
### model4
vit_small_patch16_224.augreg_in21k_ft_in1k (CV 0.1486832)
```bash
python train_20240902001446.py
```
Model weights are save to `./output_20240902001446`

## Integrate image into tabular model and submission
`tabular-with-image-submission.ipynb`
