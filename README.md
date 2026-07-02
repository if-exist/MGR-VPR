# MGR-VPR

Official repository for the paper "Multi-Grained Global-Regional Aggregation for Visual Place Recognition".

## Getting Started

We utilize the GSV-Cities dataset for training and you can download it [HERE](https://www.kaggle.com/datasets/amaralibey/gsv-cities), and refer to [VPR-datasets-downloader](https://github.com/gmberton/VPR-datasets-downloader) to prepare test datasets.

The test dataset should be organized in a directory tree as such:

```text
├── datasets_vg
    └── datasets
        └── pitts30k
            └── images
                ├── train
                │   ├── database
                │   └── queries
                ├── val
                │   ├── database
                │   └── queries
                └── test
                    ├── database
                    └── queries
```

Before training, download the DINOv2 pretrained foundation model and pass its path with `--foundation_model_path`.

## Model

MGR-VPR contains two main modules:

- `SA`: Saliency-Aware Adaptation
- `GRFA`: Global-Regional Feature Aggregation

## Train

```bash
python train.py \
  --train_dataset_folder /path/to/gsv_cities \
  --eval_datasets_folder /path/to/datasets_vg/datasets \
  --foundation_model_path /path/to/dinov2_vitl14_pretrain.pth \
  --save_dir mgr_vpr
```

## Test

```bash
python eval.py \
  --eval_datasets_folder /path/to/datasets_vg/datasets \
  --eval_dataset_name pitts30k \
  --foundation_model_path /path/to/dinov2_vitl14_pretrain.pth \
  --resume /path/to/model.pth \
  --save_dir mgr_vpr_eval
```

## Acknowledgements

Parts of this repo are inspired by the following repositories:

[EDTformer](https://github.com/Tong-Jin01/EDTformer)

[GSV-Cities](https://github.com/amaralibey/gsv-cities)

[Visual Geo-localization Benchmark](https://github.com/gmberton/deep-visual-geo-localization-benchmark)
