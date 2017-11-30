# kaggle-carvana-image-masking-challenge
48th solution for Carvana image masking challenge on Kaggle (https://www.kaggle.com/c/carvana-image-masking-challenge).

Final submission is the averaging of the U-Net based models.

## Train the models and predict masks
To train the U-Net based models and predict masks, run:

```
python model_unet12_0.5x.py
python model_unet12_1x.py
python model_unet14_0.5x.py
```

## Predictions averaging
To average the predicted masks, run:

```
python averaging.py
```
