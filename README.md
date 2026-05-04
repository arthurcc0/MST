# Medical Slice Transformer: Improved Diagnosis and Explainability on 3D Medical Images with DINOv2 
## Testing framework on UPenn HUP dataset
![MST](media/MST.jpg)
<br>
<b>Figure:  Overview of the Model Architecture and Attention Flow. </b> <br>(a) The Medical Slice Transformer framework processes individual MRI or CT slices using 2D image encoders, such as DINOv2, and then passes the encoded outputs through the Slice Transformer for downstream classification tasks. <br>(b) Visualization of attention mechanisms showing how the Slice Transformer assigns attention to specific slices and how within-slice attention is further refined to specific patches, resulting in a combined attention map highlighting regions of interest in the input volume. 

## Step 1: Run Training
## Option A: Use Trained Models
Skip training and download the weights from [Zenodo](https://doi.org/10.5281/zenodo.14500631).
### Option B: Train Models
Run Script: [scripts/main_train.py](scripts/main_train.py)
* Eg. `python scripts/main_train.py --dataset LIDC --model ResNet`
* Use `--model` to select:
    * ResNet = 3D ResNet50, 
    * ResNetSliceTrans = MST-ResNet, 
    * DinoV2ClassifierSlice = MST-DINOv2  

## Step 2: Predict & Evaluate Performance
Run Script: [scripts/main_predict.py](scripts/main_predict.py)
* Eg. `python scripts/main_predict.py --run_folder LIDC/ResNet`
* Use `--get_attention` to compute saliency maps
* Use `--get_segmentation` to compute segmentation masks and DICE score 
* Use `--use_tta` to enable Test Time Augmentation 