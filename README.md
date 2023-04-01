# NTIRE2023 SCGLANet
### Code will be release soon
### Base hardware Requirements
- : 8 A40 GPUs
## 1. Quick Test 
#### 1.1 Download the pretrained model to the dir of 'experiments/pretrained_models'
       
#### 1.2 Modify the dataroot_lq: in  'options/test/StereoSR'
        test_SCGLANet-L_4x_Track1.yml
        test_SCGLAGAN-L_4x_Track2.yml
        test_SCGLANet-L_4x_Track3.yml
#### 1.3 Run the test scripts 
        sh test.sh
#### 1.4 The final results are in 'results'
    
## 2.Quick Train

### 2.1.Data Process
         cd scripts
#### Modify the input and output path and run the following script
         python crop_stereo_img_to_subimgs.py
## 2.Modify The Config file


#### Track1 
##### The Stage 1
Modify the 'dataroot_gt' and 'dataroot_lq' about 'train dataset' and 'val dataset' 
    in 'options/train/StereoSR/train_SCGLANet_L_x4_Track1.yml'
    
    bash scripts/dist_train.sh 8 options/train/StereoSR/train_SCGLANet_L_x4_Track1.yml
##### The Stage 2
After the first stage of training, Modify the 'pixel_opt:
    type: L1Loss' to 'MSELoss' to finetune the model, and load the first stage pre-trained model, the lr is '1e-5'

    bash scripts/dist_train.sh 8 options/train/StereoSR/train_SCGLANet_L_x4_Track1.yml
    
*** After training, choose the top 4 model to use "model ensamble" by average method to improve PSNR
    
#### Track2 
Modify the 'dataroot_gt' and 'dataroot_lq' about 'train dataset' and 'val dataset' 
    in 'options/train/StereoSR/train_SCGLANet_L_x4_Track1.yml' and load Track1 pre-trained model
    
    bash scripts/dist_train.sh 8 options/train/StereoSR/train_SCGLAGAN_L_x4_Track2.yml
    
    
#### Track3 
Modify the 'dataroot_gt' and 'dataroot_lq' about 'train dataset' and 'val dataset' 
    in 'options/train/StereoSR/train_SCGLANet_L_x4_Track1.yml' and load Track1 pre-trained model
    
    bash scripts/dist_train.sh 8 options/train/StereoSR/train_SCGLANet_Lx4_Track3.yml
    
*** After training, choose the top 4 model to use "model ensemble" by average method to improve PSNR
    
    

    
    
    
    
        