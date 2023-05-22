# NTIRE2023 SCGLANet: Stereo Cross Global Learnable Attention Network 
# RealSCGALGAN: Toward Real World Stereo Image Super-Resolution via Hybrid Degradation Model and Discriminator for Implied Stereo Image Information

### News
**2023.05.22** The Dataset, including the Flickr1024RS models and StereoWeb20, are available now. The pretrained model of RealSCGLAGAN is also available now \
**2023.04.07** The Baseline, including the pretrained models and train/test configs, are available now.

### Installation
This implementation based on [BasicSR](https://github.com/xinntao/BasicSR) which is a open source toolbox for image/video restoration tasks.
    
    cd  SCGLANet
    python setup.py develop
            
### Training Base hardware Requirements
- : 8 A40/RTX3090 GPUs

### Dataset
   
   - StereoWeb20 and Flickr1024RS are available at [百度网盘](https://pan.baidu.com/s/1n-8RrVdOSnxeRljrV3hUHQ?pwd=hp6m)
   
## NTIRE2023: SCGLANet
## 1. Quick Test 
#### 1.1 Download the pretrained model to the dir of 'experiments/pretrained_models'.
#####
   *pretrained model can be download at [百度网盘](https://pan.baidu.com/s/1ELaFEP2dzOR6q9suxjDLWg?pwd=rrhy),
       
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
    

    
#### Track2 
Modify the 'dataroot_gt' and 'dataroot_lq' about 'train dataset' and 'val dataset' 
    in 'options/train/StereoSR/train_SCGLANet_L_x4_Track2.yml' and load Track1 pre-trained model
    
    bash scripts/dist_train.sh 8 options/train/StereoSR/train_SCGLAGAN_L_x4_Track2.yml
    
    
#### Track3 
Modify the 'dataroot_gt' and 'dataroot_lq' about 'train dataset' and 'val dataset' 
    in 'options/train/StereoSR/train_SCGLANet_L_x4_Track3.yml' and load Track1 pre-trained model
    
    bash scripts/dist_train.sh 8 options/train/StereoSR/train_SCGLANet_Lx4_Track3.yml
    
## RealSCGLAGAN
## 1. Quick Test 
#### 1.1 Download the pretrained model to the dir of 'experiments/pretrained_models'.
#####
   *pretrained model can be download at [百度网盘](https://pan.baidu.com/s/1r8HW4wIBw0Y4UbCTgx-pLw?pwd=sc9q),
       
  
### BibTex
    @InProceedings{Zhou2023Stereo,
    author = {Zhou, Yuanbo and Xue, Yuyang and Deng, Wei and Nie, Ruofeng and Zhang, Jiajun and others},
    title = {Stereo Cross Global Learnable Attention Module for Stereo Image Super-Resolution},
    booktitle = {CVPRW},
    year = {2023},
    }

### Contact

If you have any questions, please contact webbozhou@gmail.com 
 

    
    
    
    
        
