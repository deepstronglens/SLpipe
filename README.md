# A Modular Deep Learning Pipeline for Galaxy-Scale Strong Gravitational Lens Detection and Modeling

This repo consists of Deep Learning Pipeline for Training and Inference with 4 modules
1. Denoising (extracting clean galaxy-galaxy strong lensing images from noisy telescope data)
2. Deblending (separating the foreground lens from background source)
3. Detection (binary classification module for identifying lensed and unlensed images)
4. Modelling (regression of 3 astrophysical parameters relevent to strong lensing system)

### Data
The data used in this work is available at https://anl.box.com/s/9wz5wz8zww3jfuo6k5d7055yhsfrjyn6

1. Copy the file Array_SR_2.npz to /KDD_Submission/Denoising_Deblending/EDSR_MWCNN/experiment/Joint_model_inference/
2. Copy rest of the files to /KDD_Submission/Data/



