# Training - Lens Detection
### use the data  /KDD_Submission/Data/Array_HR.npz in dataset.py for training
python vae_quant.py --beta 0.05 --VIB -problem HEP_SL --conv --gpu 1 -n 300 -l 1e-3 -name_S VIB_label_beta_pt05_flow --save VIB_label_beta_pt05_flow --classification -x_dist bernoulli -dist flow

# Metrics/inference - Lens Detection
### use the data /KDD_Submission/Data/Array_HR.npz in dataset.py for metrics in training modality
### use the data "/KDD_Submission/Denoising_Deblending/EDSR_MWCNN/experiment/Joint_model_inference/Array_SR_2.npz" in dataset.py for metrics in Inference modality
python Inference_metrics.py --checkpt ./Results/VIB_label_beta_pt05_flow/checkpt-0000.pth

# Training - Lens Modeling
### use the data  /KDD_Submission/Data/Array_HR.npz in dataset.py for training
python vae_quant.py --beta 3 --VIB -problem HEP_SL --conv --gpu 1 -n 300  -l 1e-3 -name_S VIB_reg_const_sig_beta_3_flow  --save VIB_reg_const_sig_beta_3_flow  -dist flow

# Metrics/inference - Lens Modeling
### use the data /KDD_Submission/Data/Array_HR.npz in dataset.py for metrics in training modality
### use the data "/KDD_Submission/Denoising_Deblending/EDSR_MWCNN/experiment/Joint_model_inference/Array_SR_2.npz" in dataset.py for metrics in Inference modality
python Inference_metrics.py --checkpt ./Results/VIB_reg_const_sig_beta_3_flow/checkpt-0000.pth

