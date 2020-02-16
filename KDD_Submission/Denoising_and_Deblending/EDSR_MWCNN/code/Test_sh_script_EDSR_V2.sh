#!/bin/sh

RED='\033[0;31m'
NC='\033[0m'
GREEN='\033[0;32m'

Path_EDSR="/KDD_Submission/Data/"
Path_EDSR_MWCNN="/KDD_Submission/Denoising_Deblending/EDSR_MWCNN/"
############################ IMAGE ENHANCEMENT - Training Stage ############################

echo -e "${GREEN}***************** JOINT DENOISING and DEBLENDING TRAINING (From Scratch) *****************${NC}"
# Define the commandline arguments to use
model="EDSR"
save_folder_denoise_and_deblend_scratch=HEP_Denoise_and_Deblend_EDSR_2Loss_Scratch_20k_V4
n_resblocks=16
n_feats=256
dir_data=${Path_EDSR}/Data_denoise_new
epochs=50 
batch_size=128 

echo -e "${RED}***Folder to save the joint training from scratch results:${NC}"$save_folder_denoise_and_deblend_scratch

python main.py --model $model --scale 1 --save $save_folder_denoise_and_deblend_scratch --n_resblocks $((n_resblocks)) --n_feats $((n_feats)) --res_scale 0.1 \
--data_train Sim_data --data_test Sim_data --dir_data $dir_data --n_colors 3 --patch_size 32  --epochs $((epochs)) --batch_size $((batch_size)) \
--lr 0.0004 --ext bin  --nmodels 2 --denoise --save_results --print_every 40 --numloss 2 --Test_feed_model1_out --test_st 0 --test_end 2000 #--test_only

save_folder_denoise_and_deblend_scratch_2=HEP_Denoise_and_Deblend_EDSR_2Loss_Scratch_20k_V4_2
Folder_pretrain_model_denoise_scratch=${Path_EDSR_MWCNN}/experiment/${save_folder_denoise_and_deblend_scratch}/model/model_latest_model1.pt
Folder_pretrain_model_deblend_scratch=${Path_EDSR_MWCNN}/experiment/${save_folder_denoise_and_deblend_scratch}/model/model_latest_model2.pt
epochs=50

echo -e "${RED}***Folder to save the joint training from scratch resuming results:${NC}"$save_folder_denoise_and_deblend_scratch_2

python main.py --model $model --scale 1 --save $save_folder_denoise_and_deblend_scratch_2 --n_resblocks $((n_resblocks)) --n_feats $((n_feats)) --res_scale 0.1 \
--data_train Sim_data --data_test Sim_data --dir_data $dir_data --n_colors 3 --patch_size 32  --epochs $((epochs)) --batch_size $((batch_size)) \
--ext bin --resume 0 --lr 4.00e-4 --nmodels 2 --pre_train_model1 $Folder_pretrain_model_denoise_scratch \
--pre_train_model2 $Folder_pretrain_model_deblend_scratch --denoise --save_results --print_every 40 --numloss 2 --Test_feed_model1_out --test_st 0 --test_end 2000 #--test_only

save_folder_denoise_and_deblend_scratch_3=HEP_Denoise_and_Deblend_EDSR_2Loss_Scratch_20k_V4_3
Folder_pretrain_model_denoise_scratch=${Path_EDSR_MWCNN}/experiment/${save_folder_denoise_and_deblend_scratch_2}/model/model_latest_model1.pt
Folder_pretrain_model_deblend_scratch=${Path_EDSR_MWCNN}/experiment/${save_folder_denoise_and_deblend_scratch_2}/model/model_latest_model2.pt
epochs=50

echo -e "${RED}***Folder to save the joint training from scratch resuming results:${NC}"$save_folder_denoise_and_deblend_scratch_3

python main.py --model $model --scale 1 --save $save_folder_denoise_and_deblend_scratch_3 --n_resblocks $((n_resblocks)) --n_feats $((n_feats)) --res_scale 0.1 \
--data_train Sim_data --data_test Sim_data --dir_data $dir_data --n_colors 3 --patch_size 32  --epochs $((epochs)) --batch_size $((batch_size)) \
--ext bin --resume 0 --lr 2.00e-4 --nmodels 2 --pre_train_model1 $Folder_pretrain_model_denoise_scratch \
--pre_train_model2 $Folder_pretrain_model_deblend_scratch --denoise --save_results --print_every 40 --numloss 2 --Test_feed_model1_out --test_st 0 --test_end 2000 #--test_only

save_folder_denoise_and_deblend_scratch_4=HEP_Denoise_and_Deblend_EDSR_2Loss_Scratch_20k_V4_4
Folder_pretrain_model_denoise_scratch=${Path_EDSR_MWCNN}/experiment/${save_folder_denoise_and_deblend_scratch_3}/model/model_latest_model1.pt
Folder_pretrain_model_deblend_scratch=${Path_EDSR_MWCNN}/experiment/${save_folder_denoise_and_deblend_scratch_3}/model/model_latest_model2.pt
epochs=500

echo -e "${RED}***Folder to save the joint training from scratch resuming results:${NC}"$save_folder_denoise_and_deblend_scratch_4

python main.py --model $model --scale 1 --save $save_folder_denoise_and_deblend_scratch_4 --n_resblocks $((n_resblocks)) --n_feats $((n_feats)) --res_scale 0.1 \
--data_train Sim_data --data_test Sim_data --dir_data $dir_data --n_colors 3 --patch_size 32  --epochs $((epochs)) --batch_size $((batch_size)) \
--ext bin --resume 0 --lr 1.00e-4 --nmodels 2 --pre_train_model1 $Folder_pretrain_model_denoise_scratch \
--pre_train_model2 $Folder_pretrain_model_deblend_scratch --denoise --save_results --print_every 40 --numloss 2 --Test_feed_model1_out --test_st 0 --test_end 2000 #--test_only


Folder_pretrain_model_denoise_scratch_4=${Path_EDSR_MWCNN}/experiment/${save_folder_denoise_and_deblend_scratch_4}/model/model_latest_model1.pt
Folder_pretrain_model_deblend_scratch_4=${Path_EDSR_MWCNN}/experiment/${save_folder_denoise_and_deblend_scratch_4}/model/model_latest_model2.pt

save_folder_denoise_and_deblend_scratch_testonly=${save_folder_denoise_and_deblend_scratch_4}_Inference_2
echo -e "${RED}***Folder to save the joint training from scratch - test only:${NC}"$save_folder_denoise_and_deblend_scratch_testonly

start=0 
end=120000 
dir_data=${Path_EDSR}/Data_120k_test

python main.py --model $model --scale 1 --save $save_folder_denoise_and_deblend_scratch_testonly --n_resblocks $((n_resblocks)) --n_feats $((n_feats)) --res_scale 0.1 \
--data_train Sim_data --data_test Sim_data --dir_data $dir_data --n_colors 3 --patch_size 32  --epochs $((epochs)) --batch_size $((batch_size)) \
--ext bin --resume 0 --lr 1.00e-4 --nmodels 2 --pre_train_model1 $Folder_pretrain_model_denoise_scratch_4 \
--pre_train_model2 $Folder_pretrain_model_deblend_scratch_4 --denoise  --print_every 20 --numloss 2 \
--Test_feed_model1_out --test_only --test_st $((start)) --test_end $((end)) #--save_results


# echo -e "${GREEN}***************** CALCULATING ACCURACY METRICS *****************${NC}"

python ./metric_calc/Acc_metrics_new.py --path_results ${Path_EDSR_MWCNN}/experiment/${save_folder_denoise_and_deblend_scratch_testonly}/results/ --ind_start $start --ind_end $end