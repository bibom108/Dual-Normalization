# CUDA_VISIBLE_DEVICES=1 python -W ignore train_dn_unet.py \
#   --train_domain_list_1 t2_ss --train_domain_list_2 t2_sd --n_classes 2 \
#   --batch_size 128 --n_epochs 50 --save_step 10 --lr 0.004 --gpu_ids 1 \
#   --result_dir ./results/unet_dn_t2 --data_dir /data1/phuc/miccai-brats2018-original-dataset/MICCAI_BraTS_2018_Data_Training/npz_data

# CUDA_VISIBLE_DEVICES=0 python -W ignore test_dn_unet.py \
#   --data_dir /data1/phuc/miccai-brats2018-original-dataset/MICCAI_BraTS_2018_Data_Training/npz_data --n_classes 2 \
#   --test_domain_list flair --model_dir ./results/unet_dn_t2/model \
#   --save_label --label_dir ./results/unet_dn_t2 \
#   --batch_size 64 --gpu_ids 0 \

CUDA_VISIBLE_DEVICES=2 python -W ignore run_tent.py \
  --data_dir /data1/phuc/miccai-brats2018-original-dataset/MICCAI_BraTS_2018_Data_Training/npz_data --n_classes 2 \
  --test_domain_list flair --model_dir ./results/unet_dn_t2/model \
  --save_label --label_dir ./results/unet_dn_t2 \
  --batch_size 64 --gpu_ids 2 \

# lrs=(0.1 0.01 0.0001)
# optis=("Adam" "SGD")


# for lr in "${lrs[@]}"; do
#     for opti in "${optis[@]}"; do
#         CUDA_VISIBLE_DEVICES=1 python -W ignore run_tent.py \
#           --data_dir /data1/phuc/miccai-brats2018-original-dataset/MICCAI_BraTS_2018_Data_Training/npz_data \
#           --n_classes 2 \
#           --test_domain_list flair t1 t1ce \
#           --model_dir ./results/unet_dn_t2/model \
#           --batch_size 64 \
#           --gpu_ids 1 \
#           --opti $opti \
#           --lr $lr
#     done
# done

# steps=(1 2 5 10 20)
# episodics=(0)

# for step in "${steps[@]}"; do
#     for episodic in "${episodics[@]}"; do
#         if [[ $episodic -eq 1 ]]; then
#             CUDA_VISIBLE_DEVICES=0 python -W ignore run_tent.py \
#             --data_dir /data1/phuc/miccai-brats2018-original-dataset/MICCAI_BraTS_2018_Data_Training/npz_data \
#             --n_classes 2 \
#             --test_domain_list flair t1 t1ce \
#             --model_dir ./results/unet_dn_t2/model \
#             --batch_size 64 \
#             --step $step\
#             --gpu_ids 1 \
#             --opti Adam \
#             --lr 1e-4 \
#             --episodic $episodic
#         else
#             CUDA_VISIBLE_DEVICES=0 python -W ignore run_tent.py \
#             --data_dir /data1/phuc/miccai-brats2018-original-dataset/MICCAI_BraTS_2018_Data_Training/npz_data \
#             --n_classes 2 \
#             --test_domain_list t1 \
#             --model_dir ./results/unet_dn_t2/model \
#             --batch_size 64 \
#             --step $step\
#             --gpu_ids 1 \
#             --opti Adam \
#             --lr 1e-4
#         fi
#     done
# done


# CUDA_VISIBLE_DEVICES=1 python preprocess_func.py

# CUDA_VISIBLE_DEVICES=1 python -W ignore train_dn_unet.py \
#   --train_domain_list_1 t1ce_ss --train_domain_list_2 t1ce_sd --n_classes 2 \
#   --batch_size 128 --n_epochs 50 --save_step 10 --lr 0.004 --gpu_ids 1 \
#   --result_dir ./results/unet_dn_t1ce --data_dir /data1/phuc/miccai-brats2018-original-dataset/MICCAI_BraTS_2018_Data_Training/npz_data2

# CUDA_VISIBLE_DEVICES=1 python -W ignore test_dn_unet.py \
#   --data_dir /data1/phuc/miccai-brats2018-original-dataset/MICCAI_BraTS_2018_Data_Training/npz_data2 --n_classes 2 \
#   --test_domain_list t2 --model_dir ./results/unet_dn_t1ce/model \
#   --batch_size 64 --gpu_ids 1 \

# CUDA_VISIBLE_DEVICES=1 python -W ignore run_tent.py \
#   --data_dir /data1/phuc/miccai-brats2018-original-dataset/MICCAI_BraTS_2018_Data_Training/npz_data2 --n_classes 2 \
#   --test_domain_list flair t1 t2 --model_dir ./results/unet_dn_t1ce/model \
#   --batch_size 64 --gpu_ids 1 \

# for lr in "${lrs[@]}"; do
#     for opti in "${optis[@]}"; do
#         CUDA_VISIBLE_DEVICES=1 python -W ignore run_tent.py \
#           --data_dir /data1/phuc/miccai-brats2018-original-dataset/MICCAI_BraTS_2018_Data_Training/npz_data2 \
#           --n_classes 2 \
#           --test_domain_list flair t1 t2 \
#           --model_dir ./results/unet_dn_t1ce/model \
#           --batch_size 64 \
#           --gpu_ids 1 \
#           --opti $opti \
#           --lr $lr
#     done
# done


# for step in "${steps[@]}"; do
#     for episodic in "${episodics[@]}"; do
#         if [[ $episodic -eq 1 ]]; then
#           CUDA_VISIBLE_DEVICES=1 python -W ignore run_tent.py \
#             --data_dir /data1/phuc/miccai-brats2018-original-dataset/MICCAI_BraTS_2018_Data_Training/npz_data2 \
#             --n_classes 2 \
#             --test_domain_list flair t1 t2 \
#             --model_dir ./results/unet_dn_t1ce/model \
#             --batch_size 64 \
#             --steps $step \
#             --gpu_ids 1 \
#             --opti Adam \
#             --lr 1e-4 \
#             --episodic $episodic
#         else
#           CUDA_VISIBLE_DEVICES=1 python -W ignore run_tent.py \
#             --data_dir /data1/phuc/miccai-brats2018-original-dataset/MICCAI_BraTS_2018_Data_Training/npz_data2 \
#             --n_classes 2 \
#             --test_domain_list flair t1 t2 \
#             --model_dir ./results/unet_dn_t1ce/model \
#             --batch_size 64 \
#             --steps $step \
#             --gpu_ids 1 \
#             --opti Adam \
#             --lr 1e-4
#         fi
#     done
# done