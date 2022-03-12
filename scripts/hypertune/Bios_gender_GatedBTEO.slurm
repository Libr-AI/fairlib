#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=96:00:00
#SBATCH --mem=32G
#SBATCH --job-name=Bios_gender_GatedBTEO

#SBATCH --mail-user=xudongh1@student.unimelb.edu.au
#SBATCH --mail-type=begin
#SBATCH --mail-type=end

#SBATCH --error=/data/cephfs/punim1421/spartan_output/Bios_gender_GatedBTEO_%J.err
#SBATCH --output=/data/cephfs/punim1421/spartan_output/Bios_gender_GatedBTEO_%J.out

#SBATCH -q gpgpumse
#SBATCH -p gpgpu
#SBATCH --gres=gpu:1

# load required modules

module load gcccore/10.2.0
module load python/3.8.6
module load fosscuda/2020b
module load pytorch/1.7.1-python-3.8.6
# module load pytorch/1.4.0-python-3.7.4


cd /data/cephfs/punim1421/Fair_NLP_Classification

python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8324153 --exp_id GatedBTEO_Resampling_EO_False_0 --epochs_since_improvement 10  --BT Resampling --BTObj EO
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8909957 --exp_id GatedBTEO_Resampling_EO_True_0 --epochs_since_improvement 10  --gated --BT Resampling --BTObj EO
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7685645 --exp_id GatedBTEO_Reweighting_EO_False_0 --epochs_since_improvement 10  --BT Reweighting --BTObj EO
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 340444 --exp_id GatedBTEO_Reweighting_EO_True_0 --epochs_since_improvement 10  --gated --BT Reweighting --BTObj EO
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2369757 --exp_id GatedBTEO_Resampling_EO_False_1 --epochs_since_improvement 10  --BT Resampling --BTObj EO
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4681947 --exp_id GatedBTEO_Resampling_EO_True_1 --epochs_since_improvement 10  --gated --BT Resampling --BTObj EO
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1077810 --exp_id GatedBTEO_Reweighting_EO_False_1 --epochs_since_improvement 10  --BT Reweighting --BTObj EO
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 377828 --exp_id GatedBTEO_Reweighting_EO_True_1 --epochs_since_improvement 10  --gated --BT Reweighting --BTObj EO
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4968 --exp_id GatedBTEO_Resampling_EO_False_2 --epochs_since_improvement 10  --BT Resampling --BTObj EO
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1924335 --exp_id GatedBTEO_Resampling_EO_True_2 --epochs_since_improvement 10  --gated --BT Resampling --BTObj EO
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 588828 --exp_id GatedBTEO_Reweighting_EO_False_2 --epochs_since_improvement 10  --BT Reweighting --BTObj EO
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7098661 --exp_id GatedBTEO_Reweighting_EO_True_2 --epochs_since_improvement 10  --gated --BT Reweighting --BTObj EO
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 276721 --exp_id GatedBTEO_Resampling_EO_False_3 --epochs_since_improvement 10  --BT Resampling --BTObj EO
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 786758 --exp_id GatedBTEO_Resampling_EO_True_3 --epochs_since_improvement 10  --gated --BT Resampling --BTObj EO
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 932532 --exp_id GatedBTEO_Reweighting_EO_False_3 --epochs_since_improvement 10  --BT Reweighting --BTObj EO
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5888273 --exp_id GatedBTEO_Reweighting_EO_True_3 --epochs_since_improvement 10  --gated --BT Reweighting --BTObj EO
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1748135 --exp_id GatedBTEO_Resampling_EO_False_4 --epochs_since_improvement 10  --BT Resampling --BTObj EO
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4523748 --exp_id GatedBTEO_Resampling_EO_True_4 --epochs_since_improvement 10  --gated --BT Resampling --BTObj EO
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2814687 --exp_id GatedBTEO_Reweighting_EO_False_4 --epochs_since_improvement 10  --BT Reweighting --BTObj EO
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2155863 --exp_id GatedBTEO_Reweighting_EO_True_4 --epochs_since_improvement 10  --gated --BT Reweighting --BTObj EO
sleep 2
