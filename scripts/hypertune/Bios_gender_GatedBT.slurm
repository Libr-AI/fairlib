#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=96:00:00
#SBATCH --mem=32G
#SBATCH --job-name=Bios_gender_GatedBT

#SBATCH --mail-user=xudongh1@student.unimelb.edu.au
#SBATCH --mail-type=begin
#SBATCH --mail-type=end

#SBATCH --error=/data/cephfs/punim1421/spartan_output/Bios_gender_GatedBT_%J.err
#SBATCH --output=/data/cephfs/punim1421/spartan_output/Bios_gender_GatedBT_%J.out

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

python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5647290 --exp_id GatedBT_Reweighting_stratified_g__0 --epochs_since_improvement 10 --BT Reweighting --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5859077 --exp_id GatedBT_Reweighting_stratified_g_--gated_0 --epochs_since_improvement 10 --BT Reweighting --BTObj stratified_g--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5205929 --exp_id GatedBT_Reweighting_stratified_y__0 --epochs_since_improvement 10 --BT Reweighting --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 723454 --exp_id GatedBT_Reweighting_stratified_y_--gated_0 --epochs_since_improvement 10 --BT Reweighting --BTObj stratified_y--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6301883 --exp_id GatedBT_Reweighting_y__0 --epochs_since_improvement 10 --BT Reweighting --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5119479 --exp_id GatedBT_Reweighting_y_--gated_0 --epochs_since_improvement 10 --BT Reweighting --BTObj y--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1808338 --exp_id GatedBT_Reweighting_g__0 --epochs_since_improvement 10 --BT Reweighting --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7668727 --exp_id GatedBT_Reweighting_g_--gated_0 --epochs_since_improvement 10 --BT Reweighting --BTObj g--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9713055 --exp_id GatedBT_Reweighting_joint__0 --epochs_since_improvement 10 --BT Reweighting --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 527285 --exp_id GatedBT_Reweighting_joint_--gated_0 --epochs_since_improvement 10 --BT Reweighting --BTObj joint--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 655209 --exp_id GatedBT_Resampling_stratified_g__0 --epochs_since_improvement 10 --BT Resampling --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3448184 --exp_id GatedBT_Resampling_stratified_g_--gated_0 --epochs_since_improvement 10 --BT Resampling --BTObj stratified_g--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8333898 --exp_id GatedBT_Resampling_stratified_y__0 --epochs_since_improvement 10 --BT Resampling --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9042620 --exp_id GatedBT_Resampling_stratified_y_--gated_0 --epochs_since_improvement 10 --BT Resampling --BTObj stratified_y--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2425505 --exp_id GatedBT_Resampling_y__0 --epochs_since_improvement 10 --BT Resampling --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1259268 --exp_id GatedBT_Resampling_y_--gated_0 --epochs_since_improvement 10 --BT Resampling --BTObj y--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2854895 --exp_id GatedBT_Resampling_g__0 --epochs_since_improvement 10 --BT Resampling --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4014940 --exp_id GatedBT_Resampling_g_--gated_0 --epochs_since_improvement 10 --BT Resampling --BTObj g--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4686410 --exp_id GatedBT_Resampling_joint__0 --epochs_since_improvement 10 --BT Resampling --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4767650 --exp_id GatedBT_Resampling_joint_--gated_0 --epochs_since_improvement 10 --BT Resampling --BTObj joint--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7096301 --exp_id GatedBT_Reweighting_stratified_g__1 --epochs_since_improvement 10 --BT Reweighting --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4142237 --exp_id GatedBT_Reweighting_stratified_g_--gated_1 --epochs_since_improvement 10 --BT Reweighting --BTObj stratified_g--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9080954 --exp_id GatedBT_Reweighting_stratified_y__1 --epochs_since_improvement 10 --BT Reweighting --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2388614 --exp_id GatedBT_Reweighting_stratified_y_--gated_1 --epochs_since_improvement 10 --BT Reweighting --BTObj stratified_y--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8708479 --exp_id GatedBT_Reweighting_y__1 --epochs_since_improvement 10 --BT Reweighting --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6011380 --exp_id GatedBT_Reweighting_y_--gated_1 --epochs_since_improvement 10 --BT Reweighting --BTObj y--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4503490 --exp_id GatedBT_Reweighting_g__1 --epochs_since_improvement 10 --BT Reweighting --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4709356 --exp_id GatedBT_Reweighting_g_--gated_1 --epochs_since_improvement 10 --BT Reweighting --BTObj g--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4145101 --exp_id GatedBT_Reweighting_joint__1 --epochs_since_improvement 10 --BT Reweighting --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 474682 --exp_id GatedBT_Reweighting_joint_--gated_1 --epochs_since_improvement 10 --BT Reweighting --BTObj joint--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7628086 --exp_id GatedBT_Resampling_stratified_g__1 --epochs_since_improvement 10 --BT Resampling --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4101256 --exp_id GatedBT_Resampling_stratified_g_--gated_1 --epochs_since_improvement 10 --BT Resampling --BTObj stratified_g--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2960449 --exp_id GatedBT_Resampling_stratified_y__1 --epochs_since_improvement 10 --BT Resampling --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6341489 --exp_id GatedBT_Resampling_stratified_y_--gated_1 --epochs_since_improvement 10 --BT Resampling --BTObj stratified_y--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1783928 --exp_id GatedBT_Resampling_y__1 --epochs_since_improvement 10 --BT Resampling --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9690416 --exp_id GatedBT_Resampling_y_--gated_1 --epochs_since_improvement 10 --BT Resampling --BTObj y--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4933329 --exp_id GatedBT_Resampling_g__1 --epochs_since_improvement 10 --BT Resampling --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4092160 --exp_id GatedBT_Resampling_g_--gated_1 --epochs_since_improvement 10 --BT Resampling --BTObj g--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4993247 --exp_id GatedBT_Resampling_joint__1 --epochs_since_improvement 10 --BT Resampling --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1319834 --exp_id GatedBT_Resampling_joint_--gated_1 --epochs_since_improvement 10 --BT Resampling --BTObj joint--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9379733 --exp_id GatedBT_Reweighting_stratified_g__2 --epochs_since_improvement 10 --BT Reweighting --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3567928 --exp_id GatedBT_Reweighting_stratified_g_--gated_2 --epochs_since_improvement 10 --BT Reweighting --BTObj stratified_g--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2386817 --exp_id GatedBT_Reweighting_stratified_y__2 --epochs_since_improvement 10 --BT Reweighting --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6767801 --exp_id GatedBT_Reweighting_stratified_y_--gated_2 --epochs_since_improvement 10 --BT Reweighting --BTObj stratified_y--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5056529 --exp_id GatedBT_Reweighting_y__2 --epochs_since_improvement 10 --BT Reweighting --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6980559 --exp_id GatedBT_Reweighting_y_--gated_2 --epochs_since_improvement 10 --BT Reweighting --BTObj y--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9031773 --exp_id GatedBT_Reweighting_g__2 --epochs_since_improvement 10 --BT Reweighting --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2970076 --exp_id GatedBT_Reweighting_g_--gated_2 --epochs_since_improvement 10 --BT Reweighting --BTObj g--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1906293 --exp_id GatedBT_Reweighting_joint__2 --epochs_since_improvement 10 --BT Reweighting --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7959111 --exp_id GatedBT_Reweighting_joint_--gated_2 --epochs_since_improvement 10 --BT Reweighting --BTObj joint--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1804217 --exp_id GatedBT_Resampling_stratified_g__2 --epochs_since_improvement 10 --BT Resampling --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9163771 --exp_id GatedBT_Resampling_stratified_g_--gated_2 --epochs_since_improvement 10 --BT Resampling --BTObj stratified_g--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2820813 --exp_id GatedBT_Resampling_stratified_y__2 --epochs_since_improvement 10 --BT Resampling --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 892181 --exp_id GatedBT_Resampling_stratified_y_--gated_2 --epochs_since_improvement 10 --BT Resampling --BTObj stratified_y--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7895833 --exp_id GatedBT_Resampling_y__2 --epochs_since_improvement 10 --BT Resampling --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9178902 --exp_id GatedBT_Resampling_y_--gated_2 --epochs_since_improvement 10 --BT Resampling --BTObj y--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6370936 --exp_id GatedBT_Resampling_g__2 --epochs_since_improvement 10 --BT Resampling --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7901830 --exp_id GatedBT_Resampling_g_--gated_2 --epochs_since_improvement 10 --BT Resampling --BTObj g--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4085114 --exp_id GatedBT_Resampling_joint__2 --epochs_since_improvement 10 --BT Resampling --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 794145 --exp_id GatedBT_Resampling_joint_--gated_2 --epochs_since_improvement 10 --BT Resampling --BTObj joint--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3075188 --exp_id GatedBT_Reweighting_stratified_g__3 --epochs_since_improvement 10 --BT Reweighting --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9266599 --exp_id GatedBT_Reweighting_stratified_g_--gated_3 --epochs_since_improvement 10 --BT Reweighting --BTObj stratified_g--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8356160 --exp_id GatedBT_Reweighting_stratified_y__3 --epochs_since_improvement 10 --BT Reweighting --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2259976 --exp_id GatedBT_Reweighting_stratified_y_--gated_3 --epochs_since_improvement 10 --BT Reweighting --BTObj stratified_y--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3150289 --exp_id GatedBT_Reweighting_y__3 --epochs_since_improvement 10 --BT Reweighting --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8714135 --exp_id GatedBT_Reweighting_y_--gated_3 --epochs_since_improvement 10 --BT Reweighting --BTObj y--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7522602 --exp_id GatedBT_Reweighting_g__3 --epochs_since_improvement 10 --BT Reweighting --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4403555 --exp_id GatedBT_Reweighting_g_--gated_3 --epochs_since_improvement 10 --BT Reweighting --BTObj g--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6787737 --exp_id GatedBT_Reweighting_joint__3 --epochs_since_improvement 10 --BT Reweighting --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4537621 --exp_id GatedBT_Reweighting_joint_--gated_3 --epochs_since_improvement 10 --BT Reweighting --BTObj joint--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9615985 --exp_id GatedBT_Resampling_stratified_g__3 --epochs_since_improvement 10 --BT Resampling --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2260798 --exp_id GatedBT_Resampling_stratified_g_--gated_3 --epochs_since_improvement 10 --BT Resampling --BTObj stratified_g--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3465181 --exp_id GatedBT_Resampling_stratified_y__3 --epochs_since_improvement 10 --BT Resampling --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9087181 --exp_id GatedBT_Resampling_stratified_y_--gated_3 --epochs_since_improvement 10 --BT Resampling --BTObj stratified_y--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8818847 --exp_id GatedBT_Resampling_y__3 --epochs_since_improvement 10 --BT Resampling --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3347233 --exp_id GatedBT_Resampling_y_--gated_3 --epochs_since_improvement 10 --BT Resampling --BTObj y--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4104821 --exp_id GatedBT_Resampling_g__3 --epochs_since_improvement 10 --BT Resampling --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2015775 --exp_id GatedBT_Resampling_g_--gated_3 --epochs_since_improvement 10 --BT Resampling --BTObj g--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 342086 --exp_id GatedBT_Resampling_joint__3 --epochs_since_improvement 10 --BT Resampling --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4527218 --exp_id GatedBT_Resampling_joint_--gated_3 --epochs_since_improvement 10 --BT Resampling --BTObj joint--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1475875 --exp_id GatedBT_Reweighting_stratified_g__4 --epochs_since_improvement 10 --BT Reweighting --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6186804 --exp_id GatedBT_Reweighting_stratified_g_--gated_4 --epochs_since_improvement 10 --BT Reweighting --BTObj stratified_g--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5521870 --exp_id GatedBT_Reweighting_stratified_y__4 --epochs_since_improvement 10 --BT Reweighting --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4875157 --exp_id GatedBT_Reweighting_stratified_y_--gated_4 --epochs_since_improvement 10 --BT Reweighting --BTObj stratified_y--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1236930 --exp_id GatedBT_Reweighting_y__4 --epochs_since_improvement 10 --BT Reweighting --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4842237 --exp_id GatedBT_Reweighting_y_--gated_4 --epochs_since_improvement 10 --BT Reweighting --BTObj y--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8338750 --exp_id GatedBT_Reweighting_g__4 --epochs_since_improvement 10 --BT Reweighting --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3037742 --exp_id GatedBT_Reweighting_g_--gated_4 --epochs_since_improvement 10 --BT Reweighting --BTObj g--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8185661 --exp_id GatedBT_Reweighting_joint__4 --epochs_since_improvement 10 --BT Reweighting --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3117687 --exp_id GatedBT_Reweighting_joint_--gated_4 --epochs_since_improvement 10 --BT Reweighting --BTObj joint--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3493074 --exp_id GatedBT_Resampling_stratified_g__4 --epochs_since_improvement 10 --BT Resampling --BTObj stratified_g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8666404 --exp_id GatedBT_Resampling_stratified_g_--gated_4 --epochs_since_improvement 10 --BT Resampling --BTObj stratified_g--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3958514 --exp_id GatedBT_Resampling_stratified_y__4 --epochs_since_improvement 10 --BT Resampling --BTObj stratified_y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 299930 --exp_id GatedBT_Resampling_stratified_y_--gated_4 --epochs_since_improvement 10 --BT Resampling --BTObj stratified_y--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2701827 --exp_id GatedBT_Resampling_y__4 --epochs_since_improvement 10 --BT Resampling --BTObj y
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2224681 --exp_id GatedBT_Resampling_y_--gated_4 --epochs_since_improvement 10 --BT Resampling --BTObj y--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5246384 --exp_id GatedBT_Resampling_g__4 --epochs_since_improvement 10 --BT Resampling --BTObj g
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8630732 --exp_id GatedBT_Resampling_g_--gated_4 --epochs_since_improvement 10 --BT Resampling --BTObj g--gated
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3467614 --exp_id GatedBT_Resampling_joint__4 --epochs_since_improvement 10 --BT Resampling --BTObj joint
sleep 2
python main.py --project_dir GatedBT --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2017138 --exp_id GatedBT_Resampling_joint_--gated_4 --epochs_since_improvement 10 --BT Resampling --BTObj joint--gated
sleep 2
