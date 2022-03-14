#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=96:00:00
#SBATCH --mem=32G
#SBATCH --job-name=Moji_FSCL

#SBATCH --mail-user=xudongh1@student.unimelb.edu.au
#SBATCH --mail-type=begin
#SBATCH --mail-type=end

#SBATCH --error=/data/cephfs/punim1421/spartan_output/Moji_FSCL_%J.err
#SBATCH --output=/data/cephfs/punim1421/spartan_output/Moji_FSCL_%J.out

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

python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6936841 --exp_id FSCL_0.00630957344480193_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.00630957344480193 --fcl_lambda_g 0.00630957344480193
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9754011 --exp_id FSCL_0.15848931924611143_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.15848931924611143 --fcl_lambda_g 0.15848931924611143
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4571007 --exp_id FSCL_0.19952623149688808_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.19952623149688808 --fcl_lambda_g 0.19952623149688808
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 925025 --exp_id FSCL_0.25118864315095824_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.25118864315095824 --fcl_lambda_g 0.25118864315095824
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2118737 --exp_id FSCL_0.31622776601683794_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.31622776601683794 --fcl_lambda_g 0.31622776601683794
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6803518 --exp_id FSCL_0.3981071705534973_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.3981071705534973 --fcl_lambda_g 0.3981071705534973
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1693194 --exp_id FSCL_0.5011872336272725_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.5011872336272725 --fcl_lambda_g 0.5011872336272725
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6421249 --exp_id FSCL_0.6309573444801936_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.6309573444801936 --fcl_lambda_g 0.6309573444801936
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5979976 --exp_id FSCL_0.7943282347242822_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.7943282347242822 --fcl_lambda_g 0.7943282347242822
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9974146 --exp_id FSCL_1.0_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 1.0 --fcl_lambda_g 1.0
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4702778 --exp_id FSCL_1.2589254117941675_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 1.2589254117941675 --fcl_lambda_g 1.2589254117941675
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 670465 --exp_id FSCL_2.5118864315095824_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 2.5118864315095824 --fcl_lambda_g 2.5118864315095824
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5532065 --exp_id FSCL_3.1622776601683795_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 3.1622776601683795 --fcl_lambda_g 3.1622776601683795
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1781833 --exp_id FSCL_5.011872336272725_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 5.011872336272725 --fcl_lambda_g 5.011872336272725
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9100506 --exp_id FSCL_6.309573444801936_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 6.309573444801936 --fcl_lambda_g 6.309573444801936
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4855865 --exp_id FSCL_7.943282347242821_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 7.943282347242821 --fcl_lambda_g 7.943282347242821
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4628481 --exp_id FSCL_0.003981071705534973_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.003981071705534973 --fcl_lambda_g 0.003981071705534973
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5893493 --exp_id FSCL_0.025118864315095808_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.025118864315095808 --fcl_lambda_g 0.025118864315095808
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2795114 --exp_id FSCL_10.0_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 10.0 --fcl_lambda_g 10.0
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1385168 --exp_id FSCL_0.0025118864315095794_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0025118864315095794 --fcl_lambda_g 0.0025118864315095794
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3981101 --exp_id FSCL_0.03162277660168379_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.03162277660168379 --fcl_lambda_g 0.03162277660168379
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4761113 --exp_id FSCL_0.039810717055349734_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.039810717055349734 --fcl_lambda_g 0.039810717055349734
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9111530 --exp_id FSCL_0.005011872336272725_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.005011872336272725 --fcl_lambda_g 0.005011872336272725
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1617765 --exp_id FSCL_0.01584893192461114_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.01584893192461114 --fcl_lambda_g 0.01584893192461114
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3659132 --exp_id FSCL_3.981071705534973_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 3.981071705534973 --fcl_lambda_g 3.981071705534973
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5061962 --exp_id FSCL_0.0031622776601683794_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0031622776601683794 --fcl_lambda_g 0.0031622776601683794
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5094677 --exp_id FSCL_0.012589254117941675_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.012589254117941675 --fcl_lambda_g 0.012589254117941675
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7520361 --exp_id FSCL_0.007943282347242814_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.007943282347242814 --fcl_lambda_g 0.007943282347242814
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1891321 --exp_id FSCL_0.06309573444801933_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.06309573444801933 --fcl_lambda_g 0.06309573444801933
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2382787 --exp_id FSCL_0.1_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.1 --fcl_lambda_g 0.1
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4229955 --exp_id FSCL_0.12589254117941676_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.12589254117941676 --fcl_lambda_g 0.12589254117941676
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4112458 --exp_id FSCL_0.0199526231496888_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0199526231496888 --fcl_lambda_g 0.0199526231496888
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6631244 --exp_id FSCL_0.001584893192461114_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.001584893192461114 --fcl_lambda_g 0.001584893192461114
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 507899 --exp_id FSCL_0.05011872336272725_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.05011872336272725 --fcl_lambda_g 0.05011872336272725
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7122762 --exp_id FSCL_1.584893192461114_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 1.584893192461114 --fcl_lambda_g 1.584893192461114
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4666356 --exp_id FSCL_1.9952623149688808_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 1.9952623149688808 --fcl_lambda_g 1.9952623149688808
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1425396 --exp_id FSCL_0.07943282347242818_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.07943282347242818 --fcl_lambda_g 0.07943282347242818
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9829879 --exp_id FSCL_0.0012589254117941675_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0012589254117941675 --fcl_lambda_g 0.0012589254117941675
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1700640 --exp_id FSCL_0.01_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.01 --fcl_lambda_g 0.01
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8435062 --exp_id FSCL_0.001995262314968879_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.001995262314968879 --fcl_lambda_g 0.001995262314968879
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9566749 --exp_id FSCL_0.001_0 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.001 --fcl_lambda_g 0.001
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 139613 --exp_id FSCL_0.00630957344480193_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.00630957344480193 --fcl_lambda_g 0.00630957344480193
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7384459 --exp_id FSCL_0.15848931924611143_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.15848931924611143 --fcl_lambda_g 0.15848931924611143
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7405319 --exp_id FSCL_0.19952623149688808_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.19952623149688808 --fcl_lambda_g 0.19952623149688808
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 151060 --exp_id FSCL_0.25118864315095824_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.25118864315095824 --fcl_lambda_g 0.25118864315095824
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2838941 --exp_id FSCL_0.31622776601683794_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.31622776601683794 --fcl_lambda_g 0.31622776601683794
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6800244 --exp_id FSCL_0.3981071705534973_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.3981071705534973 --fcl_lambda_g 0.3981071705534973
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6793426 --exp_id FSCL_0.5011872336272725_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.5011872336272725 --fcl_lambda_g 0.5011872336272725
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9298419 --exp_id FSCL_0.6309573444801936_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.6309573444801936 --fcl_lambda_g 0.6309573444801936
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2903096 --exp_id FSCL_0.7943282347242822_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.7943282347242822 --fcl_lambda_g 0.7943282347242822
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5706661 --exp_id FSCL_1.0_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 1.0 --fcl_lambda_g 1.0
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6612461 --exp_id FSCL_1.2589254117941675_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 1.2589254117941675 --fcl_lambda_g 1.2589254117941675
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5824331 --exp_id FSCL_2.5118864315095824_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 2.5118864315095824 --fcl_lambda_g 2.5118864315095824
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2103808 --exp_id FSCL_3.1622776601683795_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 3.1622776601683795 --fcl_lambda_g 3.1622776601683795
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2064666 --exp_id FSCL_5.011872336272725_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 5.011872336272725 --fcl_lambda_g 5.011872336272725
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3961537 --exp_id FSCL_6.309573444801936_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 6.309573444801936 --fcl_lambda_g 6.309573444801936
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9009583 --exp_id FSCL_7.943282347242821_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 7.943282347242821 --fcl_lambda_g 7.943282347242821
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6154239 --exp_id FSCL_0.003981071705534973_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.003981071705534973 --fcl_lambda_g 0.003981071705534973
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2908940 --exp_id FSCL_0.025118864315095808_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.025118864315095808 --fcl_lambda_g 0.025118864315095808
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2548582 --exp_id FSCL_10.0_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 10.0 --fcl_lambda_g 10.0
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7348432 --exp_id FSCL_0.0025118864315095794_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0025118864315095794 --fcl_lambda_g 0.0025118864315095794
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4395750 --exp_id FSCL_0.03162277660168379_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.03162277660168379 --fcl_lambda_g 0.03162277660168379
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7039594 --exp_id FSCL_0.039810717055349734_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.039810717055349734 --fcl_lambda_g 0.039810717055349734
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4412251 --exp_id FSCL_0.005011872336272725_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.005011872336272725 --fcl_lambda_g 0.005011872336272725
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5361232 --exp_id FSCL_0.01584893192461114_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.01584893192461114 --fcl_lambda_g 0.01584893192461114
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1888383 --exp_id FSCL_3.981071705534973_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 3.981071705534973 --fcl_lambda_g 3.981071705534973
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7017546 --exp_id FSCL_0.0031622776601683794_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0031622776601683794 --fcl_lambda_g 0.0031622776601683794
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5835952 --exp_id FSCL_0.012589254117941675_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.012589254117941675 --fcl_lambda_g 0.012589254117941675
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3382611 --exp_id FSCL_0.007943282347242814_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.007943282347242814 --fcl_lambda_g 0.007943282347242814
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4417535 --exp_id FSCL_0.06309573444801933_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.06309573444801933 --fcl_lambda_g 0.06309573444801933
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4992645 --exp_id FSCL_0.1_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.1 --fcl_lambda_g 0.1
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3143058 --exp_id FSCL_0.12589254117941676_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.12589254117941676 --fcl_lambda_g 0.12589254117941676
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5845816 --exp_id FSCL_0.0199526231496888_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0199526231496888 --fcl_lambda_g 0.0199526231496888
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6815451 --exp_id FSCL_0.001584893192461114_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.001584893192461114 --fcl_lambda_g 0.001584893192461114
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1347530 --exp_id FSCL_0.05011872336272725_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.05011872336272725 --fcl_lambda_g 0.05011872336272725
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3768666 --exp_id FSCL_1.584893192461114_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 1.584893192461114 --fcl_lambda_g 1.584893192461114
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8543649 --exp_id FSCL_1.9952623149688808_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 1.9952623149688808 --fcl_lambda_g 1.9952623149688808
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6409062 --exp_id FSCL_0.07943282347242818_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.07943282347242818 --fcl_lambda_g 0.07943282347242818
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3146093 --exp_id FSCL_0.0012589254117941675_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0012589254117941675 --fcl_lambda_g 0.0012589254117941675
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5902106 --exp_id FSCL_0.01_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.01 --fcl_lambda_g 0.01
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8094576 --exp_id FSCL_0.001995262314968879_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.001995262314968879 --fcl_lambda_g 0.001995262314968879
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8616324 --exp_id FSCL_0.001_1 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.001 --fcl_lambda_g 0.001
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9955259 --exp_id FSCL_0.00630957344480193_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.00630957344480193 --fcl_lambda_g 0.00630957344480193
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5614115 --exp_id FSCL_0.15848931924611143_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.15848931924611143 --fcl_lambda_g 0.15848931924611143
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6690350 --exp_id FSCL_0.19952623149688808_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.19952623149688808 --fcl_lambda_g 0.19952623149688808
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9927147 --exp_id FSCL_0.25118864315095824_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.25118864315095824 --fcl_lambda_g 0.25118864315095824
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1229775 --exp_id FSCL_0.31622776601683794_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.31622776601683794 --fcl_lambda_g 0.31622776601683794
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2930427 --exp_id FSCL_0.3981071705534973_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.3981071705534973 --fcl_lambda_g 0.3981071705534973
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 34670 --exp_id FSCL_0.5011872336272725_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.5011872336272725 --fcl_lambda_g 0.5011872336272725
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9482502 --exp_id FSCL_0.6309573444801936_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.6309573444801936 --fcl_lambda_g 0.6309573444801936
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 256735 --exp_id FSCL_0.7943282347242822_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.7943282347242822 --fcl_lambda_g 0.7943282347242822
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8126098 --exp_id FSCL_1.0_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 1.0 --fcl_lambda_g 1.0
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2827713 --exp_id FSCL_1.2589254117941675_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 1.2589254117941675 --fcl_lambda_g 1.2589254117941675
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1428525 --exp_id FSCL_2.5118864315095824_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 2.5118864315095824 --fcl_lambda_g 2.5118864315095824
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9411098 --exp_id FSCL_3.1622776601683795_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 3.1622776601683795 --fcl_lambda_g 3.1622776601683795
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1036689 --exp_id FSCL_5.011872336272725_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 5.011872336272725 --fcl_lambda_g 5.011872336272725
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9180254 --exp_id FSCL_6.309573444801936_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 6.309573444801936 --fcl_lambda_g 6.309573444801936
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3506063 --exp_id FSCL_7.943282347242821_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 7.943282347242821 --fcl_lambda_g 7.943282347242821
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4288985 --exp_id FSCL_0.003981071705534973_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.003981071705534973 --fcl_lambda_g 0.003981071705534973
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4036634 --exp_id FSCL_0.025118864315095808_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.025118864315095808 --fcl_lambda_g 0.025118864315095808
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9872346 --exp_id FSCL_10.0_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 10.0 --fcl_lambda_g 10.0
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9712929 --exp_id FSCL_0.0025118864315095794_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0025118864315095794 --fcl_lambda_g 0.0025118864315095794
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8692421 --exp_id FSCL_0.03162277660168379_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.03162277660168379 --fcl_lambda_g 0.03162277660168379
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9748977 --exp_id FSCL_0.039810717055349734_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.039810717055349734 --fcl_lambda_g 0.039810717055349734
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8491267 --exp_id FSCL_0.005011872336272725_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.005011872336272725 --fcl_lambda_g 0.005011872336272725
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2063930 --exp_id FSCL_0.01584893192461114_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.01584893192461114 --fcl_lambda_g 0.01584893192461114
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5323926 --exp_id FSCL_3.981071705534973_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 3.981071705534973 --fcl_lambda_g 3.981071705534973
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6647731 --exp_id FSCL_0.0031622776601683794_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0031622776601683794 --fcl_lambda_g 0.0031622776601683794
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3550691 --exp_id FSCL_0.012589254117941675_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.012589254117941675 --fcl_lambda_g 0.012589254117941675
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3119767 --exp_id FSCL_0.007943282347242814_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.007943282347242814 --fcl_lambda_g 0.007943282347242814
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2760396 --exp_id FSCL_0.06309573444801933_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.06309573444801933 --fcl_lambda_g 0.06309573444801933
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3144736 --exp_id FSCL_0.1_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.1 --fcl_lambda_g 0.1
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 70069 --exp_id FSCL_0.12589254117941676_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.12589254117941676 --fcl_lambda_g 0.12589254117941676
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8604538 --exp_id FSCL_0.0199526231496888_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0199526231496888 --fcl_lambda_g 0.0199526231496888
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5037685 --exp_id FSCL_0.001584893192461114_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.001584893192461114 --fcl_lambda_g 0.001584893192461114
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1179769 --exp_id FSCL_0.05011872336272725_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.05011872336272725 --fcl_lambda_g 0.05011872336272725
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7024842 --exp_id FSCL_1.584893192461114_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 1.584893192461114 --fcl_lambda_g 1.584893192461114
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8403630 --exp_id FSCL_1.9952623149688808_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 1.9952623149688808 --fcl_lambda_g 1.9952623149688808
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2379419 --exp_id FSCL_0.07943282347242818_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.07943282347242818 --fcl_lambda_g 0.07943282347242818
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2115743 --exp_id FSCL_0.0012589254117941675_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0012589254117941675 --fcl_lambda_g 0.0012589254117941675
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5550756 --exp_id FSCL_0.01_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.01 --fcl_lambda_g 0.01
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3729282 --exp_id FSCL_0.001995262314968879_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.001995262314968879 --fcl_lambda_g 0.001995262314968879
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3926811 --exp_id FSCL_0.001_2 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.001 --fcl_lambda_g 0.001
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5508004 --exp_id FSCL_0.00630957344480193_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.00630957344480193 --fcl_lambda_g 0.00630957344480193
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 50962 --exp_id FSCL_0.15848931924611143_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.15848931924611143 --fcl_lambda_g 0.15848931924611143
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2702354 --exp_id FSCL_0.19952623149688808_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.19952623149688808 --fcl_lambda_g 0.19952623149688808
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5407631 --exp_id FSCL_0.25118864315095824_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.25118864315095824 --fcl_lambda_g 0.25118864315095824
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3274526 --exp_id FSCL_0.31622776601683794_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.31622776601683794 --fcl_lambda_g 0.31622776601683794
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8268108 --exp_id FSCL_0.3981071705534973_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.3981071705534973 --fcl_lambda_g 0.3981071705534973
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2792792 --exp_id FSCL_0.5011872336272725_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.5011872336272725 --fcl_lambda_g 0.5011872336272725
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6220316 --exp_id FSCL_0.6309573444801936_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.6309573444801936 --fcl_lambda_g 0.6309573444801936
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7633717 --exp_id FSCL_0.7943282347242822_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.7943282347242822 --fcl_lambda_g 0.7943282347242822
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6594279 --exp_id FSCL_1.0_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 1.0 --fcl_lambda_g 1.0
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3060988 --exp_id FSCL_1.2589254117941675_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 1.2589254117941675 --fcl_lambda_g 1.2589254117941675
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4238393 --exp_id FSCL_2.5118864315095824_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 2.5118864315095824 --fcl_lambda_g 2.5118864315095824
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2995290 --exp_id FSCL_3.1622776601683795_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 3.1622776601683795 --fcl_lambda_g 3.1622776601683795
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9711685 --exp_id FSCL_5.011872336272725_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 5.011872336272725 --fcl_lambda_g 5.011872336272725
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8715983 --exp_id FSCL_6.309573444801936_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 6.309573444801936 --fcl_lambda_g 6.309573444801936
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 124670 --exp_id FSCL_7.943282347242821_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 7.943282347242821 --fcl_lambda_g 7.943282347242821
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7856865 --exp_id FSCL_0.003981071705534973_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.003981071705534973 --fcl_lambda_g 0.003981071705534973
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6368685 --exp_id FSCL_0.025118864315095808_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.025118864315095808 --fcl_lambda_g 0.025118864315095808
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2511410 --exp_id FSCL_10.0_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 10.0 --fcl_lambda_g 10.0
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8435999 --exp_id FSCL_0.0025118864315095794_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0025118864315095794 --fcl_lambda_g 0.0025118864315095794
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1331108 --exp_id FSCL_0.03162277660168379_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.03162277660168379 --fcl_lambda_g 0.03162277660168379
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5683031 --exp_id FSCL_0.039810717055349734_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.039810717055349734 --fcl_lambda_g 0.039810717055349734
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8749379 --exp_id FSCL_0.005011872336272725_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.005011872336272725 --fcl_lambda_g 0.005011872336272725
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6170215 --exp_id FSCL_0.01584893192461114_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.01584893192461114 --fcl_lambda_g 0.01584893192461114
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6415955 --exp_id FSCL_3.981071705534973_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 3.981071705534973 --fcl_lambda_g 3.981071705534973
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6622151 --exp_id FSCL_0.0031622776601683794_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0031622776601683794 --fcl_lambda_g 0.0031622776601683794
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1591776 --exp_id FSCL_0.012589254117941675_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.012589254117941675 --fcl_lambda_g 0.012589254117941675
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3718773 --exp_id FSCL_0.007943282347242814_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.007943282347242814 --fcl_lambda_g 0.007943282347242814
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2198726 --exp_id FSCL_0.06309573444801933_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.06309573444801933 --fcl_lambda_g 0.06309573444801933
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6440273 --exp_id FSCL_0.1_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.1 --fcl_lambda_g 0.1
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4593177 --exp_id FSCL_0.12589254117941676_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.12589254117941676 --fcl_lambda_g 0.12589254117941676
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5781541 --exp_id FSCL_0.0199526231496888_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0199526231496888 --fcl_lambda_g 0.0199526231496888
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5393685 --exp_id FSCL_0.001584893192461114_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.001584893192461114 --fcl_lambda_g 0.001584893192461114
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8206800 --exp_id FSCL_0.05011872336272725_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.05011872336272725 --fcl_lambda_g 0.05011872336272725
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 277805 --exp_id FSCL_1.584893192461114_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 1.584893192461114 --fcl_lambda_g 1.584893192461114
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1167661 --exp_id FSCL_1.9952623149688808_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 1.9952623149688808 --fcl_lambda_g 1.9952623149688808
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8115204 --exp_id FSCL_0.07943282347242818_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.07943282347242818 --fcl_lambda_g 0.07943282347242818
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7284007 --exp_id FSCL_0.0012589254117941675_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0012589254117941675 --fcl_lambda_g 0.0012589254117941675
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6676085 --exp_id FSCL_0.01_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.01 --fcl_lambda_g 0.01
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6701188 --exp_id FSCL_0.001995262314968879_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.001995262314968879 --fcl_lambda_g 0.001995262314968879
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2618879 --exp_id FSCL_0.001_3 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.001 --fcl_lambda_g 0.001
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2634900 --exp_id FSCL_0.00630957344480193_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.00630957344480193 --fcl_lambda_g 0.00630957344480193
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6653657 --exp_id FSCL_0.15848931924611143_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.15848931924611143 --fcl_lambda_g 0.15848931924611143
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5262048 --exp_id FSCL_0.19952623149688808_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.19952623149688808 --fcl_lambda_g 0.19952623149688808
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7100103 --exp_id FSCL_0.25118864315095824_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.25118864315095824 --fcl_lambda_g 0.25118864315095824
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1850735 --exp_id FSCL_0.31622776601683794_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.31622776601683794 --fcl_lambda_g 0.31622776601683794
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4022534 --exp_id FSCL_0.3981071705534973_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.3981071705534973 --fcl_lambda_g 0.3981071705534973
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6363277 --exp_id FSCL_0.5011872336272725_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.5011872336272725 --fcl_lambda_g 0.5011872336272725
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5910212 --exp_id FSCL_0.6309573444801936_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.6309573444801936 --fcl_lambda_g 0.6309573444801936
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8897156 --exp_id FSCL_0.7943282347242822_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.7943282347242822 --fcl_lambda_g 0.7943282347242822
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3902430 --exp_id FSCL_1.0_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 1.0 --fcl_lambda_g 1.0
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9608782 --exp_id FSCL_1.2589254117941675_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 1.2589254117941675 --fcl_lambda_g 1.2589254117941675
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 748886 --exp_id FSCL_2.5118864315095824_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 2.5118864315095824 --fcl_lambda_g 2.5118864315095824
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 839272 --exp_id FSCL_3.1622776601683795_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 3.1622776601683795 --fcl_lambda_g 3.1622776601683795
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 829674 --exp_id FSCL_5.011872336272725_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 5.011872336272725 --fcl_lambda_g 5.011872336272725
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6506793 --exp_id FSCL_6.309573444801936_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 6.309573444801936 --fcl_lambda_g 6.309573444801936
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1897581 --exp_id FSCL_7.943282347242821_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 7.943282347242821 --fcl_lambda_g 7.943282347242821
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7372877 --exp_id FSCL_0.003981071705534973_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.003981071705534973 --fcl_lambda_g 0.003981071705534973
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6397711 --exp_id FSCL_0.025118864315095808_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.025118864315095808 --fcl_lambda_g 0.025118864315095808
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2596000 --exp_id FSCL_10.0_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 10.0 --fcl_lambda_g 10.0
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3731895 --exp_id FSCL_0.0025118864315095794_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0025118864315095794 --fcl_lambda_g 0.0025118864315095794
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3725390 --exp_id FSCL_0.03162277660168379_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.03162277660168379 --fcl_lambda_g 0.03162277660168379
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9030829 --exp_id FSCL_0.039810717055349734_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.039810717055349734 --fcl_lambda_g 0.039810717055349734
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4295252 --exp_id FSCL_0.005011872336272725_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.005011872336272725 --fcl_lambda_g 0.005011872336272725
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 1569916 --exp_id FSCL_0.01584893192461114_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.01584893192461114 --fcl_lambda_g 0.01584893192461114
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8796388 --exp_id FSCL_3.981071705534973_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 3.981071705534973 --fcl_lambda_g 3.981071705534973
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8824049 --exp_id FSCL_0.0031622776601683794_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0031622776601683794 --fcl_lambda_g 0.0031622776601683794
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 147394 --exp_id FSCL_0.012589254117941675_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.012589254117941675 --fcl_lambda_g 0.012589254117941675
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 7021438 --exp_id FSCL_0.007943282347242814_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.007943282347242814 --fcl_lambda_g 0.007943282347242814
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5735321 --exp_id FSCL_0.06309573444801933_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.06309573444801933 --fcl_lambda_g 0.06309573444801933
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4751035 --exp_id FSCL_0.1_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.1 --fcl_lambda_g 0.1
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 570928 --exp_id FSCL_0.12589254117941676_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.12589254117941676 --fcl_lambda_g 0.12589254117941676
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2789166 --exp_id FSCL_0.0199526231496888_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0199526231496888 --fcl_lambda_g 0.0199526231496888
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9751621 --exp_id FSCL_0.001584893192461114_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.001584893192461114 --fcl_lambda_g 0.001584893192461114
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 3565644 --exp_id FSCL_0.05011872336272725_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.05011872336272725 --fcl_lambda_g 0.05011872336272725
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 6766949 --exp_id FSCL_1.584893192461114_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 1.584893192461114 --fcl_lambda_g 1.584893192461114
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4653041 --exp_id FSCL_1.9952623149688808_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 1.9952623149688808 --fcl_lambda_g 1.9952623149688808
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 8282059 --exp_id FSCL_0.07943282347242818_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.07943282347242818 --fcl_lambda_g 0.07943282347242818
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 5679857 --exp_id FSCL_0.0012589254117941675_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.0012589254117941675 --fcl_lambda_g 0.0012589254117941675
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 4702983 --exp_id FSCL_0.01_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.01 --fcl_lambda_g 0.01
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 2644816 --exp_id FSCL_0.001995262314968879_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.001995262314968879 --fcl_lambda_g 0.001995262314968879
sleep 2
python main.py --project_dir FairSCL --dataset Moji --emb_size 2304 --num_classes 2 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9733065 --exp_id FSCL_0.001_4 --epochs_since_improvement 10 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification --FCL --fcl_lambda_y 0.001 --fcl_lambda_g 0.001
sleep 2
