# For the Moji dataset with fixed DeepMoji encodings, 
# we resure the preprocessed dataset from https://aclanthology.org/2020.acl-main.647/

mkdir -p data/deepmoji
wget https://storage.googleapis.com/ai2i/nullspace/deepmoji/pos_pos.npy -P data/deepmoji
wget https://storage.googleapis.com/ai2i/nullspace/deepmoji/pos_neg.npy -P data/deepmoji
wget https://storage.googleapis.com/ai2i/nullspace/deepmoji/neg_pos.npy -P data/deepmoji
wget https://storage.googleapis.com/ai2i/nullspace/deepmoji/neg_neg.npy -P data/deepmoji

python data/src/Moji/deepmoji_split.py --input_dir data/deepmoji --output_dir data/deepmoji