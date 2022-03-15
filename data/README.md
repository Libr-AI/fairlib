# Moji
1. Download the data from Su Lin Blodgett [dataset](https://sites.google.com/site/sulinblodgett/), described in "Demographic dialectal variation in social media: A case study of african-american english." 

    ```bash
    wget http://slanglab.cs.umass.edu/TwitterAAE/TwitterAAE-full-v1.zip

    # or directly from the site: http://slanglab.cs.umass.edu/TwitterAAE/
    ```

2. Follow [Elazar et al.](https://github.com/yanaiela/demog-text-removal) in preprocessing the dataset to get race and sentiment labels.

    ```sh
    python make_data.py /path/to/downloaded/twitteraae_all /path/to/project/data/processed/sentiment_race sentiment race
    ```

    See this [doc](https://github.com/yanaiela/demog-text-removal/edit/master/src/data/README.md) for details about the scripts and other details.

    Noticing that the prerequisites `python==2.7`, and the original scripts directly maps tokens to ids inplace, i.e., original tokens will not be stored. In order to save texts, please hack the following function 
    
    https://github.com/yanaiela/demog-text-removal/blob/f11b243c3f2f24f2179348c468b2caf76e7a3b23/src/data/make_data.py#L59

    ```python
    def to_file(output_dir, voc2id, vocab, pos_pos, pos_neg, neg_pos, neg_neg):
        if output_dir[-1] != '/':
            output_dir += '/'

        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        with open(output_dir + 'vocab', 'w') as f:
            f.writelines('\n'.join(vocab))

        for data, name in zip([pos_pos, pos_neg, neg_pos, neg_neg], ['pos_pos', 'pos_neg', 'neg_pos', 'neg_neg']):
            with open(output_dir + name, 'w') as f:
                for sen in data:
                    ids = map(lambda x: str(voc2id[x]), sen)
                    f.write(' '.join(ids) + '\n')

            with open(output_dir + name + "_text", 'w') as f:
                for sen in data:
                    ids = map(lambda x: str(x), sen)
                    f.write(' '.join(ids) + '\n')
    ```

3. Encode texts with [torchMoji](https://github.com/huggingface/torchMoji)


# Bios