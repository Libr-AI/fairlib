from fairlib.datasets.utils.download import download
from fairlib.src.utils import seed_everything
import numpy as np
import pandas as pd
import os
from pathlib import Path
from sklearn.model_selection import train_test_split


data_source = "https://bitbucket.org/lowlands/release/raw/HEAD/ACL2015/tagging_age/data/en/{}"

filenames = [
    "en.O45-UKC1_WORST-F.data.TT.tagged.gold",
    "en.O45-UKC1_WORST-M.data.TT.tagged.gold",
    "en.O45-UKH2_SOSO-F.data.TT.tagged.gold",
    "en.O45-UKH2_SOSO-M.data.TT.tagged.gold",
    "en.O45-UKN0_BEST-F.data.TT.tagged.gold",
    "en.O45-UKN0_BEST-M.data.TT.tagged.gold",
    "en.U35-UKC1_WORST-F.data.TT.tagged.gold",
    "en.U35-UKC1_WORST-M.data.TT.tagged.gold",
    "en.U35-UKH2_SOSO-F.data.TT.tagged.gold",
    "en.U35-UKH2_SOSO-M.data.TT.tagged.gold",
    "en.U35-UKN0_BEST-F.data.TT.tagged.gold",
    "en.U35-UKN0_BEST-M.data.TT.tagged.gold"
    ]


def load_web_eng(filename = ""):
    lines = list( open(filename, "r", encoding='utf8').readlines() )
    lines = [ l.strip() for l in lines]

    doc = []
    tags = []
    sent_w = []
    sent_t = []
    for l in lines:
        if l == '':
            doc.append(sent_w)
            tags.append(sent_t)
            sent_w = []
            sent_t = []
        else:
            w, t = l.split('\t')
            if t != "-NONE-":
                sent_w.append( w.lower() )
                sent_t.append( t )
    return doc, tags

def load_trustpilots(dataset_path):
    all_sents = []
    all_tags = []
    all_genders = []
    all_ages = []
    for i, filename in enumerate(filenames):
        sents, tags = load_web_eng(Path(dataset_path) / filename)
        if i < 6: 
            ages = np.array( [1] * len(sents) ) #over 45
        else:
            ages = np.array( [0] * len(sents) ) #under 35
        if i % 2 == 0:
            genders = np.array( [1] * len(sents) ) # F
        else:
            genders = np.array( [0] * len(sents) ) # M

        all_sents.extend(sents)
        all_tags.extend(tags)
        all_genders.extend(genders)
        all_ages.extend(ages)
    return all_sents, all_tags, np.array(all_genders), np.array(all_ages)

class POS:

    _NAME = "TP_POS"
    _SPLITS = ["train", "dev", "test"]

    def __init__(self, dest_folder, batch_size):
        self.dest_folder = dest_folder
        self.batch_size = batch_size

    def download_files(self):

        for filename in filenames:
            download(
                    url= data_source.format(filename), 
                    dest_folder = self.dest_folder
                    )

    def processing(self):
        TP_sents, TP_tags, TP_gender, TP_age = load_trustpilots(self.dest_folder)

        total_array = []
        for i in range(len(TP_sents)):
            total_array.append({'text':TP_sents[i], 
                                'tag_label':TP_tags[i], 
                                'age_label':int(TP_age[i]),
                                'gender_label':int(TP_gender[i])
                            })

        from sklearn.model_selection import train_test_split

        train_array, test_array = train_test_split(total_array, test_size=0.1, random_state=25042020)
        train_array, valid_array = train_test_split(train_array, test_size=0.1, random_state=25042020)

        import jsonlines
        def write2jsonl(jl_object, filePath):
            with jsonlines.open(filePath, mode='w') as writer:
                writer.write_all(jl_object)

        write2jsonl(train_array, Path(self.dest_folder) / "train_TP_POS.jsonl")
        write2jsonl(valid_array, Path(self.dest_folder) / "valid_TP_POS.jsonl")
        write2jsonl(test_array, Path(self.dest_folder) / "test_TP_POS.jsonl")

    def prepare_data(self):
        self.download_files()
        self.processing()