import re
import cltk
import os
import shutil
import json
from tqdm import tqdm
from cltk.sentence.lat import LatinPunktSentenceTokenizer
from cltk.lemmatize.lat import LatinBackoffLemmatizer
from cltk.tokenizers import LatinTokenizationProcess
from cltk.alphabet.lat import normalize_lat
from cltk.data.fetch import FetchCorpus

cltk_data_dir = cltk.utils.get_cltk_data_dir()
my_corpus_path = "custom_latin_corpus"

corpus_downloader = FetchCorpus(language="lat")
#corpus_downloader.list_corpora = ['example_distributed_latin_corpus', 'lat_text_perseus', 'lat_treebank_perseus', 'lat_text_latin_library', 'phi5', 'phi7', 'latin_proper_names_cltk', 'lat_models_cltk', 'latin_pos_lemmata_cltk', 'latin_treebank_index_thomisticus', 'latin_lexica_perseus', 'latin_training_set_sentence_cltk', 'latin_word2vec_cltk', 'latin_text_antique_digiliblt', 'latin_text_corpus_grammaticorum_latinorum', 'latin_text_poeti_ditalia', 'lat_text_tesserae']
corpus_downloader.import_corpus("lat_models_cltk")
corpus_downloader.import_corpus('lat_text_latin_library')
corpus_downloader.import_corpus('latin_training_set_sentence_cltk')
corpus_downloader.import_corpus("lat_text_perseus")
corpus_downloader.import_corpus("latin_text_corpus_grammaticorum_latinorum")
corpus_downloader.import_corpus("lat_text_tesserae")

latin_library_path = os.path.join(cltk_data_dir, "lat", "text", "lat_text_latin_library")
perseus_path = os.path.join(cltk_data_dir, "lat", "text",  "lat_text_perseus", "cltk_json")
tesserae_path = os.path.join(cltk_data_dir, "lat", "text",  "lat_text_tesserae", "texts")
corpus_grammaticorum_path = os.path.join(cltk_data_dir, "lat", "text", "latin_text_corpus_grammaticorum_latinorum") 

if not os.path.exists(my_corpus_path):
    os.mkdir(my_corpus_path)

new_perseus_path = os.path.join(
    my_corpus_path, "perseus_library"
)
if not os.path.exists(new_perseus_path):
    os.mkdir(new_perseus_path)


def get_texts_from_dict(my_dict):
    texts = list(my_dict.values())
    texts = [ re.sub(r"\s+", " ", tx) for tx in texts]

    return texts
def recursive_get_texts_from_dict(my_dict, depth):
    if depth == 1:
        return get_texts_from_dict(my_dict)
    else:
        all_texts = []
        for key, val in my_dict.items():
            nested_texts = recursive_get_texts_from_dict(
                val,
                depth=depth-1
            )
            all_texts.extend(nested_texts)
        return all_texts
def extract_text_from_perseus_dict(
        perseus_dict : dict,
)-> list[str]:
    perseus_text = perseus_dict["text"]

    # check the nest depth of dictionaries

    depth = 0
    aux_dict = perseus_text
    while True:
        try:
            aux_dict = aux_dict["0"]
            depth +=1
        except:
            break

    return recursive_get_texts_from_dict(perseus_text, depth)


for json_path in tqdm(os.listdir(perseus_path)):
    if ".json" in json_path and "english" not in json_path:
        # load data
        with open(os.path.join(perseus_path, json_path), "r") as f:
            perseus_dict = json.load(f)
        
        # extract text
        perseus_text = extract_text_from_perseus_dict(perseus_dict)

        
        
        # write to a txt file
        new_txt_path = json_path.removesuffix(".json")+".txt"
        new_txt_path = os.path.join(new_perseus_path, new_txt_path)

        with open(new_txt_path, "w") as f:
            f.writelines(line + "\n" for line in perseus_text)