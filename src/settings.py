import json
import os
import pickle
import pandas as pd

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)


modelfilename = os.path.join(BASE_DIR, 'model', '2020-05-04_crf_model.sav')

crf_vocab_file = open(os.path.join(BASE_DIR, 'model', 'crf_module_vocab.json'))

crf = pickle.load(open(modelfilename, 'rb'))

crf_vocab = json.load(crf_vocab_file)

loinc = pd.read_csv(os.path.join(BASE_DIR, 'model', 'LoincTableCore.csv'))
loinc_id_dict = dict(zip(loinc.COMPONENT, loinc.LOINC_NUM))


