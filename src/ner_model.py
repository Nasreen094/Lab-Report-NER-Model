import re
import settings
from werkzeug import secure_filename
from PIL import Image
from pytesseract import image_to_string 

import nltk
from itertools import groupby
from operator import itemgetter

crf = settings.crf
crf_vocab = settings.crf_vocab
loinc_id_dict = settings.loinc_id_dict


import flask

from flask import request, jsonify, make_response, render_template, url_for
app = flask.Flask(__name__)
app.config["DEBUG"] = True


def is_unit(word):
    if word.upper().strip().endswith('/L'):
        return True
    elif re.match(r'(^(10)\s?\^\s?[1-9]\s?(/[Uu]?[lL])?$)|(^(10)?\s?~?\s?\d/[Uu]?[Ll]$)', word):
        return True
    else:
        return False


def is_name(word):
    if any([word.upper().strip().endswith(i) for i in crf_vocab['suffix']]):
        return True
    elif any([word.upper().strip().startswith(i) for i in crf_vocab['starts']]):
        return True
    elif any([i in word.upper().strip() for i in crf_vocab['sub_names']]):
        return True

    else:
        return False


def is_range_cat(word):
    if word in crf_vocab['range_cat']:
        return True

    else:
        return False


def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
        'word[+3:]': word[+3:],
        'word[+2:]': word[+2:],
        # 'is_unit()': is_unit(word),
        #          'is_range()': is_range(word),
        'is_name()': is_name(word),
        'is_range_cat()': is_range_cat(word),
    }
    if i > 0:
        word1 = sent[i - 1][0]
        #  postag1 = sent[i - 1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:word.isdigit()': word1.isdigit(),
            '-1:is_name()': is_name(word1)
        })
    else:
        features['BOS'] = True

    if i < len(sent) - 1:
        word1 = sent[i + 1][0]
        # postag1 = sent[i + 1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:word.isdigit()': word1.isdigit(),
            '+1:is_name()': is_name(word1)
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def remove_misclassified_labels(results):
    for record in results:
        if 'TEST_NAME' in record:

            name = re.sub(
                r'((10)\s?\^\s?[1-9]\s?(/[Uu]?[lL])?)|((10)?\s?~?\s?\d/[Uu]?[Ll]$)|(\d*/[a-zA-Z]?[Ll])|(\d*\.?\d*\s?[-|—|>|<|=|-—|–|-]\s?\d+\.?\d*)',
                '', record['TEST_NAME'])

            name = re.sub(':\s*$', '', name)
            if (not re.match('^\d+\.*\d*$', name)):
                record['TEST_NAME'] = name
            else:
                record['TEST_NAME'] = ''
            if name in loinc_id_dict.keys():
                record['LOINC_ID'] = loinc_id_dict[name]

        if 'TEST_RANGE' in record:
            ranges = re.findall(
                r'\d*\.?\d*\s?/?\d*\s?[-|—|=|-—|–|-]\s?\d*\.?\d+/?\d*|[Uu][Pp]\s?[Tt][Oo0]\s?\d+|greater\s?than\s?\d*\.?\d+/?\d*|less\s?than\s?\d*\.?\d+/?\d*|[<|>]\s?\d*\.?\d+/?\d*',
                record['TEST_RANGE'], re.IGNORECASE)

            if ranges:
                record['TEST_RANGE'] = ranges[0]
            else:
                record['TEST_RANGE'] = ''
        if 'TEST_VALUE' in record:

            values = re.findall(r'\w*\.?\w+\s?[a-zA-Z]*', record['TEST_VALUE'])
            if values:
                record['TEST_VALUE'] = values[0]
        if 'TEST_UNIT' in record:

            record['TEST_UNIT'] = re.sub('#n', '', record['TEST_UNIT'])

    return results


def post_process_medical_extract(extract):
    results = []
    tests = ['TEST_UNIT:', 'TEST_VALUE:',
             'TEST_RANGE:', 'TEST_RANGE_CATEGORY:']
    for newline in extract:
        if 'TEST_NAME' in newline:
            result = ["TEST_NAME:" +
                      i for i in newline.split("TEST_NAME:")][1:]
            for i in result:
                lab = {}
                if any(c in i for c in tests):
                    lab['TEST_NAME'] = i.split('TEST_NAME: ')[1].split(' TEST_')[
                        0].split(' O: ')[0]
                    if 'TEST_UNIT' in i:
                        lab['TEST_UNIT'] = i.split('TEST_UNIT: ')[1].split(' TEST_')[
                            0].split(' O: ')[0]
                    if 'TEST_RANGE:' in i:
                        pattern = i.split('TEST_RANGE: ')[1].split(
                            ' TEST_')[0].split(' O: ')
                        lab['TEST_RANGE'] = pattern[0]

                        if (re.match('^\d*\.?\d+$', lab['TEST_RANGE'])):
                            if len(pattern) > 1:
                                O = pattern[1]
                                partial = re.findall(
                                    '^[-|=|~|—|—|-|–|-]\s?\d+', O)
                                if len(partial) > 0:
                                    lab['TEST_RANGE'] = lab['TEST_RANGE'] + \
                                        partial[0]

                        elif (re.match('^\d*\.?\d+\s?[-|=|—|—|-|–|-]$', lab['TEST_RANGE'])):
                            if len(i.split('TEST_UNIT:')) > 2:
                                unit_pattern = i.split('TEST_RANGE: ')[
                                    1].split(' TEST_UNIT: ')
                                if len(unit_pattern) > 1:
                                    partial = re.findall(
                                        '^\s?\d*\.?\d+', unit_pattern[1])
                                    if len(partial) > 0:
                                        lab['TEST_RANGE'] = lab['TEST_RANGE'] + \
                                            partial[0]

                    if 'TEST_VALUE' in i:
                        lab['TEST_VALUE'] = i.split('TEST_VALUE: ')[1].split(' TEST_')[
                            0].split(' O: ')[0]
                    if 'TEST_RANGE_CATEGORY' in i:
                        lab['TEST_RANGE_CATEGORY'] = \
                            i.split('TEST_RANGE_CATEGORY: ')[1].split(
                                ' TEST_')[0].split(' O: ')[0]
                    results.append(lab)
    processed_results = remove_misclassified_labels(results)
    return processed_results


def pre_process_medical_extract(doc):
    chars = crf_vocab['chars']

    for i in chars:
        doc = re.sub(i, str(" " + i + " "), doc)
    print(doc)
    doc = re.sub(r'\\n', str(" #n "), doc)
    doc = re.sub(r'\n', str(" #n "), doc)

    doc = re.sub("\(", " (", doc)
    doc = re.sub("\)", ") ", doc)
    doc = re.sub("\\xa0", " ", doc)
    v_x = [i for i in doc.split(' ') if (i)]
    pos = [nltk.pos_tag([x])[0][1] for x in v_x]

    test_x = list(zip(v_x, pos))
    X = [sent2features(s) for s in [test_x]]
    y = crf.predict(X)[0]
    for i in range(1, len(y)):
        if y[i] == 'NEWLINE':
            if y[i - 1] == 'TEST_NAME' and y[i + 1] == 'TEST_NAME':
                y[i] = 'SPLITLINE'
    res = ' '
    for k, g in groupby(zip(y, v_x), itemgetter(0)):
        res = res + k + ': ' + \
            ' '.join(i for i in list(list(zip(*g))[1])) + ' '
    # print(res)
    res = re.sub("NEWLINE: #n", '', res)
    # print(res)
    medical_report = ["SPLITLINE:" + i for i in res.split("SPLITLINE:")]
    # print(medical_report)
    return medical_report


def process_medical_report(medical_report):
    processed_record = []
    result = pre_process_medical_extract(medical_report)
    if result:
        processed_record = post_process_medical_extract(result)

    return processed_record


@app.route('/')
def index():
    return render_template("index.html")

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   results = {}
   if request.method == 'POST':
      f = request.files['file']
      f.save(secure_filename(f.filename))
      medical_report=image_to_string(Image.open(secure_filename(f.filename)),lang='eng')
      
      results = process_medical_report(medical_report)
        
      return render_template("index.html", results=results, num_of_results=len(results))
      
@app.route('/extract_entities', methods=['POST'])
def extract_entities():
    
    results = {}

    if request.method == 'POST':
                # choice = request.form['taskoption']
        medical_report = request.form['rawtext']

        results = process_medical_report(medical_report)
        print(results)
    return render_template("index.html", results=results, num_of_results=len(results))


'''
        med_object = {
            'entity': medical_report
        }
 
        response_json_body = flask.jsonify(med_object)
        

        return make_response(response_json_body, 200)


'''

if __name__ == '__main__':
    app.run(debug=True)
