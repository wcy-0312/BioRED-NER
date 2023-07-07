import nltk
import json

nltk.download("punkt")


def create_lab(tokens):
    label = []
    for t in range(len(tokens)):
        temp = []
        sent = tokens[t]
        for e in range(len(sent)):
            temp.append('o')
        label.append(temp)
    return label


def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def expolore(info):
    all_text = []
    relations = []
    for i in range(len(info)):
        documents = info[i]
        ident = documents['id']
        passages = documents['passages']
        relations = documents['relations']
        for j in passages:
            all_text.append(j)
    return all_text, relations


# ['infons', 'offset', 'text', 'sentences', 'annotations', 'relations']
def search_ner(passages):
    train_text = []
    train_label = []
    for p in passages:
        annotations = p['annotations']
        start_offset = p['offset']
        infons = p['infons']
        text = p['text']
        print('text:', text)

        sentences = nltk.sent_tokenize(text)
        tokens = [nltk.tokenize.word_tokenize(sent) for sent in sentences]
        labels = create_lab(tokens)

        for ann in range(len(annotations)):
            info_id = annotations[ann]
            idx, infons, entity, locations = info_id['id'], info_id['infons'], info_id['text'], info_id['locations'][0]
            offset, length = locations['offset'], locations['length']
            tp = infons['type']
            element = text[offset - start_offset:offset - start_offset + length]
            print('element:', element)
            print('entity:', entity)
            element = nltk.tokenize.word_tokenize(element)
            # import sys
            # sys.exit()

            for t in range(len(tokens)):
                token = tokens[t]
                if element[0] in token:
                    f_start = tokens[t].index(element[0])
                    len_sent = tokens[t][f_start:f_start + len(element)]
                    if len_sent == element:
                        for tlab in range(f_start, f_start + len(element)):
                            if tlab == f_start and labels[t][tlab] == 'o':
                                labels[t][tlab] = 'B-' + tp
                            elif tlab != f_start and labels[t][tlab] == 'o':
                                labels[t][tlab] = 'I-' + tp

        for t, l in zip(tokens, labels):
            train_text.append(t)
            train_label.append(l)
    return train_text, train_label


num2text = {'0': 'ChemicalEntity', '1': 'CellLine', '2': 'SequenceVariant', '3': 'OrganismTaxon',
            '4': 'DiseaseOrPhenotypicFeature', '5': 'GeneOrGeneProduct'}
text2num = {v: k for k, v in num2text.items()}

train1 = '../BC8_BioRED_Subtask1_BioCJSON/bc8_biored_task1_train.json'
val1 = '../BC8_BioRED_Subtask1_BioCJSON/bc8_biored_task1_val.json'

data = load_json(val1)
documents = data['documents']
passages, relations = expolore(documents)
train_text, train_label = search_ner(passages)

dict_name = {}
dict_name['train'] = train_text
dict_name['labels'] = train_label

# with open("val1.json",'w',encoding='utf-8') as f:
#     json.dump(dict_name, f,ensure_ascii=False)
