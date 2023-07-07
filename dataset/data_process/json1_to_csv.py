import json
import pandas as pd


def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def dict_to_csv(data, name):
    df = pd.DataFrame(data)
    df.to_csv(name, index=False)


if __name__ == '__main__':
    data_path = '../JSON1/train1.json'
    data = load_json(data_path)
    sentences = data['train']
    labels = data['labels']

    data_dict = {'sentence_index': [], 'words': [], 'tags': []}

    for idx, (sentence, label) in enumerate(zip(sentences, labels)):
        for word, tag in zip(sentence, label):
            if tag == 'o':
                tag = 'O'
            data_dict['sentence_index'].append(idx)
            data_dict['words'].append(word)
            data_dict['tags'].append(tag)

    dict_to_csv(data_dict, 'val.csv')


