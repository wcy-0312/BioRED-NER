import json


def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def json2txt(json_p, txt_p):
    f_json = load_json(json_p)

    # convert 'o' to 'O
    for i in range(len(f_json['labels'])):
        for j in range(len(f_json['labels'][i])):
            if f_json['labels'][i][j] == 'o':
                f_json['labels'][i][j] = 'O'

    # save
    with open(txt_p, 'w') as f:
        for words_sent, tags_sent in zip(f_json['train'], f_json['labels']):
            for word, tag in zip(words_sent, tags_sent):
                f.write(f"{word} {tag}\n")
            f.write("\n")


json2txt(json_p='bio/train1.json', txt_p='train.txt')
json2txt(json_p='bio/val1.json', txt_p='val.txt')