import json
import pandas as pd


def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


if __name__ == '__main__':
    # 指定 CSV 檔案路徑
    train_json = load_json('../JSON1/train1.json')
    concatenated_string = "".join(train_json['train'][0])
    print(train_json['train'][0])
    print(concatenated_string)

