import pandas as pd
import os
import urllib


def main():
    print("start")
    output_dir = "./data/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print("names")
    urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/hadley/data-baby-names/master/baby-names.csv",
            "./data/names.csv")
    names = pd.read_csv("./data/names.csv")
    toInput = names.name.values
