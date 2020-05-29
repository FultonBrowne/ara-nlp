import pandas as pd
import os
from download import download


def main():
    print("start")
    output_dir = "./data/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print("names")
    path = download("https://raw.githubusercontent.com/hadley/data-baby-names/master/baby-names.csv", "./data/names.csv", replace=True)
    names = pd.read_csv("./data/names.csv")
    toInput = names.name.values
    callList = []
    textList = []
    emailList = []
    for i in toInput:
        callList.append("call " + i)
        textList.append("text " + i)
        textList.append("message " + i)
    navList = []
    weatherList = []
    download("https://developers.google.com/adwords/api/docs/appendix/geo/geotargets-2020-03-03.csv", "./data/places.csv")
    pd.read_csv("./data/places.csv")
    mainData = pd.read_csv("data.csv", sep="\t")
main()
