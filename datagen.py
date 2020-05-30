import pandas as pd
import os
from download import download

0

class Template(object):
    data = ""
    label = ""

    # The class "constructor" - It's actually an initializer 
    def __init__(self, data, label):
        self.data = data
        self.label = label
def toFormat(data, label):
    return Template(data, label)
def main():
    print("start")
    output_dir = "./data/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print("names")
    path = download("https://raw.githubusercontent.com/hadley/data-baby-names/master/baby-names.csv", "./data/names.csv", replace=True)
    names = pd.read_csv("./data/names.csv")
    toInput = names.name.values
    allList = []
    for i in toInput:
        allList.append(toFormat("call " + i,"call"))
        allList.append(toFormat("dial " + i,"call"))
        allList.append(toFormat("text " + i, "text"))
        allList.append(toFormat("message " + i, "text"))
    download("https://developers.google.com/adwords/api/docs/appendix/geo/geotargets-2020-03-03.csv", "./data/places.csv")
    places = pd.read_csv("./data/places.csv").names.values
    for p in places:
        allList.append(Template("directions to " + p, "nav"))
        allList.append(Template("how far is " + p, "nav"))
        allList.append(Template("weather in " + p, "weath"))
        allList.append(Template("what is the weather in " + p), "weath")
    mainData = pd.read_csv("data.csv", sep="\t")
    print(len(allList))
    for t in allList:
        print(len(mainData))
        mainData.loc[len(mainData)+1] = [len(mainData)+1, t.data, t.label]
    print("done")
main()
