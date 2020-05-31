import pandas as pd
import os
from download import download

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
    placesLink="https://arafilestore.file.core.windows.net/ara-server-files/FastFoodRestaurants.csv?sv=2019-02-02&ss=bfqt&srt=sco&sp=rwdlacup&se=2024-04-01T22:11:11Z&st=2019-12-19T15:11:11Z&spr=https&sig=lfjMHSahA6fw8enCbx0hFTE1uAVJWvPmC4m6blVSuuo%3D"
    download(placesLink, "./data/ff.csv")
    download("https://developers.google.com/adwords/api/docs/appendix/geo/geotargets-2020-03-03.csv", "./data/places.csv")
    places = pd.read_csv("./data/places.csv").names.values
    for p in places:
        allList.append(Template("directions to " + p, "nav"))
        allList.append(Template("how far is " + p, "nav"))
        allList.append(Template("weather in " + p, "weath"))
        allList.append(Template("what is the weather in " + p, "weath"))
    ff = pd.read_csv("./data/ff.csv")
    for f in ff.names.values:
        allList.append(Template("directions to " + f, "nav"))
        allList.append(Template(f + " near by","nav"))
        allList.append(Template("where is the nearest " + f, "nav"))
    mainData = pd.read_csv("data.csv", sep="\t")
    print(len(allList))
    for t in allList:
        print(len(mainData))
        mainData.loc[len(mainData)+1] = [len(mainData)+1, t.data, t.label]
    download("https://arafilestore.file.core.windows.net/ara-server-files/datasets_456949_861300_top10s.csv?sv=2019-02-02&ss=bfqt&srt=sco&sp=rwdlacup&se=2024-04-01T22:11:11Z&st=2019-12-19T15:11:11Z&spr=https&sig=lfjMHSahA6fw8enCbx0hFTE1uAVJWvPmC4m6blVSuuo%3D", "./data/music.csv")
    music = pd.read_csv("./data/music.csv")
    songs = music.name.values
    for s in songs:
        allList.append(Template("play " + s, "music"))
    mainData.to_csv("data.csv", sep='\t')
    print("done")
main()
