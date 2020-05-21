import pandas as pd

data = {'sentences':  ['First value', 'Second value'],
        'labels': ['First value', 'Second value'],
         
        }

df = pd.DataFrame (data, columns = ['sentences','labels'])
df.to_csv("data1.csv", sep='\t', encoding='utf-8')
