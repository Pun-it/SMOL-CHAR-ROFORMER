import pandas as pd

"""
Dataset stolen from here : 
https://github.com/ArchanGhosh/Robert-Frost-Collection-A-Kaggle-Dataset/blob/main/robert_frost_collection.csv
"""

def get_text():
    data = pd.read_csv('robert_frost_collection.csv')
    data = data.dropna()
    random_text = data['Content'] 

    random_text = ' '.join(x for x in random_text)
    return random_text