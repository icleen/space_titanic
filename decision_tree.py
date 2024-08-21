import numpy as np

import pandas as pd
pd.options.mode.copy_on_write = True

import sklearn as skl
from sklearn import tree

data = pd.read_csv("data/train.csv")

# ['PassengerId', 'HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Name', 'Transported']
headers = [c for c in data.columns]
print(headers)
print(data.Transported.value_counts())

seldata = data[['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'Age', 'VIP']]
cabin = [c if not pd.isnull(c) else 'n/n/n' for c in data['Cabin']]
for cab in cabin:
    try:
        cab.split('/')
    except Exception as e:
        print(e)
        import pdb; pdb.set_trace()
cabin = [c.split('/') for c in cabin]
cabindeck = np.array([c[0] for c in cabin])
cabinnum = np.array([c[1] for c in cabin])
cabinside = np.array([c[2] for c in cabin])

def categorize(arr):
    categories = np.unique(arr)
    for ci, cat in enumerate(categories):
        arr[arr == cat] = ci
    return arr.astype(int)

seldata['CabinDeck'] = categorize(cabindeck)
seldata['CabinNum'] = cabinnum.astype(int)
seldata['CabinSide'] = categorize(cabinside)

seldata['HomePlanet'] = categorize(seldata['HomePlanet'])
seldata['Destination'] = categorize(seldata['Destination'])

print(seldata.head(3))

seldata = seldata[['HomePlanet', 'CryoSleep', 'CabinDeck', 'CabinNum', 'CabinSide', 'Destination', 'Age', 'VIP']]

print(seldata.head(3))

enc = skl.preprocessing.OneHotEncoder()

import pdb; pdb.set_trace()
enc.fit(seldata['CabinSide'])

print(enc.categories_)

trans = enc.transform(seldata['CabinSide'])
print(trans)

import pdb; pdb.set_trace()

clf = tree.DecisionTreeClassifier()
clf = clf.fit(seldata[['HomePlanet', 'CryoSleep', 'CabinDeck', 'CabinNum', 'CabinSide', 'Destination', 'Age', 'VIP']], data['Transported'])

import pdb; pdb.set_trace()

