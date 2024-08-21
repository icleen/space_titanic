import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
pd.options.mode.copy_on_write = True

import sklearn as skl
from sklearn import tree


def process_data(data):
    headers = [c for c in data.columns]
    print(headers)
    print(data.Transported.value_counts())
    seldata = data[['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'Age', 'VIP']]
    seldata['HomePlanet'] = seldata['HomePlanet'].fillna('na')
    seldata['CryoSleep'] = seldata['CryoSleep'].fillna(False)
    seldata['Cabin'] = seldata['Cabin'].fillna('n/n/n')
    seldata['Destination'] = seldata['Destination'].fillna('na')
    seldata['Age'] = seldata['Age'].fillna(-1)
    seldata['VIP'] = seldata['VIP'].fillna(False)
    cabin = [c if not pd.isnull(c) else 'n/n/n' for c in data['Cabin']]
    for cab in cabin:
        try:
            cab.split('/')
        except Exception as e:
            print(e)
            import pdb; pdb.set_trace()
    cabin = [c.split('/') for c in cabin]
    cabindeck = np.array([c[0] for c in cabin])
    cabinnum = np.array([float(c[1]) if c[1] != 'n' else float('nan') for c in cabin])
    cabinside = np.array([c[2] for c in cabin])

    def categorize(arr):
        if isinstance(arr, pd.core.series.Series):
            arr = arr.to_numpy()
        try:
            categories = np.unique(arr)
        except Exception as e:
            print(e)
            import pdb; pdb.set_trace()
        print(categories)
        arr = arr.copy()
        for ci, cat in enumerate(categories):
            arr[arr == cat] = ci
        try:
            return arr.astype(float)
        except Exception as e:
            print(e)
            import pdb; pdb.set_trace()

    seldata['CabinDeck'] = categorize(cabindeck)
    seldata['CabinNum'] = cabinnum.astype(float)
    seldata['CabinSide'] = categorize(cabinside)

    # if pd.isnull(seldata['HomePlanet']).any():
    #     seldata['HomePlanet'] = 
    seldata['HomePlanet'] = categorize(seldata['HomePlanet'].to_numpy())
    seldata['Destination'] = categorize(seldata['Destination'].to_numpy())

    seldata = seldata[['HomePlanet', 'CryoSleep', 'CabinDeck', 'CabinNum', 'CabinSide', 'Destination', 'Age', 'VIP']]

    print(seldata.head(3))

    return seldata


def main():

    data = pd.read_csv("data/train.csv")
    # ['PassengerId', 'HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Name', 'Transported']

    tdata = process_data(data)

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(tdata[['HomePlanet', 'CryoSleep', 'CabinDeck', 'CabinNum', 'CabinSide', 'Destination', 'Age', 'VIP']], data['Transported'])

    tree.plot_tree(clf)

    validation_data = pd.read_csv("data/valid.csv")

    vdata = process_data(validation_data)

    preds = clf.predict(vdata[['HomePlanet', 'CryoSleep', 'CabinDeck', 'CabinNum', 'CabinSide', 'Destination', 'Age', 'VIP']])
    correct = preds == validation_data['Transported'].to_numpy()
    print('correct perc:', correct.sum() / len(correct))


if __name__ == '__main__':
    main()


