import csv
import pandas as pd
from sklearn import tree
# Load CSV (using python)
filename = 'path to the csv file'
raw_data = open(filename, 'rt')
reader = pd.read_csv(filename, delimiter = ',')
df = pd.DataFrame(reader)

df.Label[df.Label == 'Apple'] = 0
df.Label[df.Label == 'Orange'] = 1
df.Texture[df.Texture == 'Smooth'] = 1
df.Texture[df.Texture == 'Bumpy'] = 0
features = df[['Weight', 'Texture']].values.tolist()
labels = df['Label'].values.tolist()
print(features)
print(labels)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)
print (clf.predict([[90,0]]))