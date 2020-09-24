#!/usr/bin/env python

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import svm
from pymatgen.ext.matproj import MPRester

# Initialise own apikey
apikey = 'NthZ9vFMzvVhRxGnFG3L'
# initialise MPRester using api key
mpr = MPRester(apikey)
# Initialise array eV and formulas
eV = []
formulas = []
# Query for searching materials project database
# data = mpr.query("**O3", ['pretty_formula', 'e_above_hull', 'diel', 'is_compatible'])

# filter data from query
# for mat in data:

    # if mat['e_above_hull'] == 0 and mat['is_compatible'] is True:

    # formulas.append(mat["pretty_formula"])
    # eV.append(mat["e_above_hull"])

   # print(mat)

# print(formulas)
# print(eV)
# save formula and eV to material variable
# materials = {'Formula': formulas, 'eV': eV}
# Initialise data frame of materials
# df = pd.DataFrame(materials, columns=['Formula', 'eV'])
# print(df)
# Export the data frame to a csv file
# df.to_csv(r'C:\Users\Joe Cant\Desktop\FYP MATERIAL DISCOVERY\eVData.csv', sep=',', header=True)

# Load data from csv file
ABO3 = pd.read_csv(r'C:\Users\Joe Cant\Desktop\FYP MATERIAL DISCOVERY\DFT-ABO3-Simplified.csv', sep=',', header=0)
print(ABO3.head())

# Select columns wanted as features
cols = [col for col in ABO3.columns if col in ['Tolerance factor', 'Stability [eV/atom]']]

# Initialise data and target respectively
data = ABO3[cols]
target = ABO3['Label']

# Split data into training and testing sets
data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.5,)

# Initialise classification model and fit to data
SVM = svm.SVC(kernel='linear')
SVM.fit(data_train, target_train)

# Predict target
target_pred = SVM.predict(data_test)

# show accuracy of classifier
print("Accuracy:", metrics.accuracy_score(target_train, target_pred))
