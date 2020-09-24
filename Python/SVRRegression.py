import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# Load csv from file
BandGap = pd.read_csv(r'C:\Users\Joe Cant\Desktop\FYP MATERIAL DISCOVERY\BandGapTest.csv', sep=',', header=0)
# show top 10 entries
print(BandGap.head(10))

# Select columns wanted as features
cols = [col for col in BandGap.columns if col in ['Tolerance factor', 'Stability [eV/atom]', 'Valence A',
                                                  'Valence B', 'Formation energy [eV/atom]']]

# Initialise data and target respectively
X = BandGap[cols]
y = BandGap['Band gap [eV]'].astype(int)
# Scale and normalise the data
X = preprocessing.normalize(X, norm='l2')
# X = preprocessing.scale(X)

# Select top 3 features
X = SelectKBest(chi2, k=3).fit_transform(X, y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialise regression model and fit to data
svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svr_rbf.fit(X_train, y_train)
# Predict target
y_pred = svr_rbf.predict(X_test)

# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))

# Initialise pandas data frame with results and display
dfsvr = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
dfsvr1 = dfsvr.head(25)
print(dfsvr1)

# plot results as a bar chart and display
dfsvr1.plot(kind='bar', figsize=(10, 8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
