import pandas as pd
import seaborn
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold

# Load csv from file
BandGap = pd.read_csv(r'C:\Users\Joe Cant\Desktop\FYP MATERIAL DISCOVERY\BandGapTest.csv', sep=',', header=0)
# show top 10 entries
print(BandGap.head(10))

# show 'Band gap [eV]' features most common results as graph
plt.figure(figsize=(15, 10))
plt.tight_layout()
distplot = seaborn.distplot(BandGap['Band gap [eV]'])
plt.show()

# Select columns wanted as features
cols = [col for col in BandGap.columns if col in ['Tolerance factor', 'Stability [eV/atom]', 'Valence A',
                                                  'Valence B', 'Formation energy [eV/atom]']]

# Initialise data and target respectively
X = BandGap[cols]
y = BandGap['Band gap [eV]']
# Scale and normalise the data
X = preprocessing.scale(X)
X = preprocessing.normalize(X, norm='l1')

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Initialise regression model
LinReg = LinearRegression()
# Remove features above 80% variance threshold
sel = VarianceThreshold()
sel.fit_transform(X)

# Fitting the data to model
LinReg.fit(X, y)
# Predict target
y_pred = LinReg.predict(X_test)

# The coefficients
print('Coefficients: \n', LinReg.coef_)
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))

# Initialise pandas data frame with results and display
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df1 = df.head(25)
print(df1)

# plot results as a bar chart and display
df1.plot(kind='bar', figsize=(10, 8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

