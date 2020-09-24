import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectPercentile, chi2

# Load csv from file
BandGap = pd.read_csv(r'C:\Users\Joe Cant\Desktop\FYP MATERIAL DISCOVERY\BandGapTest.csv', sep=',', header=0)
# show top 10 entries
print(BandGap.head(10))

# Select columns wanted as features
cols = [col for col in BandGap.columns if col in ['Tolerance factor', 'Stability [eV/atom]', 'Valence A',
                                                  'Valence B', 'Formation energy [eV/atom]']]

# Initialise polynomial features
polynomial_features = PolynomialFeatures(degree=2)

# Initialise data and target respectively
X = BandGap[cols]
y = BandGap['Band gap [eV]'].astype(int)
# Scale and normalise the data
# X = preprocessing.scale(X)
X = preprocessing.normalize(X, norm='l2')

# Select top 60% of features
X = SelectPercentile(chi2, percentile=60).fit_transform(X, y)
X_poly = polynomial_features.fit_transform(X)

# Split data into training and testing sets
X_poly_train, X_poly_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.3, random_state=0)

# Initialise regression model and fit to data
LinReg = LinearRegression()
LinReg.fit(X_poly_train, y_train)

# Predict target
y_pred = LinReg.predict(X_poly_test)

# The coefficients
print('Coefficients: \n', LinReg.coef_)
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))

# Initialise pandas data frame with results
df2 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df3 = df2.head(25)
print(df3)

# plot results as a bar chart and display
df3.plot(kind='bar', figsize=(10, 8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
