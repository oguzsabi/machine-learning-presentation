import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, Normalizer, StandardScaler 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neural_network import MLPRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from scipy import stats

# Load the dataset
file_path = './datasets/spotify_songs.csv'
data = pd.read_csv(file_path, encoding='ISO-8859-1')
data = data.drop_duplicates(subset='track_id')

columns_to_exclude = [col for col in data.columns if col.startswith('track_') and col != 'track_popularity'] + [col for col in data.columns if col.startswith('playlist_')]
data_selected = data.drop(columns=columns_to_exclude)

# Calculate Z-scores
z_scores = stats.zscore(data_selected)
outliers = (np.abs(z_scores) > 3).any(axis=1)
# print(data_selected[outliers])

data_selected = data_selected[~outliers]

numerical_transformer = StandardScaler()
numerical_cols = data_selected.drop('track_popularity', axis=1).columns.values

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols)
    ]
)

X = data_selected.drop('track_popularity', axis=1)
y = data_selected['track_popularity']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

X_train_transformed_df = pd.DataFrame(X_train_transformed, columns=numerical_cols)
print(X_train_transformed_df)

# RF
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train_transformed, y_train)
y_pred_rf = rf_regressor.predict(X_test_transformed)

# SVR
svr = SVR(kernel='linear')
svr.fit(X_train_transformed, y_train)
y_pred_svr = svr.predict(X_test_transformed)

# LR
pr = PolynomialFeatures(degree=1)
X_poly = pr.fit_transform(X_train_transformed)

lr = LinearRegression()
lr.fit(X_poly, y_train)
X_poly_test = pr.fit_transform(X_test_transformed)

y_pred_poly = lr.predict(X_poly_test)

# Ridge
ridge = Ridge(alpha=1)
ridge.fit(X_train_transformed, y_train)
y_pred_ridge = ridge.predict(X_test_transformed)

# Lasso Regression
lasso = Lasso(alpha=1.0)
lasso.fit(X_train_transformed, y_train)
y_pred_lasso = lasso.predict(X_test_transformed)

# Multilayer Perceptron
mlp = MLPRegressor(hidden_layer_sizes=(30,50,50), activation='relu', solver='adam', max_iter=10000, random_state=42, learning_rate='adaptive')
mlp.fit(X_train_transformed, y_train)
y_pred_mlp = mlp.predict(X_test_transformed)

print(y_test)
# print(y_pred_rf)
print(f"R2 score for RF: {r2_score(y_test, y_pred_rf)}")
print(f"R2 score for SVR: {r2_score(y_test, y_pred_svr)}")
print(f"R2 score for POLY: {r2_score(y_test, y_pred_poly)}")
print(f"R2 score for Ridge: {r2_score(y_test, y_pred_ridge)}")
print(f"R2 score for Lasso: {r2_score(y_test, y_pred_lasso)}")
print(f"RMSE for MLP: {np.sqrt(mean_squared_error(y_test, y_pred_mlp))}")
print(f"RMSE for RF: {np.sqrt(mean_squared_error(y_test, y_pred_rf))}")
print(f"MAE for RF: {mean_absolute_error(y_test, y_pred_rf)}")
print(f"MAE for SVR: {mean_absolute_error(y_test, y_pred_svr)}")
print(f"MAE for POLY: {mean_absolute_error(y_test, y_pred_poly)}")
print(f"MAE for Ridge: {mean_absolute_error(y_test, y_pred_ridge)}")
print(f"MAE for Lasso: {mean_absolute_error(y_test, y_pred_lasso)}")
print(f"MAPE for Lasso: {mean_absolute_percentage_error(y_test, y_pred_lasso)}")
print("MAPE for Lasso: {:.100}".format(mean_absolute_percentage_error(y_test, y_pred_lasso)))
# print(f"MAPE for MLP: {mean_absolute_percentage_error(y_test, y_pred_mlp)}")
print("Mean of y_train: {:.10}".format(np.mean(y_train)))
print("Median of y_train: {:.10}".format(np.median(y_train)))
