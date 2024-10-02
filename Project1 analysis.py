import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import reciprocal
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score



df = pd.read_csv("Project_1_Data.csv")

print(df.info())

# Step 2
x = df['X'].values
y = df['Y'].values
z = df['Z'].values
Step = df['Step'].values

plt.plot(Step, x, label= 'X')
plt.plot(Step, y, label= 'Y')
plt.plot(Step, z, label= 'Z')


plt.xlabel('Step')
plt.ylabel('Values')
plt.title('Line Plot')
plt.legend()
plt.show()

# Step 3 Correlation
corr_matrix = df.corr()
sns.heatmap(np.abs(corr_matrix))

# Step 4 Classification
X = np.column_stack((x, y, z))
Y = Step

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

param_grid_log = {'C': [0.01, 0.1, 1, 10, 100]}
param_grid_for = {'n_estimators': [3, 10, 30], 'max_depth': [2, 4, 6, 8]}
param_grid_KN = {'n_neighbors': [3, 5, 7, 9]}

log_model = LogisticRegression()
for_model = RandomForestClassifier()
KN_model = KNeighborsClassifier()
# CV_model = RandomizedSearchCV()

grid_log = GridSearchCV(log_model, param_grid_log, cv=5)
grid_log.fit(X_train, Y_train)
Y_pred_log = grid_log.predict(X_test)

grid_for = GridSearchCV(for_model, param_grid_for, cv=5)
grid_for.fit(X_train, Y_train)
Y_pred_for = grid_for.predict(X_test)

grid_KN = GridSearchCV(KN_model, param_grid_KN, cv=5)
grid_KN.fit(X_train, Y_train)
Y_pred_KN = grid_KN.predict(X_test)

accuracy_log = accuracy_score(Y_test, Y_pred_log)
accuracy_for = accuracy_score(Y_test, Y_pred_for)
accuracy_KN = accuracy_score(Y_test, Y_pred_KN)

print(f"Logistic Regression Accuracy: {accuracy_log}")
print(f"Random Forest Accuracy: {accuracy_for}")
print(f"KNeighbors Accuracy: {accuracy_KN}")

# randomized ssearch cv

"""GridSearchCV"""
param_grid = {
    'n_estimators': [10, 30, 50],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
    }

CV_model = RandomForestClassifier()
grid_search = RandomizedSearchCV(CV_model, param_distributions=param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=1)
grid_search.fit(X_train, Y_train)
Y_pred_CV = grid_search.predict(X_test)

best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)
best_modelCV = grid_search.best_estimator_


accuracy_CV = accuracy_score(Y_test, Y_pred_CV)
print(f"CV Accuracy: {accuracy_CV}")