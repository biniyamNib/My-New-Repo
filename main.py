import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Load the datasets
ratings = pd.read_csv('ml-latest-small\\ratings.csv')
movies = pd.read_csv('ml-latest-small\\movies.csv')


print(ratings.isnull().sum())
print(movies.isnull().sum())


sns.histplot(ratings['rating'], bins=10, kde=True)
plt.title('Distribution of Ratings')
plt.show()

# Merge datasets on movieId
data = pd.merge(ratings, movies, on='movieId') 

# Add a binary column indicating high-rated movies (e.g., rating >= 4.0)
data['high_rating'] = data['rating'] >= 4.0

# Plot relationship between genres and high_rating
sns.countplot(x='high_rating', hue='genres', data=data)
plt.title('High Rating by Genre')
plt.xticks(rotation=90)
plt.show()


# Encode genres as numerical features
label_encoder = LabelEncoder()
data['genres_encoded'] = label_encoder.fit_transform(data['genres'])


scaler = MinMaxScaler()
data[['userId', 'movieId', 'rating']] = scaler.fit_transform(data[['userId', 'movieId', 'rating']])


X = data[['userId', 'movieId', 'genres_encoded', 'rating']]
y = data['high_rating']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


param_grid = {
    'n_estimators': [50, 100, 200],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [None, 10, 20, 30],
    'criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best parameters
print(grid_search.best_params_)


y_pred = grid_search.predict(X_test)


print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ROC Curve
fpr, tpr, _ = roc_curve(y_test, grid_search.predict_proba(X_test)[:,1])
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc_score(y_test, y_pred))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()


dummy_classifier = DummyClassifier(strategy='most_frequent')
dummy_classifier.fit(X_train, y_train)
y_dummy_pred = dummy_classifier.predict(X_test)

print("Baseline Accuracy:", accuracy_score(y_test, y_dummy_pred))
print("Baseline Precision:", precision_score(y_test, y_dummy_pred))
print("Baseline Recall:", recall_score(y_test, y_dummy_pred))
print("Baseline F1 Score:", f1_score(y_test, y_dummy_pred))

# Save the model
joblib.dump(grid_search.best_estimator_, 'movie_rating_model.pkl')