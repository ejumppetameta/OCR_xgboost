import os
import pickle
import pandas as pd
import numpy as np
import nltk
from collections import Counter
from sqlalchemy import create_engine
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedShuffleSplit, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, average_precision_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, RandomOverSampler
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from scipy.stats import uniform, randint

# Download necessary NLTK resources
nltk.download('wordnet')
nltk.download('omw-1.4')

# Import custom preprocessor (ensure it includes any additional cleaning like lemmatization if needed)
from preprocessing import custom_preprocessor

def determine_type(desc):
    desc_upper = desc.upper()
    return 'CR' if 'CR' in desc_upper else 'DR'

def build_vectorizer():
    """
    Builds a TF-IDF Vectorizer with character n-grams and sublinear TF scaling.
    """
    return TfidfVectorizer(
        stop_words='english',
        preprocessor=custom_preprocessor,
        ngram_range=(1, 3),
        sublinear_tf=True
    )

def load_data(engine):
    query = "SELECT description, category, type FROM train_data"
    try:
        return pd.read_sql(query, engine)
    except Exception as e:
        print("Error loading data from MySQL:", e)
        return None

def filter_small_categories(data, min_samples=2):
    """
    Removes categories that have fewer than the specified number of samples.
    """
    category_counts = data['category'].value_counts()
    small_categories = category_counts[category_counts < min_samples].index
    filtered_data = data[~data['category'].isin(small_categories)]
    print(f"Filtered out {len(small_categories)} small categories: {small_categories}")
    return filtered_data

def train_category_model(X_train, y_train):
    category_counts = Counter(y_train)
    min_samples = min(category_counts.values())

    # Use RandomOverSampler if any fold might have too few samples (e.g. less than 4)
    if min_samples < 4:
        oversampler = RandomOverSampler(random_state=42)
    else:
        oversampler = SMOTE(random_state=42, k_neighbors=max(1, min_samples - 1))

    # Compute class weights for the imbalance
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

    # Define base classifier and parameter distribution for hyperparameter tuning.
    base_clf = XGBClassifier(
        use_label_encoder=False,
        eval_metric="mlogloss",
        objective='multi:softprob',
        scale_pos_weight=weight_dict
    )
    param_dist = {
        'clf__n_estimators': randint(100, 300),
        'clf__max_depth': randint(3, 10),
        'clf__learning_rate': uniform(0.01, 0.2),
        'clf__subsample': uniform(0.5, 0.5),
        'clf__colsample_bytree': uniform(0.5, 0.5)
    }

    pipeline = ImbPipeline([
        ('oversample', oversampler),
        ('clf', base_clf)
    ])

    # Use RandomizedSearchCV for hyperparameter tuning.
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=20,
        scoring='accuracy',
        cv=3,
        random_state=42,
        verbose=1,
        n_jobs=-1
    )

    search.fit(X_train, y_train)
    print("Best parameters found:", search.best_params_)
    return search.best_estimator_

def train_type_model(X, y_type):
    return XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5
    ).fit(X, y_type)

def plot_precision_recall_curve(y_true, y_pred_proba, classes):
    """Plot precision-recall curve for each class."""
    for i, class_name in enumerate(classes):
        precision, recall, _ = precision_recall_curve(y_true == i, y_pred_proba[:, i])
        avg_precision = average_precision_score(y_true == i, y_pred_proba[:, i])
        plt.plot(recall, precision, label=f'{class_name} (AP={avg_precision:.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')
    plt.show()

def main():
    MYSQL_USER = os.getenv("MYSQL_USER", "laraveluser")
    MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "secret")
    MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
    MYSQL_PORT = os.getenv("MYSQL_PORT", "3306")
    MYSQL_DB = os.getenv("MYSQL_DB", "laravel")

    connection_string = f"mysql+mysqlconnector://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}?auth_plugin=mysql_native_password"
    engine = create_engine(connection_string)

    # Load data from database
    data = load_data(engine)
    if data is None:
        return

    if 'type' not in data or data['type'].isnull().all():
        data['type'] = data['description'].apply(determine_type)
        print("Generated 'type' column based on description.")

    print("Category Distribution Before Filtering:")
    print(data['category'].value_counts())

    # Filter out categories with too few samples
    data = filter_small_categories(data, min_samples=2)

    # Vectorization
    vectorizer = build_vectorizer()
    X_all = vectorizer.fit_transform(data['description'])

    le_category, le_type = LabelEncoder(), LabelEncoder()
    y_category = le_category.fit_transform(data['category'])
    y_type = le_type.fit_transform(data['type'])

    # Stratified split for ensuring balanced class distribution
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in sss.split(X_all, y_category):
        X_train, X_test = X_all[train_index], X_all[test_index]
        y_train, y_test = y_category[train_index], y_category[test_index]

    # Train category model with hyperparameter tuning
    model_category = train_category_model(X_train, y_train)
    y_pred = model_category.predict(X_test)

    print("Category Model - Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    target_names = le_category.inverse_transform(np.unique(y_test))
    print("Category Model - Classification Report:")
    print(classification_report(y_test, y_pred, labels=np.unique(y_test), target_names=target_names, zero_division=0))

    # Get prediction probabilities for precision-recall curve
    y_pred_proba = model_category.predict_proba(X_test)

    # Plot Precision-Recall curves
    plot_precision_recall_curve(y_test, y_pred_proba, target_names)

    # Train type model
    model_type = train_type_model(X_all, y_type)

    # Save models and encoders
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for name, model in [('model_category.pkl', model_category), ('model_type.pkl', model_type), ('vectorizer.pkl', vectorizer), ('le_category.pkl', le_category), ('le_type.pkl', le_type)]:
        with open(os.path.join(script_dir, name), 'wb') as f:
            pickle.dump(model, f)

    print("Models, vectorizer, and label encoders saved successfully.")

if __name__ == '__main__':
    main()
