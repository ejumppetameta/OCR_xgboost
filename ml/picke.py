import pickle
vectorizer = pickle.load(open('ml/model.pkl', 'rb'))  # or 'ml/vectorizer.pkl'
print(hasattr(vectorizer, 'idf_'))  # Should print True
