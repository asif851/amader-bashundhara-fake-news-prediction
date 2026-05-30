import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# 1. Load the dataset robustly 

print("Loading dataset...")
file_path = 'Authentic-48K.csv'

# Using 'utf-8-sig' and python engine bypasses faulty formatting issues 
df = pd.read_csv(
    file_path, 
    encoding='utf-8-sig', 
    engine='python', 
    on_bad_lines='skip'
)

#  Clean missing values

df = df.dropna(subset=['headline', 'category'])

#  Define features (X) and target variable (y)
X = df['headline']
y = df['category']

#  Split the data into Training (80%) and Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# Text Vectorization (With Token Pattern suited for Bangla script)

print("Vectorizing text data...")
# The added token_pattern ensures the vectorizer captures Bangla words completely
vectorizer = TfidfVectorizer(max_features=10000, token_pattern=r'[\u0980-\u09FF\w]+')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Initialize and Train the Model
print("Training Logistic Regression model...")
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train_tfidf, y_train)

#Evaluate the Model
y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
print("\n  MODEL EVALUATION ")
print(f"Overall Accuracy: {accuracy * 100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))

#Test the model with a custom headline
print("\n LIVE TEST ")
sample_headline = [" "]
sample_tfidf = vectorizer.transform(sample_headline)
predicted_category = model.predict(sample_tfidf)
print(f"Headline: {sample_headline[0]}")
print(f"Predicted Category: {predicted_category[0]}")
