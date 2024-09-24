import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np

# Function to perform k-mer encoding
def kmer_encoding(sequences, k=3, *, vectorizer=None):
    # Convert sequence list into a string format
    seqs = [" ".join([seq[i:i+k] for i in range(len(seq)-k+1)]) for seq in sequences]
    
    # Apply k-mer counting using CountVectorizer
    if vectorizer is None:
        vectorizer = CountVectorizer()
        kmer_features = vectorizer.fit_transform(seqs)
    else:
        kmer_features = vectorizer.transform(seqs)
    
    # Return as a pandas DataFrame
    return pd.DataFrame(kmer_features.toarray(), columns=vectorizer.get_feature_names_out())


# Load data
dataA = pd.read_csv("Train.csv")
dataB = pd.read_csv("Test.csv")

# Encode the miRNA_Sequence and gene_Sequence for training data
vectorizer_mirna = CountVectorizer()
vectorizer_gene = CountVectorizer()

vectorizer_mirna.fit(dataA['miRNA_Sequence'])
vectorizer_gene.fit(dataA['gene_Sequence'])

X_train_mirna = kmer_encoding(dataA['miRNA_Sequence'], k=3, vectorizer=vectorizer_mirna)
X_test_mirna = kmer_encoding(dataB['miRNA_Sequence'], k=3, vectorizer=vectorizer_mirna)

X_train_gene = kmer_encoding(dataA['gene_Sequence'], k=3, vectorizer=vectorizer_gene)
X_test_gene = kmer_encoding(dataB['gene_Sequence'], k=3, vectorizer=vectorizer_gene)

# Normalize the gene_length feature
scaler = StandardScaler()
X_train_length = scaler.fit_transform(dataA[['gene_length']])
X_test_length = scaler.transform(dataB[['gene_length']])

# Concatenate all features for training data
X_train = np.hstack((X_train_mirna, X_train_gene, X_train_length))

# Concatenate all features for test data
X_test = np.hstack((X_test_mirna, X_test_gene, X_test_length))

# Save the encoded data to new CSV files
# Convert the normalized gene_length back to DataFrames for saving
X_train_length_df = pd.DataFrame(X_train_length, columns=['gene_length'])
X_test_length_df = pd.DataFrame(X_test_length, columns=['gene_length'])
train_encoded = pd.concat([X_train_mirna, X_train_gene, X_train_length_df], axis=1)
test_encoded = pd.concat([X_test_mirna, X_test_gene, X_test_length_df], axis=1)

train_encoded.to_csv("Encoded_TrainData.csv", index=False)
test_encoded.to_csv("Encoded_TestData.csv", index=False)
print("Encoding Done")
# Prepare the target
y_train = dataA['Target_Score']

# Create SVM classifier
svm_model = SVC(kernel='linear')  # You can experiment with different kernels

# Train the model
svm_model.fit(X_train, y_train)
print("Model Fitted")

# Predict target scores for dataB
y_pred = svm_model.predict(X_test)
print("Predicted Target Scores for dataB:", y_pred)

# If you want to evaluate performance on the training set, you can do:
y_train_pred = svm_model.predict(X_train)

# Evaluate model performance on the training set
accuracy = accuracy_score(y_train, y_train_pred)
print("Training Accuracy:", accuracy)

mse = mean_squared_error(y_train, y_train_pred)
print("Mean Squared Error (Training Set):", mse)
