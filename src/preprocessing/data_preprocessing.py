import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

# Download required NLTK data
# nltk.download('punkt_tab')
# nltk.download('stopwords')
# nltk.download('wordnet')


class ResumeTextPreprocessor:
    def __init__(self, max_words=10000, max_length=500):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english')) - {
            'experience', 'experienced', 'skills', 'skill', 'responsible',
            'responsibilities', 'include', 'includes', 'including', 'work',
            'project', 'projects', 'team', 'teams', 'led', 'manager', 'managing',
            'management', 'develop', 'developer', 'development', 'engineer', 'engineering'
        }
        self.max_words = max_words
        self.max_length = max_length
        self.tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
        self.label_encoder = LabelEncoder()
        self.onehot_encoder = OneHotEncoder(sparse_output=False)  # Ensure sparse=False for one-hot encoding
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        self.section_headers = r'(PROFESSIONAL EXPERIENCE|WORK EXPERIENCE|EXPERIENCE|'
        self.section_headers += r'EDUCATION|SKILLS|CERTIFICATIONS|PROJECTS|'
        self.section_headers += r'CORE COMPETENCIES|PROFESSIONAL SUMMARY|SUMMARY|'
        self.section_headers += r'TECHNICAL SKILLS|TECHNOLOGIES|EMPLOYMENT HISTORY)'
        
    def clean_resume_text(self, text):
        """Clean and preprocess resume text"""
        if not isinstance(text, str):
            return ''
        
        # Decode bytes if needed
        if isinstance(text, bytes):
            text = text.decode('utf-8', errors='ignore')
        
        # Handle escaped newlines first
        text = text.replace('\\n', ' ')
        
        # Replace literal newlines with spaces
        text = text.replace('\n', ' ')
        
        # Convert to lowercase
        text = text.lower()
        
        # Replace common resume separators with spaces
        text = re.sub(r'[|•⋅·⚫∙◦≫→■●]', ' ', text)
        
        # Remove encoded characters (like \xe2\x80\x93)
        text = re.sub(r'\\x[a-fA-F0-9]{2}', ' ', text)
        
        # Replace multiple types of quotation marks with standard ones
        text = re.sub(r'[""''`]', '"', text)
        
        # Remove email addresses but keep domain for context
        text = re.sub(r'\S+@(\S+)', r'\1', text)
        
        # Remove URLs but keep domain
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 
                     lambda x: ' '.join(x.group().split('/')[-1].split('.')), text)
        
        # Remove phone numbers
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', ' phone ', text)
        
        # Remove postal codes while preserving state abbreviations
        text = re.sub(r'\b[A-Z]{2}\s+\d{5}(?:-\d{4})?\b', ' ', text)
        
        # Remove section headers to focus on content
        text = re.sub(self.section_headers, ' ', text, flags=re.IGNORECASE)
        
        # Standardize dates
        text = re.sub(r'\b(19|20)\d{2}\b', ' year ', text)
        text = re.sub(r'\b(jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|september|oct|october|nov|november|dec|december)\b',
                     ' month ', text)
        
        # Remove special characters but preserve hyphenated terms
        text = re.sub(r'[^a-zA-Z\s\-]', ' ', text)
        
        # Preserve hyphenated terms by replacing hyphen with space
        text = re.sub(r'-', ' ', text)
        
        # Handle multiple spaces
        text = ' '.join(text.split())
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize, preserving terms longer than 2 characters
        cleaned_tokens = []
        for token in tokens:
            if token not in self.stop_words and len(token) > 2:
                # Additional check for any remaining 'n' artifacts at the start of words
                if token.startswith('n') and len(token) > 1:
                    token = token[1:]
                if len(token) > 2:  # Check length again after potential 'n' removal
                    cleaned_tokens.append(self.lemmatizer.lemmatize(token))
        
        return ' '.join(cleaned_tokens)

    def process_and_check(self, text):
        """Process text and print before/after for problematic words"""
        cleaned = self.clean_resume_text(text)
        
        # Find words that might still have 'n' artifact
        original_words = text.split()
        cleaned_words = cleaned.split()
        
        problematic = [word for word in cleaned_words if word.startswith('n')]
        
        if problematic:
            print("Found potentially problematic words:")
            for word in problematic:
                print(f"Problem word: {word}")
        
        return cleaned

    def prepare_data(self, df, text_column, label_column, test_size=0.2, val_size=0.1):
        """Prepare data for training"""
        # Clean text data
        print("Cleaning text data...")
        df['cleaned_text'] = df[text_column].apply(self.process_and_check)
        
        # Fit tokenizer on training data
        print("Tokenizing text...")
        self.tokenizer.fit_on_texts(df['cleaned_text'])
        
        # Convert text to sequences
        sequences = self.tokenizer.texts_to_sequences(df['cleaned_text'])
        
        # Pad sequences
        padded_sequences = pad_sequences(sequences, maxlen=self.max_length, padding='post', truncating='post')
        
        # Encode labels
        # encoded_labels = self.label_encoder.fit_transform(df[label_column])
        encoded_labels = self.onehot_encoder.fit_transform(df[label_column].values.reshape(-1, 1))
        
        # Split data into train, validation, and test sets
        print("Splitting data...")
        X_train_temp, X_test, y_train_temp, y_test = train_test_split(
            padded_sequences, encoded_labels, test_size=test_size, stratify=encoded_labels, random_state=42
        )
        
        # Further split training data into train and validation
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_temp, y_train_temp, test_size=val_size_adjusted, stratify=y_train_temp, random_state=42
        )
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'vocab_size': len(self.tokenizer.word_index) + 1,
            'num_classes': encoded_labels.shape[1]  # Use this instead of len(self.onehot_encoder.categories_[0])
        }
    
    def encode_single_text(self, text):
        """Encode a single text for prediction"""
        cleaned_text = self.clean_text(text)
        sequence = self.tokenizer.texts_to_sequences([cleaned_text])
        padded_sequence = pad_sequences(sequence, maxlen=self.max_length, padding='post', truncating='post')
        return padded_sequence
    
    def decode_label(self, encoded_label):
        """Convert encoded label back to original category"""
        # return self.label_encoder.inverse_transform([encoded_label])[0]
        return self.onehot_encoder.inverse_transform([encoded_label])[0]

# Example usage and testing
'''
preprocessor = ResumeTextPreprocessor()

# Test with your sample text
sample_text = """your resume text"""
cleaned_text = preprocessor.process_and_check(sample_text)

# Print first few words to verify
print("First 50 words of cleaned text:")
print(' '.join(cleaned_text.split()[:50]))
'''


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import gensim
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize
import tensorflow as tf

class NLPPreprocessor:
    def __init__(self, max_words=10000, max_length=500, embedding_dim=100, TFDataset: bool = False, PyTorch: bool = False):
        self.max_words = max_words
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
        self.label_encoder = LabelEncoder()
        self.onehot_encoder = OneHotEncoder(sparse_output=False)
        self.word2vec_model = None
        self.TFDataset = TFDataset
        self.PyTorch = PyTorch
        
    def analyze_text_lengths(self, texts):
        """Analyze text lengths to help determine optimal max_length"""
        lengths = [len(text.split()) for text in texts]
        print(f"Average length: {np.mean(lengths):.2f}")
        print(f"Median length: {np.median(lengths):.2f}")
        print(f"95th percentile length: {np.percentile(lengths, 95):.2f}")
        print(f"Max length: {max(lengths)}")
        return lengths
    
    def analyze_vocabulary(self, texts):
        """Analyze vocabulary to help determine optimal max_words"""
        all_words = ' '.join(texts).split()
        word_freq = Counter(all_words)
        print(f"Total unique words: {len(word_freq)}")
        print(f"Words appearing only once: {sum(1 for count in word_freq.values() if count == 1)}")
        return word_freq
    
    def create_word2vec_embeddings(self, texts):
        """Create Word2Vec embeddings from the training data"""
        # Tokenize texts into sentences of words
        tokenized_texts = [word_tokenize(text.lower()) for text in texts]
        
        # Train Word2Vec model
        self.word2vec_model = Word2Vec(sentences=tokenized_texts,
                                    vector_size=self.embedding_dim,
                                    window=5,
                                    min_count=1,
                                    workers=4)
        
        # Create embedding matrix
        embedding_matrix = np.zeros((self.max_words, self.embedding_dim))
        for word, i in self.tokenizer.word_index.items():
            if i >= self.max_words:
                break
            try:
                embedding_matrix[i] = self.word2vec_model.wv[word]
            except KeyError:
                embedding_matrix[i] = np.random.normal(0, 0.1, self.embedding_dim)
                
        return embedding_matrix
    
    def prepare_data(self, texts, labels, use_word2vec=False):
        """Prepare text data and labels for model training"""
        # Analyze text characteristics
        print("Analyzing text characteristics...")
        self.analyze_text_lengths(texts)
        self.analyze_vocabulary(texts)
        
        # Fit tokenizer on texts
        print("Tokenizing texts...")
        self.tokenizer.fit_on_texts(texts)
        
        # Convert texts to sequences
        sequences = self.tokenizer.texts_to_sequences(texts)
        
        # Pad sequences
        padded_sequences = pad_sequences(sequences, maxlen=self.max_length,
                                    padding='post', truncating='post')
        
        # Encode labels
        # encoded_labels = self.label_encoder.fit_transform(labels)
        encoded_labels = self.onehot_encoder.fit_transform(labels)
        
        # Create Word2Vec embeddings if requested
        embedding_matrix = None
        if use_word2vec:
            print("Creating Word2Vec embeddings...")
            embedding_matrix = self.create_word2vec_embeddings(texts)
        
        # Split data
        print("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            padded_sequences, encoded_labels, 
            test_size=0.2, stratify=encoded_labels,
            random_state=42
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=0.2, stratify=y_train,
            random_state=42
        )
        
        if self.TFDataset:
            # Prepare TF datasets for training
            print("Creating TF datasets...")
            train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)) \
                .shuffle(10000).batch(32)
            val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(32)
            test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32)
            
            return {
                'train_dataset': train_dataset,
                'val_dataset': val_dataset,
                'test_dataset': test_dataset,
                'X_train': X_train,
                'X_val': X_val,
                'X_test': X_test,
                'y_train': y_train,
                'y_val': y_val,
                'y_test': y_test,
                'vocab_size': len(self.tokenizer.word_index) + 1,
                # 'num_classes': len(self.label_encoder.classes_),
                'num_classes': len(self.onehot_encoder.get_feature_names_out()),
                'embedding_matrix': embedding_matrix
            }


        if self.PyTorch:
            # Prepare Pytorch datasets for training
            print("Creating PyTorch datasets...")
            train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
            val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
            test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

            return {
                'train_loader': train_loader,
                'val_loader': val_loader,
                'test_loader': test_loader,
                'X_train': X_train,
                'X_val': X_val,
                'X_test': X_test,
                'y_train': y_train,
                'y_val': y_val,
                'y_test': y_test,
                'vocab_size': len(self.tokenizer.word_index) + 1,
                'num_classes': len(self.label_encoder.classes_),
                'embedding_matrix': embedding_matrix
            }

        
        return { # Returns simple splitted Numpy Array
                'X_train': X_train,
                'X_val': X_val,
                'X_test': X_test,
                'y_train': y_train,
                'y_val': y_val,
                'y_test': y_test,
                'vocab_size': len(self.tokenizer.word_index) + 1,
                'num_classes': len(self.label_encoder.classes_),
                'embedding_matrix': embedding_matrix
            }

    def predict_process(self, text):
        """Process a single text for prediction"""
        # Convert to sequence
        sequence = self.tokenizer.texts_to_sequences([text])
        # Pad sequence
        padded = pad_sequences(sequence, maxlen=self.max_length,
                            padding='post', truncating='post')
        return padded
    
    def decode_predictions(self, predictions):
        """Convert predicted label indices back to original labels"""
        return self.label_encoder.inverse_transform(predictions)

# Example usage:
'''
# Initialize preprocessor
preprocessor = NLPPreprocessor(max_words=10000, max_length=500, embedding_dim=100)

# Prepare your cleaned data
data = preprocessor.prepare_data(
    texts=df['cleaned_resume'],
    labels=df['category'],
    use_word2vec=True  # Set to True if you want to use Word2Vec embeddings
)

# Create and compile your model
model = create_model(
    vocab_size=data['vocab_size'],
    embedding_dim=100,
    num_classes=data['num_classes'],
    embedding_matrix=data['embedding_matrix']  # Pass this if using Word2Vec
)

# Train the model using the TF datasets
history = model.fit(
    data['train_dataset'],
    validation_data=data['val_dataset'],
    epochs=10
)

# Make predictions on new text
new_text = "Your cleaned resume text here"
processed_text = preprocessor.predict_process(new_text)
predictions = model.predict(processed_text)
predicted_label = preprocessor.decode_predictions(np.argmax(predictions, axis=1))
'''



## IMBALANCED DATASET PROCESSOR ##

import tensorflow as tf
import numpy as np
from typing import List, Dict, Union, Tuple
from collections import Counter
from sklearn.utils import resample

class ImbalancedNLPHandler:
    def __init__(self,
                 preprocessor: NLPPreprocessor,
                 strategy: str = "oversample"):
        """
        Combines NLPPreprocessor with imbalanced data handling.
        
        Args:
            preprocessor: Existing NLPPreprocessor instance
            strategy: "weighted", "oversample", or "undersample"
        """
        self.preprocessor = preprocessor
        self.strategy = strategy
    
    def calculate_class_weights(self, labels: List) -> Dict:
        """Calculate class weights for imbalanced data."""
        class_counts = Counter(labels)
        total = len(labels)
        weights = {cls: total / count for cls, count in class_counts.items()}
        
        # Normalize weights
        weight_sum = sum(weights.values())
        weights = {cls: weight / weight_sum for cls, weight in weights.items()}
        
        return weights
    
    def oversample(self, texts: List[str], labels: List) -> Tuple[List[str], List]:
        """Oversample minority classes."""
        df = pd.DataFrame({'text': texts, 'label': labels})
        class_counts = Counter(labels)
        majority_size = max(class_counts.values())
        
        balanced_dfs = []
        for label in class_counts.keys():
            class_df = df[df['label'] == label]
            if len(class_df) < majority_size:
                resampled = resample(class_df,
                                   replace=True,
                                   n_samples=majority_size,
                                   random_state=42)
                balanced_dfs.append(resampled)
            else:
                balanced_dfs.append(class_df)
        
        balanced_df = pd.concat(balanced_dfs)
        return balanced_df['text'].tolist(), balanced_df['label'].tolist()
    
    def undersample(self, texts: List[str], labels: List) -> Tuple[List[str], List]:
        """Undersample majority classes."""
        df = pd.DataFrame({'text': texts, 'label': labels})
        class_counts = Counter(labels)
        minority_size = min(class_counts.values())
        
        balanced_dfs = []
        for label in class_counts.keys():
            class_df = df[df['label'] == label]
            if len(class_df) > minority_size:
                resampled = resample(class_df,
                                   replace=False,
                                   n_samples=minority_size,
                                   random_state=42)
                balanced_dfs.append(resampled)
            else:
                balanced_dfs.append(class_df)
        
        balanced_df = pd.concat(balanced_dfs)
        return balanced_df['text'].tolist(), balanced_df['label'].tolist()

    def prepare_balanced_data(self, texts: List[str], labels: List, use_word2vec: bool = False):
        """Prepare data with imbalance handling."""
        # Apply balancing strategy if needed
        if self.strategy == "oversample":
            texts, labels = self.oversample(texts, labels)
        elif self.strategy == "undersample":
            texts, labels = self.undersample(texts, labels)
        
        # Use existing preprocessor to prepare data
        data = self.preprocessor.prepare_data(texts, labels, use_word2vec)
        
        # Add class weights if using weighted strategy
        if self.strategy == "weighted":
            class_weights = self.calculate_class_weights(data['y_train'])
            data['class_weights'] = class_weights
            
            # Update datasets to use sample weights
            weights = [class_weights[label] for label in data['y_train']]
            data['train_dataset'] = tf.data.Dataset.from_tensor_slices(
                (data['X_train'], data['y_train'], weights)
            ).shuffle(10000).batch(32)
        
        return data
    
    def get_class_distribution(self, labels: List) -> Dict:
        """Calculate class distribution percentages."""
        total = len(labels)
        class_counts = Counter(labels)
        return {label: count/total * 100 for label, count in class_counts.items()}

        