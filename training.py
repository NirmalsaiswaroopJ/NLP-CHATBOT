import os
import json
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# NLTK and Text Processing
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('wordnet')

# Hugging Face Transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TFAutoModelForSequenceClassification
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class IntentClassifier:
    def __init__(self, intents_file='data.json', model_name='bert-base-uncased'):
        """
        Initialize the Intent Classifier with advanced preprocessing and multiple model support
        
        Args:
            intents_file (str): Path to intents JSON file
            model_name (str): Hugging Face transformer model name
        """
        # Initialize components
        self.lemmatizer = WordNetLemmatizer()
        self.ignore_words = ['?', '!', '.', ',']
        
        # Load intents
        with open(intents_file, 'r') as f:
            self.intents = json.load(f)
        
        # Hugging Face Tokenizer and Model
        self.hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model_name = model_name
        
        # Preprocessing attributes
        self.words = []
        self.classes = []
        self.documents = []
        
        # Label Encoder
        self.label_encoder = LabelEncoder()
        
    def preprocess_data(self):
        """
        Advanced preprocessing of intent data
        """
        # Reset lists
        self.words = []
        self.classes = []
        self.documents = []
        
        # Process intents
        for intent in self.intents['intents']:
            for pattern in intent['patterns']:
                # Tokenize and lemmatize
                words = word_tokenize(pattern.lower())
                words = [self.lemmatizer.lemmatize(word) for word in words if word not in self.ignore_words]
                
                # Add to documents
                self.documents.append((words, intent['tag']))
                
                # Collect words
                self.words.extend(words)
                
                # Collect classes
                if intent['tag'] not in self.classes:
                    self.classes.append(intent['tag'])
        
        # Remove duplicates and sort
        self.words = sorted(list(set(self.words)))
        self.classes = sorted(list(set(self.classes)))
        
        # Encode labels
        self.label_encoder.fit(self.classes)
        
        # Save preprocessed data
        with open('texts.pkl', 'wb') as f:
            pickle.dump(self.words, f)
        with open('labels.pkl', 'wb') as f:
            pickle.dump(self.classes, f)
        
        return self
    
    def create_training_data(self):
        """
        Create training data with bag of words approach
        """
        # Prepare training data
        training_sentences = []
        training_labels = []
        
        for doc in self.documents:
            # Bag of words
            bag = [1 if w in doc[0] else 0 for w in self.words]
            
            # One-hot encode labels
            label = self.label_encoder.transform([doc[1]])[0]
            
            training_sentences.append(bag)
            training_labels.append(label)
        
        # Convert to numpy arrays
        X = np.array(training_sentences)
        y = keras.utils.to_categorical(training_labels)
        
        return X, y
    
    def build_keras_model(self, input_shape):
        """
        Build an advanced Neural Network model
        """
        model = Sequential([
            Dense(256, input_shape=(input_shape,), activation='relu'),
            BatchNormalization(),
            Dropout(0.6),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dense(len(self.classes), activation='softmax')
        ])
        
        # Compile with advanced optimizer
        model.compile(
            loss='categorical_crossentropy', 
            optimizer=Adam(learning_rate=0.001),
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self, epochs=300, validation_split=0.2):
        """
        Train the model with advanced training techniques
        """
        # Preprocess data
        self.preprocess_data()
        
        # Create training data
        X, y = self.create_training_data()
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_split, random_state=42)
        
        # Build model
        model = self.build_keras_model(X.shape[1])
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=15, 
            restore_best_weights=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.2, 
            patience=5, 
            min_lr=0.00001
        )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=16,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Save model
        model.save('keras_intent_model.h5')
        
        return model, history
    
    def train_huggingface_model(self, epochs=5):
        """
        Train a Hugging Face transformer model for intent classification
        """
        # Prepare data for Hugging Face
        texts = [' '.join(doc[0]) for doc in self.documents]
        labels = self.label_encoder.transform([doc[1] for doc in self.documents])
        
        # Tokenize
        encodings = self.hf_tokenizer(
            texts, 
            truncation=True, 
            padding=True, 
            max_length=128
        )
        
        # Convert to PyTorch tensors
        dataset = {
            'input_ids': torch.tensor(encodings['input_ids']),
            'attention_mask': torch.tensor(encodings['attention_mask']),
            'labels': torch.tensor(labels)
        }
        
        # Load pre-trained model
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, 
            num_labels=len(self.classes)
        )
        
        # TODO: Add training loop for PyTorch model
        # This would involve setting up a training loop with optimizer, 
        # potentially using Hugging Face Trainer for more advanced training
        
        return model
    
    def predict_intent(self, text):
        """
        Predict intent using both Keras and Hugging Face models
        """
        # Preprocess input
        words = word_tokenize(text.lower())
        words = [self.lemmatizer.lemmatize(word) for word in words if word not in self.ignore_words]
        
        # Create bag of words
        bag = [1 if w in words else 0 for w in self.words]
        
        # Load Keras model
        keras_model = keras.models.load_model('keras_intent_model.h5')
        
        # Predict using Keras model
        keras_pred = keras_model.predict(np.array([bag]))[0]
        keras_intent_index = np.argmax(keras_pred)
        keras_intent = self.classes[keras_intent_index]
        
        return keras_intent, keras_pred[keras_intent_index]

def main():
    # Initialize and train classifier
    classifier = IntentClassifier()
    
    # Train Keras model
    keras_model, history = classifier.train_model()
    
    # Optional: Train Hugging Face model
    # classifier.train_huggingface_model()
    
    print("Models trained successfully!")
    
    # Example prediction
    test_text = "I want to make an appointment"
    predicted_intent, confidence = classifier.predict_intent(test_text)
    print(f"Predicted Intent: {predicted_intent}")
    print(f"Confidence: {confidence*100:.2f}%")

if __name__ == "__main__":
    main()