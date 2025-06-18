import pandas as pd
import numpy as np
import pickle
import os
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

import matplotlib.pyplot as plt
import seaborn as sns


class SentimentModelTrainer:
    def __init__(self):
        self.vectorizer = None
        self.model = None
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
    
    def load_data(self, file_path):
        """
        Load Twitter/social media data from CSV file
        Expected columns: Name, Handle, Content, etc.
        """
        try:
            df = pd.read_csv(file_path)
            print(f"Data loaded successfully! Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            
            # Check if Content column exists
            if 'Content' not in df.columns:
                print(f"Available columns: {list(df.columns)}")
                raise ValueError("Column 'Content' not found in data")
            
            # Clean data - remove rows with missing content
            df = df.dropna(subset=['Content'])
            df = df[df['Content'].str.strip() != '']  # Remove empty strings
            
            print(f"Data after cleaning: {df.shape}")
            
            return df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def label_sentiment_manually(self, df, sample_size=None):
        """
        Manual labeling interface for sentiment annotation
        """
        if sample_size:
            df_sample = df.sample(n=min(sample_size, len(df)), random_state=42)
        else:
            df_sample = df.copy()
        
        labels = []
        
        print("\n=== SENTIMENT LABELING ===")
        print("For each post, enter:")
        print("0 = Negative")
        print("1 = Neutral") 
        print("2 = Positive")
        print("q = Quit labeling")
        print("-" * 50)
        
        for idx, row in df_sample.iterrows():
            print(f"\nPost {idx + 1}/{len(df_sample)}:")
            print(f"Author: {row.get('Name', 'Unknown')}")
            print(f"Content: {row['Content'][:200]}...")
            
            while True:
                label = input("Enter sentiment (0/1/2/q): ").strip().lower()
                if label == 'q':
                    print("Labeling stopped.")
                    break
                elif label in ['0', '1', '2']:
                    labels.append(int(label))
                    break
                else:
                    print("Invalid input. Please enter 0, 1, 2, or q")
            
            if label == 'q':
                break
        
        # Return only the labeled portion
        labeled_df = df_sample.iloc[:len(labels)].copy()
        labeled_df['sentiment'] = labels
        
        return labeled_df
    
    def auto_label_with_keywords(self, df):
        """
        Automatically label data using keyword-based approach
        (This creates initial labels that you can review/correct)
        """
        def get_sentiment(text):
            if not isinstance(text, str):
                return 1  # neutral for missing text
            
            text = text.lower()
            
            # Enhanced Swahili positive words
            positive_words = [
                'mzuri', 'nzuri', 'vizuri', 'maendeleo', 'mafanikio', 'furaha', 'raha',
                'kazi nzuri', 'faida', 'heri', 'baraka', 'amani', 'upendo', 'ushindi',
                'bora', 'safi', 'kamili', 'mzalendo', 'heshima', 'tumaini', 'mema',
                'fanaka', 'stahimilivu', 'imara', 'shwari', 'salama',
                'umoja', 'uwazi', 'uwajibikaji', 'uadilifu', 'demokrasia', 'uhuru', 'usawa',
                'kiongozi bora', 'sauti ya wananchi', 'suluhisho', 'mapinduzi ya kimaendeleo',
                'sera nzuri', 'utawala bora', 'ahadi kutimizwa', 'mgombea bora',
                'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
                'love', 'happy', 'proud', 'success', 'progress', 'development',
                'visionary', 'integrity', 'accountable', 'fairness', 'freedom', 'justice'
            ]


            # Enhanced Swahili negative words
            negative_words = [
                'mbaya', 'vibaya', 'tatizo', 'matatizo', 'changamoto', 'huzuni',
                'hasira', 'upungufu', 'kasoro', 'kosa', 'makosa', 'shida', 'fitina',
                'uongozi mbaya', 'rushwa', 'ufisadi', 'dhuluma', 'unyangavu', 'kero',
                'udhalilishaji', 'unyanyasaji', 'maovu', 'kinyama', 'kikatili',
                'ubaguzi', 'uhuni wa kisiasa', 'maneno ya chuki', 'mgombea dhaifu',
                'ahadi hewa', 'sera mbaya', 'udanganyifu', 'uongo', 'kutelekezwa',
                'kupuuza wananchi', 'ubadhirifu', 'kushindwa kutekeleza', 'ukandamizaji',
                'bad', 'terrible', 'awful', 'hate', 'angry', 'sad', 'problem',
                'issue', 'corruption', 'failed', 'failure', 'disappointed', 'poor',
                'dictatorship', 'injustice', 'lies', 'rigged', 'fraud', 'inequality'
            ]

                        
            positive_count = sum(1 for word in positive_words if word in text)
            negative_count = sum(1 for word in negative_words if word in text)
            
            if positive_count > negative_count:
                return 2  # Positive
            elif negative_count > positive_count:
                return 0  # Negative
            else:
                return 1  # Neutral
        
        df['sentiment'] = df['Content'].apply(get_sentiment)
        
        print("Auto-labeling completed!")
        print("Sentiment distribution:")
        print(df['sentiment'].value_counts())
        
        return df
    
    def preprocess_text(self, text):
        """Enhanced preprocessing for Swahili social media text"""
        if not isinstance(text, str):
            return ""
        
        # Keep basic Swahili punctuation
        text = re.sub(r'[^\w\s\'\-]', ' ', text.lower())
        
        # Remove URLs and user mentions but keep hashtag content
        text = re.sub(r'http\S+|www\S+|https\S+|@\w+', '', text)
        text = re.sub(r'#(\w+)', r'\1', text)  # Keep hashtag text
        
        # Remove isolated numbers/single letters
        text = ' '.join([w for w in text.split() if len(w) > 1 or w in ['ni', 'ya', 'na']])
        
        return text.strip()
    
    # (Add this import at the top of your script)

    def train_model(self, df):
        """Train logistic regression model with imbalance handling."""
        try:
            # Preprocess texts
            print("Preprocessing texts...")
            texts = []
            labels = []
            
            for idx, row in df.iterrows():
                processed_text = self.preprocess_text(row['Content'])
                if processed_text and len(processed_text.strip()) > 5:
                    texts.append(processed_text)
                    labels.append(row['sentiment'])
                else:
                    print(f"Skipping empty/short text: '{row['Content'][:50]}...'")
            
            if len(texts) < 10:
                print("Not enough valid texts found for training!")
                return False

            print(f"Training on {len(texts)} valid samples (out of {len(df)} total)")
            
            # Check label distribution
            unique_labels_list = np.unique(labels)
            label_counts = {label: labels.count(label) for label in unique_labels_list}
            print(f"Label distribution: {label_counts}")
            
            if any(count < 2 for count in label_counts.values()):
                print("Warning: Some classes have very few samples. Model performance may be poor.")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                texts, labels, test_size=0.2, random_state=42,
                stratify=labels if len(set(labels)) > 1 else None
            )
            
            # Vectorize text - IMPORTANT CHANGE HERE
            print("Creating TF-IDF vectors...")
            self.vectorizer = TfidfVectorizer(
                max_features=2000,
                min_df=1,  # Lowered to 1 to include rare but important features
                max_df=0.85,
                stop_words=None,
                analyzer='char_wb',
                ngram_range=(2, 4) # Focusing on slightly longer character sequences
            )
            
            X_train_vec = self.vectorizer.fit_transform(X_train)
            X_test_vec = self.vectorizer.transform(X_test)
            
            # Calculate class weights - IMPORTANT CHANGE HERE
            class_weights_values = compute_class_weight(
                'balanced',
                classes=np.unique(y_train),
                y=y_train
            )
            class_weights_dict = dict(zip(np.unique(y_train), class_weights_values))
            print(f"Using calculated class weights: {class_weights_dict}")

            # Train model
            print("Training logistic regression model...")
            self.model = LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight=class_weights_dict, # Using calculated weights
                C=1.0, # Using default regularization to start
                solver='liblinear'
            )
            
            self.model.fit(X_train_vec, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test_vec)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"\nModel Performance:")
            print(f"Accuracy: {accuracy:.3f}")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred,
                                        target_names=['Negative', 'Neutral', 'Positive'],
                                        zero_division=0))
            
            # Plot confusion matrix
            self.plot_confusion_matrix(y_test, y_pred)
            
            return True
            
        except Exception as e:
            print(f"Error training model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Negative', 'Neutral', 'Positive'],
                   yticklabels=['Negative', 'Neutral', 'Positive'])
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig('models/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self):
        """Save trained model and vectorizer"""
        try:
            if self.model and self.vectorizer:
                # Save model
                with open('models/sentiment_model.pkl', 'wb') as f:
                    pickle.dump(self.model, f)
                
                # Save vectorizer
                with open('models/vectorizer.pkl', 'wb') as f:
                    pickle.dump(self.vectorizer, f)
                
                print("Model and vectorizer saved successfully in 'models/' directory!")
                return True
            else:
                print("No model to save. Please train the model first.")
                return False
                
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def test_model(self, sample_texts=None):
        """Test the trained model with sample texts"""
        if not self.model or not self.vectorizer:
            print("Model not trained yet!")
            return
        
        if sample_texts is None:
            sample_texts = [
                "Hii ni furaha kubwa sana! Maendeleo mazuri.",
                "Hili ni tatizo kubwa, hali mbaya sana.",
                "Leo ni siku ya kawaida tu.",
                "Great progress in development!",
                "This is a terrible situation."
            ]
        
        print("\n=== MODEL TESTING ===")
        sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        
        for text in sample_texts:
            processed = self.preprocess_text(text)
            if processed:
                vec = self.vectorizer.transform([processed])
                pred = self.model.predict(vec)[0]
                prob = self.model.predict_proba(vec)[0]
                
                print(f"\nText: {text}")
                print(f"Prediction: {sentiment_map[pred]}")
                print(f"Confidence: {max(prob):.3f}")

# Main execution script
def main():
    trainer = SentimentModelTrainer()
    
    print("=== SENTIMENT ANALYSIS MODEL TRAINER ===")
    
    # Step 1: Load data
    # file_path = input("Enter path to your CSV file: ").strip()
    file_path = "2025-04-30_10-18-33_tweets_1-100.csv"  # default filename
    
    df = trainer.load_data(file_path)
    if df is None:
        return
    
    # Step 2: Label data
    print("\nChoose labeling method:")
    print("1. Auto-label using keywords (recommended for initial training)")
    print("2. Manual labeling (more accurate but time-consuming)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "2":
        sample_size = input("Enter number of samples to label (or press Enter for all): ").strip()
        sample_size = int(sample_size) if sample_size.isdigit() else None
        labeled_df = trainer.label_sentiment_manually(df, sample_size)
    else:
        labeled_df = trainer.auto_label_with_keywords(df)
    
    if len(labeled_df) == 0:
        print("No labeled data available. Exiting.")
        return
    
    # Step 3: Train model
    print("\nStarting model training...")
    success = trainer.train_model(labeled_df)
    
    if success:
        # Step 4: Save model
        trainer.save_model()
        
        # Step 5: Test model
        trainer.test_model()
        
        print("\n=== TRAINING COMPLETED ===")
        print("Files created:")
        print("- models/sentiment_model.pkl")
        print("- models/vectorizer.pkl")
        print("- models/confusion_matrix.png")
    else:
        print("Training failed. Please check your data and try again.")

if __name__ == "__main__":
    main()