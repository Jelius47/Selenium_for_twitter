from flask import Flask, jsonify, request
from flask_cors import CORS
import sqlite3
from datetime import datetime, timedelta
import time
import random
import io
import pandas as pd
import re
from collections import Counter
import ast
import uuid
import traceback
import json
import os
from threading import Thread
import pickle
import re

app = Flask(__name__)
# CORS(app)  # Enable CORS for all routes
# Enable CORS for all routes - THIS IS CRUCIAL
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:3000", "http://127.0.0.1:3000"],  # Add your frontend URLs
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})


class TwitterDataProcessor:
    def __init__(self, db_path='sentiment.db'):
        self.db_path = db_path
        
    def init_db(self):
        """Initialize database with all required columns"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Drop existing table to recreate with proper schema
        # c.execute('DROP TABLE IF EXISTS comments')
        
        # Create table with correct column names matching your CSV
        c.execute('''CREATE TABLE IF NOT EXISTS comments
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     name TEXT,
                     handle TEXT,
                     timestamp TEXT,
                     verified BOOLEAN DEFAULT 0,
                     content TEXT,
                     comments_count TEXT ,
                     retweets TEXT,
                     likes TEXT,
                     analytics TEXT,
                     tags TEXT DEFAULT '[]',
                     mentions TEXT DEFAULT '[]',
                     emojis TEXT DEFAULT '[]',
                     profile_image TEXT,
                     tweet_link TEXT,
                     tweet_id TEXT,
                     party TEXT,
                     sentiment_label TEXT,
                     date TEXT)''')

        c.execute('''CREATE UNIQUE INDEX IF NOT EXISTS unique_tweet_id ON comments (tweet_id)''')

        c.execute('''CREATE TABLE IF NOT EXISTS word_cloud
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                    word TEXT,
                    count INTEGER,
                    sentiment_type TEXT)''')
        
        c.execute('''CREATE TABLE IF NOT EXISTS processing_jobs
                    (job_id TEXT PRIMARY KEY,
                    filename TEXT,
                    status TEXT,
                    created_at TEXT,
                    processed_count INTEGER)''')
        
        conn.commit()
        conn.close()
    
    def extract_party_from_tags_and_content(self, tags_str, content):
        """Extract political party affiliation from tags and content"""
        content_lower = content.lower() if content else ""
        tags_lower = str(tags_str).lower() if tags_str else ""
        
        # Enhanced party hashtag mapping
        party_keywords = {
            'CCM': [
                '#mamayukokazi', '#mamayukokazini', '#mamaanafanikisha', 
                '#kazinaututunasongambele', '#sisinitanzania', '#matokeochanya',
                '#katibanasheria', '#nchiyangukwanza', '#kaziiendelee', 
                '#ssh', '#ccm', '#mslac', '#tanzaniyasamia', '#proudlyccm',
                'mama', 'samia', 'ccm', 'chama cha mapinduzi'
            ],
            'CHADEMA': [
                '#noreformsnoelection', '#katibampya', '#mwenyekitinyasa',
                '#sasabasi', '#nguvuyaumma', '#strongertogether',
                '#tusikubalitenakushirikichaguzibilakatibampya',
                'chadema', 'tundu', 'lissu', 'freeman mbowe', 'opposition'
            ],
            'CUF': [
                '#cuf', '#zanzibar', '#seif sharif', 'cuf', 'civic united front'
            ],
            'ACT-Wazalendo': [
                '#taifalawe', '#maslahiyawote', '#ajirakwawotemaslahiyawote',
                '#opereshenilinademokrasia', '#hifadhijamii', '#thefutureispurple',
                'act', 'wazalendo', 'zitto kabwe'
            ]
        }
        
        # Combine tags and content for analysis
        text_to_analyze = f"{tags_lower} {content_lower}"
        
        # Score each party based on keyword matches
        party_scores = {}
        for party, keywords in party_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in text_to_analyze:
                    score += 1
            if score > 0:
                party_scores[party] = score
        
        # Return party with highest score
        if party_scores:
            return max(party_scores, key=party_scores.get)
        
        return 'Unknown'
    
    

    

    # You must define a helper function with the SAME preprocessing logic used for training
    def _preprocess_text_for_prediction(text):
        """
        This helper function MUST EXACTLY MATCH the preprocessing from the training script.
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # The same multi-step cleaning process from the trainer
        text = re.sub(r'[^\w\s\'\-]', ' ', text)
        text = re.sub(r'http\S+|www\S+|https\S+|@\w+', '', text)
        text = re.sub(r'#(\w+)', r'\1', text)
        text = ' '.join([w for w in text.split() if len(w) > 1 or w in ['ni', 'ya', 'na']])
        
        return text.strip()

    # This should be a method in your class
    def analyze_sentiment(self, text, model_path=r'model_preparation\models\sentiment_model.pkl', vectorizer_path=r'model_preparation\models\vectorizer.pkl'):
        """
        Enhanced sentiment analysis for Swahili and English.
        """
        if not text or not text.strip():
            return 'Neutral'
        
        try:
            # Load model and vectorizer
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            with open(vectorizer_path, 'rb') as f:
                vectorizer = pickle.load(f)
            
            # --- THIS IS THE CRITICAL FIX ---
            # Use the correct, matching preprocessing function
            text = text.lower()
        
        # The same multi-step cleaning process from the trainer
            text = re.sub(r'[^\w\s\'\-]', ' ', text)
            text = re.sub(r'http\S+|www\S+|https\S+|@\w+', '', text)
            text = re.sub(r'#(\w+)', r'\1', text)
            text = ' '.join([w for w in text.split() if len(w) > 1 or w in ['ni', 'ya', 'na']])
            processed_text = text
            
            if not processed_text:
                return 'Neutral' # Return Neutral if text is empty after cleaning (e.g., was only a URL)
                
            # Transform text and predict
            text_vector = vectorizer.transform([processed_text])
            prediction = model.predict(text_vector)[0]
            
            # Map prediction to sentiment
            sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
            return sentiment_map.get(prediction, 'Neutral')
        
        except FileNotFoundError:
            print(f"MODEL/VECTORIZER NOT FOUND. Check paths. Falling back to keyword method.")
            return self.fallback_keyword_analysis(text) # Assuming fallback is another method
        except Exception as e:
            print(f"An error occurred with the ML model: {e}. Falling back to keyword method.")
            return self.fallback_keyword_analysis(text) # Assuming fallback is another method

    # You should move the fallback logic to its own method for cleaner code
    def fallback_keyword_analysis(self, text):
        text = text.lower()
        positive_words = [
            'mzuri', 'nzuri', 'vizuri', 'maendeleo', 'mafanikio', 'furaha', 'raha',
            'kazi', 'faida', 'heri', 'baraka', 'amani', 'upendo', 'ushindi', 'bora',
            'safi', 'kamili', 'mzalendo', 'heshima', 'tumaini', 'good', 'great',
            'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'happy',
            'proud', 'success', 'progress', 'development'
        ]
        negative_words = [
            'mbaya', 'vibaya', 'tatizo', 'matatizo', 'changamoto', 'huzuni',
            'hasira', 'upungufu', 'kasoro', 'kosa', 'makosa', 'shida',
            'uongozi mbaya', 'rushwa', 'ufisadi', 'dhuluma', 'unyangavu', 'bad',
            'terrible', 'awful', 'hate', 'angry', 'sad', 'problem', 'issue',
            'corruption', 'failed', 'failure', 'disappointed'
        ]
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)
        
        if positive_count > negative_count:
            return 'Positive'
        elif negative_count > positive_count:
            return 'Negative'
        else:
            return 'Neutral'
        
    def extract_words_for_cloud(self, text):
        """Extract meaningful words for word cloud"""
        if not text:
            return []
            
        # Remove URLs, mentions, hashtag symbols
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'[@#]', ' ', text)
        
        # Swahili and English stopwords
        stopwords = {
            'na', 'ya', 'wa', 'za', 'la', 'ni', 'si', 'kwa', 'katika', 'kwenye',
            'hii', 'huu', 'hao', 'wao', 'sisi', 'wewe', 'mimi', 'yeye', 'the',
            'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'between', 'among', 'is', 'are', 'was',
            'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
            'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must'
        }
        
        # Extract meaningful words (3+ characters, not stopwords)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        meaningful_words = [word for word in words if word not in stopwords]
        
        return meaningful_words

    def process_csv_file(self, file_stream):
        """Process CSV file and insert data into database"""
        try:
            # Generate job ID
            job_id = str(uuid.uuid4())[:8]
            filename = getattr(file_stream, 'filename', 'uploaded_file.csv')
            
            # Create processing job record
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            c.execute('''INSERT INTO processing_jobs 
                        (job_id, filename, status, created_at, processed_count) 
                        VALUES (?, ?, ?, ?, ?)''',
                    (job_id, filename, 'processing', 
                    datetime.now().isoformat(), 0))
            conn.commit()
            
            # Read CSV file
            df = pd.read_csv(file_stream)
            print(f"CSV columns: {df.columns.tolist()}")
            print(f"Processing {len(df)} rows...")
            
            processed_count = 0
            word_counter = Counter()
            sentiment_word_mapping = {'Positive': [], 'Negative': [], 'Neutral': []}
            
            for index, row in df.iterrows():
                try:
                    # Extract all fields from your CSV structure
                    name = str(row.get('Name', ''))
                    handle = str(row.get('Handle', ''))
                    timestamp = str(row.get('Timestamp', ''))
                    verified = bool(row.get('Verified', False))
                    content = str(row.get('Content', ''))
                    comments_count = str(row.get('Comments', 0))
                    retweets = str(row.get('Retweets', 0))
                    likes = str(row.get('Likes', 0))
                    analytics = str(row.get('Analytics', 0))
                    tags = str(row.get('Tags', '[]'))
                    mentions = str(row.get('Mentions', '[]'))
                    emojis = str(row.get('Emojis', '[]'))
                    profile_image = str(row.get('Profile Image', ''))
                    tweet_link = str(row.get('Tweet Link', ''))
                    tweet_id = str(row.get('Tweet ID', ''))
                    
                    # Skip empty content
                    if not content or content == 'nan' or content.strip() == '':
                        continue
                    
                    # Extract party affiliation
                    party = self.extract_party_from_tags_and_content(tags, content)
                    
                    # Analyze sentiment
                    sentiment = self.analyze_sentiment(content)
                    
                    # Format timestamp
                    try:
                        if timestamp and timestamp != 'nan':
                            parsed_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                            formatted_timestamp = parsed_time.isoformat()
                        else:
                            formatted_timestamp = datetime.now().isoformat()
                    except:
                        formatted_timestamp = datetime.now().isoformat()
                    
                    # Insert into database with all fields
                    c.execute('''INSERT OR IGNORE INTO comments 
                                (name, handle, timestamp, verified, content, comments_count, 
                                 retweets, likes, analytics, tags, mentions, emojis, 
                                 profile_image, tweet_link, tweet_id, party, sentiment_label, date) 
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                            (name, handle, timestamp, verified, content, comments_count,
                             retweets, likes, analytics, tags, mentions, emojis,
                             profile_image, tweet_link, tweet_id, party, sentiment, formatted_timestamp))
                                        
                    # Extract words for word cloud
                    words = self.extract_words_for_cloud(content)
                    for word in words:
                        word_counter[word] += 1
                        sentiment_word_mapping[sentiment].append(word)
                    
                    processed_count += 1
                    
                    if processed_count % 100 == 0:
                        print(f"Processed {processed_count} records...")
                        
                except Exception as e:
                    print(f"Error processing row {index}: {e}")
                    continue
            
            # Update word cloud
            self.update_word_cloud(c, word_counter, sentiment_word_mapping)
            
            # Update job status
            c.execute('''UPDATE processing_jobs 
                        SET status = ?, processed_count = ? 
                        WHERE job_id = ?''',
                    ('completed', processed_count, job_id))
            
            conn.commit()
            conn.close()
            
            print(f"Processing completed! Processed {processed_count} records.")
            return job_id, processed_count
            
        except Exception as e:
            # Update job status to failed
            try:
                conn = sqlite3.connect(self.db_path)
                c = conn.cursor()
                c.execute('''UPDATE processing_jobs 
                            SET status = ? 
                            WHERE job_id = ?''',
                        ('failed', job_id))
                conn.commit()
                conn.close()
            except:
                pass
            
            print(f"Error processing file: {e}")
            print(f"FULL TRACEBACK: {traceback.format_exc()}")
            raise e
        
    def update_word_cloud(self, cursor, word_counter, sentiment_mapping):
        """Update word cloud table with new word counts"""
        # Clear existing word cloud data
        cursor.execute('DELETE FROM word_cloud')
        
        # Count words by sentiment
        for sentiment, words in sentiment_mapping.items():
            sentiment_lower = sentiment.lower()
            word_count = Counter(words)
            
            for word, count in word_count.most_common(50):  # Top 50 words per sentiment
                cursor.execute('''INSERT INTO word_cloud (word, count, sentiment_type) 
                                VALUES (?, ?, ?)''',
                             (word, count, sentiment_lower))

# Initializing database
processor = TwitterDataProcessor("sentiment.db")



class ScrapingJob:
    def __init__(self, job_id, parameters, target_type, target_value, max_tweets_requested, csv_file_path=None):
        self.id = None
        self.job_id = job_id
        self.status = 'pending'
        self.parameters = parameters
        self.tweets_scraped = 0
        self.started_at = datetime.utcnow()
        self.completed_at = None
        self.error_message = None
        self.target_type = target_type
        self.target_value = target_value
        self.max_tweets_requested = max_tweets_requested
        self.progress_percentage = 0.0
        self.result_summary = None
        self.duration_seconds = None
        self.csv_file_path = csv_file_path
        self.tweets = []

    def to_dict(self):
        return {
            'id': self.id,
            'job_id': self.job_id,
            'status': self.status,
            'target_type': self.target_type,
            'target_value': self.target_value,
            'tweets_scraped': self.tweets_scraped,
            'max_tweets_requested': self.max_tweets_requested,
            'progress_percentage': self.progress_percentage,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'duration_seconds': self.duration_seconds,
            'error_message': self.error_message
        }

    def update_progress(self, tweets_scraped, percentage=None):
        self.tweets_scraped = tweets_scraped
        if percentage is not None:
            self.progress_percentage = min(100.0, max(0.0, float(percentage)))
        elif self.max_tweets_requested and self.max_tweets_requested > 0:
            self.progress_percentage = min(100.0, (tweets_scraped / self.max_tweets_requested) * 100)

    def mark_completed(self, tweets_scraped, result_summary=None):
        self.status = 'completed'
        self.tweets_scraped = tweets_scraped
        self.completed_at = datetime.utcnow()
        self.progress_percentage = 100.0
        if result_summary:
            self.result_summary = json.dumps(result_summary)
        if self.started_at:
            self.duration_seconds = (self.completed_at - self.started_at).total_seconds()

    def mark_failed(self, error_message):
        self.status = 'failed'
        self.error_message = str(error_message)
        self.completed_at = datetime.utcnow()
        if self.started_at:
            self.duration_seconds = (self.completed_at - self.started_at).total_seconds()

    def mark_cancelled(self):
        self.status = 'cancelled'
        self.completed_at = datetime.utcnow()
        if self.started_at:
            self.duration_seconds = (self.completed_at - self.started_at).total_seconds()

    def save_to_db(self):
        try:
            conn = sqlite3.connect('sentiment.db')
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO scraping_jobs 
                (job_id, status, parameters, target_type, target_value, max_tweets_requested, 
                 tweets_scraped, progress_percentage, started_at, completed_at, error_message, 
                 result_summary, duration_seconds, csv_file_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                self.job_id, self.status, self.parameters, self.target_type, self.target_value,
                self.max_tweets_requested, self.tweets_scraped, self.progress_percentage,
                self.started_at, self.completed_at, self.error_message, self.result_summary,
                self.duration_seconds, self.csv_file_path
            ))
            conn.commit()
        except Exception as e:
            print(f"Error saving job to DB: {e}")

    def update_in_db(self):
        try:
            conn = sqlite3.connect('sentiment.db')
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE scraping_jobs 
                SET status=?, tweets_scraped=?, progress_percentage=?, completed_at=?, 
                    error_message=?, result_summary=?, duration_seconds=?
                WHERE job_id=?
            """, (
                self.status, self.tweets_scraped, self.progress_percentage, self.completed_at,
                self.error_message, self.result_summary, self.duration_seconds, self.job_id
            ))
            conn.commit()
        except sqlite3.Error as e:
            print(f"Database error: {e}")
        except Exception as e:
            print(f"Exception in update_in_db: {e}")
        finally:
            if conn:
                conn.close()

    @classmethod
    def get_by_job_id(cls, job_id):
        try:
            conn = sqlite3.connect('sentiment.db')
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM scraping_jobs WHERE job_id=?", (job_id,))
            row = cursor.fetchone()
            if row:
                job = cls.__new__(cls)
                job.job_id = row[1]
                job.status = row[2]
                job.parameters = row[3]
                job.target_type = row[4]
                job.target_value = row[5]
                job.max_tweets_requested = row[6]
                job.tweets_scraped = row[7]
                job.progress_percentage = row[8]
                job.started_at = row[9]
                job.completed_at = row[10]
                job.error_message = row[11]
                job.result_summary = row[12]
                job.duration_seconds = row[13]
                job.csv_file_path = row[14]
                return job
        except sqlite3.Error as e:
            print(f"Database error: {e}")
        except Exception as e:
            print(f"Exception in get_by_job_id: {e}")
        finally:
            if conn:
                conn.close()
        return None

def extract_hashtags(text):
    if not text:
        return []
    return list(set(re.findall(r'#(\w+)', text.lower())))

def extract_mentions(text):
    if not text:
        return []
    return list(set(re.findall(r'@(\w+)', text.lower())))


def run_scraper(job_id, data):
    with current_app.app_context():
        job = None
        try:
            job = ScrapingJob.get_by_job_id(job_id)
            if not job:
                current_app.logger.error(f"Job {job_id} not found")
                return
            
            job.status = 'running'
            job.update_in_db()

            from scraper.twitter_scraper import Twitter_Scraper

            USER_MAIL = data.get('email')
            USER_UNAME = data.get('username')
            USER_PASSWORD = data.get('password')

            if not all([USER_MAIL, USER_UNAME, USER_PASSWORD]):
                raise ValueError("Email, username, and password are required")

            max_tweets = min(int(data.get('tweets', 50)), 1000)
            target = data.get('target_username') or data.get('hashtag') or data.get('query')
            if not target:
                raise ValueError("No target specified")

            scraper = Twitter_Scraper(
                mail=USER_MAIL,
                username=USER_UNAME,
                password=USER_PASSWORD,
            )
            
            scraper.login()

            tweets = []
            for i, tweet_data in enumerate(scraper.scrape_tweets(
                max_tweets=max_tweets,
                scrape_username=data.get('target_username'),
                scrape_hashtag=data.get('hashtag'),
                scrape_query=data.get('query'),
                scrape_latest=data.get('latest', False),
                scrape_top=data.get('top', False)
            )):
                tweets.append(tweet_data)
                
                if (i + 1) % 10 == 0:
                    job.update_progress(i + 1)
                    job.update_in_db()

            conn = sqlite3.connect('sentiment.db')
            cursor= conn.cursor()
            for tweet_data in tweets:
                cursor.execute("""
                    INSERT INTO tweets 
                    (tweet_id, name, handle, timestamp, verified, content, comments, 
                     retweets, likes, analytics, profile_image, tweet_link, job_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    tweet_data.get('tweet_id', str(int(time.time() * 1000))),
                    tweet_data.get('name', ''),
                    tweet_data.get('handle', '').lower() if tweet_data.get('handle') else '',
                    datetime.fromtimestamp(tweet_data.get('timestamp', time.time())),
                    tweet_data.get('verified', False),
                    tweet_data.get('content', ''),
                    tweet_data.get('comments', 0),
                    tweet_data.get('retweets', 0),
                    tweet_data.get('likes', 0),
                    tweet_data.get('analytics', 0),
                    tweet_data.get('profile_image', ''),
                    tweet_data.get('tweet_link', ''),
                    job_id
                ))

                tweet_id = tweet_data.get('tweet_id', str(int(time.time() * 1000)))
                for hashtag in extract_hashtags(tweet_data.get('content', '')):
                    cursor.execute("INSERT INTO hashtags (tweet_id, hashtag) VALUES (?, ?)",
                                 (tweet_id, hashtag))
                
                for mention in extract_mentions(tweet_data.get('content', '')):
                    cursor.execute("INSERT INTO mentions (tweet_id, mention) VALUES (?, ?)",
                                 (tweet_id, mention))

            cursor.commit()

            job.mark_completed(
                tweets_scraped=len(tweets),
                result_summary={
                    'target': target,
                    'tweets_scraped': len(tweets),
                    'success': True
                }
            )
            job.update_in_db()

        except Exception as e:
            current_app.logger.error(f"Job {job_id} failed: {str(e)}")
            if job:
                job.mark_failed(str(e))
                job.update_in_db()
            raise

@app.route('/api/scrape', methods=['POST'])
def start_scraping():
    try:
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        if not all([data.get('email'), data.get('username'), data.get('password')]):
            return jsonify({
                'error': 'Email, username, and password are required'
            }), 400
        
        if not any([data.get('target_username'), data.get('hashtag'), data.get('query')]):
            return jsonify({
                'error': 'Must specify one of: target_username, hashtag, or query'
            }), 400
        
        try:
            max_tweets = int(data.get('tweets', 50))
            if max_tweets <= 0 or max_tweets > 1000:
                return jsonify({
                    'error': 'tweets parameter must be between 1 and 1000'
                }), 400
        except (ValueError, TypeError):
            return jsonify({'error': 'tweets parameter must be a valid integer'}), 400
        
        job_id = f"job_{int(time.time())}_{hash(json.dumps(data, sort_keys=True)) % 10000:04d}"
        
        if data.get('target_username'):
            target_type = 'username'
            target_value = data.get('target_username')
        elif data.get('hashtag'):
            target_type = 'hashtag'
            target_value = data.get('hashtag')
        else:
            target_type = 'query'
            target_value = data.get('query')
        
        job = ScrapingJob(
            job_id=job_id,
            parameters=json.dumps(data),
            target_type=target_type,
            target_value=target_value,
            max_tweets_requested=max_tweets
        )
        
        job.save_to_db()
        
        thread = Thread(target=run_scraper, args=(job_id, data))
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'job_id': job_id,
            'status': 'started',
            'message': 'Scraping job started successfully',
            'details': {
                'target_type': target_type,
                'target': target_value,
                'max_tweets': max_tweets
            }
        }), 202
    
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        current_app.logger.error(f"Error starting scraping job: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/scrape/<job_id>/status', methods=['GET'])
def get_job_status_(job_id):
    try:
        job = ScrapingJob.get_by_job_id(job_id)
        if not job:
            return jsonify({'error': 'Job not found'}), 404
        
        return jsonify(job.to_dict()), 200
    
    except Exception as e:
        current_app.logger.error(f"Error getting job status: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/scrape/<job_id>/cancel', methods=['POST'])
def cancel_job(job_id):
    try:
        job = ScrapingJob.get_by_job_id(job_id)
        if not job:
            return jsonify({'error': 'Job not found'}), 404
        
        if job.status in ['completed', 'failed', 'cancelled']:
            return jsonify({'error': f'Job already {job.status}'}), 400
        
        job.mark_cancelled()
        job.update_in_db()
        
        return jsonify({
            'job_id': job_id,
            'status': 'cancelled',
            'message': 'Job cancelled successfully'
        }), 200
    
    except Exception as e:
        current_app.logger.error(f"Error cancelling job: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500
# API Routes
@app.route('/')
def index():
    return jsonify({
        'message': 'Twitter Analytics API',
        'version': '1.0.0',
        'status': 'operational',
        'endpoints': {
            '/api/tweets': 'GET tweets with filters',
            '/api/scrape': 'POST new scraping job',
            '/api/jobs': 'GET all jobs',
            '/api/jobs/<job_id>': 'GET job status',
            '/api/analytics': 'GET engagement analytics',
            '/api/hashtags': 'GET hashtag analytics',
            '/api/users': 'GET user analytics'
        }
    })


@app.route('/api/analytics', methods=['GET'])
def get_analytics():
    """Get analytics data with filtering support"""
    try:
        import sqlite3
        
        # Get query parameters
        parties = request.args.get('parties', '').split(',') if request.args.get('parties') else []
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        conn = sqlite3.connect('sentiment.db')
        c = conn.cursor()
        
        # First, let's check what data exists in the table
        c.execute("SELECT COUNT(*) FROM comments")
        total_records = c.fetchone()[0]
        print(f"Total records in comments table: {total_records}")
        
        c.execute("SELECT COUNT(*) FROM comments WHERE content IS NOT NULL AND content != ''")
        records_with_content = c.fetchone()[0]
        print(f"Records with content: {records_with_content}")
        
        c.execute("SELECT COUNT(*) FROM comments WHERE sentiment_label IS NOT NULL AND sentiment_label != ''")
        records_with_sentiment = c.fetchone()[0]
        print(f"Records with sentiment: {records_with_sentiment}")
        
        c.execute("SELECT DISTINCT sentiment_label FROM comments WHERE sentiment_label IS NOT NULL")
        sentiment_labels = c.fetchall()
        print(f"Available sentiment labels: {sentiment_labels}")
        
        c.execute("SELECT DISTINCT party FROM comments WHERE party IS NOT NULL AND party != ''")
        available_parties = c.fetchall()
        print(f"Available parties: {available_parties}")
        
        # Build base query with more flexible filtering
        base_query = "FROM comments WHERE 1=1"
        params = []
        
        # Only filter by content if we have records with content
        if records_with_content > 0:
            base_query += " AND content IS NOT NULL AND content != ''"
        
        # Only filter by sentiment if we have records with sentiment
        if records_with_sentiment > 0:
            base_query += " AND sentiment_label IS NOT NULL AND sentiment_label != ''"
        
        if parties and parties != [''] and parties != ['null']:
            # Clean up party names (remove empty strings)
            clean_parties = [p.strip() for p in parties if p.strip()]
            if clean_parties:
                placeholders = ','.join(['?' for _ in clean_parties])
                base_query += f" AND party IN ({placeholders})"
                params.extend(clean_parties)
        
        if start_date:
            base_query += " AND date >= ?"
            params.append(start_date)
        
        if end_date:
            base_query += " AND date <= ?"
            params.append(end_date)
        
        print(f"Base query: {base_query}")
        print(f"Params: {params}")
        
        # Get overall summary counts with more robust handling
        summary_query = f"""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN LOWER(TRIM(sentiment_label)) = 'positive' THEN 1 ELSE 0 END) as positive,
                SUM(CASE WHEN LOWER(TRIM(sentiment_label)) = 'negative' THEN 1 ELSE 0 END) as negative,
                SUM(CASE WHEN LOWER(TRIM(sentiment_label)) = 'neutral' THEN 1 ELSE 0 END) as neutral
            {base_query}
        """
        
        c.execute(summary_query, params)
        summary_result = c.fetchone()
        print(f"Summary query result: {summary_result}")
        
        # Get party sentiment breakdown
        party_query = f"""
            SELECT 
                party,
                COUNT(*) as total,
                SUM(CASE WHEN LOWER(TRIM(sentiment_label)) = 'positive' THEN 1 ELSE 0 END) as positive,
                SUM(CASE WHEN LOWER(TRIM(sentiment_label)) = 'negative' THEN 1 ELSE 0 END) as negative,
                SUM(CASE WHEN LOWER(TRIM(sentiment_label)) = 'neutral' THEN 1 ELSE 0 END) as neutral
            {base_query} AND party IS NOT NULL AND party != ''
            GROUP BY party
            ORDER BY total DESC
        """
        
        c.execute(party_query, params)
        party_sentiment_data = c.fetchall()
        print(f"Party sentiment data: {party_sentiment_data}")
        
        # Get overall sentiment distribution
        sentiment_query = f"""
            SELECT 
                TRIM(sentiment_label) as sentiment_label, 
                COUNT(*) as count 
            {base_query} 
            GROUP BY TRIM(sentiment_label)
            ORDER BY CASE LOWER(TRIM(sentiment_label))
                WHEN 'positive' THEN 1
                WHEN 'negative' THEN 2
                WHEN 'neutral' THEN 3
                ELSE 4
            END
        """
        
        c.execute(sentiment_query, params)
        sentiment_data = c.fetchall()
        print(f"Sentiment distribution data: {sentiment_data}")
        
        # Get daily trends - simplified query
        daily_query = f"""
            SELECT 
                date,
                SUM(CASE WHEN LOWER(TRIM(sentiment_label)) = 'positive' THEN 1 ELSE 0 END) as positive,
                SUM(CASE WHEN LOWER(TRIM(sentiment_label)) = 'negative' THEN 1 ELSE 0 END) as negative,
                SUM(CASE WHEN LOWER(TRIM(sentiment_label)) = 'neutral' THEN 1 ELSE 0 END) as neutral
            {base_query} AND date IS NOT NULL AND date != ''
            GROUP BY date
            ORDER BY date DESC
            LIMIT 7
        """
        
        c.execute(daily_query, params)
        daily_trends_data = c.fetchall()
        print(f"Daily trends data: {daily_trends_data}")
        
        conn.close()
        
        # Format the response exactly like the sample structure
        response = {
            'analytics': {
                'summary': {
                    'total_comments': summary_result[0] if summary_result and summary_result[0] is not None else 0,
                    'positive_count': summary_result[1] if summary_result and len(summary_result) > 1 and summary_result[1] is not None else 0,
                    'negative_count': summary_result[2] if summary_result and len(summary_result) > 2 and summary_result[2] is not None else 0,
                    'neutral_count': summary_result[3] if summary_result and len(summary_result) > 3 and summary_result[3] is not None else 0
                },
                'party_sentiment': [
                    {
                        'party': row[0] if row[0] else 'Unknown',
                        'total': row[1] if row[1] is not None else 0,
                        'positive': row[2] if row[2] is not None else 0,
                        'negative': row[3] if row[3] is not None else 0,
                        'neutral': row[4] if row[4] is not None else 0
                    }
                    for row in party_sentiment_data if row and len(row) >= 5
                ],
                'overall_sentiment': [
                    {
                        'sentiment': row[0] if row[0] else 'Unknown',
                        'count': row[1] if row[1] is not None else 0
                    }
                    for row in sentiment_data if row and len(row) >= 2 and row[0] and row[0].strip()
                ],
                'daily_trends': [
                    {
                        'date': row[0].split('T')[0] if row[0] and 'T' in row[0] else row[0] if row[0] else 'Unknown',
                        'positive': row[1] if row[1] is not None else 0,
                        'negative': row[2] if row[2] is not None else 0,
                        'neutral': row[3] if row[3] is not None else 0
                    }
                    for row in daily_trends_data if row and len(row) >= 4
                ]
            }
        }
        
        print("Final response:", response)
        return jsonify(response)
        
    except Exception as e:
        print(f"Analytics error: {str(e)}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'message': f'Error retrieving analytics: {str(e)}'
        }), 500
        
@app.route('/api/wordcloud', methods=['GET'])
def get_wordcloud():
    """Get word cloud data with filtering"""
    try:
        import sqlite3
        
        # Get query parameters
        parties = request.args.get('parties', '').split(',') if request.args.get('parties') else []
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        conn = sqlite3.connect('sentiment.db')
        c = conn.cursor()
        
        # Build base query
        base_query = "FROM word_cloud wc JOIN comments c ON 1=1 WHERE c.content IS NOT NULL"
        params = []
        
        if parties and parties != ['']:
            placeholders = ','.join(['?' for _ in parties])
            base_query += f" AND c.party IN ({placeholders})"
            params.extend(parties)
        
        if start_date:
            base_query += " AND c.date >= ?"
            params.append(start_date)
        
        if end_date:
            base_query += " AND c.date <= ?"
            params.append(end_date)
        
        # Get word cloud data by sentiment
        result = {}
        for sentiment in ['positive', 'negative', 'neutral']:
            c.execute(f"""
                SELECT word, SUM(count) as total_count 
                FROM word_cloud 
                WHERE sentiment_type = ? 
                GROUP BY word 
                ORDER BY total_count DESC 
                LIMIT 20
            """, (sentiment,))
            
            words_data = c.fetchall()
            result[sentiment] = [
                {'word': word, 'count': count}
                for word, count in words_data
            ]
        
        conn.close()
        
        return jsonify({
            'success': True,
            **result
        })
        
    except Exception as e:
        print(f"WordCloud error: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Error retrieving word cloud: {str(e)}'
        }), 500
@app.route('/api/comments', methods=['GET'])
def get_comments():
    """Get processed comments with pagination"""
    try:
        import sqlite3
        
        # Get query parameters
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 20))
        sentiment = request.args.get('sentiment')
        party = request.args.get('party')
        
        offset = (page - 1) * per_page
        
        conn = sqlite3.connect('sentiment.db')
        c = conn.cursor()
        
        # Build query with filters
        query = '''SELECT content, party, sentiment_label, date, handle, name, likes, retweets 
                   FROM comments WHERE content IS NOT NULL AND content != ""'''
        conditions = []
        params = []
        
        if sentiment:
            conditions.append('sentiment_label = ?')
            params.append(sentiment)
        
        if party:
            conditions.append('party = ?')
            params.append(party)
        
        if conditions:
            query += ' AND ' + ' AND '.join(conditions)
        
        query += ' ORDER BY date DESC LIMIT ? OFFSET ?'
        params.extend([per_page, offset])
        
        c.execute(query, params)
        comments = c.fetchall()
        
        # Get total count
        count_query = 'SELECT COUNT(*) FROM comments WHERE content IS NOT NULL AND content != ""'
        count_params = []
        if conditions:
            count_query += ' AND ' + ' AND '.join(conditions)
            count_params = params[:-2]
        
        c.execute(count_query, count_params)
        total_count = c.fetchone()[0]
        
        conn.close()
        
        # Format response
        formatted_comments = []
        for comment in comments:
            formatted_comments.append({
                'comment': comment[0],
                'party': comment[1],
                'sentiment_label': comment[2],
                'date': comment[3],
                'handle': comment[4],
                'name': comment[5],
                'likes': comment[6],
                'retweets': comment[7]
            })
        # print(formatted_comments)
        return jsonify({
            'success': True,
            'comments': formatted_comments,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': total_count,
                'pages': (total_count + per_page - 1) // per_page
            }
        })
        
    except Exception as e:
        print(f"Comments error: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Error retrieving comments: {str(e)}'
        }), 500
    
@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and file.filename.endswith('.csv'):
        try:
            processor = TwitterDataProcessor()
            processor.init_db()  # Ensure DB is initialized
            
            # Process the file directly from memory
            job_id, processed_count = processor.process_csv_file(file)
            
            return jsonify({
                'success': True,
                'job_id': job_id,
                'processed_count': processed_count,
                'message': f'Successfully processed {processed_count} records'
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': 'Processing failed',
                'details': str(e)
            }), 500
    
    return jsonify({'error': 'Invalid file type. Only CSV accepted.'}), 400


@app.route('/api/parties', methods=['GET'])
def get_parties():
    """Get list of available parties"""
    try:
        import sqlite3
        
        conn = sqlite3.connect('sentiment.db')
        c = conn.cursor()
        
        c.execute('SELECT DISTINCT party FROM comments WHERE party IS NOT NULL ORDER BY party')
        parties = [row[0] for row in c.fetchall()]
        
        conn.close()
        
        return jsonify({
            'success': True,
            'parties': parties
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error retrieving parties: {str(e)}'
        }), 500

@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    """Get comprehensive statistics - formatted like demo data"""
    try:
        import sqlite3
        
        conn = sqlite3.connect('sentiment.db')
        c = conn.cursor()
        
        # Get correct column names
        c.execute("PRAGMA table_info(comments)")
        columns = c.fetchall()
        
        sentiment_col = None
        for col in columns:
            if 'sentiment' in col[1].lower():
                sentiment_col = col[1]
                break
        
        if not sentiment_col:
            sentiment_col = 'sentiment_label'
        
        # Get total comments
        c.execute('SELECT COUNT(*) FROM comments')
        total_comments = c.fetchone()[0]
        
        # Get sentiment distribution
        c.execute(f'SELECT {sentiment_col}, COUNT(*) FROM comments GROUP BY {sentiment_col}')
        sentiment_distribution = c.fetchall()
        
        # Get party distribution
        c.execute('SELECT party, COUNT(*) FROM comments WHERE party IS NOT NULL GROUP BY party ORDER BY COUNT(*) DESC')
        party_distribution = c.fetchall()
        
        # Get recent jobs (if you have a processing_jobs table)
        try:
            c.execute('''SELECT job_id, filename, status, created_at, processed_count 
                        FROM processing_jobs 
                        ORDER BY created_at DESC 
                        LIMIT 5''')
            recent_jobs = c.fetchall()
        except:
            recent_jobs = []
        
        conn.close()
        
        return jsonify({
            'success': True,
            'statistics': {
                'total_comments': total_comments,
                'sentiment_distribution': [
                    {'sentiment': sentiment, 'count': count} 
                    for sentiment, count in sentiment_distribution if sentiment
                ],
                'party_distribution': [
                    {'party': party, 'count': count} 
                    for party, count in party_distribution
                ],
                'recent_jobs': [
                    {
                        'job_id': job[0],
                        'filename': job[1],
                        'status': job[2],
                        'created_at': job[3],
                        'processed_count': job[4]
                    }
                    for job in recent_jobs
                ]
            }
        })
        
    except Exception as e:
        print(f"Statistics error: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Error retrieving statistics: {str(e)}'
        }), 500

# Add new route to get data in demo format for frontend compatibility
@app.route('/api/demo-format', methods=['GET'])
def get_demo_format():
    """Get all data in demo format for easy frontend integration"""
    try:
        import sqlite3
        
        conn = sqlite3.connect('sentiment.db')
        c = conn.cursor()
        
        # Get summary
        c.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN sentiment_lable = 'Positive' THEN 1 ELSE 0 END) as positive,
                SUM(CASE WHEN sentiment_lable = 'Negative' THEN 1 ELSE 0 END) as negative,
                SUM(CASE WHEN sentiment_lable = 'Neutral' THEN 1 ELSE 0 END) as neutral
            FROM comments
        """)
        summary = c.fetchone()
        
        # Get party sentiment
        c.execute("""
            SELECT 
                party,
                COUNT(*) as total,
                SUM(CASE WHEN sentiment_lable = 'Positive' THEN 1 ELSE 0 END) as positive,
                SUM(CASE WHEN sentiment_lable = 'Negative' THEN 1 ELSE 0 END) as negative,
                SUM(CASE WHEN sentiment_lable = 'Neutral' THEN 1 ELSE 0 END) as neutral
            FROM comments 
            WHERE party IS NOT NULL
            GROUP BY party
            ORDER BY total DESC
        """)
        party_sentiment = c.fetchall()
        
        # Get recent comments
        c.execute('SELECT comments, party, sentiment_lable, date, handle FROM comments ORDER BY date DESC LIMIT 10')
        recent_comments = c.fetchall()
        
        # Get parties
        c.execute('SELECT DISTINCT party FROM comments WHERE party IS NOT NULL ORDER BY party')
        parties = [row[0] for row in c.fetchall()]
        
        conn.close()
        
        return jsonify({
            'success': True,
            'data': {
                'analytics': {
                    'summary': {
                        'total_comments': summary[0],
                        'positive_count': summary[1],
                        'negative_count': summary[2],
                        'neutral_count': summary[3]
                    },
                    'party_sentiment': [
                        {
                            'party': row[0],
                            'total': row[1],
                            'positive': row[2],
                            'negative': row[3],
                            'neutral': row[4]
                        }
                        for row in party_sentiment
                    ],
                    'overall_sentiment': [
                        {'sentiment': 'Positive', 'count': summary[1]},
                        {'sentiment': 'Negative', 'count': summary[2]},
                        {'sentiment': 'Neutral', 'count': summary[3]}
                    ]
                },
                'comments': [
                    {
                        'comment': comment[0],
                        'party': comment[1],
                        'sentiment_label': comment[2],
                        'date': comment[3],
                        'handle': comment[4]
                    }
                    for comment in recent_comments
                ],
                'parties': parties
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error retrieving demo format data: {str(e)}'
        }), 500

# Keep your existing routes
@app.route('/api/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({
        'success': True,
        'status': 'Server is running',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/processing-jobs', methods=['GET'])
def get_processing_jobs():
    """Get all processing jobs with optional filtering"""
    try:
        import sqlite3
        
        # Get query parameters
        status = request.args.get('status')
        limit = int(request.args.get('limit', 10))
        
        conn = sqlite3.connect('sentiment.db')
        c = conn.cursor()
        
        # Build query
        if status:
            c.execute('''SELECT job_id, filename, status, created_at, processed_count 
                        FROM processing_jobs 
                        WHERE status = ? 
                        ORDER BY created_at DESC 
                        LIMIT ?''', (status, limit))
        else:
            c.execute('''SELECT job_id, filename, status, created_at, processed_count 
                        FROM processing_jobs 
                        ORDER BY created_at DESC 
                        LIMIT ?''', (limit,))
        
        jobs = c.fetchall()
        conn.close()
        
        return jsonify({
            'success': True,
            'jobs': [
                {
                    'job_id': job[0],
                    'filename': job[1],
                    'status': job[2],
                    'created_at': job[3],
                    'processed_count': job[4]
                }
                for job in jobs
            ]
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error retrieving processing jobs: {str(e)}'
        }), 500

@app.route('/api/jobs/<job_id>', methods=['GET'])
def get_job_status(job_id):
    """Get processing job status"""
    try:
        import sqlite3
        
        conn = sqlite3.connect('sentiment.db')
        c = conn.cursor()
        
        c.execute('''SELECT job_id, filename, status, created_at, processed_count 
                    FROM processing_jobs WHERE job_id = ?''', (job_id,))
        job = c.fetchone()
        
        conn.close()
        
        if job:
            return jsonify({
                'success': True,
                'job': {
                    'job_id': job[0],
                    'filename': job[1],
                    'status': job[2],
                    'created_at': job[3],
                    'processed_count': job[4]
                }
            })
        else:
            return jsonify({'success': False, 'message': 'Job not found'}), 404
            
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error retrieving job: {str(e)}'
        }), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'message': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'message': 'Internal server error'
    }), 500

if __name__ == '__main__':
    # Initialize database
    processor.init_db()
    app.run(debug=True, port=5000)