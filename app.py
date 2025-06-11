import os
import re
import json
import time
import csv
from datetime import datetime, timedelta
from threading import Thread
from collections import Counter

import pandas as pd
from flask import Flask, request, jsonify, current_app
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func, and_
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Database Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///twitter_analytics.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_size': 10,
    'pool_recycle': 300,
    'pool_pre_ping': True
}
db = SQLAlchemy(app)

# Database Models
class Tweet(db.Model):
    __tablename__ = 'tweets'
    
    id = db.Column(db.Integer, primary_key=True)
    tweet_id = db.Column(db.String(50), unique=True, nullable=False)
    name = db.Column(db.String(100), nullable=False)
    handle = db.Column(db.String(50), nullable=False, index=True)
    timestamp = db.Column(db.DateTime, nullable=False, index=True)
    verified = db.Column(db.Boolean, default=False)
    content = db.Column(db.Text, nullable=False)
    comments = db.Column(db.Integer, default=0)
    retweets = db.Column(db.Integer, default=0)
    likes = db.Column(db.Integer, default=0)
    analytics = db.Column(db.Integer, default=0)
    profile_image = db.Column(db.String(200))
    tweet_link = db.Column(db.String(200))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    job_id = db.Column(db.String(50), db.ForeignKey('scraping_jobs.job_id'))
    
    hashtags = db.relationship('Hashtag', backref='tweet', lazy='dynamic')
    mentions = db.relationship('Mention', backref='tweet', lazy='dynamic')
    
    def to_dict(self):
        return {
            'id': self.id,
            'tweet_id': self.tweet_id,
            'name': self.name,
            'handle': self.handle,
            'timestamp': self.timestamp.isoformat(),
            'verified': self.verified,
            'content': self.content,
            'comments': self.comments,
            'retweets': self.retweets,
            'likes': self.likes,
            'analytics': self.analytics,
            'profile_image': self.profile_image,
            'tweet_link': self.tweet_link,
            'created_at': self.created_at.isoformat(),
            'hashtags': [h.hashtag for h in self.hashtags],
            'mentions': [m.mention for m in self.mentions]
        }

class Hashtag(db.Model):
    __tablename__ = 'hashtags'
    
    id = db.Column(db.Integer, primary_key=True)
    tweet_id = db.Column(db.String(50), db.ForeignKey('tweets.tweet_id'), nullable=False)
    hashtag = db.Column(db.String(100), nullable=False, index=True)
    
    __table_args__ = (
        db.Index('idx_hashtag_tweet', 'hashtag', 'tweet_id'),
    )

class Mention(db.Model):
    __tablename__ = 'mentions'
    
    id = db.Column(db.Integer, primary_key=True)
    tweet_id = db.Column(db.String(50), db.ForeignKey('tweets.tweet_id'), nullable=False)
    mention = db.Column(db.String(100), nullable=False, index=True)
    
    __table_args__ = (
        db.Index('idx_mention_tweet', 'mention', 'tweet_id'),
    )

class ScrapingJob(db.Model):
    __tablename__ = 'scraping_jobs'
    
    id = db.Column(db.Integer, primary_key=True)
    job_id = db.Column(db.String(50), unique=True, nullable=False, index=True)
    status = db.Column(db.String(20), default='pending', index=True)
    parameters = db.Column(db.Text)
    tweets_scraped = db.Column(db.Integer, default=0)
    started_at = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime)
    error_message = db.Column(db.Text)
    target_type = db.Column(db.String(20))
    target_value = db.Column(db.String(255), index=True)
    max_tweets_requested = db.Column(db.Integer)
    progress_percentage = db.Column(db.Float, default=0.0)
    result_summary = db.Column(db.Text)
    duration_seconds = db.Column(db.Float)
    csv_file_path = db.Column(db.String(500))
    
    tweets = db.relationship('Tweet', backref='job', lazy='dynamic')
    
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
        db.session.commit()
    
    def mark_completed(self, tweets_scraped, result_summary=None):
        self.status = 'completed'
        self.tweets_scraped = tweets_scraped
        self.completed_at = datetime.utcnow()
        self.progress_percentage = 100.0
        if result_summary:
            self.result_summary = json.dumps(result_summary)
        if self.started_at:
            self.duration_seconds = (self.completed_at - self.started_at).total_seconds()
        db.session.commit()
    
    def mark_failed(self, error_message):
        self.status = 'failed'
        self.error_message = str(error_message)
        self.completed_at = datetime.utcnow()
        if self.started_at:
            self.duration_seconds = (self.completed_at - self.started_at).total_seconds()
        db.session.commit()
    
    def mark_cancelled(self):
        self.status = 'cancelled'
        self.completed_at = datetime.utcnow()
        if self.started_at:
            self.duration_seconds = (self.completed_at - self.started_at).total_seconds()
        db.session.commit()

# Helper Functions
def extract_hashtags(text):
    return list(set(re.findall(r'#(\w+)', text.lower())))

def extract_mentions(text):
    return list(set(re.findall(r'@(\w+)', text.lower())))

def parse_csv_to_db(csv_file_path, job_id=None):
    try:
        df = pd.read_csv(csv_file_path)
        tweets_added = 0
        
        for _, row in df.iterrows():
            if 'Tweet ID' not in row:
                continue
                
            # Check if tweet exists
            if Tweet.query.filter_by(tweet_id=str(row['Tweet ID'])).first():
                continue
                
            # Parse timestamp
            timestamp_str = row.get('Timestamp', '')
            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                timestamp = datetime.utcnow()
            
            # Create tweet
            tweet = Tweet(
                tweet_id=str(row['Tweet ID']),
                name=row.get('Name', ''),
                handle=row.get('Handle', '').lower(),
                timestamp=timestamp,
                verified=str(row.get('Verified', 'False')).lower() == 'true',
                content=row.get('Content', ''),
                comments=int(row.get('Comments', 0)),
                retweets=int(row.get('Retweets', 0)),
                likes=int(row.get('Likes', 0)),
                analytics=int(row.get('Analytics', 0)),
                profile_image=row.get('Profile Image', ''),
                tweet_link=row.get('Tweet Link', ''),
                job_id=job_id
            )
            
            db.session.add(tweet)
            db.session.flush()  # Get the tweet ID
            
            # Process hashtags
            for hashtag in extract_hashtags(tweet.content):
                db.session.add(Hashtag(
                    tweet_id=tweet.tweet_id,
                    hashtag=hashtag.lower()
                ))
            
            # Process mentions
            for mention in extract_mentions(tweet.content):
                db.session.add(Mention(
                    tweet_id=tweet.tweet_id,
                    mention=mention.lower()
                ))
            
            tweets_added += 1
            
            # Commit every 100 records to manage memory
            if tweets_added % 100 == 0:
                db.session.commit()
        
        db.session.commit()
        return tweets_added
    
    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Error parsing CSV: {str(e)}")
        raise

def run_scraper(job_id, data):
    """Background task to run the scraper and save results"""
    # Use the global app instance defined in this module
    global app

    with app.app_context():
        try:
            job = ScrapingJob.query.filter_by(job_id=job_id).first()
            if not job:
                app.logger.error(f"Job {job_id} not found")
                return
            
            job.status = 'running'
            db.session.commit()

            from scraper.twitter_scraper import Twitter_Scraper

            USER_MAIL = data.get('mail', os.getenv("TWITTER_MAIL"))
            USER_UNAME = data.get('username', os.getenv("TWITTER_USERNAME"))
            USER_PASSWORD = data.get('password', os.getenv("TWITTER_PASSWORD"))

            if not all([USER_MAIL, USER_UNAME, USER_PASSWORD]):
                raise ValueError("Missing Twitter credentials")

            max_tweets = min(int(data.get('tweets', 50)), 1000)
            target = data.get('target_username') or data.get('hashtag') or data.get('query')
            if not target:
                raise ValueError("No target specified")

            try:
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
                    if i % 10 == 0:
                        job.update_progress(i + 1)

                for tweet_data in tweets:
                    tweet = Tweet(
                        tweet_id=tweet_data.get('tweet_id', str(int(time.time() * 1000))),
                        name=tweet_data.get('name', ''),
                        handle=tweet_data.get('handle', '').lower(),
                        timestamp=datetime.fromtimestamp(tweet_data.get('timestamp', time.time())),
                        verified=tweet_data.get('verified', False),
                        content=tweet_data.get('content', ''),
                        comments=tweet_data.get('comments', 0),
                        retweets=tweet_data.get('retweets', 0),
                        likes=tweet_data.get('likes', 0),
                        analytics=tweet_data.get('analytics', 0),
                        profile_image=tweet_data.get('profile_image', ''),
                        tweet_link=tweet_data.get('tweet_link', ''),
                        job_id=job_id
                    )
                    db.session.add(tweet)

                    for hashtag in extract_hashtags(tweet.content):
                        db.session.add(Hashtag(tweet_id=tweet.tweet_id, hashtag=hashtag))
                    for mention in extract_mentions(tweet.content):
                        db.session.add(Mention(tweet_id=tweet.tweet_id, mention=mention))

                db.session.commit()

                job.mark_completed(
                    tweets_scraped=len(tweets),
                    result_summary={
                        'target': target,
                        'tweets_scraped': len(tweets),
                        'success': True
                    }
                )

            except Exception as e:
                app.logger.error(f"Scraping error: {str(e)}")
                job.mark_failed(str(e))
                raise

        except Exception as e:
            app.logger.error(f"Job {job_id} failed: {str(e)}")
            if 'job' in locals():
                job.mark_failed(str(e))
            raise

# API Routes
@app.route('/')
def index():
    return jsonify({
        'message': 'Twitter Analytics API',
        'version': '1.0.0',
        'endpoints': {
            'GET /api/tweets': 'Get all tweets with optional filters',
            'POST /api/scrape': 'Start a new scraping job',
            'GET /api/jobs/<job_id>': 'Get scraping job status',
            'GET /api/analytics': 'Get analytics data',
            'GET /api/hashtags': 'Get hashtag analytics',
            'GET /api/users': 'Get user analytics'
        }
    })

# @app.route('/api/tweets', methods=['GET'])
# def get_tweets():
#     try:
#         # Query parameters
#         page = request.args.get('page', 1, type=int)
#         per_page = request.args.get('per_page', 50, type=int)
#         username = request.args.get('username')
#         hashtag = request.args.get('hashtag')
#         date_from = request.args.get('date_from')
#         date_to = request.args.get('date_to')
#         verified_only = request.args.get('verified_only', 'false').lower() == 'true'
        
#         # Build query
#         query = Tweet.query
        
#         if username:
#             query = query.filter(Tweet.handle.ilike(f'%{username}%'))
        
#         if verified_only:
#             query = query.filter(Tweet.verified == True)
        
#         if date_from:
#             date_from_obj = datetime.fromisoformat(date_from)
#             query = query.filter(Tweet.timestamp >= date_from_obj)
        
#         if date_to:
#             date_to_obj = datetime.fromisoformat(date_to)
#             query = query.filter(Tweet.timestamp <= date_to_obj)
        
#         if hashtag:
#             hashtag_tweet_ids = db.session.query(Hashtag.tweet_id).filter(
#                 Hashtag.hashtag.ilike(f'%{hashtag}%')
#             ).subquery()
#             query = query.filter(Tweet.tweet_id.in_(hashtag_tweet_ids))
        
#         # Order by timestamp (newest first)
#         query = query.order_by(Tweet.timestamp.desc())
        
#         # Paginate
#         tweets = query.paginate(page=page, per_page=per_page, error_out=False)
        
#         # Get hashtags and mentions for each tweet
#         result = []
#         for tweet in tweets.items:
#             tweet_dict = tweet.to_dict()
            
#             # Get hashtags
#             hashtags = db.session.query(Hashtag.hashtag).filter_by(tweet_id=tweet.tweet_id).all()
#             tweet_dict['tags'] = [h.hashtag for h in hashtags]
            
#             # Get mentions
#             mentions = db.session.query(Mention.mention).filter_by(tweet_id=tweet.tweet_id).all()
#             tweet_dict['mentions'] = [m.mention for m in mentions]
            
#             result.append(tweet_dict)
        
#         return jsonify({
#             'tweets': result,
#             'pagination': {
#                 'page': tweets.page,
#                 'pages': tweets.pages,
#                 'per_page': tweets.per_page,
#                 'total': tweets.total,
#                 'has_next': tweets.has_next,
#                 'has_prev': tweets.has_prev
#             }
#         })
    
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500
    
# @app.route('/api/scrape', methods=['POST'])
# def start_scraping():
#     try:
#         data = request.get_json()
#         job_id = f"job_{int(time.time())}_{hash(str(data)) % 10000}"
        
#         # When creating a job:
#         job = ScrapingJob(
#             job_id=job_id,
#             parameters=json.dumps(data),
#             target_type='username' if data.get('target_username') else 'hashtag' if data.get('hashtag') else 'query',
#             target_value=data.get('target_username') or data.get('hashtag') or data.get('query'),
#             max_tweets_requested=data.get('tweets', 50)
#         )

#         # During scraping (in your run_scraper function):
#         job.update_progress(tweets_scraped=current_count)

#         # When completed:
#         job.mark_completed(tweets_scraped=final_count, result_summary=json.dumps(summary))
#         db.session.add(job)
#         db.session.commit()
        
#         thread = Thread(target=run_scraper, args=(job_id, data))
#         thread.daemon = True
#         thread.start()
        
#         return jsonify({
#             'job_id': job_id,
#             'status': 'started',
#             'message': 'Scraping job started successfully'
#         })
    
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# @app.route('/api/jobs/<job_id>', methods=['GET'])
# def get_job_status(job_id):
#     try:
#         job = ScrapingJob.query.filter_by(job_id=job_id).first()
#         if not job:
#             return jsonify({'error': 'Job not found'}), 404
        
#         return jsonify({
#             'job_id': job.job_id,
#             'status': job.status,
#             'tweets_scraped': job.tweets_scraped,
#             'started_at': job.started_at.isoformat(),
#             'completed_at': job.completed_at.isoformat() if job.completed_at else None,
#             'error_message': job.error_message,
#             'parameters': json.loads(job.parameters)
#         })
    
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# @app.route('/api/analytics', methods=['GET'])
# def get_analytics():
#     try:
#         # Date filter
#         days = request.args.get('days', 30, type=int)
#         date_from = datetime.utcnow() - timedelta(days=days)
        
#         # Basic stats
#         total_tweets = Tweet.query.filter(Tweet.timestamp >= date_from).count()
#         total_engagement = db.session.query(
#             db.func.sum(Tweet.likes + Tweet.retweets + Tweet.comments)
#         ).filter(Tweet.timestamp >= date_from).scalar() or 0
        
#         total_impressions = db.session.query(
#             db.func.sum(Tweet.analytics)
#         ).filter(Tweet.timestamp >= date_from).scalar() or 0
        
#         avg_engagement = total_engagement / total_tweets if total_tweets > 0 else 0
        
#         # Top users by engagement
#         top_users = db.session.query(
#             Tweet.handle,
#             Tweet.name,
#             Tweet.verified,
#             db.func.sum(Tweet.likes + Tweet.retweets + Tweet.comments).label('total_engagement'),
#             db.func.sum(Tweet.likes).label('total_likes'),
#             db.func.sum(Tweet.retweets).label('total_retweets'),
#             db.func.sum(Tweet.comments).label('total_comments')
#         ).filter(Tweet.timestamp >= date_from).group_by(Tweet.handle).order_by(
#             db.func.sum(Tweet.likes + Tweet.retweets + Tweet.comments).desc()
#         ).limit(10).all()
        
#         # Daily engagement trend
#         daily_stats = db.session.query(
#             db.func.date(Tweet.timestamp).label('date'),
#             db.func.count(Tweet.id).label('tweet_count'),
#             db.func.sum(Tweet.likes + Tweet.retweets + Tweet.comments).label('engagement'),
#             db.func.sum(Tweet.analytics).label('impressions')
#         ).filter(Tweet.timestamp >= date_from).group_by(
#             db.func.date(Tweet.timestamp)
#         ).order_by(db.func.date(Tweet.timestamp)).all()
        
#         return jsonify({
#             'summary': {
#                 'total_tweets': total_tweets,
#                 'total_engagement': total_engagement,
#                 'avg_engagement': round(avg_engagement, 2),
#                 'total_impressions': total_impressions
#             },
#             'top_users': [
#                 {
#                     'handle': user.handle,
#                     'name': user.name,
#                     'verified': user.verified,
#                     'total_engagement': user.total_engagement,
#                     'total_likes': user.total_likes,
#                     'total_retweets': user.total_retweets,
#                     'total_comments': user.total_comments
#                 }
#                 for user in top_users
#             ],
#             'daily_trends': [
#                 {
#                     'date': str(day.date),
#                     'tweet_count': day.tweet_count,
#                     'engagement': day.engagement or 0,
#                     'impressions': day.impressions or 0
#                 }
#                 for day in daily_stats
#             ]
#         })
    
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# @app.route('/api/hashtags', methods=['GET'])
# def get_hashtag_analytics():
#     try:
#         days = request.args.get('days', 30, type=int)
#         limit = request.args.get('limit', 50, type=int)
#         date_from = datetime.utcnow() - timedelta(days=days)
        
#         # Top hashtags with engagement
#         hashtag_stats = db.session.query(
#             Hashtag.hashtag,
#             db.func.count(Hashtag.id).label('count'),
#             db.func.sum(Tweet.likes + Tweet.retweets + Tweet.comments).label('total_engagement')
#         ).join(Tweet, Tweet.tweet_id == Hashtag.tweet_id).filter(
#             Tweet.timestamp >= date_from
#         ).group_by(Hashtag.hashtag).order_by(
#             db.func.count(Hashtag.id).desc()
#         ).limit(limit).all()
        
#         return jsonify({
#             'hashtags': [
#                 {
#                     'hashtag': stat.hashtag,
#                     'count': stat.count,
#                     'total_engagement': stat.total_engagement or 0
#                 }
#                 for stat in hashtag_stats
#             ]
#         })
    
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# @app.route('/api/users', methods=['GET'])
# def get_user_analytics():
#     try:
#         days = request.args.get('days', 30, type=int)
#         limit = request.args.get('limit', 50, type=int)
#         date_from = datetime.utcnow() - timedelta(days=days)
        
#         user_stats = db.session.query(
#             Tweet.handle,
#             Tweet.name,
#             Tweet.verified,
#             db.func.count(Tweet.id).label('tweet_count'),
#             db.func.sum(Tweet.likes).label('total_likes'),
#             db.func.sum(Tweet.retweets).label('total_retweets'),
#             db.func.sum(Tweet.comments).label('total_comments'),
#             db.func.sum(Tweet.analytics).label('total_impressions'),
#             db.func.avg(Tweet.likes + Tweet.retweets + Tweet.comments).label('avg_engagement')
#         ).filter(Tweet.timestamp >= date_from).group_by(Tweet.handle).order_by(
#             db.func.sum(Tweet.likes + Tweet.retweets + Tweet.comments).desc()
#         ).limit(limit).all()
        
#         return jsonify({
#             'users': [
#                 {
#                     'handle': user.handle,
#                     'name': user.name,
#                     'verified': user.verified,
#                     'tweet_count': user.tweet_count,
#                     'total_likes': user.total_likes or 0,
#                     'total_retweets': user.total_retweets or 0,
#                     'total_comments': user.total_comments or 0,
#                     'total_impressions': user.total_impressions or 0,
#                     'avg_engagement': round(user.avg_engagement or 0, 2)
#                 }
#                 for user in user_stats
#             ]
#         })
    
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# @app.route('/api/jobs', methods=['GET'])
# def get_all_jobs():
#     try:
#         jobs = ScrapingJob.query.order_by(ScrapingJob.started_at.desc()).limit(50).all()
        
#         return jsonify({
#             'jobs': [
#                 {
#                     'job_id': job.job_id,
#                     'status': job.status,
#                     'tweets_scraped': job.tweets_scraped,
#                     'started_at': job.started_at.isoformat(),
#                     'completed_at': job.completed_at.isoformat() if job.completed_at else None,
#                     'parameters': json.loads(job.parameters)
#                 }
#                 for job in jobs
#             ]
#         })
    
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# # Initialize database
# # @app.before_first_request
# def create_tables():
#     db.create_all()

# if __name__ == '__main__':
#     with app.app_context():
#         db.create_all()
#     app.run(debug=True, host='0.0.0.0', port=5000) 


@app.route('/api/scrape', methods=['POST'])
def start_scraping():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        job_id = f"job_{int(time.time())}_{hash(str(data)) % 10000}"
        
        # Create job record
        job = ScrapingJob(
            job_id=job_id,
            parameters=json.dumps(data),
            target_type='username' if data.get('target_username') else 'hashtag' if data.get('hashtag') else 'query',
            target_value=data.get('target_username') or data.get('hashtag') or data.get('query'),
            max_tweets_requested=data.get('tweets', 50)
        )
        
        db.session.add(job)
        db.session.commit()
        
        # Start scraping in background thread
        thread = Thread(target=run_scraper, args=(job_id, data))
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'job_id': job_id,
            'status': 'started',
            'message': 'Scraping job started successfully'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/jobs/<job_id>', methods=['GET'])
def get_job_status(job_id):
    try:
        job = ScrapingJob.query.filter_by(job_id=job_id).first()
        if not job:
            return jsonify({'error': 'Job not found'}), 404
        
        return jsonify(job.to_dict())
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/jobs', methods=['GET'])
def get_all_jobs():
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 20, type=int)
        
        jobs = ScrapingJob.query.order_by(ScrapingJob.started_at.desc()).paginate(
            page=page, per_page=per_page, error_out=False
        )
        
        return jsonify({
            'jobs': [job.to_dict() for job in jobs.items],
            'total': jobs.total,
            'pages': jobs.pages,
            'current_page': page
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/analytics', methods=['GET'])
def get_analytics():
    try:
        # Date filter
        days = request.args.get('days', 30, type=int)
        date_from = datetime.utcnow() - timedelta(days=days)
        
        # Basic stats
        total_tweets = Tweet.query.filter(Tweet.timestamp >= date_from).count()
        total_engagement = db.session.query(
            db.func.sum(Tweet.likes + Tweet.retweets + Tweet.comments)
        ).filter(Tweet.timestamp >= date_from).scalar() or 0
        
        total_impressions = db.session.query(
            db.func.sum(Tweet.analytics)
        ).filter(Tweet.timestamp >= date_from).scalar() or 0
        
        avg_engagement = total_engagement / total_tweets if total_tweets > 0 else 0
        
        # Top users by engagement
        top_users = db.session.query(
            Tweet.handle,
            Tweet.name,
            Tweet.verified,
            db.func.sum(Tweet.likes + Tweet.retweets + Tweet.comments).label('total_engagement'),
            db.func.sum(Tweet.likes).label('total_likes'),
            db.func.sum(Tweet.retweets).label('total_retweets'),
            db.func.sum(Tweet.comments).label('total_comments')
        ).filter(Tweet.timestamp >= date_from).group_by(Tweet.handle).order_by(
            db.func.sum(Tweet.likes + Tweet.retweets + Tweet.comments).desc()
        ).limit(10).all()
        
        # Daily engagement trend
        daily_stats = db.session.query(
            db.func.date(Tweet.timestamp).label('date'),
            db.func.count(Tweet.id).label('tweet_count'),
            db.func.sum(Tweet.likes + Tweet.retweets + Tweet.comments).label('engagement'),
            db.func.sum(Tweet.analytics).label('impressions')
        ).filter(Tweet.timestamp >= date_from).group_by(
            db.func.date(Tweet.timestamp)
        ).order_by(db.func.date(Tweet.timestamp)).all()
        
        return jsonify({
            'summary': {
                'total_tweets': total_tweets,
                'total_engagement': total_engagement,
                'avg_engagement': round(avg_engagement, 2),
                'total_impressions': total_impressions
            },
            'top_users': [
                {
                    'handle': user.handle,
                    'name': user.name,
                    'verified': user.verified,
                    'total_engagement': user.total_engagement,
                    'total_likes': user.total_likes,
                    'total_retweets': user.total_retweets,
                    'total_comments': user.total_comments
                }
                for user in top_users
            ],
            'daily_trends': [
                {
                    'date': str(day.date),
                    'tweet_count': day.tweet_count,
                    'engagement': day.engagement or 0,
                    'impressions': day.impressions or 0
                }
                for day in daily_stats
            ]
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/hashtags', methods=['GET'])
def get_hashtag_analytics():
    try:
        days = request.args.get('days', 30, type=int)
        limit = request.args.get('limit', 50, type=int)
        date_from = datetime.utcnow() - timedelta(days=days)
        
        # Top hashtags with engagement
        hashtag_stats = db.session.query(
            Hashtag.hashtag,
            db.func.count(Hashtag.id).label('count'),
            db.func.sum(Tweet.likes + Tweet.retweets + Tweet.comments).label('total_engagement')
        ).join(Tweet, Tweet.tweet_id == Hashtag.tweet_id).filter(
            Tweet.timestamp >= date_from
        ).group_by(Hashtag.hashtag).order_by(
            db.func.count(Hashtag.id).desc()
        ).limit(limit).all()
        
        return jsonify({
            'hashtags': [
                {
                    'hashtag': stat.hashtag,
                    'count': stat.count,
                    'total_engagement': stat.total_engagement or 0
                }
                for stat in hashtag_stats
            ]
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/users', methods=['GET'])
def get_user_analytics():
    try:
        days = request.args.get('days', 30, type=int)
        limit = request.args.get('limit', 50, type=int)
        date_from = datetime.utcnow() - timedelta(days=days)
        
        user_stats = db.session.query(
            Tweet.handle,
            Tweet.name,
            Tweet.verified,
            db.func.count(Tweet.id).label('tweet_count'),
            db.func.sum(Tweet.likes).label('total_likes'),
            db.func.sum(Tweet.retweets).label('total_retweets'),
            db.func.sum(Tweet.comments).label('total_comments'),
            db.func.sum(Tweet.analytics).label('total_impressions'),
            db.func.avg(Tweet.likes + Tweet.retweets + Tweet.comments).label('avg_engagement')
        ).filter(Tweet.timestamp >= date_from).group_by(Tweet.handle).order_by(
            db.func.sum(Tweet.likes + Tweet.retweets + Tweet.comments).desc()
        ).limit(limit).all()
        
        return jsonify({
            'users': [
                {
                    'handle': user.handle,
                    'name': user.name,
                    'verified': user.verified,
                    'tweet_count': user.tweet_count,
                    'total_likes': user.total_likes or 0,
                    'total_retweets': user.total_retweets or 0,
                    'total_comments': user.total_comments or 0,
                    'total_impressions': user.total_impressions or 0,
                    'avg_engagement': round(user.avg_engagement or 0, 2)
                }
                for user in user_stats
            ]
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Initialize database
def create_tables():
    db.create_all()


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, host='0.0.0.0', port=5000)