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
            if pd.isna(row.get('Tweet ID')):
                continue

            if Tweet.query.filter_by(tweet_id=str(row['Tweet ID'])).first():
                continue

            # Parse timestamp
            try:
                timestamp = datetime.fromisoformat(str(row.get('Timestamp', '')).replace('Z', '+00:00'))
            except Exception:
                timestamp = datetime.utcnow()

            tweet = Tweet(
                tweet_id=str(row['Tweet ID']),
                name=row.get('Name', ''),
                handle=row.get('Handle', '').lower(),
                timestamp=timestamp,
                verified=str(row.get('Verified', 'False')).lower() == 'true',
                content=row.get('Content', ''),
                comments=str(row.get('Comments', 0)),
                retweets=str(row.get('Retweets', 0)),
                likes=str(row.get('Likes', 0)),
                analytics=str(row.get('Analytics', 0)),
                profile_image=row.get('Profile Image', ''),
                tweet_link=row.get('Tweet Link', ''),
                job_id=job_id
            )

            db.session.add(tweet)

            # Use stringified lists from CSV
            try:
                hashtags = ast.literal_eval(row.get('Tags', '[]'))
                for tag in hashtags:
                    db.session.add(Hashtag(tweet_id=tweet.tweet_id, hashtag=tag.lower()))
            except Exception:
                pass

            try:
                mentions = ast.literal_eval(row.get('Mentions', '[]'))
                for mention in mentions:
                    db.session.add(Mention(tweet_id=tweet.tweet_id, mention=mention.lower()))
            except Exception:
                pass

            tweets_added += 1

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

            if not any([USER_MAIL, USER_UNAME, USER_PASSWORD]):
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

@app.route('/api/scrape', methods=['POST'])
def start_scraping():
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400
    
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    try:
        # Validate input
        if not any([data.get('target_username'), data.get('hashtag'), data.get('query')]):
            return jsonify({'error': 'Must specify target_username, hashtag, or query'}), 400
        
        # Create job
        job_id = f"job_{int(time.time())}_{hash(json.dumps(data)) % 10000:04d}"
        job = ScrapingJob(
            job_id=job_id,
            parameters=json.dumps(data),
            target_type='username' if data.get('target_username') else 'hashtag' if data.get('hashtag') else 'query',
            target_value=data.get('target_username') or data.get('hashtag') or data.get('query'),
            max_tweets_requested=min(int(data.get('tweets', 50)), 1000)  # Limit to 1000 max
        )
        
        db.session.add(job)
        db.session.commit()
        
        # Start background job
        thread = Thread(target=run_scraper, args=(job_id, data))
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'job_id': job_id,
            'status': 'started',
            'message': 'Scraping job started successfully',
            'details': {
                'target': job.target_value,
                'max_tweets': job.max_tweets_requested
            }
        }), 202
    
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/jobs', methods=['GET'])
def get_all_jobs():
    try:
        page = request.args.get('page', 1, type=int)
        per_page = min(request.args.get('per_page', 20, type=int), 100)
        
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

@app.route('/api/jobs/<job_id>', methods=['GET'])
def get_job_status(job_id):
    try:
        job = ScrapingJob.query.filter_by(job_id=job_id).first_or_404()
        
        response = job.to_dict()
        
        # Add tweet count if job is completed
        if job.status == 'completed':
            response['tweets'] = Tweet.query.filter_by(job_id=job_id).count()
        else:
            job.status == "incomplete"
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 404 if '404' in str(e) else 500


@app.route('/api/tweets', methods=['GET'])
def get_tweets():
    try:
        # Pagination
        page = request.args.get('page', 1, type=int)
        per_page = min(request.args.get('per_page', 20, type=int), 100)
        
        # Filters
        query = Tweet.query
        job_id = request.args.get('job_id')
        if job_id:
            query = query.filter_by(job_id=job_id)
        
        handle = request.args.get('handle')
        if handle:
            query = query.filter(Tweet.handle.ilike(f"%{handle}%"))
        
        hashtag = request.args.get('hashtag')
        if hashtag:
            query = query.join(Hashtag).filter(Hashtag.hashtag.ilike(f"%{hashtag.lower()}%"))
        
        mention = request.args.get('mention')
        if mention:
            query = query.join(Mention).filter(Mention.mention.ilike(f"%{mention.lower()}%"))
        
        date_from = request.args.get('date_from')
        if date_from:
            try:
                date_from = datetime.fromisoformat(date_from)
                query = query.filter(Tweet.timestamp >= date_from)
            except ValueError:
                pass
        
        date_to = request.args.get('date_to')
        if date_to:
            try:
                date_to = datetime.fromisoformat(date_to)
                query = query.filter(Tweet.timestamp <= date_to)
            except ValueError:
                pass
        
        # Sorting
        sort = request.args.get('sort', 'timestamp')
        order = request.args.get('order', 'desc')
        if sort in ['timestamp', 'likes', 'retweets', 'comments']:
            sort_attr = getattr(Tweet, sort)
            query = query.order_by(sort_attr.desc() if order == 'desc' else sort_attr.asc())
        
        # Execute query
        tweets = query.paginate(page=page, per_page=per_page, error_out=False)
        
        return jsonify({
            'tweets': [tweet.to_dict() for tweet in tweets.items],
            'total': tweets.total,
            'pages': tweets.pages,
            'current_page': page
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analytics', methods=['GET'])
def get_analytics():
    try:
        days = min(request.args.get('days', 30, type=int), 365)  # Max 1 year
        date_from = datetime.utcnow() - timedelta(days=days)
        
        # Basic stats
        stats = db.session.query(
            func.count(Tweet.id).label('total_tweets'),
            func.sum(Tweet.likes + Tweet.retweets + Tweet.comments).label('total_engagement'),
            func.sum(Tweet.analytics).label('total_impressions'),
            func.avg(Tweet.likes + Tweet.retweets + Tweet.comments).label('avg_engagement')
        ).filter(Tweet.timestamp >= date_from).first()
        
        # Daily trends
        daily_stats = db.session.query(
            func.date(Tweet.timestamp).label('date'),
            func.count(Tweet.id).label('tweet_count'),
            func.sum(Tweet.likes + Tweet.retweets + Tweet.comments).label('engagement'),
            func.sum(Tweet.analytics).label('impressions')
        ).filter(Tweet.timestamp >= date_from).group_by(
            func.date(Tweet.timestamp)
        ).order_by(func.date(Tweet.timestamp)).all()
        
        return jsonify({
            'summary': {
                'total_tweets': stats.total_tweets or 0,
                'total_engagement': stats.total_engagement or 0,
                'avg_engagement': round(float(stats.avg_engagement or 0), 2),
                'total_impressions': stats.total_impressions or 0,
                'time_period': f"Last {days} days"
            },
            'daily_trends': [
                {
                    'date': str(day.date),
                    'tweet_count': day.tweet_count or 0,
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
        days = min(request.args.get('days', 30, type=int), 365)
        limit = min(request.args.get('limit', 50, type=int), 100)
        date_from = datetime.utcnow() - timedelta(days=days)
        
        # Top hashtags
        hashtags = db.session.query(
            Hashtag.hashtag,
            func.count(Hashtag.id).label('count'),
            func.sum(Tweet.likes + Tweet.retweets + Tweet.comments).label('engagement')
        ).join(Tweet).filter(
            Tweet.timestamp >= date_from
        ).group_by(Hashtag.hashtag).order_by(
            func.count(Hashtag.id).desc()
        ).limit(limit).all()
        
        return jsonify({
            'hashtags': [
                {
                    'hashtag': f"#{h.hashtag}",
                    'count': h.count,
                    'engagement': h.engagement or 0
                }
                for h in hashtags
            ],
            'time_period': f"Last {days} days"
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
            'top_users': [
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
    
# Error Handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(400)
def bad_request(error):
    return jsonify({'error': 'Bad request'}), 400

# Initialize database
def create_tables():
    with app.app_context():
        db.create_all()
        folder_path = "tweets/"
        for filename in os.listdir(folder_path):
            if filename.endswith(".csv"):
                csv_file_path = os.path.join(folder_path, filename)
                print(f"Processing {csv_file_path} ...")
                try:
                    tweets_added = parse_csv_to_db(csv_file_path)
                    print(f"Added {tweets_added} tweets from {filename}")
                except Exception as e:
                    print(f"Failed to process {filename}: {str(e)}")


if __name__ == '__main__':
    create_tables()
   
    

    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)), debug=os.getenv('DEBUG', 'False') == 'True')