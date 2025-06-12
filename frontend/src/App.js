import React, { useState, useEffect } from 'react';
import { LineChart, Line, BarChart, Bar, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Search, Users, Hash, TrendingUp, MessageCircle, Repeat, Heart, Eye, RefreshCw, Play, Clock, CheckCircle, XCircle } from 'lucide-react';

const API_BASE = 'http://localhost:5000/api';

const Dashboard = () => { 
  const [activeTab, setActiveTab] = useState('overview');
  const [tweets, setTweets] = useState([]);
  const [analytics, setAnalytics] = useState(null);
  const [hashtags, setHashtags] = useState([]);
  const [users, setUsers] = useState([]);
  const [jobs, setJobs] = useState([]);
  const [loading, setLoading] = useState(false);
  const [filters, setFilters] = useState({
    days: 30,
    username: '',
    hashtag: '',
    verified_only: false
  });

  const [scrapeForm, setScrapeForm] = useState({
    tweets: 100,
    target_username: '',
    hashtag: '',
    query: '',
    latest: false,
    top: false
  });

  // Fetch data functions
  const fetchTweets = async () => {
    try {
      const params = new URLSearchParams();
      if (filters.username) params.append('username', filters.username);
      if (filters.hashtag) params.append('hashtag', filters.hashtag);
      if (filters.verified_only) params.append('verified_only', 'true');
      params.append('per_page', '100');

      const response = await fetch(`${API_BASE}/tweets?${params}`);
      const data = await response.json();
      setTweets(data.tweets || []);
    } catch (error) {
      console.error('Sorry Error fetching tweets:', error);
    }
  };

  const fetchAnalytics = async () => {
    try {
      const response = await fetch(`${API_BASE}/analytics?days=${filters.days}`);
      const data = await response.json();
      setAnalytics(data);
    } catch (error) {
      console.error('Error fetching analytics:', error);
    }
  };

  const fetchHashtags = async () => {
    try {
      const response = await fetch(`${API_BASE}/hashtags?days=${filters.days}&limit=20`);
      const data = await response.json();
      setHashtags(data.hashtags || []);
    } catch (error) {
      console.error('Error fetching hashtags:', error);
    }
  };

  const fetchUsers = async () => {
    try {
      const response = await fetch(`${API_BASE}/users?days=${filters.days}&limit=20`);
      const data = await response.json();
      setUsers(data.users || []);
    } catch (error) {
      console.error('Error fetching users:', error);
    }
  };

  const fetchJobs = async () => {
    try {
      const response = await fetch(`${API_BASE}/jobs`);
      const data = await response.json();
      setJobs(data.jobs || []);
    } catch (error) {
      console.error('Error fetching jobs:', error);
    }
  };

  const refreshData = async () => {
    setLoading(true);
    await Promise.all([
      fetchTweets(),
      fetchAnalytics(),
      fetchHashtags(),
      fetchUsers(),
      fetchJobs()
    ]);
    setLoading(false);
  };

  const startScraping = async () => {
    try {
      setLoading(true);
      const response = await fetch(`${API_BASE}/scrape`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(scrapeForm)
      });
      const data = await response.json();
      alert(`Scraping job started: ${data.job_id}`);
      fetchJobs();
    } catch (error) {
      console.error('Error starting scrape:', error);
      alert('Error starting scrape job');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    refreshData();
  }, [filters.days]);

  useEffect(() => {
    const interval = setInterval(fetchJobs, 10000); // Refresh jobs every 10 seconds
    return () => clearInterval(interval);
  }, []);

  const getStatusIcon = (status) => {
    switch (status) {
      case 'pending': return <Clock className="w-4 h-4 text-yellow-500" />;
      case 'running': return <RefreshCw className="w-4 h-4 text-blue-500 animate-spin" />;
      case 'completed': return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'failed': return <XCircle className="w-4 h-4 text-red-500" />;
      default: return null;
    }
  };

  const formatNumber = (num) => {
    if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
    if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
    return num?.toString() || '0';
  };

  const COLORS = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', '#f97316', '#06b6d4', '#84cc16'];

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center space-x-4">
              <h1 className="text-2xl font-bold text-gray-900">Twitter Analytics Dashboard</h1>
            </div>
            <div className="flex items-center space-x-4">
              <select
                value={filters.days}
                onChange={(e) => setFilters({...filters, days: parseInt(e.target.value)})}
                className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value={7}>Last 7 days</option>
                <option value={30}>Last 30 days</option>
                <option value={90}>Last 90 days</option>
              </select>
              <button
                onClick={refreshData}
                disabled={loading}
                className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 flex items-center space-x-2"
              >
                <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
                <span>Refresh</span>
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="border-b border-gray-200">
          <nav className="-mb-px flex space-x-8">
            {[
              { id: 'overview', name: 'Overview', icon: TrendingUp },
              { id: 'tweets', name: 'Tweets', icon: MessageCircle },
              { id: 'hashtags', name: 'Hashtags', icon: Hash },
              { id: 'users', name: 'Users', icon: Users },
              { id: 'scraper', name: 'Scraper', icon: Search }
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center space-x-2 py-4 px-1 border-b-2 font-medium text-sm ${
                  activeTab === tab.id
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <tab.icon className="w-4 h-4" />
                <span>{tab.name}</span>
              </button>
            ))}
          </nav>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Overview Tab */}
        {activeTab === 'overview' && analytics && (
          <div className="space-y-6">
            {/* Stats Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <div className="bg-white rounded-lg shadow p-6">
                <div className="flex items-center">
                  <MessageCircle className="w-8 h-8 text-blue-500" />
                  <div className="ml-4">
                    <p className="text-sm font-medium text-gray-600">Total Tweets</p>
                    <p className="text-2xl font-semibold text-gray-900">{formatNumber(analytics.summary.total_tweets)}</p>
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-lg shadow p-6">
                <div className="flex items-center">
                  <Heart className="w-8 h-8 text-red-500" />
                  <div className="ml-4">
                    <p className="text-sm font-medium text-gray-600">Total Engagement</p>
                    <p className="text-2xl font-semibold text-gray-900">{formatNumber(analytics.summary.total_engagement)}</p>
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-lg shadow p-6">
                <div className="flex items-center">
                  <TrendingUp className="w-8 h-8 text-green-500" />
                  <div className="ml-4">
                    <p className="text-sm font-medium text-gray-600">Avg Engagement</p>
                    <p className="text-2xl font-semibold text-gray-900">{analytics.summary.avg_engagement}</p>
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-lg shadow p-6">
                <div className="flex items-center">
                  <Eye className="w-8 h-8 text-purple-500" />
                  <div className="ml-4">
                    <p className="text-sm font-medium text-gray-600">Impressions</p>
                    <p className="text-2xl font-semibold text-gray-900">{formatNumber(analytics.summary.total_impressions)}</p>
                  </div>
                </div>
              </div>
            </div>

            {/* Charts */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Daily Trends */}
              <div className="bg-white rounded-lg shadow p-6">
                <h3 className="text-lg font-medium text-gray-900 mb-4">Daily Engagement Trends</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={analytics.daily_trends}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Line type="monotone" dataKey="engagement" stroke="#3b82f6" strokeWidth={2} />
                    <Line type="monotone" dataKey="tweet_count" stroke="#10b981" strokeWidth={2} />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              {/* Top Users */}
              <div className="bg-white rounded-lg shadow p-6">
                <h3 className="text-lg font-medium text-gray-900 mb-4">Top Users by Engagement</h3>
                <div className="space-y-3">
                  {(analytics.top_users || []).slice(0, 8).map((user, index) => (
                    <div key={user.handle} className="flex items-center justify-between">
                      <div className="flex items-center">
                        <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center text-blue-600 font-medium text-sm">
                          {index + 1}
                        </div>
                        <div className="ml-3">
                          <p className="text-sm font-medium text-gray-900">{user.name}</p>
                          <p className="text-xs text-gray-500">{user.handle}</p>
                        </div>
                        {user.verified && <div className="ml-2 w-4 h-4 bg-blue-500 rounded-full"></div>}
                      </div>
                      <div className="text-sm font-medium text-gray-900">
                        {formatNumber(user.total_engagement)}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Tweets Tab */}
        {activeTab === 'tweets' && (
          <div className="space-y-6">
            {/* Filters */}
            <div className="bg-white rounded-lg shadow p-6">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <input
                  type="text"
                  placeholder="Filter by username..."
                  value={filters.username}
                  onChange={(e) => setFilters({...filters, username: e.target.value})}
                  className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
                <input
                  type="text"
                  placeholder="Filter by hashtag..."
                  value={filters.hashtag}
                  onChange={(e) => setFilters({...filters, hashtag: e.target.value})}
                  className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
                <label className="flex items-center">
                  <input
                    type="checkbox"
                    checked={filters.verified_only}
                    onChange={(e) => setFilters({...filters, verified_only: e.target.checked})}
                    className="mr-2"
                  />
                  Verified only
                </label>
              </div>
              <button
                onClick={fetchTweets}
                className="mt-4 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
              >
                Apply Filters
              </button>
            </div>

            {/* Tweets List */}
            <div className="bg-white rounded-lg shadow">
              <div className="px-6 py-4 border-b border-gray-200">
                <h3 className="text-lg font-medium text-gray-900">Recent Tweets ({tweets.length})</h3>
              </div>
              <div className="divide-y divide-gray-200 max-h-96 overflow-y-auto">
                {tweets.map((tweet) => (
                  <div key={tweet.id} className="p-6">
                    <div className="flex items-start space-x-3">
                      <div className="flex-1">
                        <div className="flex items-center space-x-2">
                          <p className="font-medium text-gray-900">{tweet.name}</p>
                          <p className="text-gray-500">@{tweet.handle}</p>
                          {tweet.verified && <div className="w-4 h-4 bg-blue-500 rounded-full"></div>}
                          <p className="text-sm text-gray-500">
                            {new Date(tweet.timestamp).toLocaleDateString()}
                          </p>
                        </div>
                        <p className="mt-2 text-gray-700">{tweet.content}</p>
                        <div className="mt-3 flex items-center space-x-6 text-sm text-gray-500">
                          <div className="flex items-center space-x-1">
                            <MessageCircle className="w-4 h-4" />
                            <span>{tweet.comments}</span>
                          </div>
                          <div className="flex items-center space-x-1">
                            <Repeat className="w-4 h-4" />
                            <span>{tweet.retweets}</span>
                          </div>
                          <div className="flex items-center space-x-1">
                            <Heart className="w-4 h-4" />
                            <span>{tweet.likes}</span>
                          </div>
                          <div className="flex items-center space-x-1">
                            <Eye className="w-4 h-4" />
                            <span>{formatNumber(tweet.analytics)}</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Hashtags Tab */}
        {activeTab === 'hashtags' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-white rounded-lg shadow p-6">
              <h3 className="text-lg font-medium text-gray-900 mb-4">Top Hashtags</h3>
              <ResponsiveContainer width="100%" height={400}>
                <BarChart data={hashtags.slice(0, 10)}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="hashtag" angle={-45} textAnchor="end" height={100} />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="count" fill="#3b82f6" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="bg-white rounded-lg shadow p-6">
              <h3 className="text-lg font-medium text-gray-900 mb-4">Hashtag Engagement</h3>
              <div className="space-y-3 max-h-96 overflow-y-auto">
                {hashtags.map((hashtag, index) => (
                  <div key={hashtag.hashtag} className="flex items-center justify-between p-3 border rounded-lg">
                    <div className="flex items-center">
                      <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center text-blue-600 font-medium text-sm">
                        {index + 1}
                      </div>
                      <div className="ml-3">
                        <p className="font-medium text-gray-900">{hashtag.hashtag}</p>
                        <p className="text-sm text-gray-500">{hashtag.count} tweets</p>
                      </div>
                    </div>
                    <div className="text-sm font-medium text-gray-900">
                      {formatNumber(hashtag.total_engagement)} eng.
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Users Tab */}
        {activeTab === 'users' && (
          <div className="bg-white rounded-lg shadow">
            <div className="px-6 py-4 border-b border-gray-200">
              <h3 className="text-lg font-medium text-gray-900">User Analytics</h3>
            </div>
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">User</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Tweets</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Likes</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Retweets</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Comments</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Avg Eng.</th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {users.map((user) => (
                    <tr key={user.handle}>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex items-center">
                          <div>
                            <div className="text-sm font-medium text-gray-900 flex items-center">
                              {user.name}
                              {user.verified && <div className="ml-2 w-4 h-4 bg-blue-500 rounded-full"></div>}
                            </div>
                            <div className="text-sm text-gray-500">@{user.handle}</div>
                          </div>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{user.tweet_count}</td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{formatNumber(user.total_likes)}</td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{formatNumber(user.total_retweets)}</td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{formatNumber(user.total_comments)}</td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{user.avg_engagement}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Scraper Tab */}
        {activeTab === 'scraper' && (
          <div className="space-y-6">
            {/* Scrape Form */}
            <div className="bg-white rounded-lg shadow p-6">
              <h3 className="text-lg font-medium text-gray-900 mb-4">Start New Scraping Job</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <input
                  type="number"
                  placeholder="Number of tweets"
                  value={scrapeForm.tweets}
                  onChange={(e) => setScrapeForm({...scrapeForm, tweets: parseInt(e.target.value)})}
                  className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
                <input
                  type="text"
                  placeholder="Target username"
                  value={scrapeForm.target_username}
                  onChange={(e) => setScrapeForm({...scrapeForm, target_username: e.target.value})}
                  className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
                <input
                  type="text"
                  placeholder="Hashtag"
                  value={scrapeForm.hashtag}
                  onChange={(e) => setScrapeForm({...scrapeForm, hashtag: e.target.value})}
                  className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
                <input
                  type="text"
                  placeholder="Search query"
                  value={scrapeForm.query}
                  onChange={(e) => setScrapeForm({...scrapeForm, query: e.target.value})}
                  className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
              <div className="mt-4 flex items-center space-x-4">
                <label className="flex items-center">
                  <input
                    type="checkbox"
                    checked={scrapeForm.latest}
                    onChange={(e) => setScrapeForm({...scrapeForm, latest: e.target.checked})}
                    className="mr-2"
                  />
                  Latest tweets
                </label>
                <label className="flex items-center">
                  <input
                    type="checkbox"
                    checked={scrapeForm.top}
                    onChange={(e) => setScrapeForm({...scrapeForm, top: e.target.checked})}
                    className="mr-2"
                  />
                  Top tweets
                </label>
              </div>
              <button
                onClick={startScraping}
                disabled={loading}
                className="mt-4 px-6 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:opacity-50 flex items-center space-x-2"
              >
                <Play className="w-4 h-4" />
                <span>Start Scraping</span>
              </button>
            </div>

            {/* Jobs List */}
            <div className="bg-white rounded-lg shadow">
              <div className="px-6 py-4 border-b border-gray-200">
                <h3 className="text-lg font-medium text-gray-900">Scraping Jobs</h3>
              </div>
              <div className="divide-y divide-gray-200">
                {jobs.map((job) => (
                  <div key={job.job_id} className="p-6">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-3">
                        {getStatusIcon(job.status)}
                        <div>
                          <p className="font-medium text-gray-900">{job.job_id}</p>
                          <p className="text-sm text-gray-500">
                            Started: {new Date(job.started_at).toLocaleString()}
                          </p>
                        </div>
                      </div>
                      <div className="text-right">
                        <p className="text-sm font-medium text-gray-900">
                          Status: <span className={`capitalize ${
                            job.status === 'completed' ? 'text-green-600' : 
                            job.status === 'failed' ? 'text-red-600' : 
                            job.status === 'running' ? 'text-blue-600' : 'text-yellow-600'
                          }`}>{job.status}</span>
                        </p>
                        {job.tweets_scraped > 0 && (
                          <p className="text-sm text-gray-500">Tweets: {job.tweets_scraped}</p>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Dashboard;
