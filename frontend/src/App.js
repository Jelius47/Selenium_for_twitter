import React, { useState, useEffect } from 'react';
import { LineChart, Line, BarChart, Bar, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Upload, Users, Hash, TrendingUp, MessageCircle, Filter, RefreshCw, Play, Clock, CheckCircle, XCircle, FileText, Calendar, BarChart3, Cloud } from 'lucide-react';

const SentimentDashboard = () => {
  const [activeTab, setActiveTab] = useState('overview');
  const [comments, setComments] = useState([]);
  const [analytics, setAnalytics] = useState(null);
  const [parties, setParties] = useState([]);
  const [wordCloud, setWordCloud] = useState({});
  const [loading, setLoading] = useState(false);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [processingJobs, setProcessingJobs] = useState([]);
  const [demoMode, setDemoMode] = useState(true);
  
  const [filters, setFilters] = useState({
    selectedParties: [],
    dateRange: {
      start: '',
      end: ''
    },
    sentiment: 'all'
  });

  // Demo data for offline viewing
  const demoData = {
    analytics: {
      summary: {
        total_comments: 15420,
        positive_count: 6820,
        negative_count: 4320,
        neutral_count: 4280
      },
      party_sentiment: [
        { party: 'CCM', total: 8500, positive: 3800, negative: 2200, neutral: 2500 },
        { party: 'CHADEMA', total: 4200, positive: 1800, negative: 1400, neutral: 1000 },
        { party: 'CUF', total: 1800, positive: 800, negative: 400, neutral: 600 },
        { party: 'ACT-Wazalendo', total: 920, positive: 420, negative: 320, neutral: 180 }
      ],
      overall_sentiment: [
        { sentiment: 'Positive', count: 6820 },
        { sentiment: 'Negative', count: 4320 },
        { sentiment: 'Neutral', count: 4280 }
      ],
      daily_trends: [
        { date: '2024-01-01', positive: 45, negative: 25, neutral: 30 },
        { date: '2024-01-02', positive: 52, negative: 28, neutral: 35 },
        { date: '2024-01-03', positive: 48, negative: 32, neutral: 28 },
        { date: '2024-01-04', positive: 55, negative: 20, neutral: 32 },
        { date: '2024-01-05', positive: 60, negative: 22, neutral: 38 },
        { date: '2024-01-06', positive: 58, negative: 25, neutral: 35 },
        { date: '2024-01-07', positive: 62, negative: 18, neutral: 40 }
      ]
    },
    comments: [
      {
        comment: "Serikali imefanya kazi nzuri katika sekta ya afya mwaka huu",
        party: "CCM",
        sentiment_label: "Positive",
        date: "2024-01-15T10:30:00Z"
      },
      {
        comment: "Tunahitaji mabadiliko makubwa katika utawala wa nchi",
        party: "CHADEMA",
        sentiment_label: "Negative",
        date: "2024-01-14T15:45:00Z"
      },
      {
        comment: "Hali ya uchumi bado ni changamoto kubwa kwa wananchi",
        party: "CUF",
        sentiment_label: "Negative",
        date: "2024-01-13T09:20:00Z"
      },
      {
        comment: "Mradi wa barabara mpya umechangia maendeleo ya vijijini",
        party: "CCM",
        sentiment_label: "Positive",
        date: "2024-01-12T14:10:00Z"
      },
      {
        comment: "Suala la elimu bado linahitaji uongozi bora zaidi",
        party: "ACT-Wazalendo",
        sentiment_label: "Neutral",
        date: "2024-01-11T11:55:00Z"
      }
    ],
    parties: ['CCM', 'CHADEMA', 'CUF', 'ACT-Wazalendo'],
    wordCloud: {
      positive: [
        { word: 'maendeleo', count: 450 },
        { word: 'mzuri', count: 380 },
        { word: 'kazi', count: 320 },
        { word: 'nzuri', count: 280 },
        { word: 'faida', count: 240 },
        { word: 'furaha', count: 220 },
        { word: 'mafanikio', count: 200 },
        { word: 'vizuri', count: 180 },
        { word: 'kubwa', count: 160 },
        { word: 'bora', count: 140 }
      ],
      negative: [
        { word: 'changamoto', count: 320 },
        { word: 'tatizo', count: 280 },
        { word: 'mbaya', count: 240 },
        { word: 'hasara', count: 200 },
        { word: 'upungufu', count: 180 },
        { word: 'kasoro', count: 160 },
        { word: 'uhalifu', count: 140 },
        { word: 'utengano', count: 120 },
        { word: 'gharama', count: 100 },
        { word: 'ukosefu', count: 80 }
      ],
      neutral: [
        { word: 'serikali', count: 500 },
        { word: 'wananchi', count: 420 },
        { word: 'nchi', count: 380 },
        { word: 'mkutano', count: 320 },
        { word: 'utawala', count: 280 },
        { word: 'siasa', count: 240 },
        { word: 'chama', count: 200 },
        { word: 'uchaguzi', count: 180 },
        { word: 'rais', count: 160 },
        { word: 'bunge', count: 140 }
      ]
    },
    processingJobs: [
      {
        job_id: 'job_001',
        filename: 'political_comments_jan_2024.csv',
        status: 'completed',
        created_at: '2024-01-15T10:00:00Z',
        processed_count: 1500
      },
      {
        job_id: 'job_002',
        filename: 'social_media_comments.csv',
        status: 'processing',
        created_at: '2024-01-15T14:30:00Z',
        processed_count: 800
      }
    ]
  };

  const API_BASE = 'http://localhost:5000/api';

  // Fetch data functions
  const fetchComments = async () => {
    if (demoMode) {
      setComments(demoData.comments);
      return;
    }
    
    try {
      const params = new URLSearchParams();
      if (filters.selectedParties.length > 0) {
        params.append('parties', filters.selectedParties.join(','));
      }
      if (filters.dateRange.start) params.append('start_date', filters.dateRange.start);
      if (filters.dateRange.end) params.append('end_date', filters.dateRange.end);
      if (filters.sentiment !== 'all') params.append('sentiment', filters.sentiment);

      const response = await fetch(`${API_BASE}/comments?${params}`);
      const data = await response.json();
      setComments(data.comments || []);
    } catch (error) {
      console.error('Error fetching comments:', error);
      setComments(demoData.comments);
    }
  };

  const fetchAnalytics = async () => {
  if (demoMode) {
    setAnalytics(demoData.analytics);
    return;
  }

  try {
    const params = new URLSearchParams();
    if (filters.selectedParties.length > 0) {
      params.append('parties', filters.selectedParties.join(','));
    }
    if (filters.dateRange.start) params.append('start_date', filters.dateRange.start);
    if (filters.dateRange.end) params.append('end_date', filters.dateRange.end);

    const response = await fetch(`${API_BASE}/analytics?${params}`);
    const data = await response.json();
    setAnalytics(data.analytics ?? data);  // ✅ Fallback handles both structures
  } catch (error) {
    console.error('Error fetching analytics:', error);
    setAnalytics(demoData.analytics);
  }
};


  const fetchParties = async () => {
    if (demoMode) {
      setParties(demoData.parties);
      return;
    }
    
    try {
      const response = await fetch(`${API_BASE}/parties`);
      const data = await response.json();
      setParties(data.parties || []);
    } catch (error) {
      console.error('Error fetching parties:', error);
      setParties(demoData.parties);
    }
  };

  const fetchWordCloud = async () => {
    if (demoMode) {
      setWordCloud(demoData.wordCloud);
      return;
    }
    
    try {
      const params = new URLSearchParams();
      if (filters.selectedParties.length > 0) {
        params.append('parties', filters.selectedParties.join(','));
      }
      if (filters.dateRange.start) params.append('start_date', filters.dateRange.start);
      if (filters.dateRange.end) params.append('end_date', filters.dateRange.end);

      const response = await fetch(`${API_BASE}/wordcloud?${params}`);
      const data = await response.json();
      setWordCloud(data);
    } catch (error) {
      console.error('Error fetching word cloud data:', error);
      setWordCloud(demoData.wordCloud);
    }
  };

  const fetchProcessingJobs = async () => {
    if (demoMode) {
      setProcessingJobs(demoData.processingJobs);
      return;
    }
    
    try {
      const response = await fetch(`${API_BASE}/processing-jobs`);
      const data = await response.json();
      setProcessingJobs(data.jobs || []);
    } catch (error) {
      console.error('Error fetching processing jobs:', error);
      setProcessingJobs(demoData.processingJobs);
    }
  };
const uploadFile = async (file) => {
  // Check if demo mode is active first
  if (demoMode) {
    console.log('Demo mode active - simulating upload');
    
    // Simulate API delay for better UX
    setLoading(true);
    await new Promise(resolve => setTimeout(resolve, 1500));
    setLoading(false);
    
    // Create a demo job for the UI
    const demoJob = {
      job_id: `demo_${Date.now()}`,
      filename: file.name,
      status: 'completed',
      created_at: new Date().toISOString(),
      processed_count: Math.floor(Math.random() * 1000) + 500
    };
    
    setProcessingJobs(prev => [demoJob, ...prev]);
    alert(`Demo mode: File "${file.name}" uploaded successfully (simulated)`);
    return;
  }

  try {
    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    console.log('Starting file upload:', file.name);
    
    const response = await fetch(`${API_BASE}/upload`, {
      method: 'POST',
      body: formData
      // Don't set Content-Type header - let browser set it with boundary
    });

    console.log('Upload response status:', response.status);
    
    // Check if response is OK (status 200-299)
    if (!response.ok) {
      // Try to get error message from response
      let errorMsg = `Server error: ${response.status}`;
      try {
        const errorData = await response.json();
        errorMsg = errorData.message || errorMsg;
      } catch (e) {
        console.error('Error parsing error response:', e);
      }
      throw new Error(errorMsg);
    }

    const data = await response.json();
    console.log('Upload response data:', data);
    
    if (!data.success) {
      throw new Error(data.message || 'Upload failed (server reported failure)');
    }

    // Success case
    alert(`File uploaded successfully! ${data.processed_count} records processed`);
    
    // Refresh data
    await Promise.all([
      fetchProcessingJobs(),
      refreshData()
    ]);
    
  } catch (error) {
    console.error('Upload error:', error);
    
    // More specific error messages
    let userMessage = 'Error uploading file';
    if (error.name === 'TypeError' && error.message.includes('Failed to fetch')) {
      userMessage = 'Failed to connect to server. Is the backend running?';
    } else if (error.message.includes('NetworkError')) {
      userMessage = 'Network error. Please check your connection.';
    } else {
      userMessage = error.message || userMessage;
    }
    
    alert(userMessage);
  } finally {
    setLoading(false);
  }
};

  const refreshData = async () => {
    setLoading(true);
    await Promise.all([
      fetchComments(),
      fetchAnalytics(),
      fetchParties(),
      fetchWordCloud(),
      fetchProcessingJobs()
    ]);
    setLoading(false);
  };

  useEffect(() => {
    refreshData();
  }, []);

  useEffect(() => {
    if (!demoMode) {
      const interval = setInterval(fetchProcessingJobs, 10000);
      return () => clearInterval(interval);
    }
  }, [demoMode]);

  const getStatusIcon = (status) => {
    switch (status) {
      case 'pending': return <Clock className="w-4 h-4 text-yellow-500" />;
      case 'processing': return <RefreshCw className="w-4 h-4 text-blue-500 animate-spin" />;
      case 'completed': return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'failed': return <XCircle className="w-4 h-4 text-red-500" />;
      default: return null;
    }
  };

  const getSentimentColor = (sentiment) => {
    switch (sentiment.toLowerCase()) {
      case 'positive': return 'text-green-600 bg-green-100';
      case 'negative': return 'text-red-600 bg-red-100';
      case 'neutral': return 'text-gray-600 bg-gray-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const formatNumber = (num) => {
    if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
    if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
    return num?.toString() || '0';
  };

  const SENTIMENT_COLORS = {
    'Positive': '#10b981',
    'Negative': '#ef4444',
    'Neutral': '#6b7280'
  };

  // const handleFileUpload = (event) => {
  //   const file = event.target.files[0];
  //   if (file && file.type === 'text/csv') {
  //     setUploadedFile(file);
  //     uploadFile(file);
  //   } else {
  //     alert('Please upload a CSV file');
  //   }
  // };

const handleFileSelect = async (event) => {
  const file = event.target.files[0];
  if (!file) return;

  console.log('Selected file:', {
    name: file.name,
    size: file.size,
    type: file.type,
    lastModified: new Date(file.lastModified)
  });

  try {
    await uploadFile(file);
  } catch (error) {
    console.error('Upload failed:', error);
    alert(`Upload failed: ${error.message}`);
  } finally {
    event.target.value = ''; // Reset input
  }
};
  const applyFilters = () => {
    refreshData();
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* Modern Header with Gradient */}
      <div className="bg-white shadow-lg border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <div className="flex items-center space-x-4">
              <div className="w-10 h-10 bg-gradient-to-r from-blue-600 to-indigo-600 rounded-xl flex items-center justify-center">
                <BarChart3 className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
                  Tanzania Political Sentiment Analyzer
                </h1>
                <p className="text-gray-600 text-sm">Analyze Swahili comments and political sentiment insights</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <button
                onClick={() => setDemoMode(!demoMode)}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
                  demoMode 
                    ? 'bg-green-100 text-green-700 hover:bg-green-200' 
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                {demoMode ? 'Demo Mode' : 'Live Mode'}
              </button>
              <button
                onClick={refreshData}
                disabled={loading}
                className="px-6 py-3 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-xl hover:from-blue-700 hover:to-indigo-700 disabled:opacity-50 flex items-center space-x-2 transition-all duration-200 shadow-lg hover:shadow-xl"
              >
                <RefreshCw className={`w-5 h-5 ${loading ? 'animate-spin' : ''}`} />
                <span>Refresh Data</span>
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="border-b border-gray-200 bg-white/80 backdrop-blur-sm rounded-b-2xl shadow-sm">
          <nav className="-mb-px flex space-x-8 px-6">
            {[
              { id: 'overview', name: 'Overview', icon: TrendingUp },
              { id: 'upload', name: 'Upload Data', icon: Upload },
              { id: 'comments', name: 'Comments', icon: MessageCircle },
              { id: 'parties', name: 'Political Parties', icon: Users },
              { id: 'trends', name: 'Sentiment Trends', icon: BarChart3 },
              { id: 'wordcloud', name: 'Word Analysis', icon: Cloud }
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center space-x-2 py-4 px-2 border-b-2 font-medium text-sm transition-all duration-200 ${
                  activeTab === tab.id
                    ? 'border-blue-500 text-blue-600 bg-blue-50/50'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300 hover:bg-gray-50/50'
                } rounded-t-lg`}
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
          <div className="space-y-8">
            {/* Stats Cards with Modern Design */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <div className="bg-white rounded-2xl shadow-lg p-6 border border-gray-100 hover:shadow-xl transition-shadow duration-300">
                <div className="flex items-center">
                  <div className="w-12 h-12 bg-gradient-to-r from-blue-500 to-blue-600 rounded-xl flex items-center justify-center">
                    <MessageCircle className="w-6 h-6 text-white" />
                  </div>
                  <div className="ml-4">
                    <p className="text-sm font-medium text-gray-600">Total Comments</p>
                    <p className="text-2xl font-bold text-gray-900">{formatNumber(analytics.summary?.total_comments || 0)}</p>
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-2xl shadow-lg p-6 border border-gray-100 hover:shadow-xl transition-shadow duration-300">
                <div className="flex items-center">
                  <div className="w-12 h-12 bg-gradient-to-r from-green-500 to-emerald-600 rounded-xl flex items-center justify-center">
                    <TrendingUp className="w-6 h-6 text-white" />
                  </div>
                  <div className="ml-4">
                    <p className="text-sm font-medium text-gray-600">Positive Sentiment</p>
                    <p className="text-2xl font-bold text-gray-900">{analytics.summary?.positive_count || 0}</p>
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-2xl shadow-lg p-6 border border-gray-100 hover:shadow-xl transition-shadow duration-300">
                <div className="flex items-center">
                  <div className="w-12 h-12 bg-gradient-to-r from-red-500 to-red-600 rounded-xl flex items-center justify-center">
                    <TrendingUp className="w-6 h-6 text-white transform rotate-180" />
                  </div>
                  <div className="ml-4">
                    <p className="text-sm font-medium text-gray-600">Negative Sentiment</p>
                    <p className="text-2xl font-bold text-gray-900">{analytics.summary?.negative_count || 0}</p>
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-2xl shadow-lg p-6 border border-gray-100 hover:shadow-xl transition-shadow duration-300">
                <div className="flex items-center">
                  <div className="w-12 h-12 bg-gradient-to-r from-purple-500 to-purple-600 rounded-xl flex items-center justify-center">
                    <Users className="w-6 h-6 text-white" />
                  </div>
                  <div className="ml-4">
                    <p className="text-sm font-medium text-gray-600">Political Parties</p>
                    <p className="text-2xl font-bold text-gray-900">{parties.length}</p>
                  </div>
                </div>
              </div>
            </div>

            {/* Charts */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              {/* Sentiment Distribution */}
              <div className="bg-white rounded-2xl shadow-lg p-6 border border-gray-100">
                <h3 className="text-xl font-bold text-gray-900 mb-6">Sentiment Distribution by Party</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={analytics.party_sentiment || []}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                    <XAxis dataKey="party" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="positive" fill="#10b981" name="Positive" />
                    <Bar dataKey="negative" fill="#ef4444" name="Negative" />
                    <Bar dataKey="neutral" fill="#6b7280" name="Neutral" />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {/* Sentiment Pie Chart */}
              <div className="bg-white rounded-2xl shadow-lg p-6 border border-gray-100">
                <h3 className="text-xl font-bold text-gray-900 mb-6">Overall Sentiment Distribution</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={analytics.overall_sentiment || []}
                      cx="50%"
                      cy="50%"
                      outerRadius={100}
                      fill="#8884d8"
                      dataKey="count"
                      label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                    >
                      {(analytics.overall_sentiment || []).map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={SENTIMENT_COLORS[entry.sentiment]} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        )}

        {/* Upload Tab */}
        {activeTab === 'upload' && (
          <div className="space-y-8">
            {/* Upload Section */}
            <div className="bg-white rounded-2xl shadow-lg p-8 border border-gray-100">
              <h3 className="text-2xl font-bold text-gray-900 mb-6">Upload CSV File</h3>
              {demoMode && (
                <div className="mb-4 p-4 bg-blue-50 border border-blue-200 rounded-lg">
                  <p className="text-blue-800 text-sm">
                    <strong>Demo Mode:</strong> File upload is simulated. Switch to Live Mode to upload actual files.
                  </p>
                </div>
              )}
              <div className="border-2 border-dashed border-gray-300 rounded-xl p-8 text-center hover:border-blue-400 transition-colors duration-200">
                <Upload className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                <p className="text-gray-600 mb-4">
                  Upload your CSV file with Swahili comments. Required columns: Comment, Party, Date
                </p>

                <input
                  type="file"
                  accept=".csv"
                  onChange={handleFileSelect}
                  // onChange={uploadFile}
                  className="hidden"
                  id="csv-upload"
                />
                           



                <label
                  htmlFor="csv-upload"
                  className="px-6 py-3 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-xl hover:from-blue-700 hover:to-indigo-700 cursor-pointer inline-flex items-center space-x-2 transition-all duration-200 shadow-lg hover:shadow-xl"
                >
                  <Upload className="w-5 h-5" />
                  <span>Choose CSV File</span>
                </label>
              </div>
            </div>

            {/* Processing Jobs */}
            <div className="bg-white rounded-2xl shadow-lg border border-gray-100">
              <div className="px-8 py-6 border-b border-gray-200">
                <h3 className="text-xl font-bold text-gray-900">Processing Jobs</h3>
              </div>
              <div className="divide-y divide-gray-200">
                {processingJobs.map((job) => (
                  <div key={job.job_id} className="p-6">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-3">
                        {getStatusIcon(job.status)}
                        <div>
                          <p className="font-medium text-gray-900">{job.filename}</p>
                          <p className="text-sm text-gray-500">
                            Started: {new Date(job.created_at).toLocaleString()}
                          </p>
                        </div>
                      </div>
                      <div className="text-right">
                        <p className="text-sm font-medium text-gray-900">
                          Status: <span className={`capitalize ${
                            job.status === 'completed' ? 'text-green-600' : 
                            job.status === 'failed' ? 'text-red-600' : 
                            job.status === 'processing' ? 'text-blue-600' : 'text-yellow-600'
                          }`}>{job.status}</span>
                        </p>
                        {job.processed_count > 0 && (
                          <p className="text-sm text-gray-500">Processed: {job.processed_count}</p>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
                {processingJobs.length === 0 && (
                  <div className="p-8 text-center text-gray-500">
                    No processing jobs found. Upload a CSV file to get started.
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Comments Tab */}
        {activeTab === 'comments' && (
          <div className="space-y-6">
            {/* Filters */}
            <div className="bg-white rounded-2xl shadow-lg p-6 border border-gray-100">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Filter Options</h3>
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <select
                  multiple
                  value={filters.selectedParties}
                  onChange={(e) => setFilters({
                    ...filters, 
                    selectedParties: Array.from(e.target.selectedOptions, option => option.value)
                  })}
                  className="px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  {parties.map(party => (
                    <option key={party} value={party}>{party}</option>
                  ))}
                </select>
                <input
                  type="date"
                  value={filters.dateRange.start}
                  onChange={(e) => setFilters({
                    ...filters, 
                    dateRange: { ...filters.dateRange, start: e.target.value }
                  })}
                  className="px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
                <input
                  type="date"
                  value={filters.dateRange.end}
                  onChange={(e) => setFilters({
                    ...filters, 
                    dateRange: { ...filters.dateRange, end: e.target.value }
                  })}
                  className="px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
                <select
                  value={filters.sentiment}
                  onChange={(e) => setFilters({...filters, sentiment: e.target.value})}
                  className="px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="all">All Sentiments</option>
                  <option value="positive">Positive</option>
                  <option value="negative">Negative</option>
                  <option value="neutral">Neutral</option>
                </select>
              </div>
              <div className="mt-4">
                <button
                  onClick={applyFilters}
                  className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors duration-200"
                >
                  Apply Filters
                </button>
              </div>
            </div>

            {/* Comments List */}
            <div className="bg-white rounded-2xl shadow-lg border border-gray-100">
              <div className="px-6 py-4 border-b border-gray-200">
                <h3 className="text-lg font-semibold text-gray-900">Comments ({comments.length})</h3>
              </div>
              <div className="divide-y divide-gray-200">
                {comments.length > 0 ? (
                  comments.map((comment, index) => (
                    <div key={index} className="p-6">
                      <div className="flex justify-between items-start">
                        <div>
                          <p className="text-gray-800 mb-2">{comment.comment}</p>
                          <div className="flex items-center space-x-4">
                            <span className="text-sm text-gray-500">
                              <span className="font-medium">Party:</span> {comment.party}
                            </span>
                            <span className="text-sm text-gray-500">
                              <span className="font-medium">Date:</span> {new Date(comment.date).toLocaleString()}
                            </span>
                          </div>
                        </div>
                        <span className={`px-3 py-1 rounded-full text-xs font-medium ${getSentimentColor(comment.sentiment_label)}`}>
                          {comment.sentiment_label}
                        </span>
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="p-8 text-center text-gray-500">
                    No comments found matching your filters.
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Political Parties Tab */}
        {activeTab === 'parties' && (
          <div className="space-y-8">
            <div className="bg-white rounded-2xl shadow-lg p-6 border border-gray-100">
              <h3 className="text-xl font-bold text-gray-900 mb-6">Political Parties Analysis</h3>
              
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                {/* Party Sentiment Breakdown */}
                <div>
                  <h4 className="text-lg font-semibold text-gray-800 mb-4">Sentiment by Party</h4>
                  <div className="space-y-4">
                    {analytics?.party_sentiment?.map((partyData) => (
                      <div key={partyData.party} className="bg-gray-50 rounded-lg p-4">
                        <div className="flex justify-between items-center mb-2">
                          <h5 className="font-medium text-gray-900">{partyData.party}</h5>
                          <span className="text-sm text-gray-500">Total: {partyData.total}</span>
                        </div>
                        <div className="space-y-2">
                          <div className="flex items-center">
                            <div className="w-16 text-sm text-green-600">Positive</div>
                            <div className="flex-1 bg-gray-200 rounded-full h-2.5">
                              <div 
                                className="bg-green-500 h-2.5 rounded-full" 
                                style={{ width: `${(partyData.positive / partyData.total) * 100}%` }}
                              ></div>
                            </div>
                            <div className="w-16 text-right text-sm text-gray-600">{partyData.positive}</div>
                          </div>
                          <div className="flex items-center">
                            <div className="w-16 text-sm text-red-600">Negative</div>
                            <div className="flex-1 bg-gray-200 rounded-full h-2.5">
                              <div 
                                className="bg-red-500 h-2.5 rounded-full" 
                                style={{ width: `${(partyData.negative / partyData.total) * 100}%` }}
                              ></div>
                            </div>
                            <div className="w-16 text-right text-sm text-gray-600">{partyData.negative}</div>
                          </div>
                          <div className="flex items-center">
                            <div className="w-16 text-sm text-gray-600">Neutral</div>
                            <div className="flex-1 bg-gray-200 rounded-full h-2.5">
                              <div 
                                className="bg-gray-500 h-2.5 rounded-full" 
                                style={{ width: `${(partyData.neutral / partyData.total) * 100}%` }}
                              ></div>
                            </div>
                            <div className="w-16 text-right text-sm text-gray-600">{partyData.neutral}</div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Party Comparison Chart */}
                <div>
                  <h4 className="text-lg font-semibold text-gray-800 mb-4">Party Comparison</h4>
                  <ResponsiveContainer width="100%" height={400}>
                    <BarChart
                      data={analytics?.party_sentiment || []}
                      layout="vertical"
                      margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis type="number" />
                      <YAxis dataKey="party" type="category" />
                      <Tooltip />
                      <Legend />
                      <Bar dataKey="positive" fill="#10b981" name="Positive" />
                      <Bar dataKey="negative" fill="#ef4444" name="Negative" />
                      <Bar dataKey="neutral" fill="#6b7280" name="Neutral" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Sentiment Trends Tab */}
        {activeTab === 'trends' && analytics && (
          <div className="space-y-8">
            <div className="bg-white rounded-2xl shadow-lg p-6 border border-gray-100">
              <h3 className="text-xl font-bold text-gray-900 mb-6">Sentiment Trends Over Time</h3>
              
              <div className="grid grid-cols-1 gap-8">
                {/* Daily Trends Line Chart */}
                <div>
                  <h4 className="text-lg font-semibold text-gray-800 mb-4">Daily Sentiment Trends</h4>
                  <ResponsiveContainer width="100%" height={400}>
                    <LineChart
                      data={analytics.daily_trends || []}
                      margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="date" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Line type="monotone" dataKey="positive" stroke="#10b981" name="Positive" />
                      <Line type="monotone" dataKey="negative" stroke="#ef4444" name="Negative" />
                      <Line type="monotone" dataKey="neutral" stroke="#6b7280" name="Neutral" />
                    </LineChart>
                  </ResponsiveContainer>
                </div>

                {/* Stacked Area Chart */}
                <div>
                  <h4 className="text-lg font-semibold text-gray-800 mb-4">Sentiment Distribution Over Time</h4>
                  <ResponsiveContainer width="100%" height={400}>
                    <LineChart
                      data={analytics.daily_trends || []}
                      margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="date" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Line 
                        type="monotone" 
                        dataKey="positive" 
                        stroke="#10b981" 
                        fill="#10b981" 
                        name="Positive" 
                        stackId="1" 
                      />
                      <Line 
                        type="monotone" 
                        dataKey="negative" 
                        stroke="#ef4444" 
                        fill="#ef4444" 
                        name="Negative" 
                        stackId="1" 
                      />
                      <Line 
                        type="monotone" 
                        dataKey="neutral" 
                        stroke="#6b7280" 
                        fill="#6b7280" 
                        name="Neutral" 
                        stackId="1" 
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Word Analysis Tab */}
        {activeTab === 'wordcloud' && (
          <div className="space-y-8">
            <div className="bg-white rounded-2xl shadow-lg p-6 border border-gray-100">
              <h3 className="text-xl font-bold text-gray-900 mb-6">Word Frequency Analysis</h3>
              
              <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                {/* Positive Words */}
                <div className="bg-green-50 rounded-xl p-6">
                  <div className="flex items-center justify-between mb-4">
                    <h4 className="text-lg font-semibold text-green-800">Positive Words</h4>
                    <div className="w-8 h-8 bg-green-100 rounded-full flex items-center justify-center">
                      <TrendingUp className="w-4 h-4 text-green-600" />
                    </div>
                  </div>
                  <div className="space-y-3">
                    {wordCloud.positive?.map((wordData, index) => (
                      <div key={index} className="flex items-center justify-between">
                        <span className="text-green-800 font-medium">{wordData.word}</span>
                        <span className="text-green-600">{wordData.count}</span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Negative Words */}
                <div className="bg-red-50 rounded-xl p-6">
                  <div className="flex items-center justify-between mb-4">
                    <h4 className="text-lg font-semibold text-red-800">Negative Words</h4>
                    <div className="w-8 h-8 bg-red-100 rounded-full flex items-center justify-center">
                      <TrendingUp className="w-4 h-4 text-red-600 transform rotate-180" />
                    </div>
                  </div>
                  <div className="space-y-3">
                    {wordCloud.negative?.map((wordData, index) => (
                      <div key={index} className="flex items-center justify-between">
                        <span className="text-red-800 font-medium">{wordData.word}</span>
                        <span className="text-red-600">{wordData.count}</span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Neutral Words */}
                <div className="bg-gray-50 rounded-xl p-6">
                  <div className="flex items-center justify-between mb-4">
                    <h4 className="text-lg font-semibold text-gray-800">Neutral Words</h4>
                    <div className="w-8 h-8 bg-gray-100 rounded-full flex items-center justify-center">
                      <Hash className="w-4 h-4 text-gray-600" />
                    </div>
                  </div>
                  <div className="space-y-3">
                    {wordCloud.neutral?.map((wordData, index) => (
                      <div key={index} className="flex items-center justify-between">
                        <span className="text-gray-800 font-medium">{wordData.word}</span>
                        <span className="text-gray-600">{wordData.count}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* Word Cloud Visualization */}
              <div className="mt-8">
                <h4 className="text-lg font-semibold text-gray-800 mb-4">Word Cloud Visualization</h4>
                <div className="bg-gray-50 rounded-xl p-8 text-center">
                  <div className="flex flex-wrap justify-center gap-4">
                    {wordCloud.positive?.slice(0, 15).map((wordData, index) => (
                      <span 
                        key={`positive-${index}`} 
                        className="inline-block px-3 py-1 rounded-full text-green-700 bg-green-100"
                        style={{ fontSize: `${Math.min(24, 14 + (wordData.count / 50))}px` }}
                      >
                        {wordData.word}
                      </span>
                    ))}
                    {wordCloud.negative?.slice(0, 15).map((wordData, index) => (
                      <span 
                        key={`negative-${index}`} 
                        className="inline-block px-3 py-1 rounded-full text-red-700 bg-red-100"
                        style={{ fontSize: `${Math.min(24, 14 + (wordData.count / 50))}px` }}
                      >
                        {wordData.word}
                      </span>
                    ))}
                    {wordCloud.neutral?.slice(0, 15).map((wordData, index) => (
                      <span 
                        key={`neutral-${index}`} 
                        className="inline-block px-3 py-1 rounded-full text-gray-700 bg-gray-200"
                        style={{ fontSize: `${Math.min(24, 14 + (wordData.count / 50))}px` }}
                      >
                        {wordData.word}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default SentimentDashboard;