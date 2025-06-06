#!/usr/bin/env python3
"""
Snowflake Cost Management Tool - Web Application

This module implements a Flask web application that demonstrates the
capabilities of the Snowflake cost management tool with AI features.
"""

from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import json
import os
import logging
from datetime import datetime, timedelta
import joblib

# Import our models
from warehouse_sizing_model import WarehouseSizingModel, WarehouseSizingRecommender
from query_optimization_model import QueryOptimizationModel, QueryOptimizationRecommender

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Initialize models
warehouse_model = None
query_model = None

# Initialize recommenders
warehouse_recommender = None
query_recommender = None

# Model paths
WAREHOUSE_MODEL_PATH = 'warehouse_sizing_model.joblib'
QUERY_MODEL_PATH = 'query_optimization_model.joblib'

# Sample data for demonstration
sample_data = None


def load_models():
    """
    Load pre-trained models if available, otherwise create new ones.
    """
    global warehouse_model, query_model, warehouse_recommender, query_recommender
    
    # Load warehouse sizing model
    if os.path.exists(WAREHOUSE_MODEL_PATH):
        logger.info(f"Loading warehouse sizing model from {WAREHOUSE_MODEL_PATH}")
        warehouse_model = WarehouseSizingModel(WAREHOUSE_MODEL_PATH)
    else:
        logger.info("Creating new warehouse sizing model")
        warehouse_model = WarehouseSizingModel()
    
    # Load query optimization model
    if os.path.exists(QUERY_MODEL_PATH):
        logger.info(f"Loading query optimization model from {QUERY_MODEL_PATH}")
        query_model = QueryOptimizationModel(QUERY_MODEL_PATH)
    else:
        logger.info("Creating new query optimization model")
        query_model = QueryOptimizationModel()
    
    # Create recommenders
    warehouse_recommender = WarehouseSizingRecommender(warehouse_model)
    query_recommender = QueryOptimizationRecommender(query_model)


def generate_sample_data():
    """
    Generate sample data for demonstration purposes.
    """
    global sample_data
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 100
    
    # Generate timestamps for the last 30 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    timestamps = [start_date + timedelta(
        seconds=np.random.randint(0, 30*24*60*60)
    ) for _ in range(n_samples)]
    
    # Generate warehouse names
    warehouse_names = [f"WH_{i}" for i in range(1, 6)]
    warehouse_distribution = np.random.choice(
        warehouse_names,
        size=n_samples
    )
    
    # Generate warehouse sizes
    warehouse_sizes = np.random.choice(
        ['X-SMALL', 'SMALL', 'MEDIUM', 'LARGE', 'X-LARGE'],
        size=n_samples,
        p=[0.3, 0.3, 0.2, 0.15, 0.05]
    )
    
    # Generate credits used
    credits_used = []
    for size in warehouse_sizes:
        size_idx = ['X-SMALL', 'SMALL', 'MEDIUM', 'LARGE', 'X-LARGE'].index(size)
        # Larger warehouses use more credits
        mean_credits = 0.5 * (2**size_idx)
        credits_used.append(np.random.lognormal(mean=np.log(mean_credits), sigma=0.5))
    
    # Generate sample queries
    sample_queries = [
        "SELECT * FROM customers",
        "SELECT id, name FROM customers WHERE region = 'WEST'",
        "SELECT c.id, c.name, o.order_id FROM customers c JOIN orders o ON c.id = o.customer_id",
        "SELECT * FROM customers WHERE UPPER(name) = 'JOHN'",
        "SELECT * FROM customers c, orders o WHERE c.id = o.customer_id",
        "SELECT DISTINCT name FROM customers",
        "SELECT id FROM (SELECT id FROM customers WHERE region = 'WEST')",
        "SELECT * FROM customers ORDER BY name",
        "SELECT COUNT(*) FROM customers GROUP BY region",
        "SELECT * FROM customers WHERE name = 'John' OR name = 'Jane'"
    ]
    
    # Generate query texts
    query_texts = []
    for _ in range(n_samples):
        query_idx = np.random.randint(0, len(sample_queries))
        query_texts.append(sample_queries[query_idx])
    
    # Generate bytes scanned
    bytes_scanned = np.random.lognormal(mean=15, sigma=2, size=n_samples)
    
    # Generate rows produced
    rows_produced = np.random.lognormal(mean=5, sigma=2, size=n_samples)
    
    # Generate execution times
    execution_times = np.random.lognormal(mean=7, sigma=1.5, size=n_samples)
    
    # Create DataFrame for warehouse metrics
    warehouse_data = {
        'WAREHOUSE_NAME': warehouse_distribution,
        'WAREHOUSE_SIZE': warehouse_sizes,
        'START_TIME': timestamps,
        'CREDITS_USED': credits_used
    }
    
    # Create DataFrame for query history
    query_data = {
        'QUERY_ID': [f'query_{i}' for i in range(n_samples)],
        'QUERY_TEXT': query_texts,
        'WAREHOUSE_NAME': warehouse_distribution,
        'WAREHOUSE_SIZE': warehouse_sizes,
        'START_TIME': timestamps,
        'BYTES_SCANNED': bytes_scanned,
        'ROWS_PRODUCED': rows_produced,
        'EXECUTION_TIME': execution_times,
        'COMPILATION_TIME': np.random.lognormal(mean=4, sigma=1, size=n_samples),
        'BYTES_SPILLED_TO_LOCAL_STORAGE': np.random.lognormal(mean=10, sigma=2, size=n_samples),
        'BYTES_SPILLED_TO_REMOTE_STORAGE': np.random.lognormal(mean=5, sigma=2, size=n_samples),
        'PARTITIONS_SCANNED': np.random.randint(1, 100, size=n_samples),
        'PARTITIONS_TOTAL': np.random.randint(50, 200, size=n_samples)
    }
    
    # Ensure partitions_scanned <= partitions_total
    for i in range(n_samples):
        query_data['PARTITIONS_SCANNED'][i] = min(
            query_data['PARTITIONS_SCANNED'][i], 
            query_data['PARTITIONS_TOTAL'][i]
        )
    
    # Create DataFrames
    warehouse_df = pd.DataFrame(warehouse_data)
    query_df = pd.DataFrame(query_data)
    
    # Store sample data
    sample_data = {
        'warehouse_metrics': warehouse_df,
        'query_history': query_df
    }
    
    # Generate warehouse usage summary
    warehouse_summary = warehouse_df.groupby('WAREHOUSE_NAME').agg({
        'CREDITS_USED': 'sum'
    }).reset_index()
    
    warehouse_summary['COST'] = warehouse_summary['CREDITS_USED'] * 3  # Assuming $3 per credit
    
    # Generate size distribution
    size_distribution = warehouse_df.groupby('WAREHOUSE_SIZE').size().reset_index()
    size_distribution.columns = ['WAREHOUSE_SIZE', 'COUNT']
    
    # Generate daily usage
    daily_usage = warehouse_df.copy()
    daily_usage['DATE'] = daily_usage['START_TIME'].dt.date
    daily_usage = daily_usage.groupby('DATE').agg({
        'CREDITS_USED': 'sum'
    }).reset_index()
    
    # Convert to lists for JSON serialization
    warehouse_summary_list = warehouse_summary.to_dict('records')
    size_distribution_list = size_distribution.to_dict('records')
    daily_usage_list = [{
        'DATE': date.strftime('%Y-%m-%d'),
        'CREDITS_USED': credits
    } for date, credits in zip(daily_usage['DATE'], daily_usage['CREDITS_USED'])]
    
    # Store processed data for API endpoints
    sample_data['warehouse_summary'] = warehouse_summary_list
    sample_data['size_distribution'] = size_distribution_list
    sample_data['daily_usage'] = daily_usage_list
    
    # Generate query optimization candidates
    query_candidates = query_df.sample(5).to_dict('records')
    sample_data['query_candidates'] = query_candidates
    
    # Generate warehouse sizing recommendations
    warehouse_recommendations = []
    for name in warehouse_names:
        warehouse_df_filtered = warehouse_df[warehouse_df['WAREHOUSE_NAME'] == name]
        if len(warehouse_df_filtered) > 0:
            current_size = warehouse_df_filtered['WAREHOUSE_SIZE'].iloc[0]
            
            # Generate a recommendation (simplified for demo)
            size_idx = ['X-SMALL', 'SMALL', 'MEDIUM', 'LARGE', 'X-LARGE'].index(current_size)
            
            # Randomly decide to upsize, downsize, or keep the same
            action = np.random.choice(['upsize', 'downsize', 'same'], p=[0.3, 0.3, 0.4])
            
            if action == 'upsize' and size_idx < 4:
                recommended_size = ['X-SMALL', 'SMALL', 'MEDIUM', 'LARGE', 'X-LARGE'][size_idx + 1]
                reason = "Warehouse is frequently queued and spilling data"
                savings = -50  # Negative savings (cost increase)
            elif action == 'downsize' and size_idx > 0:
                recommended_size = ['X-SMALL', 'SMALL', 'MEDIUM', 'LARGE', 'X-LARGE'][size_idx - 1]
                reason = "Warehouse is oversized for current workload"
                savings = 50  # Positive savings
            else:
                recommended_size = current_size
                reason = "Current size is optimal for workload"
                savings = 0
            
            warehouse_recommendations.append({
                'WAREHOUSE_NAME': name,
                'CURRENT_SIZE': current_size,
                'RECOMMENDED_SIZE': recommended_size,
                'REASON': reason,
                'ESTIMATED_SAVINGS': savings
            })
    
    sample_data['warehouse_recommendations'] = warehouse_recommendations


@app.route('/')
def index():
    """
    Render the main dashboard page.
    """
    return render_template('index.html')


@app.route('/api/warehouse/summary')
def warehouse_summary():
    """
    API endpoint for warehouse usage summary.
    """
    return jsonify(sample_data['warehouse_summary'])


@app.route('/api/warehouse/size_distribution')
def warehouse_size_distribution():
    """
    API endpoint for warehouse size distribution.
    """
    return jsonify(sample_data['size_distribution'])


@app.route('/api/warehouse/daily_usage')
def warehouse_daily_usage():
    """
    API endpoint for daily warehouse usage.
    """
    return jsonify(sample_data['daily_usage'])


@app.route('/api/warehouse/recommendations')
def warehouse_recommendations():
    """
    API endpoint for warehouse sizing recommendations.
    """
    return jsonify(sample_data['warehouse_recommendations'])


@app.route('/api/query/candidates')
def query_candidates():
    """
    API endpoint for query optimization candidates.
    """
    return jsonify(sample_data['query_candidates'])


@app.route('/api/warehouse/recommend', methods=['POST'])
def recommend_warehouse_size():
    """
    API endpoint for warehouse sizing recommendation.
    """
    data = request.json
    
    # Extract query features
    query_features = {
        'BYTES_SCANNED': data.get('bytes_scanned', 0),
        'EXECUTION_TIME': data.get('execution_time', 0),
        'COMPILATION_TIME': data.get('compilation_time', 0),
        'BYTES_SPILLED_TO_LOCAL_STORAGE': data.get('bytes_spilled_local', 0),
        'BYTES_SPILLED_TO_REMOTE_STORAGE': data.get('bytes_spilled_remote', 0),
        'PARTITIONS_SCANNED': data.get('partitions_scanned', 0),
        'PARTITIONS_TOTAL': data.get('partitions_total', 100),
        'PARTITION_SCAN_RATIO': data.get('partitions_scanned', 0) / max(data.get('partitions_total', 100), 1),
        'SPILL_RATIO': (data.get('bytes_spilled_local', 0) + data.get('bytes_spilled_remote', 0)) / max(data.get('bytes_scanned', 1), 1),
        'HOUR_OF_DAY': datetime.now().hour,
        'DAY_OF_WEEK': datetime.now().weekday(),
        'IS_WEEKEND': 1 if datetime.now().weekday() >= 5 else 0,
        'QUERY_TYPE': data.get('query_type', 'SELECT')
    }
    
    current_size = data.get('current_size', 'SMALL')
    
    # Get recommendation
    if warehouse_recommender:
        recommendation = warehouse_recommender.recommend_warehouse_size(
            query_features, current_size
        )
        return jsonify(recommendation)
    else:
        return jsonify({
            'error': 'Warehouse sizing model not loaded'
        }), 500


@app.route('/api/query/analyze', methods=['POST'])
def analyze_query():
    """
    API endpoint for query analysis and optimization.
    """
    data = request.json
    query_text = data.get('query_text', '')
    
    # Extract query metrics if provided
    query_metrics = None
    if 'metrics' in data:
        query_metrics = data['metrics']
    
    # Analyze query
    if query_recommender:
        analysis = query_recommender.analyze_query(query_text, query_metrics)
        return jsonify(analysis)
    else:
        return jsonify({
            'error': 'Query optimization model not loaded'
        }), 500


@app.route('/api/train/warehouse', methods=['POST'])
def train_warehouse_model():
    """
    API endpoint for training the warehouse sizing model.
    """
    # In a real application, this would use actual data from Snowflake
    # For demonstration, we'll use our sample data
    if sample_data and 'query_history' in sample_data:
        try:
            # Train model
            metrics = warehouse_model.train(
                sample_data['query_history'],
                model_path=WAREHOUSE_MODEL_PATH
            )
            
            # Update recommender
            global warehouse_recommender
            warehouse_recommender = WarehouseSizingRecommender(warehouse_model)
            
            return jsonify({
                'success': True,
                'metrics': metrics
            })
        except Exception as e:
            logger.error(f"Error training warehouse model: {e}")
            return jsonify({
                'error': str(e)
            }), 500
    else:
        return jsonify({
            'error': 'No training data available'
        }), 400


@app.route('/api/train/query', methods=['POST'])
def train_query_model():
    """
    API endpoint for training the query optimization model.
    """
    # In a real application, this would use actual data from Snowflake
    # For demonstration, we'll use our sample data
    if sample_data and 'query_history' in sample_data:
        try:
            # Train model
            metrics = query_model.train(
                sample_data['query_history'],
                model_path=QUERY_MODEL_PATH
            )
            
            # Update recommender
            global query_recommender
            query_recommender = QueryOptimizationRecommender(query_model)
            
            return jsonify({
                'success': True,
                'metrics': metrics
            })
        except Exception as e:
            logger.error(f"Error training query model: {e}")
            return jsonify({
                'error': str(e)
            }), 500
    else:
        return jsonify({
            'error': 'No training data available'
        }), 400


def create_templates():
    """
    Create HTML templates for the web application.
    """
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Create index.html template
    index_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Snowflake Cost Management Tool</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            padding-top: 20px;
        }
        .card {
            margin-bottom: 20px;
        }
        .recommendation-card {
            border-left: 4px solid #007bff;
        }
        .recommendation-card.savings {
            border-left-color: #28a745;
        }
        .recommendation-card.cost {
            border-left-color: #dc3545;
        }
        .query-card {
            border-left: 4px solid #17a2b8;
        }
        pre {
            background-color: #f8f9fa;
            padding: 10px;
            border-radius: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <header class="mb-4">
            <h1 class="display-4">Snowflake Cost Management Tool</h1>
            <p class="lead">AI-powered optimization for Snowflake environments</p>
        </header>
        
        <div class="row">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5>Warehouse Usage Summary</h5>
                    </div>
                    <div class="card-body">
                        <canvas id="warehouseUsageChart"></canvas>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5>Warehouse Size Distribution</h5>
                    </div>
                    <div class="card-body">
                        <canvas id="warehouseSizeChart"></canvas>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row">
            <div class="col-12">
                <div class="card">
                    <div class="card-header">
                        <h5>Daily Credit Usage</h5>
                    </div>
                    <div class="card-body">
                        <canvas id="dailyUsageChart"></canvas>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row">
            <div class="col-12">
                <h2>Warehouse Sizing Recommendations</h2>
                <div id="warehouseRecommendations"></div>
            </div>
        </div>
        
        <div class="row mt-4">
            <div class="col-12">
                <h2>Query Optimization Candidates</h2>
                <div id="queryCandidates"></div>
            </div>
        </div>
        
        <div class="row mt-4">
            <div class="col-12">
                <div class="card">
                    <div class="card-header">
                        <h5>Query Analyzer</h5>
                    </div>
                    <div class="card-body">
                        <div class="mb-3">
                            <label for="queryInput" class="form-label">Enter SQL Query</label>
                            <textarea class="form-control" id="queryInput" rows="5"></textarea>
                        </div>
                        <button class="btn btn-primary" id="analyzeButton">Analyze Query</button>
                        <div class="mt-3" id="queryAnalysisResult"></div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row mt-4">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5>Train Warehouse Sizing Model</h5>
                    </div>
                    <div class="card-body">
                        <p>Train the warehouse sizing model using sample data.</p>
                        <button class="btn btn-success" id="trainWarehouseButton">Train Model</button>
                        <div class="mt-3" id="warehouseTrainingResult"></div>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5>Train Query Optimization Model</h5>
                    </div>
                    <div class="card-body">
                        <p>Train the query optimization model using sample data.</p>
                        <button class="btn btn-success" id="trainQueryButton">Train Model</button>
                        <div class="mt-3" id="queryTrainingResult"></div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Fetch data and initialize charts when page loads
        document.addEventListener('DOMContentLoaded', function() {
            fetchWarehouseSummary();
            fetchWarehouseSizeDistribution();
            fetchDailyUsage();
            fetchWarehouseRecommendations();
            fetchQueryCandidates();
            
            // Set up event listeners
            document.getElementById('analyzeButton').addEventListener('click', analyzeQuery);
            document.getElementById('trainWarehouseButton').addEventListener('click', trainWarehouseModel);
            document.getElementById('trainQueryButton').addEventListener('click', trainQueryModel);
        });
        
        // Fetch warehouse usage summary
        function fetchWarehouseSummary() {
            fetch('/api/warehouse/summary')
                .then(response => response.json())
                .then(data => {
                    const ctx = document.getElementById('warehouseUsageChart').getContext('2d');
                    new Chart(ctx, {
                        type: 'bar',
                        data: {
                            labels: data.map(item => item.WAREHOUSE_NAME),
                            datasets: [{
                                label: 'Credits Used',
                                data: data.map(item => item.CREDITS_USED),
                                backgroundColor: 'rgba(54, 162, 235, 0.5)',
                                borderColor: 'rgba(54, 162, 235, 1)',
                                borderWidth: 1
                            }, {
                                label: 'Cost ($)',
                                data: data.map(item => item.COST),
                                backgroundColor: 'rgba(255, 99, 132, 0.5)',
                                borderColor: 'rgba(255, 99, 132, 1)',
                                borderWidth: 1,
                                yAxisID: 'y1'
                            }]
                        },
                        options: {
                            scales: {
                                y: {
                                    beginAtZero: true,
                                    title: {
                                        display: true,
                                        text: 'Credits'
                                    }
                                },
                                y1: {
                                    beginAtZero: true,
                                    position: 'right',
                                    title: {
                                        display: true,
                                        text: 'Cost ($)'
                                    },
                                    grid: {
                                        drawOnChartArea: false
                                    }
                                }
                            }
                        }
                    });
                })
                .catch(error => console.error('Error fetching warehouse summary:', error));
        }
        
        // Fetch warehouse size distribution
        function fetchWarehouseSizeDistribution() {
            fetch('/api/warehouse/size_distribution')
                .then(response => response.json())
                .then(data => {
                    const ctx = document.getElementById('warehouseSizeChart').getContext('2d');
                    new Chart(ctx, {
                        type: 'pie',
                        data: {
                            labels: data.map(item => item.WAREHOUSE_SIZE),
                            datasets: [{
                                data: data.map(item => item.COUNT),
                                backgroundColor: [
                                    'rgba(255, 99, 132, 0.5)',
                                    'rgba(54, 162, 235, 0.5)',
                                    'rgba(255, 206, 86, 0.5)',
                                    'rgba(75, 192, 192, 0.5)',
                                    'rgba(153, 102, 255, 0.5)'
                                ],
                                borderColor: [
                                    'rgba(255, 99, 132, 1)',
                                    'rgba(54, 162, 235, 1)',
                                    'rgba(255, 206, 86, 1)',
                                    'rgba(75, 192, 192, 1)',
                                    'rgba(153, 102, 255, 1)'
                                ],
                                borderWidth: 1
                            }]
                        }
                    });
                })
                .catch(error => console.error('Error fetching warehouse size distribution:', error));
        }
        
        // Fetch daily usage
        function fetchDailyUsage() {
            fetch('/api/warehouse/daily_usage')
                .then(response => response.json())
                .then(data => {
                    const ctx = document.getElementById('dailyUsageChart').getContext('2d');
                    new Chart(ctx, {
                        type: 'line',
                        data: {
                            labels: data.map(item => item.DATE),
                            datasets: [{
                                label: 'Credits Used',
                                data: data.map(item => item.CREDITS_USED),
                                backgroundColor: 'rgba(75, 192, 192, 0.5)',
                                borderColor: 'rgba(75, 192, 192, 1)',
                                borderWidth: 2,
                                tension: 0.1
                            }]
                        },
                        options: {
                            scales: {
                                y: {
                                    beginAtZero: true
                                }
                            }
                        }
                    });
                })
                .catch(error => console.error('Error fetching daily usage:', error));
        }
        
        // Fetch warehouse recommendations
        function fetchWarehouseRecommendations() {
            fetch('/api/warehouse/recommendations')
                .then(response => response.json())
                .then(data => {
                    const container = document.getElementById('warehouseRecommendations');
                    container.innerHTML = '';
                    
                    data.forEach(rec => {
                        const cardClass = rec.ESTIMATED_SAVINGS > 0 ? 'savings' : 
                                         rec.ESTIMATED_SAVINGS < 0 ? 'cost' : '';
                        
                        const card = document.createElement('div');
                        card.className = `card recommendation-card ${cardClass} mb-3`;
                        
                        card.innerHTML = `
                            <div class="card-body">
                                <h5 class="card-title">Warehouse: ${rec.WAREHOUSE_NAME}</h5>
                                <p class="card-text">
                                    <strong>Current Size:</strong> ${rec.CURRENT_SIZE}<br>
                                    <strong>Recommended Size:</strong> ${rec.RECOMMENDED_SIZE}<br>
                                    <strong>Reason:</strong> ${rec.REASON}<br>
                                    <strong>Estimated Savings:</strong> ${rec.ESTIMATED_SAVINGS > 0 ? '+' : ''}${rec.ESTIMATED_SAVINGS}%
                                </p>
                                <button class="btn btn-sm btn-primary apply-recommendation">Apply Recommendation</button>
                            </div>
                        `;
                        
                        container.appendChild(card);
                    });
                    
                    // Add event listeners to buttons
                    document.querySelectorAll('.apply-recommendation').forEach(button => {
                        button.addEventListener('click', function() {
                            alert('Recommendation applied! (This is a demo)');
                        });
                    });
                })
                .catch(error => console.error('Error fetching warehouse recommendations:', error));
        }
        
        // Fetch query candidates
        function fetchQueryCandidates() {
            fetch('/api/query/candidates')
                .then(response => response.json())
                .then(data => {
                    const container = document.getElementById('queryCandidates');
                    container.innerHTML = '';
                    
                    data.forEach(query => {
                        const card = document.createElement('div');
                        card.className = 'card query-card mb-3';
                        
                        card.innerHTML = `
                            <div class="card-body">
                                <h5 class="card-title">Query ID: ${query.QUERY_ID}</h5>
                                <p class="card-text">
                                    <strong>Warehouse:</strong> ${query.WAREHOUSE_NAME} (${query.WAREHOUSE_SIZE})<br>
                                    <strong>Execution Time:</strong> ${(query.EXECUTION_TIME / 1000).toFixed(2)} seconds<br>
                                    <strong>Bytes Scanned:</strong> ${(query.BYTES_SCANNED / (1024 * 1024)).toFixed(2)} MB
                                </p>
                                <pre>${query.QUERY_TEXT}</pre>
                                <button class="btn btn-sm btn-info analyze-query" data-query="${encodeURIComponent(query.QUERY_TEXT)}">Analyze Query</button>
                            </div>
                        `;
                        
                        container.appendChild(card);
                    });
                    
                    // Add event listeners to buttons
                    document.querySelectorAll('.analyze-query').forEach(button => {
                        button.addEventListener('click', function() {
                            const queryText = decodeURIComponent(this.getAttribute('data-query'));
                            document.getElementById('queryInput').value = queryText;
                            analyzeQuery();
                        });
                    });
                })
                .catch(error => console.error('Error fetching query candidates:', error));
        }
        
        // Analyze query
        function analyzeQuery() {
            const queryText = document.getElementById('queryInput').value;
            if (!queryText) {
                alert('Please enter a SQL query');
                return;
            }
            
            fetch('/api/query/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    query_text: queryText
                })
            })
            .then(response => response.json())
            .then(data => {
                const resultContainer = document.getElementById('queryAnalysisResult');
                
                let html = '<div class="card mt-3">';
                html += '<div class="card-header"><h5>Query Analysis Results</h5></div>';
                html += '<div class="card-body">';
                
                // Show formatted query
                html += '<h6>Formatted Query:</h6>';
                html += `<pre>${data.formatted_query}</pre>`;
                
                // Show structure analysis
                html += '<h6>Query Structure:</h6>';
                html += '<ul>';
                for (const [key, value] of Object.entries(data.structure_analysis)) {
                    html += `<li><strong>${key}:</strong> ${value}</li>`;
                }
                html += '</ul>';
                
                // Show model prediction if available
                if (data.model_prediction) {
                    html += '<h6>Efficiency Classification:</h6>';
                    html += `<p><strong>Class:</strong> ${data.model_prediction.efficiency_class} `;
                    html += `(Confidence: ${(data.model_prediction.confidence * 100).toFixed(2)}%)</p>`;
                }
                
                // Show recommendations
                if (data.pattern_recommendations && data.pattern_recommendations.length > 0) {
                    html += '<h6>Optimization Recommendations:</h6>';
                    html += '<ul>';
                    data.pattern_recommendations.forEach(rec => {
                        html += `<li><strong>${rec.recommendation}</strong><br>`;
                        html += `<em>Impact: ${rec.impact}</em></li>`;
                    });
                    html += '</ul>';
                } else {
                    html += '<p>No specific optimization recommendations.</p>';
                }
                
                html += '</div></div>';
                
                resultContainer.innerHTML = html;
            })
            .catch(error => {
                console.error('Error analyzing query:', error);
                document.getElementById('queryAnalysisResult').innerHTML = 
                    '<div class="alert alert-danger">Error analyzing query</div>';
            });
        }
        
        // Train warehouse sizing model
        function trainWarehouseModel() {
            document.getElementById('warehouseTrainingResult').innerHTML = 
                '<div class="alert alert-info">Training in progress...</div>';
            
            fetch('/api/train/warehouse', {
                method: 'POST'
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    document.getElementById('warehouseTrainingResult').innerHTML = 
                        `<div class="alert alert-success">
                            Model trained successfully!<br>
                            Accuracy: ${(data.metrics.accuracy * 100).toFixed(2)}%
                        </div>`;
                } else {
                    document.getElementById('warehouseTrainingResult').innerHTML = 
                        `<div class="alert alert-danger">Error: ${data.error}</div>`;
                }
            })
            .catch(error => {
                console.error('Error training warehouse model:', error);
                document.getElementById('warehouseTrainingResult').innerHTML = 
                    '<div class="alert alert-danger">Error training model</div>';
            });
        }
        
        // Train query optimization model
        function trainQueryModel() {
            document.getElementById('queryTrainingResult').innerHTML = 
                '<div class="alert alert-info">Training in progress...</div>';
            
            fetch('/api/train/query', {
                method: 'POST'
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    document.getElementById('queryTrainingResult').innerHTML = 
                        `<div class="alert alert-success">
                            Model trained successfully!<br>
                            Accuracy: ${(data.metrics.accuracy * 100).toFixed(2)}%
                        </div>`;
                } else {
                    document.getElementById('queryTrainingResult').innerHTML = 
                        `<div class="alert alert-danger">Error: ${data.error}</div>`;
                }
            })
            .catch(error => {
                console.error('Error training query model:', error);
                document.getElementById('queryTrainingResult').innerHTML = 
                    '<div class="alert alert-danger">Error training model</div>';
            });
        }
    </script>
</body>
</html>
    """
    
    with open('templates/index.html', 'w') as f:
        f.write(index_html)


if __name__ == '__main__':
    # Load models
    load_models()
    
    # Generate sample data
    generate_sample_data()
    
    # Create templates
    create_templates()
    
    # Run the application
    app.run(host='0.0.0.0', port=5000, debug=True)

