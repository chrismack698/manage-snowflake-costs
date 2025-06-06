#!/usr/bin/env python3
"""
Test script for Snowflake Cost Management Tool models

This script tests the warehouse sizing and query optimization models
to ensure they work correctly.
"""

import logging
import pandas as pd
import numpy as np
from warehouse_sizing_model import WarehouseSizingModel, WarehouseSizingRecommender
from query_optimization_model import QueryOptimizationModel, QueryOptimizationRecommender

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_sample_data(n_samples=100):
    """
    Generate sample data for testing.
    
    Args:
        n_samples (int): Number of samples to generate
        
    Returns:
        pandas.DataFrame: Sample data
    """
    np.random.seed(42)
    
    # Generate timestamps
    timestamps = [pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(1, 30)) 
                 for _ in range(n_samples)]
    
    # Generate query types
    query_types = np.random.choice(
        ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'MERGE', 'CREATE', 'DROP'],
        size=n_samples,
        p=[0.7, 0.1, 0.05, 0.05, 0.05, 0.03, 0.02]
    )
    
    # Generate warehouse sizes
    warehouse_sizes = np.random.choice(
        ['X-SMALL', 'SMALL', 'MEDIUM', 'LARGE', 'X-LARGE'],
        size=n_samples,
        p=[0.3, 0.3, 0.2, 0.15, 0.05]
    )
    
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
    
    # Create DataFrame
    data = {
        'QUERY_ID': [f'query_{i}' for i in range(n_samples)],
        'QUERY_TEXT': query_texts,
        'QUERY_TYPE': query_types,
        'START_TIME': timestamps,
        'WAREHOUSE_SIZE': warehouse_sizes,
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
        data['PARTITIONS_SCANNED'][i] = min(
            data['PARTITIONS_SCANNED'][i], 
            data['PARTITIONS_TOTAL'][i]
        )
    
    return pd.DataFrame(data)


def test_warehouse_sizing_model():
    """
    Test the warehouse sizing model.
    """
    logger.info("Testing warehouse sizing model...")
    
    # Generate sample data
    sample_df = generate_sample_data()
    
    # Create and train model
    model = WarehouseSizingModel()
    metrics = model.train(sample_df)
    
    # Print training metrics
    logger.info(f"Model accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Best parameters: {metrics['best_params']}")
    
    # Create recommender
    recommender = WarehouseSizingRecommender(model)
    
    # Test recommendation
    sample_query = {
        'BYTES_SCANNED': 5 * 10**8,  # 500 MB
        'EXECUTION_TIME': 15000,  # 15 seconds
        'COMPILATION_TIME': 2000,  # 2 seconds
        'BYTES_SPILLED_TO_LOCAL_STORAGE': 0,
        'BYTES_SPILLED_TO_REMOTE_STORAGE': 0,
        'PARTITIONS_SCANNED': 20,
        'PARTITIONS_TOTAL': 100,
        'PARTITION_SCAN_RATIO': 0.2,
        'SPILL_RATIO': 0,
        'HOUR_OF_DAY': 14,  # 2 PM
        'DAY_OF_WEEK': 2,  # Wednesday
        'IS_WEEKEND': 0,
        'QUERY_TYPE': 'SELECT'
    }
    
    # Get recommendation
    recommendation = recommender.recommend_warehouse_size(
        sample_query, current_size='SMALL'
    )
    
    # Print recommendation
    logger.info("Warehouse size recommendation:")
    logger.info(f"Current size: {recommendation['current_size']}")
    logger.info(f"Recommended size: {recommendation['recommended_size']}")
    logger.info(f"Confidence: {recommendation['confidence']:.4f}")
    logger.info(f"Reasoning: {recommendation['reasoning']}")
    logger.info(f"Expected impact: {recommendation['expected_impact']['description']}")
    
    return True


def test_query_optimization_model():
    """
    Test the query optimization model.
    """
    logger.info("Testing query optimization model...")
    
    # Generate sample data
    sample_df = generate_sample_data()
    
    # Create and train model
    model = QueryOptimizationModel()
    metrics = model.train(sample_df)
    
    # Print training metrics
    logger.info(f"Model accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Best parameters: {metrics['best_params']}")
    
    # Create recommender
    recommender = QueryOptimizationRecommender(model)
    
    # Test query analysis
    sample_query = "SELECT * FROM customers WHERE UPPER(name) = 'JOHN'"
    sample_metrics = {
        'BYTES_SCANNED': 5 * 10**8,  # 500 MB
        'ROWS_PRODUCED': 1000,
        'EXECUTION_TIME': 15000,  # 15 seconds
        'COMPILATION_TIME': 2000,  # 2 seconds
        'BYTES_SPILLED_TO_LOCAL_STORAGE': 0,
        'BYTES_SPILLED_TO_REMOTE_STORAGE': 0,
        'PARTITIONS_SCANNED': 20,
        'PARTITIONS_TOTAL': 100,
        'QUERY_TYPE': 'SELECT',
        'WAREHOUSE_SIZE': 'SMALL'
    }
    
    # Get analysis
    analysis = recommender.analyze_query(sample_query, sample_metrics)
    
    # Print analysis
    logger.info("Query analysis:")
    logger.info(f"Structure analysis: {analysis['structure_analysis']}")
    
    if analysis['model_prediction']:
        logger.info(f"Efficiency class: {analysis['model_prediction']['efficiency_class']}")
        logger.info(f"Confidence: {analysis['model_prediction']['confidence']:.4f}")
    
    logger.info("Recommendations:")
    for rec in analysis['pattern_recommendations']:
        logger.info(f"- {rec['recommendation']} ({rec['impact']})")
    
    return True


def main():
    """
    Main function to run tests.
    """
    logger.info("Starting model tests...")
    
    # Test warehouse sizing model
    warehouse_result = test_warehouse_sizing_model()
    
    # Test query optimization model
    query_result = test_query_optimization_model()
    
    # Print overall results
    if warehouse_result and query_result:
        logger.info("All tests passed!")
    else:
        logger.error("Some tests failed!")


if __name__ == "__main__":
    main()

