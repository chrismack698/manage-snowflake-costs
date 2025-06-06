#!/usr/bin/env python3
"""
Warehouse Sizing Recommendation Model for Snowflake Cost Management Tool

This module implements a machine learning model that recommends optimal
warehouse sizes based on query characteristics and workload patterns.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import logging
from datetime import datetime, timedelta
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define warehouse sizes
WAREHOUSE_SIZES = [
    'X-SMALL', 'SMALL', 'MEDIUM', 'LARGE', 'X-LARGE',
    '2X-LARGE', '3X-LARGE', '4X-LARGE'
]

class WarehouseSizingModel:
    """
    Machine learning model for recommending optimal Snowflake warehouse sizes
    based on query characteristics and workload patterns.
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the warehouse sizing model.
        
        Args:
            model_path (str, optional): Path to a saved model file. If provided,
                                       the model will be loaded from this file.
        """
        self.model = None
        self.preprocessor = None
        self.features = None
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            logger.info(f"Model loaded from {model_path}")
    
    def _extract_features(self, query_history_df):
        """
        Extract relevant features from query history data.
        
        Args:
            query_history_df (pandas.DataFrame): DataFrame containing query history data
            
        Returns:
            pandas.DataFrame: DataFrame with extracted features
        """
        # Extract time-based features
        query_history_df['START_TIME'] = pd.to_datetime(query_history_df['START_TIME'])
        query_history_df['HOUR_OF_DAY'] = query_history_df['START_TIME'].dt.hour
        query_history_df['DAY_OF_WEEK'] = query_history_df['START_TIME'].dt.dayofweek
        query_history_df['IS_WEEKEND'] = query_history_df['DAY_OF_WEEK'].isin([5, 6]).astype(int)
        
        # Extract query complexity features
        features_df = pd.DataFrame()
        features_df['BYTES_SCANNED'] = query_history_df['BYTES_SCANNED']
        features_df['EXECUTION_TIME'] = query_history_df['EXECUTION_TIME']
        features_df['COMPILATION_TIME'] = query_history_df['COMPILATION_TIME']
        features_df['BYTES_SPILLED_TO_LOCAL_STORAGE'] = query_history_df['BYTES_SPILLED_TO_LOCAL_STORAGE'].fillna(0)
        features_df['BYTES_SPILLED_TO_REMOTE_STORAGE'] = query_history_df['BYTES_SPILLED_TO_REMOTE_STORAGE'].fillna(0)
        features_df['PARTITIONS_SCANNED'] = query_history_df['PARTITIONS_SCANNED'].fillna(0)
        features_df['PARTITIONS_TOTAL'] = query_history_df['PARTITIONS_TOTAL'].fillna(0)
        
        # Calculate derived features
        features_df['PARTITION_SCAN_RATIO'] = (
            features_df['PARTITIONS_SCANNED'] / features_df['PARTITIONS_TOTAL']
        ).fillna(0)
        
        features_df['SPILL_RATIO'] = (
            (features_df['BYTES_SPILLED_TO_LOCAL_STORAGE'] + 
             features_df['BYTES_SPILLED_TO_REMOTE_STORAGE']) / 
            features_df['BYTES_SCANNED']
        ).fillna(0)
        
        # Add time-based features
        features_df['HOUR_OF_DAY'] = query_history_df['HOUR_OF_DAY']
        features_df['DAY_OF_WEEK'] = query_history_df['DAY_OF_WEEK']
        features_df['IS_WEEKEND'] = query_history_df['IS_WEEKEND']
        
        # Add query type features
        features_df['QUERY_TYPE'] = query_history_df['QUERY_TYPE']
        
        # Add target variable (current warehouse size)
        features_df['WAREHOUSE_SIZE'] = query_history_df['WAREHOUSE_SIZE']
        
        return features_df
    
    def _create_preprocessor(self):
        """
        Create a preprocessor for feature transformation.
        
        Returns:
            ColumnTransformer: Scikit-learn preprocessor for feature transformation
        """
        # Define numeric and categorical features
        numeric_features = [
            'BYTES_SCANNED', 'EXECUTION_TIME', 'COMPILATION_TIME',
            'BYTES_SPILLED_TO_LOCAL_STORAGE', 'BYTES_SPILLED_TO_REMOTE_STORAGE',
            'PARTITIONS_SCANNED', 'PARTITIONS_TOTAL', 'PARTITION_SCAN_RATIO',
            'SPILL_RATIO', 'HOUR_OF_DAY'
        ]
        
        categorical_features = ['DAY_OF_WEEK', 'IS_WEEKEND', 'QUERY_TYPE']
        
        # Create preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ]
        )
        
        return preprocessor
    
    def train(self, query_history_df, model_path=None):
        """
        Train the warehouse sizing model using query history data.
        
        Args:
            query_history_df (pandas.DataFrame): DataFrame containing query history data
            model_path (str, optional): Path to save the trained model
            
        Returns:
            dict: Training metrics
        """
        logger.info("Extracting features from query history data")
        features_df = self._extract_features(query_history_df)
        
        # Define features and target
        X = features_df.drop('WAREHOUSE_SIZE', axis=1)
        y = features_df['WAREHOUSE_SIZE']
        
        # Store feature names for later use
        self.features = list(X.columns)
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        logger.info("Creating preprocessor and model pipeline")
        self.preprocessor = self._create_preprocessor()
        
        # Create pipeline with preprocessor and model
        pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('classifier', RandomForestClassifier(random_state=42))
        ])
        
        # Define hyperparameter grid for tuning
        param_grid = {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [None, 10, 20],
            'classifier__min_samples_split': [2, 5, 10]
        }
        
        # Perform grid search for hyperparameter tuning
        logger.info("Performing hyperparameter tuning")
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        # Get best model
        self.model = grid_search.best_estimator_
        
        # Evaluate model
        logger.info("Evaluating model performance")
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': grid_search.best_score_,
            'best_params': grid_search.best_params_,
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        logger.info(f"Model training completed with accuracy: {metrics['accuracy']:.4f}")
        
        # Save model if path is provided
        if model_path:
            self.save_model(model_path)
            logger.info(f"Model saved to {model_path}")
        
        return metrics
    
    def predict(self, query_features):
        """
        Predict the optimal warehouse size for a given query.
        
        Args:
            query_features (dict or pandas.DataFrame): Features of the query
            
        Returns:
            str: Recommended warehouse size
            dict: Additional information including confidence scores
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded. Call train() or load_model() first.")
        
        # Convert dict to DataFrame if necessary
        if isinstance(query_features, dict):
            query_features = pd.DataFrame([query_features])
        
        # Ensure all required features are present
        for feature in self.features:
            if feature not in query_features.columns:
                query_features[feature] = 0
        
        # Make prediction
        prediction = self.model.predict(query_features)[0]
        
        # Get prediction probabilities
        probabilities = self.model.predict_proba(query_features)[0]
        
        # Get class labels (warehouse sizes)
        classes = self.model.classes_
        
        # Create confidence scores dictionary
        confidence_scores = {size: prob for size, prob in zip(classes, probabilities)}
        
        # Sort sizes by confidence score
        sorted_sizes = sorted(
            confidence_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Return prediction and additional information
        result = {
            'recommended_size': prediction,
            'confidence': confidence_scores[prediction],
            'alternatives': sorted_sizes[1:3],  # Next 2 best alternatives
            'all_scores': confidence_scores
        }
        
        return prediction, result
    
    def save_model(self, model_path):
        """
        Save the trained model to a file.
        
        Args:
            model_path (str): Path to save the model
        """
        if self.model is None:
            raise ValueError("No trained model to save")
        
        model_data = {
            'model': self.model,
            'features': self.features,
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'version': '1.0'
            }
        }
        
        joblib.dump(model_data, model_path)
    
    def load_model(self, model_path):
        """
        Load a trained model from a file.
        
        Args:
            model_path (str): Path to the saved model
        """
        model_data = joblib.load(model_path)
        
        self.model = model_data['model']
        self.features = model_data['features']
        
        # Extract preprocessor from the pipeline
        if hasattr(self.model, 'named_steps') and 'preprocessor' in self.model.named_steps:
            self.preprocessor = self.model.named_steps['preprocessor']


class WarehouseSizingRecommender:
    """
    Recommender system that uses the WarehouseSizingModel to provide
    warehouse sizing recommendations based on workload patterns.
    """
    
    def __init__(self, model=None):
        """
        Initialize the recommender.
        
        Args:
            model (WarehouseSizingModel, optional): Pre-trained model
        """
        self.model = model if model else WarehouseSizingModel()
        self.sla_constraints = {}
        self.cost_constraints = {}
    
    def set_sla_constraints(self, constraints):
        """
        Set SLA constraints for recommendations.
        
        Args:
            constraints (dict): Dictionary of SLA constraints
        """
        self.sla_constraints = constraints
    
    def set_cost_constraints(self, constraints):
        """
        Set cost constraints for recommendations.
        
        Args:
            constraints (dict): Dictionary of cost constraints
        """
        self.cost_constraints = constraints
    
    def recommend_warehouse_size(self, query_features, current_size=None):
        """
        Recommend warehouse size based on query features and constraints.
        
        Args:
            query_features (dict): Features of the query
            current_size (str, optional): Current warehouse size
            
        Returns:
            dict: Recommendation including size, reasoning, and expected impact
        """
        # Get model prediction
        predicted_size, prediction_info = self.model.predict(query_features)
        
        # Apply SLA constraints
        if self.sla_constraints:
            # Example: If query is marked as high priority, ensure minimum size
            if query_features.get('priority') == 'high' and self._size_index(predicted_size) < self._size_index('MEDIUM'):
                predicted_size = 'MEDIUM'
                reason = "Upgraded due to high priority SLA requirements"
            else:
                reason = "Based on query characteristics and historical performance"
        else:
            reason = "Based on query characteristics and historical performance"
        
        # Apply cost constraints
        if self.cost_constraints:
            # Example: If we're over budget, cap the size
            if self.cost_constraints.get('over_budget', False) and self._size_index(predicted_size) > self._size_index('LARGE'):
                predicted_size = 'LARGE'
                reason = "Capped due to budget constraints"
        
        # Calculate expected impact
        impact = self._calculate_impact(query_features, current_size, predicted_size)
        
        # Create recommendation
        recommendation = {
            'recommended_size': predicted_size,
            'current_size': current_size,
            'confidence': prediction_info['confidence'],
            'reasoning': reason,
            'expected_impact': impact,
            'alternatives': prediction_info['alternatives']
        }
        
        return recommendation
    
    def _size_index(self, size):
        """
        Get the index of a warehouse size in the size hierarchy.
        
        Args:
            size (str): Warehouse size
            
        Returns:
            int: Index of the size
        """
        try:
            return WAREHOUSE_SIZES.index(size)
        except ValueError:
            return -1
    
    def _calculate_impact(self, query_features, current_size, recommended_size):
        """
        Calculate the expected impact of changing warehouse size.
        
        Args:
            query_features (dict): Features of the query
            current_size (str): Current warehouse size
            recommended_size (str): Recommended warehouse size
            
        Returns:
            dict: Expected impact on performance and cost
        """
        if current_size is None or current_size == recommended_size:
            return {
                'performance_change': 0,
                'cost_change': 0,
                'description': "No change in warehouse size"
            }
        
        current_idx = self._size_index(current_size)
        recommended_idx = self._size_index(recommended_size)
        
        # Calculate size difference
        size_diff = recommended_idx - current_idx
        
        # Estimate performance change (simplified model)
        # Each size up roughly doubles performance
        if size_diff > 0:
            perf_change = (2 ** size_diff - 1) * 100  # percentage improvement
        else:
            perf_change = (1 - 2 ** abs(size_diff)) * 100  # percentage degradation
        
        # Estimate cost change (simplified model)
        # Each size up doubles cost
        cost_change = (2 ** size_diff - 1) * 100  # percentage change
        
        # Create description
        if size_diff > 0:
            description = (
                f"Upgrading from {current_size} to {recommended_size} is expected to "
                f"improve performance by approximately {perf_change:.1f}% at a "
                f"{cost_change:.1f}% increase in cost."
            )
        else:
            description = (
                f"Downgrading from {current_size} to {recommended_size} is expected to "
                f"reduce performance by approximately {abs(perf_change):.1f}% with a "
                f"{abs(cost_change):.1f}% decrease in cost."
            )
        
        return {
            'performance_change': perf_change,
            'cost_change': cost_change,
            'description': description
        }


def fetch_query_history(conn, days=30):
    """
    Fetch query history data from Snowflake.
    
    Args:
        conn: Snowflake connection object
        days (int): Number of days of history to fetch
        
    Returns:
        pandas.DataFrame: Query history data
    """
    query = f"""
    SELECT
        QUERY_ID,
        QUERY_TEXT,
        DATABASE_NAME,
        SCHEMA_NAME,
        QUERY_TYPE,
        SESSION_ID,
        USER_NAME,
        ROLE_NAME,
        WAREHOUSE_NAME,
        WAREHOUSE_SIZE,
        WAREHOUSE_TYPE,
        CLUSTER_NUMBER,
        QUERY_TAG,
        EXECUTION_STATUS,
        ERROR_CODE,
        ERROR_MESSAGE,
        START_TIME,
        END_TIME,
        TOTAL_ELAPSED_TIME,
        BYTES_SCANNED,
        PERCENTAGE_SCANNED_FROM_CACHE,
        BYTES_WRITTEN,
        BYTES_WRITTEN_TO_RESULT,
        BYTES_READ_FROM_RESULT,
        ROWS_PRODUCED,
        ROWS_INSERTED,
        ROWS_UPDATED,
        ROWS_DELETED,
        ROWS_UNLOADED,
        BYTES_DELETED,
        PARTITIONS_SCANNED,
        PARTITIONS_TOTAL,
        BYTES_SPILLED_TO_LOCAL_STORAGE,
        BYTES_SPILLED_TO_REMOTE_STORAGE,
        BYTES_SENT_OVER_THE_NETWORK,
        COMPILATION_TIME,
        EXECUTION_TIME,
        QUEUED_PROVISIONING_TIME,
        QUEUED_REPAIR_TIME,
        QUEUED_OVERLOAD_TIME,
        TRANSACTION_BLOCKED_TIME,
        OUTBOUND_DATA_TRANSFER_CLOUD,
        OUTBOUND_DATA_TRANSFER_REGION,
        OUTBOUND_DATA_TRANSFER_BYTES,
        INBOUND_DATA_TRANSFER_CLOUD,
        INBOUND_DATA_TRANSFER_REGION,
        INBOUND_DATA_TRANSFER_BYTES
    FROM
        SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY
    WHERE
        START_TIME >= DATEADD(day, -{days}, CURRENT_TIMESTAMP())
        AND WAREHOUSE_SIZE IS NOT NULL
        AND EXECUTION_STATUS = 'SUCCESS'
    ORDER BY
        START_TIME DESC
    """
    
    # Execute query and fetch results
    cursor = conn.cursor()
    cursor.execute(query)
    
    # Convert to DataFrame
    df = pd.DataFrame.from_records(
        iter(cursor), 
        columns=[col[0] for col in cursor.description]
    )
    
    cursor.close()
    
    return df


def main():
    """
    Main function for demonstration purposes.
    """
    # Create a sample dataset for demonstration
    np.random.seed(42)
    
    # Generate sample data
    n_samples = 1000
    
    # Create timestamps for the last 30 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    timestamps = [start_date + timedelta(
        seconds=np.random.randint(0, 30*24*60*60)
    ) for _ in range(n_samples)]
    
    # Generate query types
    query_types = np.random.choice(
        ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'MERGE', 'CREATE', 'DROP'],
        size=n_samples,
        p=[0.7, 0.1, 0.05, 0.05, 0.05, 0.03, 0.02]
    )
    
    # Generate warehouse sizes
    warehouse_sizes = np.random.choice(
        WAREHOUSE_SIZES[:5],  # Use only the first 5 sizes
        size=n_samples,
        p=[0.3, 0.3, 0.2, 0.15, 0.05]
    )
    
    # Generate bytes scanned based on warehouse size
    bytes_scanned = []
    for size in warehouse_sizes:
        size_idx = WAREHOUSE_SIZES.index(size)
        # Larger warehouses tend to process more data
        mean_bytes = 10**6 * (2**size_idx)
        bytes_scanned.append(np.random.lognormal(mean=np.log(mean_bytes), sigma=1))
    
    # Generate execution times based on warehouse size and bytes scanned
    execution_times = []
    for i, (size, bytes) in enumerate(zip(warehouse_sizes, bytes_scanned)):
        size_idx = WAREHOUSE_SIZES.index(size)
        # Larger warehouses process data faster
        mean_time = bytes / (10**6 * (2**size_idx)) * 1000  # in ms
        execution_times.append(np.random.lognormal(mean=np.log(mean_time), sigma=0.5))
    
    # Create DataFrame
    data = {
        'QUERY_ID': [f'query_{i}' for i in range(n_samples)],
        'QUERY_TEXT': [f'Sample query {i}' for i in range(n_samples)],
        'QUERY_TYPE': query_types,
        'START_TIME': timestamps,
        'WAREHOUSE_SIZE': warehouse_sizes,
        'BYTES_SCANNED': bytes_scanned,
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
    
    sample_df = pd.DataFrame(data)
    
    # Train model
    logger.info("Training warehouse sizing model on sample data")
    model = WarehouseSizingModel()
    metrics = model.train(sample_df, model_path='warehouse_sizing_model.joblib')
    
    # Print training metrics
    logger.info(f"Model accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Best parameters: {metrics['best_params']}")
    
    # Create recommender
    recommender = WarehouseSizingRecommender(model)
    
    # Set SLA constraints
    recommender.set_sla_constraints({
        'high_priority_minimum_size': 'MEDIUM'
    })
    
    # Set cost constraints
    recommender.set_cost_constraints({
        'over_budget': False,
        'max_size': 'X-LARGE'
    })
    
    # Generate a sample query for recommendation
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
        'QUERY_TYPE': 'SELECT',
        'priority': 'normal'
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


if __name__ == "__main__":
    main()

