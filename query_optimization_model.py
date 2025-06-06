#!/usr/bin/env python3
"""
Query Optimization Recommendation Model for Snowflake Cost Management Tool

This module implements a machine learning model that identifies inefficient
queries and provides optimization recommendations.
"""

import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import logging
from datetime import datetime
import os
import sqlparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define query efficiency classes
EFFICIENCY_CLASSES = ['EFFICIENT', 'NEEDS_OPTIMIZATION', 'INEFFICIENT']

# Define optimization patterns
OPTIMIZATION_PATTERNS = {
    'select_all': {
        'pattern': r'SELECT\s+\*\s+FROM',
        'recommendation': 'Specify only needed columns instead of using SELECT *',
        'impact': 'Reduces data transfer and improves query performance'
    },
    'missing_where': {
        'pattern': r'FROM\s+\w+(?:\.\w+)*\s+(?!WHERE|JOIN|GROUP|ORDER|LIMIT|HAVING)',
        'recommendation': 'Add a WHERE clause to filter data',
        'impact': 'Reduces data scanned and improves query performance'
    },
    'function_in_where': {
        'pattern': r'WHERE\s+\w+\s*\(\w+\)',
        'recommendation': 'Avoid using functions on filtered columns in WHERE clauses',
        'impact': 'Enables query pruning and improves performance'
    },
    'or_condition': {
        'pattern': r'WHERE\s+.*\s+OR\s+',
        'recommendation': 'Consider restructuring queries with OR conditions',
        'impact': 'Improves query optimization and execution'
    },
    'cross_join': {
        'pattern': r'FROM\s+\w+(?:\.\w+)*\s*,\s*\w+(?:\.\w+)*',
        'recommendation': 'Use explicit JOIN syntax instead of comma-separated tables',
        'impact': 'Improves query readability and optimization'
    },
    'multiple_joins': {
        'pattern': r'JOIN.*JOIN.*JOIN',
        'recommendation': 'Review multiple joins for optimization opportunities',
        'impact': 'Reduces complexity and improves performance'
    },
    'subquery': {
        'pattern': r'\(\s*SELECT',
        'recommendation': 'Consider using CTEs instead of subqueries',
        'impact': 'Improves readability and may enhance performance'
    },
    'group_by_all': {
        'pattern': r'GROUP\s+BY\s+\d+(?:\s*,\s*\d+)*',
        'recommendation': 'Use column names in GROUP BY instead of positions',
        'impact': 'Improves readability and maintainability'
    },
    'distinct': {
        'pattern': r'SELECT\s+DISTINCT',
        'recommendation': 'Evaluate if DISTINCT is necessary',
        'impact': 'Reduces processing overhead if not needed'
    },
    'order_by': {
        'pattern': r'ORDER\s+BY',
        'recommendation': 'Ensure ORDER BY is necessary for the query',
        'impact': 'Reduces processing overhead if not needed'
    }
}


class QueryOptimizationModel:
    """
    Machine learning model for identifying inefficient queries and providing
    optimization recommendations.
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the query optimization model.
        
        Args:
            model_path (str, optional): Path to a saved model file. If provided,
                                       the model will be loaded from this file.
        """
        self.model = None
        self.preprocessor = None
        self.text_vectorizer = None
        self.features = None
        self.pattern_analyzer = QueryPatternAnalyzer()
        
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
        # Extract query text features
        query_texts = query_history_df['QUERY_TEXT'].fillna('')
        
        # Create TF-IDF vectorizer for query text
        self.text_vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Transform query texts to TF-IDF features
        text_features = self.text_vectorizer.fit_transform(query_texts)
        
        # Extract performance features
        features_df = pd.DataFrame()
        features_df['BYTES_SCANNED'] = query_history_df['BYTES_SCANNED']
        features_df['ROWS_PRODUCED'] = query_history_df['ROWS_PRODUCED'].fillna(0)
        features_df['EXECUTION_TIME'] = query_history_df['EXECUTION_TIME']
        features_df['COMPILATION_TIME'] = query_history_df['COMPILATION_TIME']
        features_df['BYTES_SPILLED_TO_LOCAL_STORAGE'] = query_history_df['BYTES_SPILLED_TO_LOCAL_STORAGE'].fillna(0)
        features_df['BYTES_SPILLED_TO_REMOTE_STORAGE'] = query_history_df['BYTES_SPILLED_TO_REMOTE_STORAGE'].fillna(0)
        features_df['PARTITIONS_SCANNED'] = query_history_df['PARTITIONS_SCANNED'].fillna(0)
        features_df['PARTITIONS_TOTAL'] = query_history_df['PARTITIONS_TOTAL'].fillna(0)
        
        # Calculate derived features
        features_df['BYTES_PER_ROW'] = (
            features_df['BYTES_SCANNED'] / features_df['ROWS_PRODUCED'].replace(0, 1)
        )
        
        features_df['PARTITION_SCAN_RATIO'] = (
            features_df['PARTITIONS_SCANNED'] / features_df['PARTITIONS_TOTAL'].replace(0, 1)
        ).fillna(0)
        
        features_df['SPILL_RATIO'] = (
            (features_df['BYTES_SPILLED_TO_LOCAL_STORAGE'] + 
             features_df['BYTES_SPILLED_TO_REMOTE_STORAGE']) / 
            features_df['BYTES_SCANNED'].replace(0, 1)
        ).fillna(0)
        
        # Extract query pattern features
        pattern_features = []
        for query_text in query_texts:
            pattern_counts = self.pattern_analyzer.analyze_query(query_text)
            pattern_features.append(pattern_counts)
        
        pattern_df = pd.DataFrame(pattern_features)
        
        # Combine all features
        features_df = pd.concat([features_df, pattern_df], axis=1)
        
        # Add query type
        features_df['QUERY_TYPE'] = query_history_df['QUERY_TYPE']
        
        # Add warehouse size
        features_df['WAREHOUSE_SIZE'] = query_history_df['WAREHOUSE_SIZE']
        
        # Add target variable (efficiency class)
        # For training purposes, we'll create a synthetic target based on performance metrics
        efficiency_classes = self._create_synthetic_targets(query_history_df)
        features_df['EFFICIENCY_CLASS'] = efficiency_classes
        
        return features_df, text_features
    
    def _create_synthetic_targets(self, query_history_df):
        """
        Create synthetic target labels for training purposes.
        In a real-world scenario, these would come from expert labeling or feedback.
        
        Args:
            query_history_df (pandas.DataFrame): DataFrame containing query history data
            
        Returns:
            numpy.ndarray: Array of efficiency class labels
        """
        # Calculate efficiency metrics
        bytes_scanned = query_history_df['BYTES_SCANNED'].values
        execution_time = query_history_df['EXECUTION_TIME'].values
        rows_produced = query_history_df['ROWS_PRODUCED'].fillna(0).values
        
        # Calculate bytes per row (efficiency metric)
        bytes_per_row = bytes_scanned / np.maximum(rows_produced, 1)
        
        # Calculate execution time per byte (efficiency metric)
        time_per_byte = execution_time / np.maximum(bytes_scanned, 1)
        
        # Normalize metrics
        bytes_per_row_norm = (bytes_per_row - np.mean(bytes_per_row)) / np.std(bytes_per_row)
        time_per_byte_norm = (time_per_byte - np.mean(time_per_byte)) / np.std(time_per_byte)
        
        # Combine metrics into an efficiency score
        efficiency_score = bytes_per_row_norm + time_per_byte_norm
        
        # Classify based on efficiency score
        efficiency_classes = np.zeros(len(efficiency_score), dtype=int)
        efficiency_classes[efficiency_score > 1.0] = 2  # INEFFICIENT
        efficiency_classes[(efficiency_score > -0.5) & (efficiency_score <= 1.0)] = 1  # NEEDS_OPTIMIZATION
        efficiency_classes[efficiency_score <= -0.5] = 0  # EFFICIENT
        
        return [EFFICIENCY_CLASSES[i] for i in efficiency_classes]
    
    def _create_preprocessor(self):
        """
        Create a preprocessor for feature transformation.
        
        Returns:
            ColumnTransformer: Scikit-learn preprocessor for feature transformation
        """
        # Define numeric and categorical features
        numeric_features = [
            'BYTES_SCANNED', 'ROWS_PRODUCED', 'EXECUTION_TIME', 'COMPILATION_TIME',
            'BYTES_SPILLED_TO_LOCAL_STORAGE', 'BYTES_SPILLED_TO_REMOTE_STORAGE',
            'PARTITIONS_SCANNED', 'PARTITIONS_TOTAL', 'BYTES_PER_ROW',
            'PARTITION_SCAN_RATIO', 'SPILL_RATIO'
        ]
        
        # Add pattern features
        pattern_features = list(OPTIMIZATION_PATTERNS.keys())
        numeric_features.extend(pattern_features)
        
        categorical_features = ['QUERY_TYPE', 'WAREHOUSE_SIZE']
        
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
        Train the query optimization model using query history data.
        
        Args:
            query_history_df (pandas.DataFrame): DataFrame containing query history data
            model_path (str, optional): Path to save the trained model
            
        Returns:
            dict: Training metrics
        """
        logger.info("Extracting features from query history data")
        features_df, text_features = self._extract_features(query_history_df)
        
        # Define features and target
        X_structured = features_df.drop('EFFICIENCY_CLASS', axis=1)
        y = features_df['EFFICIENCY_CLASS']
        
        # Store feature names for later use
        self.features = list(X_structured.columns)
        
        # Split data into training and testing sets
        X_struct_train, X_struct_test, X_text_train, X_text_test, y_train, y_test = train_test_split(
            X_structured, text_features, y, test_size=0.2, random_state=42
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
        grid_search.fit(X_struct_train, y_train)
        
        # Get best model
        self.model = grid_search.best_estimator_
        
        # Evaluate model
        logger.info("Evaluating model performance")
        y_pred = self.model.predict(X_struct_test)
        
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
    
    def predict(self, query_features, query_text=None):
        """
        Predict the efficiency class of a query and provide optimization recommendations.
        
        Args:
            query_features (dict or pandas.DataFrame): Features of the query
            query_text (str, optional): Text of the SQL query
            
        Returns:
            str: Predicted efficiency class
            dict: Additional information including recommendations
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
        
        # Get class labels
        classes = self.model.classes_
        
        # Create confidence scores dictionary
        confidence_scores = {EFFICIENCY_CLASSES[cls]: prob for cls, prob in zip(classes, probabilities)}
        
        # Generate recommendations if query text is provided
        recommendations = []
        if query_text and prediction != 'EFFICIENT':
            recommendations = self.pattern_analyzer.generate_recommendations(query_text)
        
        # Return prediction and additional information
        result = {
            'efficiency_class': prediction,
            'confidence': confidence_scores[prediction],
            'all_scores': confidence_scores,
            'recommendations': recommendations
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
            'text_vectorizer': self.text_vectorizer,
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
        self.text_vectorizer = model_data['text_vectorizer']
        
        # Extract preprocessor from the pipeline
        if hasattr(self.model, 'named_steps') and 'preprocessor' in self.model.named_steps:
            self.preprocessor = self.model.named_steps['preprocessor']


class QueryPatternAnalyzer:
    """
    Analyzes SQL queries for common patterns that may indicate inefficiencies.
    """
    
    def __init__(self):
        """
        Initialize the query pattern analyzer.
        """
        self.patterns = OPTIMIZATION_PATTERNS
    
    def analyze_query(self, query_text):
        """
        Analyze a query for inefficiency patterns.
        
        Args:
            query_text (str): SQL query text
            
        Returns:
            dict: Dictionary of pattern counts
        """
        if not query_text or not isinstance(query_text, str):
            return {pattern: 0 for pattern in self.patterns}
        
        # Normalize query text
        normalized_query = self._normalize_query(query_text)
        
        # Count pattern occurrences
        pattern_counts = {}
        for pattern_name, pattern_info in self.patterns.items():
            pattern_regex = pattern_info['pattern']
            matches = re.findall(pattern_regex, normalized_query, re.IGNORECASE)
            pattern_counts[pattern_name] = len(matches)
        
        return pattern_counts
    
    def generate_recommendations(self, query_text):
        """
        Generate optimization recommendations for a query.
        
        Args:
            query_text (str): SQL query text
            
        Returns:
            list: List of recommendation dictionaries
        """
        if not query_text or not isinstance(query_text, str):
            return []
        
        # Normalize query text
        normalized_query = self._normalize_query(query_text)
        
        # Find patterns and generate recommendations
        recommendations = []
        for pattern_name, pattern_info in self.patterns.items():
            pattern_regex = pattern_info['pattern']
            matches = re.findall(pattern_regex, normalized_query, re.IGNORECASE)
            
            if matches:
                recommendation = {
                    'pattern': pattern_name,
                    'recommendation': pattern_info['recommendation'],
                    'impact': pattern_info['impact'],
                    'matches': len(matches)
                }
                recommendations.append(recommendation)
        
        # Sort recommendations by number of matches (descending)
        recommendations.sort(key=lambda x: x['matches'], reverse=True)
        
        return recommendations
    
    def _normalize_query(self, query_text):
        """
        Normalize a SQL query for pattern matching.
        
        Args:
            query_text (str): SQL query text
            
        Returns:
            str: Normalized query text
        """
        # Parse the query
        try:
            parsed = sqlparse.parse(query_text)
            if not parsed:
                return query_text.upper()
            
            # Format the query
            formatted = sqlparse.format(
                query_text,
                keyword_case='upper',
                identifier_case='lower',
                strip_comments=True,
                reindent=True
            )
            
            return formatted
        except:
            # If parsing fails, just uppercase the query
            return query_text.upper()


class QueryOptimizationRecommender:
    """
    Recommender system that uses the QueryOptimizationModel to provide
    query optimization recommendations.
    """
    
    def __init__(self, model=None):
        """
        Initialize the recommender.
        
        Args:
            model (QueryOptimizationModel, optional): Pre-trained model
        """
        self.model = model if model else QueryOptimizationModel()
        self.pattern_analyzer = QueryPatternAnalyzer()
    
    def analyze_query(self, query_text, query_metrics=None):
        """
        Analyze a query and provide optimization recommendations.
        
        Args:
            query_text (str): SQL query text
            query_metrics (dict, optional): Performance metrics of the query
            
        Returns:
            dict: Analysis results including recommendations
        """
        # Generate pattern-based recommendations
        pattern_recommendations = self.pattern_analyzer.generate_recommendations(query_text)
        
        # If query metrics are provided, use the model for prediction
        model_prediction = None
        if query_metrics and self.model:
            try:
                efficiency_class, prediction_info = self.model.predict(query_metrics, query_text)
                model_prediction = {
                    'efficiency_class': efficiency_class,
                    'confidence': prediction_info['confidence'],
                    'all_scores': prediction_info['all_scores']
                }
            except Exception as e:
                logger.error(f"Error making model prediction: {e}")
        
        # Parse and format the query for better readability
        formatted_query = self._format_query(query_text)
        
        # Generate query structure analysis
        structure_analysis = self._analyze_query_structure(query_text)
        
        # Combine all analysis results
        analysis = {
            'query_text': query_text,
            'formatted_query': formatted_query,
            'structure_analysis': structure_analysis,
            'pattern_recommendations': pattern_recommendations,
            'model_prediction': model_prediction
        }
        
        return analysis
    
    def _format_query(self, query_text):
        """
        Format a SQL query for better readability.
        
        Args:
            query_text (str): SQL query text
            
        Returns:
            str: Formatted query text
        """
        try:
            formatted = sqlparse.format(
                query_text,
                keyword_case='upper',
                identifier_case='lower',
                reindent=True,
                indent_width=4
            )
            return formatted
        except:
            return query_text
    
    def _analyze_query_structure(self, query_text):
        """
        Analyze the structure of a SQL query.
        
        Args:
            query_text (str): SQL query text
            
        Returns:
            dict: Query structure analysis
        """
        try:
            # Parse the query
            parsed = sqlparse.parse(query_text)
            if not parsed:
                return {'error': 'Failed to parse query'}
            
            stmt = parsed[0]
            
            # Identify query type
            query_type = stmt.get_type()
            
            # Count tables
            tables = []
            from_seen = False
            for token in stmt.tokens:
                if token.is_keyword and token.value.upper() == 'FROM':
                    from_seen = True
                elif from_seen and token.ttype is None:
                    # This might be a table reference
                    tables.append(token.value)
                    from_seen = False
            
            # Count joins
            join_count = len(re.findall(r'\bJOIN\b', query_text, re.IGNORECASE))
            
            # Check for subqueries
            subquery_count = len(re.findall(r'\(\s*SELECT', query_text, re.IGNORECASE))
            
            # Check for aggregations
            has_aggregation = bool(re.search(r'\b(SUM|AVG|MIN|MAX|COUNT)\s*\(', query_text, re.IGNORECASE))
            
            # Check for GROUP BY
            has_group_by = bool(re.search(r'\bGROUP\s+BY\b', query_text, re.IGNORECASE))
            
            # Check for ORDER BY
            has_order_by = bool(re.search(r'\bORDER\s+BY\b', query_text, re.IGNORECASE))
            
            # Check for DISTINCT
            has_distinct = bool(re.search(r'\bDISTINCT\b', query_text, re.IGNORECASE))
            
            # Return structure analysis
            return {
                'query_type': query_type,
                'table_count': len(tables),
                'join_count': join_count,
                'subquery_count': subquery_count,
                'has_aggregation': has_aggregation,
                'has_group_by': has_group_by,
                'has_order_by': has_order_by,
                'has_distinct': has_distinct
            }
        except Exception as e:
            logger.error(f"Error analyzing query structure: {e}")
            return {'error': str(e)}


def main():
    """
    Main function for demonstration purposes.
    """
    # Create a sample dataset for demonstration
    np.random.seed(42)
    
    # Generate sample data
    n_samples = 1000
    
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
    
    sample_df = pd.DataFrame(data)
    
    # Train model
    logger.info("Training query optimization model on sample data")
    model = QueryOptimizationModel()
    metrics = model.train(sample_df, model_path='query_optimization_model.joblib')
    
    # Print training metrics
    logger.info(f"Model accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Best parameters: {metrics['best_params']}")
    
    # Create recommender
    recommender = QueryOptimizationRecommender(model)
    
    # Analyze a sample query
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
    
    # Add pattern features
    pattern_analyzer = QueryPatternAnalyzer()
    pattern_counts = pattern_analyzer.analyze_query(sample_query)
    sample_metrics.update(pattern_counts)
    
    # Get analysis
    analysis = recommender.analyze_query(sample_query, sample_metrics)
    
    # Print analysis
    logger.info("Query analysis:")
    logger.info(f"Formatted query:\n{analysis['formatted_query']}")
    logger.info(f"Structure analysis: {analysis['structure_analysis']}")
    
    if analysis['model_prediction']:
        logger.info(f"Efficiency class: {analysis['model_prediction']['efficiency_class']}")
        logger.info(f"Confidence: {analysis['model_prediction']['confidence']:.4f}")
    
    logger.info("Recommendations:")
    for rec in analysis['pattern_recommendations']:
        logger.info(f"- {rec['recommendation']} ({rec['impact']})")


if __name__ == "__main__":
    main()

