# Snowflake Cost Management Tool with AI

This project implements an AI-powered cost management tool for Snowflake, similar to Keebo.ai, with features for warehouse sizing recommendations and query optimization.

## Overview

The Snowflake Cost Management Tool uses machine learning to analyze Snowflake usage patterns and provide recommendations for:

1. **Warehouse Sizing**: Automatically recommends optimal warehouse sizes based on workload patterns
2. **Query Optimization**: Identifies inefficient queries and suggests improvements
3. **Cost Monitoring**: Tracks and visualizes Snowflake credit usage and costs
4. **Anomaly Detection**: Identifies unusual spending patterns

## Components

The tool consists of the following components:

1. **Warehouse Sizing Model** (`warehouse_sizing_model.py`): ML model that recommends optimal warehouse sizes based on query characteristics and workload patterns.

2. **Query Optimization Model** (`query_optimization_model.py`): ML model that identifies inefficient queries and provides optimization recommendations.

3. **Web Application** (`app.py`): Flask application that provides a user interface for the cost management tool.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/chrismack698/manage-snowflake-costs.git
cd manage-snowflake-costs
```

2. Install dependencies:
```bash
pip install flask pandas numpy scikit-learn joblib sqlparse
```

3. Run the application:
```bash
python app.py
```

4. Open your browser and navigate to `http://localhost:5000`

## Usage

### Dashboard

The dashboard provides an overview of Snowflake usage, including:

- Warehouse usage summary
- Warehouse size distribution
- Daily credit usage
- Warehouse sizing recommendations
- Query optimization candidates

### Warehouse Sizing Recommendations

The tool analyzes warehouse usage patterns and provides recommendations for optimal warehouse sizes. For each warehouse, it shows:

- Current size
- Recommended size
- Reason for the recommendation
- Estimated cost savings

### Query Optimization

The query analyzer identifies inefficient queries and provides optimization recommendations. It shows:

- Query structure analysis
- Efficiency classification
- Specific optimization recommendations with expected impact

### Training Models

The tool includes functionality to train the machine learning models using your own Snowflake data:

1. Connect to your Snowflake account
2. Extract query history and warehouse metrics
3. Train the models using the extracted data
4. Save the trained models for future use

## Integration with Snowflake

To integrate with your Snowflake account:

1. Update the connection settings in `config.py` (create this file based on `config.example.py`)
2. Ensure your Snowflake user has access to the ACCOUNT_USAGE views
3. Run the data collection script to gather historical data
4. Train the models using your own data

## Customization

The tool can be customized to fit your specific needs:

- Adjust the warehouse sizing algorithm parameters
- Modify the query optimization patterns
- Customize the dashboard visualizations
- Add additional metrics and recommendations

## Architecture

The tool follows a modular architecture:

1. **Data Collection Layer**: Gathers data from Snowflake
2. **Data Processing Layer**: Transforms and prepares data for analysis
3. **AI/ML Layer**: Implements machine learning models for recommendations
4. **Optimization Engine**: Applies optimization strategies
5. **User Interface**: Provides visualizations and controls

## Future Enhancements

Planned enhancements include:

1. Real-time monitoring and alerts
2. Automated implementation of recommendations
3. Integration with Snowflake Resource Monitors
4. Advanced anomaly detection
5. Workload forecasting
6. Multi-account support

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

