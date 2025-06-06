#!/usr/bin/env python3
"""
Snowflake Cost Management Tool - Data Collection Script

This script collects data from Snowflake for training the cost management models.
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import snowflake.connector
import argparse
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import config
try:
    import config
    SNOWFLAKE_CONFIG = config.SNOWFLAKE_CONFIG
    DATA_COLLECTION = config.DATA_COLLECTION
except ImportError:
    logger.warning("Config file not found. Using default settings.")
    SNOWFLAKE_CONFIG = {
        'account': os.environ.get('SNOWFLAKE_ACCOUNT', ''),
        'user': os.environ.get('SNOWFLAKE_USER', ''),
        'password': os.environ.get('SNOWFLAKE_PASSWORD', ''),
        'warehouse': os.environ.get('SNOWFLAKE_WAREHOUSE', ''),
        'database': os.environ.get('SNOWFLAKE_DATABASE', ''),
        'schema': os.environ.get('SNOWFLAKE_SCHEMA', ''),
        'role': os.environ.get('SNOWFLAKE_ROLE', '')
    }
    DATA_COLLECTION = {
        'history_days': 30,
        'query_limit': 10000,
        'refresh_interval': 24
    }


def connect_to_snowflake():
    """
    Connect to Snowflake using the configuration settings.
    
    Returns:
        snowflake.connector.connection.SnowflakeConnection: Snowflake connection
    """
    try:
        conn = snowflake.connector.connect(
            account=SNOWFLAKE_CONFIG['account'],
            user=SNOWFLAKE_CONFIG['user'],
            password=SNOWFLAKE_CONFIG['password'],
            warehouse=SNOWFLAKE_CONFIG['warehouse'],
            database=SNOWFLAKE_CONFIG['database'],
            schema=SNOWFLAKE_CONFIG['schema'],
            role=SNOWFLAKE_CONFIG['role']
        )
        logger.info("Connected to Snowflake")
        return conn
    except Exception as e:
        logger.error(f"Error connecting to Snowflake: {e}")
        raise


def fetch_query_history(conn, days=30, limit=10000):
    """
    Fetch query history data from Snowflake.
    
    Args:
        conn: Snowflake connection object
        days (int): Number of days of history to fetch
        limit (int): Maximum number of queries to fetch
        
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
    LIMIT {limit}
    """
    
    logger.info(f"Fetching query history for the last {days} days (limit: {limit} queries)")
    
    try:
        # Execute query and fetch results
        cursor = conn.cursor()
        cursor.execute(query)
        
        # Convert to DataFrame
        df = pd.DataFrame.from_records(
            iter(cursor), 
            columns=[col[0] for col in cursor.description]
        )
        
        cursor.close()
        
        logger.info(f"Fetched {len(df)} queries")
        return df
    except Exception as e:
        logger.error(f"Error fetching query history: {e}")
        raise


def fetch_warehouse_metering(conn, days=30):
    """
    Fetch warehouse metering history from Snowflake.
    
    Args:
        conn: Snowflake connection object
        days (int): Number of days of history to fetch
        
    Returns:
        pandas.DataFrame: Warehouse metering data
    """
    query = f"""
    SELECT
        WAREHOUSE_NAME,
        WAREHOUSE_ID,
        START_TIME,
        END_TIME,
        CREDITS_USED,
        CREDITS_USED_COMPUTE,
        CREDITS_USED_CLOUD_SERVICES
    FROM
        SNOWFLAKE.ACCOUNT_USAGE.WAREHOUSE_METERING_HISTORY
    WHERE
        START_TIME >= DATEADD(day, -{days}, CURRENT_TIMESTAMP())
    ORDER BY
        START_TIME DESC
    """
    
    logger.info(f"Fetching warehouse metering history for the last {days} days")
    
    try:
        # Execute query and fetch results
        cursor = conn.cursor()
        cursor.execute(query)
        
        # Convert to DataFrame
        df = pd.DataFrame.from_records(
            iter(cursor), 
            columns=[col[0] for col in cursor.description]
        )
        
        cursor.close()
        
        logger.info(f"Fetched {len(df)} warehouse metering records")
        return df
    except Exception as e:
        logger.error(f"Error fetching warehouse metering history: {e}")
        raise


def fetch_warehouse_events(conn, days=30):
    """
    Fetch warehouse events history from Snowflake.
    
    Args:
        conn: Snowflake connection object
        days (int): Number of days of history to fetch
        
    Returns:
        pandas.DataFrame: Warehouse events data
    """
    query = f"""
    SELECT
        WAREHOUSE_NAME,
        WAREHOUSE_ID,
        EVENT_NAME,
        EVENT_TIMESTAMP,
        EVENT_STATE,
        EVENT_REASON
    FROM
        SNOWFLAKE.ACCOUNT_USAGE.WAREHOUSE_EVENTS_HISTORY
    WHERE
        EVENT_TIMESTAMP >= DATEADD(day, -{days}, CURRENT_TIMESTAMP())
    ORDER BY
        EVENT_TIMESTAMP DESC
    """
    
    logger.info(f"Fetching warehouse events history for the last {days} days")
    
    try:
        # Execute query and fetch results
        cursor = conn.cursor()
        cursor.execute(query)
        
        # Convert to DataFrame
        df = pd.DataFrame.from_records(
            iter(cursor), 
            columns=[col[0] for col in cursor.description]
        )
        
        cursor.close()
        
        logger.info(f"Fetched {len(df)} warehouse events")
        return df
    except Exception as e:
        logger.error(f"Error fetching warehouse events history: {e}")
        raise


def fetch_storage_usage(conn, days=30):
    """
    Fetch storage usage history from Snowflake.
    
    Args:
        conn: Snowflake connection object
        days (int): Number of days of history to fetch
        
    Returns:
        pandas.DataFrame: Storage usage data
    """
    query = f"""
    SELECT
        USAGE_DATE,
        STORAGE_BYTES,
        STAGE_BYTES,
        FAILSAFE_BYTES,
        AVERAGE_DATABASE_BYTES,
        AVERAGE_FAILSAFE_BYTES
    FROM
        SNOWFLAKE.ACCOUNT_USAGE.STORAGE_USAGE
    WHERE
        USAGE_DATE >= DATEADD(day, -{days}, CURRENT_DATE())
    ORDER BY
        USAGE_DATE DESC
    """
    
    logger.info(f"Fetching storage usage history for the last {days} days")
    
    try:
        # Execute query and fetch results
        cursor = conn.cursor()
        cursor.execute(query)
        
        # Convert to DataFrame
        df = pd.DataFrame.from_records(
            iter(cursor), 
            columns=[col[0] for col in cursor.description]
        )
        
        cursor.close()
        
        logger.info(f"Fetched {len(df)} storage usage records")
        return df
    except Exception as e:
        logger.error(f"Error fetching storage usage history: {e}")
        raise


def save_data(data_dict, output_dir='data'):
    """
    Save collected data to CSV files.
    
    Args:
        data_dict (dict): Dictionary of DataFrames to save
        output_dir (str): Directory to save the files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each DataFrame to a CSV file
    for name, df in data_dict.items():
        if df is not None and not df.empty:
            file_path = os.path.join(output_dir, f"{name}.csv")
            df.to_csv(file_path, index=False)
            logger.info(f"Saved {len(df)} records to {file_path}")
        else:
            logger.warning(f"No data to save for {name}")
    
    # Save collection metadata
    metadata = {
        'collection_time': datetime.now().isoformat(),
        'history_days': DATA_COLLECTION['history_days'],
        'record_counts': {name: len(df) if df is not None else 0 for name, df in data_dict.items()}
    }
    
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved collection metadata to {metadata_path}")


def main():
    """
    Main function to collect data from Snowflake.
    """
    parser = argparse.ArgumentParser(description='Collect data from Snowflake for cost management models')
    parser.add_argument('--days', type=int, default=DATA_COLLECTION['history_days'],
                        help='Number of days of history to collect')
    parser.add_argument('--limit', type=int, default=DATA_COLLECTION['query_limit'],
                        help='Maximum number of queries to collect')
    parser.add_argument('--output-dir', type=str, default='data',
                        help='Directory to save the collected data')
    args = parser.parse_args()
    
    try:
        # Connect to Snowflake
        conn = connect_to_snowflake()
        
        # Collect data
        query_history = fetch_query_history(conn, days=args.days, limit=args.limit)
        warehouse_metering = fetch_warehouse_metering(conn, days=args.days)
        warehouse_events = fetch_warehouse_events(conn, days=args.days)
        storage_usage = fetch_storage_usage(conn, days=args.days)
        
        # Close connection
        conn.close()
        
        # Save data
        data_dict = {
            'query_history': query_history,
            'warehouse_metering': warehouse_metering,
            'warehouse_events': warehouse_events,
            'storage_usage': storage_usage
        }
        
        save_data(data_dict, output_dir=args.output_dir)
        
        logger.info("Data collection completed successfully")
    except Exception as e:
        logger.error(f"Error collecting data: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

