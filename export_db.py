import os
import csv
import sqlite3
import json
import logging
from datetime import datetime

def export_db_to_csv(db_path: str, output_dir: str):
    """
    Export all tables from SQLite database to CSV files.
    
    Args:
        db_path (str): Path to the SQLite database file
        output_dir (str): Directory where CSV files will be saved
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get list of all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    for table in tables:
        table_name = table[0]
        
        # Get column names
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = [column[1] for column in cursor.fetchall()]
        
        # Get all rows
        cursor.execute(f"SELECT * FROM {table_name};")
        rows = cursor.fetchall()
        
        # Create CSV file
        output_file = os.path.join(output_dir, f"{table_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            writer.writerow(columns)
            
            # Write data rows
            for row in rows:
                # Process any JSON fields
                processed_row = []
                for value in row:
                    if isinstance(value, str) and (value.startswith('[') or value.startswith('{')):
                        try:
                            # Try to parse JSON strings and format them
                            processed_value = json.dumps(json.loads(value))
                        except json.JSONDecodeError:
                            processed_value = value
                    else:
                        processed_value = value
                    processed_row.append(processed_value)
                writer.writerow(processed_row)
                
        logging.info(f"Exported {table_name} to {output_file}")
    
    conn.close()
    logging.info("Database export completed successfully")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Get database path from environment variable or use default
    db_path = os.getenv('DB_PATH', './data/content.db')
    
    # Set output directory in current working directory
    output_dir = os.path.join(os.getcwd(), 'database_exports')
    
    try:
        export_db_to_csv(db_path, output_dir)
        print(f"Database exported successfully to {output_dir}")
    except Exception as e:
        logging.error(f"Error exporting database: {e}")
        raise