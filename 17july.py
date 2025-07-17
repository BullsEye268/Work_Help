import pandas as pd
import os
from datetime import datetime

def append_to_output_file(new_data_df, output_file_path):
“””
Append new data to existing CSV file, or create new file if it doesn’t exist.

```
Args:
    new_data_df: pandas DataFrame with new data to append
    output_file_path: path to the output CSV file
"""

# Check if output file exists
if os.path.exists(output_file_path):
    # Read existing data
    try:
        existing_df = pd.read_csv(output_file_path)
        
        # Append new data
        combined_df = pd.concat([existing_df, new_data_df], ignore_index=True)
        
        print(f"Appended {len(new_data_df)} rows to existing file. Total rows: {len(combined_df)}")
        
    except Exception as e:
        print(f"Error reading existing file: {e}")
        print("Creating new file instead...")
        combined_df = new_data_df
        
else:
    # Create new file
    combined_df = new_data_df
    print(f"Created new file with {len(new_data_df)} rows")

# Save the combined data
try:
    combined_df.to_csv(output_file_path, index=False)
    print(f"Successfully saved to: {output_file_path}")
    
except Exception as e:
    print(f"Error saving file: {e}")
```

def main():
“””
Main function for your daily script
“””

```
# Your existing code for finding and analyzing the file goes here
# For example:
# input_file = find_input_file()  # Your existing function
# analyzed_data = analyze_file(input_file)  # Your existing function

# Example: Create sample data (replace with your actual analyzed data)
new_data = pd.DataFrame({
    'timestamp': [datetime.now()],
    'file_processed': ['example_file.txt'],
    'metric_1': [123.45],
    'metric_2': [67.89],
    'status': ['success']
})

# Define output file path
output_file = 'daily_analysis_results.csv'

# Append to output file
append_to_output_file(new_data, output_file)
```

# Alternative: More robust version with error handling and logging

def append_with_logging(new_data_df, output_file_path, log_file=‘script.log’):
“””
Enhanced version with logging and better error handling
“””
import logging

```
# Setup logging
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

try:
    if os.path.exists(output_file_path):
        existing_df = pd.read_csv(output_file_path)
        combined_df = pd.concat([existing_df, new_data_df], ignore_index=True)
        logging.info(f"Appended {len(new_data_df)} rows. Total: {len(combined_df)}")
    else:
        combined_df = new_data_df
        logging.info(f"Created new file with {len(new_data_df)} rows")
    
    # Remove duplicates if needed (optional)
    # combined_df = combined_df.drop_duplicates()
    
    combined_df.to_csv(output_file_path, index=False)
    logging.info(f"Successfully saved to: {output_file_path}")
    
except Exception as e:
    logging.error(f"Error in append_with_logging: {e}")
    raise
```

# For scheduling the script daily, you can use:

# 1. Windows Task Scheduler

# 2. Linux/Mac cron job:

# 0 9 * * * /usr/bin/python3 /path/to/your/script.py

# 3. Python scheduler library (schedule)

if **name** == “**main**”:
main()
