import pandas as pd
import os
from datetime import datetime

import pandas as pd
import os
from datetime import datetime

def append_to_output_file(new_data_df, output_file_path, key_column):
    """
    Append new data to existing CSV file, updating existing rows if key_column matches.
    
    Args:
        new_data_df: pandas DataFrame with new data to append
        output_file_path: path to the output CSV file
        key_column: column name to check for existing values
    """
    
    # Check if output file exists
    if os.path.exists(output_file_path):
        # Read existing data
        try:
            existing_df = pd.read_csv(output_file_path)
            
            # Find rows that need to be updated
            existing_keys = existing_df[key_column].values
            new_keys = new_data_df[key_column].values
            
            # Separate new data into update and append
            update_mask = new_data_df[key_column].isin(existing_keys)
            update_data = new_data_df[update_mask]
            append_data = new_data_df[~update_mask]
            
            # Update existing rows
            for idx, row in update_data.iterrows():
                key_value = row[key_column]
                existing_df.loc[existing_df[key_column] == key_value, :] = row.values
            
            # Append new rows
            if len(append_data) > 0:
                combined_df = pd.concat([existing_df, append_data], ignore_index=True)
            else:
                combined_df = existing_df.copy()
            
            print(f"Updated {len(update_data)} rows, appended {len(append_data)} rows. Total rows: {len(combined_df)}")
            
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


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Example data - replace with your actual x and y arrays
x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([1.2, 1.8, 3.1, 5.2, 8.9, 15.1, 25.8, 44.2, 75.3, 128.5, 218.7])

# Define the three exponential functions
def exponential(x, a, b):
    """Standard exponential: y = a * e^(b*x)"""
    return a * np.exp(b * x)

def power_law(x, a, b):
    """Power law: y = a * x^b"""
    return a * np.power(x, b)

def exponential_offset(x, a, b, c):
    """Exponential with offset: y = a * e^(b*x) + c"""
    return a * np.exp(b * x) + c

# Fit the curves
try:
    # Fit standard exponential
    popt1, _ = curve_fit(exponential, x, y, p0=[1, 0.1])
    
    # Fit power law (need to avoid x=0 if present)
    x_power = x[x > 0]
    y_power = y[x > 0]
    popt2, _ = curve_fit(power_law, x_power, y_power, p0=[1, 2])
    
    # Fit exponential with offset
    popt3, _ = curve_fit(exponential_offset, x, y, p0=[1, 0.1, 0])
    
    # Generate smooth curves for plotting
    x_smooth = np.linspace(x.min(), x.max(), 200)
    y1 = exponential(x_smooth, *popt1)
    y2 = power_law(x_smooth, *popt2)
    y3 = exponential_offset(x_smooth, *popt3)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot original data
    plt.scatter(x, y, color='red', s=50, zorder=5, label='Data')
    
    # Plot fitted curves
    plt.plot(x_smooth, y1, 'b-', linewidth=2, 
             label=f'Exponential: y = {popt1[0]:.2f}·e^({popt1[1]:.2f}x)')
    plt.plot(x_smooth, y2, 'g-', linewidth=2, 
             label=f'Power law: y = {popt2[0]:.2f}·x^{popt2[1]:.2f}')
    plt.plot(x_smooth, y3, 'm-', linewidth=2, 
             label=f'Exp+offset: y = {popt3[0]:.2f}·e^({popt3[1]:.2f}x) + {popt3[2]:.2f}')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Exponential Curve Fitting Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Print fitted parameters
    print("Fitted Parameters:")
    print(f"1. Exponential: a={popt1[0]:.4f}, b={popt1[1]:.4f}")
    print(f"2. Power law: a={popt2[0]:.4f}, b={popt2[1]:.4f}")
    print(f"3. Exponential with offset: a={popt3[0]:.4f}, b={popt3[1]:.4f}, c={popt3[2]:.4f}")
    
except Exception as e:
    print(f"Error in curve fitting: {e}")
    print("Try adjusting initial parameter guesses or check your data.")
