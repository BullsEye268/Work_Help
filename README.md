# Work_Help

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import inv
import warnings
warnings.filterwarnings(‘ignore’)

class StockKalmanFilter:
“””
Kalman Filter for stock trend analysis
Estimates price level and trend (velocity) simultaneously
“””

```
def __init__(self, process_variance=1e-5, observation_variance=1e-3):
    """
    Initialize Kalman Filter
    
    Parameters:
    process_variance: How much we expect the true price/trend to change (lower = smoother)
    observation_variance: How much noise we expect in price observations (lower = trust data more)
    """
    self.process_var = process_variance
    self.obs_var = observation_variance
    
    # State vector: [price, trend/velocity]
    self.state = np.array([0.0, 0.0])
    
    # State covariance matrix
    self.P = np.eye(2) * 1.0
    
    # State transition matrix (constant velocity model)
    self.F = np.array([[1.0, 1.0],  # price = previous_price + trend
                      [0.0, 1.0]])  # trend = previous_trend
    
    # Observation matrix (we only observe price, not trend directly)
    self.H = np.array([[1.0, 0.0]])
    
    # Process noise covariance
    self.Q = np.array([[self.process_var, 0.0],
                      [0.0, self.process_var]])
    
    # Observation noise covariance
    self.R = np.array([[self.obs_var]])
    
def predict(self):
    """Prediction step"""
    # Predict state
    self.state = self.F @ self.state
    
    # Predict covariance
    self.P = self.F @ self.P @ self.F.T + self.Q
    
def update(self, observation):
    """Update step with new price observation"""
    # Innovation (prediction error)
    y = observation - self.H @ self.state
    
    # Innovation covariance
    S = self.H @ self.P @ self.H.T + self.R
    
    # Kalman gain
    K = self.P @ self.H.T @ inv(S)
    
    # Update state
    self.state = self.state + K @ y
    
    # Update covariance
    I = np.eye(len(self.state))
    self.P = (I - K @ self.H) @ self.P
    
def get_state(self):
    """Return current state [price_estimate, trend_estimate]"""
    return self.state.copy()
```

def apply_kalman_filter(df, price_column=‘close’, process_var=1e-5, obs_var=1e-3,
trading_hours_only=True, market_open=8, market_close=18):
“””
Apply Kalman filter to stock data

```
Parameters:
df: DataFrame with stock data (must have datetime index)
price_column: Column name containing prices
process_var: Process noise variance (tune for smoothness)
obs_var: Observation noise variance (tune for responsiveness)
trading_hours_only: If True, handles gaps between trading sessions
market_open: Market opening hour (24-hour format)
market_close: Market closing hour (24-hour format)

Returns:
DataFrame with original data plus Kalman estimates
"""

# Ensure datetime index
if not isinstance(df.index, pd.DatetimeIndex):
    raise ValueError("DataFrame must have a datetime index for trading hours analysis")

# Sort by time to ensure chronological order
df = df.sort_index()

# Initialize filter
kf = StockKalmanFilter(process_var, obs_var)

# Initialize with first price
kf.state[0] = df[price_column].iloc[0]

# Storage for results
filtered_prices = []
trends = []

prev_timestamp = None

# Process each observation
for timestamp, row in df.iterrows():
    price = row[price_column]
    
    if trading_hours_only and prev_timestamp is not None:
        # Check if this is the start of a new trading session
        is_new_session = (
            timestamp.date() != prev_timestamp.date() or  # New day
            (prev_timestamp.hour >= market_close and timestamp.hour >= market_open) or  # Gap over close
            (timestamp - prev_timestamp).total_seconds() > 3600 * 2  # Gap > 2 hours
        )
        
        if is_new_session:
            # Reset trend component for new session, keep price estimate
            # This prevents overnight gaps from being treated as large trend moves
            kf.state[1] = 0.0  # Reset trend to zero
            # Increase uncertainty about trend after gap
            kf.P[1, 1] *= 2.0
    
    # Predict
    kf.predict()
    
    # Update with observation
    kf.update(np.array([price]))
    
    # Store results
    state = kf.get_state()
    filtered_prices.append(state[0])
    trends.append(state[1])
    
    prev_timestamp = timestamp

# Create result DataFrame
result_df = df.copy()
result_df['kalman_price'] = filtered_prices
result_df['kalman_trend'] = trends

# Add trend classifications
result_df['trend_signal'] = np.where(result_df['kalman_trend'] > 0, 1, 
                                    np.where(result_df['kalman_trend'] < 0, -1, 0))

return result_df
```

def analyze_trends(df, trend_threshold=0.01, market_open=8, market_close=18):
“””
Analyze trends over different time horizons for trading hours data

```
Parameters:
df: DataFrame with Kalman filter results
trend_threshold: Minimum trend magnitude to consider significant
market_open: Market opening hour (24-hour format)
market_close: Market closing hour (24-hour format)
"""

# Ensure datetime index
if not isinstance(df.index, pd.DatetimeIndex):
    raise ValueError("DataFrame must have a datetime index")

results = {}

# Filter for trading hours only
trading_hours_mask = (df.index.hour >= market_open) & (df.index.hour < market_close)
df_trading = df[trading_hours_mask].copy()

# 1. Daily trends (within each trading day)
df_trading['date'] = df_trading.index.date
daily_trends = []

for date in df_trading['date'].unique():
    day_data = df_trading[df_trading['date'] == date].copy()
    if len(day_data) > 3:  # Need at least a few data points
        trend_strength = day_data['kalman_trend'].mean()
        trend_consistency = (day_data['kalman_trend'] > 0).mean()  # % of time trending up
        trend_volatility = day_data['kalman_trend'].std()
        
        # Calculate intraday price change
        price_change = day_data['kalman_price'].iloc[-1] - day_data['kalman_price'].iloc[0]
        price_change_pct = (price_change / day_data['kalman_price'].iloc[0]) * 100
        
        daily_trends.append({
            'date': date,
            'avg_trend': trend_strength,
            'trend_consistency': trend_consistency,
            'trend_volatility': trend_volatility,
            'price_change': price_change,
            'price_change_pct': price_change_pct,
            'trend_direction': 'up' if trend_strength > trend_threshold else 'down' if trend_strength < -trend_threshold else 'sideways',
            'session_start_price': day_data['kalman_price'].iloc[0],
            'session_end_price': day_data['kalman_price'].iloc[-1],
            'data_points': len(day_data)
        })

results['daily_trends'] = pd.DataFrame(daily_trends)

# 2. Multi-day trends (2-3 trading days)
# Calculate rolling trends based on trading hours only
df_trading['trend_2day'] = df_trading['kalman_trend'].rolling(window=20, center=True).mean()  # ~2 days of hourly data
df_trading['trend_3day'] = df_trading['kalman_trend'].rolling(window=30, center=True).mean()  # ~3 days of hourly data

# 3. Intraday pattern analysis
df_trading['hour'] = df_trading.index.hour
hourly_patterns = df_trading.groupby('hour').agg({
    'kalman_trend': ['mean', 'std', 'count'],
    'kalman_price': ['mean', 'std']
}).round(4)

results['hourly_patterns'] = hourly_patterns

# 4. Session analysis (morning vs afternoon trends)
df_trading['session_period'] = df_trading['hour'].apply(
    lambda x: 'morning' if x < 12 else 'afternoon'
)

session_analysis = df_trading.groupby(['date', 'session_period']).agg({
    'kalman_trend': ['mean', 'std'],
    'kalman_price': ['first', 'last']
}).round(4)

results['session_analysis'] = session_analysis

# 5. Trend strength and persistence (trading hours only)
df_trading['trend_strength'] = np.abs(df_trading['kalman_trend'])
df_trading['trend_persistence'] = df_trading['trend_signal'].rolling(window=5).apply(
    lambda x: np.abs(x.sum()) / len(x), raw=True
)

# 6. Gap analysis (overnight changes)
gap_analysis = []
for date in df_trading['date'].unique():
    day_data = df_trading[df_trading['date'] == date]
    if len(day_data) > 0:
        # Find previous trading day
        prev_date = date - pd.Timedelta(days=1)
        while prev_date not in df_trading['date'].values and (date - prev_date).days < 5:
            prev_date = prev_date - pd.Timedelta(days=1)
        
        if prev_date in df_trading['date'].values:
            prev_day_data = df_trading[df_trading['date'] == prev_date]
            if len(prev_day_data) > 0:
                prev_close = prev_day_data['kalman_price'].iloc[-1]
                current_open = day_data['kalman_price'].iloc[0]
                gap = current_open - prev_close
                gap_pct = (gap / prev_close) * 100
                
                gap_analysis.append({
                    'date': date,
                    'prev_close': prev_close,
                    'current_open': current_open,
                    'gap': gap,
                    'gap_pct': gap_pct,
                    'gap_direction': 'up' if gap > 0 else 'down' if gap < 0 else 'flat'
                })

results['gap_analysis'] = pd.DataFrame(gap_analysis)

# Add the processed trading hours data
results['processed_df'] = df_trading
results['full_df'] = df  # Keep original for reference

return results
```

def plot_kalman_analysis(df, start_date=None, end_date=None, figsize=(15, 12)):
“””
Create comprehensive plots of Kalman filter analysis
“””

```
# Filter date range if specified
if start_date or end_date:
    if start_date:
        df = df[df.index >= start_date]
    if end_date:
        df = df[df.index <= end_date]

fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)

# Plot 1: Price and filtered price
axes[0].plot(df.index, df.iloc[:, 0], alpha=0.7, label='Observed Price', color='lightblue')
axes[0].plot(df.index, df['kalman_price'], label='Kalman Filtered Price', color='navy', linewidth=2)
axes[0].set_ylabel('Price')
axes[0].legend()
axes[0].set_title('Stock Price vs Kalman Filtered Price')
axes[0].grid(True, alpha=0.3)

# Plot 2: Trend (velocity)
axes[1].plot(df.index, df['kalman_trend'], color='green', linewidth=1.5)
axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
axes[1].fill_between(df.index, df['kalman_trend'], 0, 
                    where=(df['kalman_trend'] > 0), color='green', alpha=0.3, label='Uptrend')
axes[1].fill_between(df.index, df['kalman_trend'], 0, 
                    where=(df['kalman_trend'] < 0), color='red', alpha=0.3, label='Downtrend')
axes[1].set_ylabel('Trend (Velocity)')
axes[1].legend()
axes[1].set_title('Estimated Trend/Velocity')
axes[1].grid(True, alpha=0.3)

# Plot 3: Multi-day trends
if 'trend_2day' in df.columns and 'trend_3day' in df.columns:
    axes[2].plot(df.index, df['trend_2day'], label='2-Day Trend', color='orange', linewidth=2)
    axes[2].plot(df.index, df['trend_3day'], label='3-Day Trend', color='purple', linewidth=2)
    axes[2].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    axes[2].set_ylabel('Multi-Day Trend')
    axes[2].legend()
    axes[2].set_title('2-Day and 3-Day Rolling Trends')
    axes[2].grid(True, alpha=0.3)

# Plot 4: Trend strength and persistence
if 'trend_strength' in df.columns and 'trend_persistence' in df.columns:
    ax4_twin = axes[3].twinx()
    axes[3].plot(df.index, df['trend_strength'], color='darkgreen', alpha=0.7, label='Trend Strength')
    ax4_twin.plot(df.index, df['trend_persistence'], color='orange', alpha=0.7, label='Trend Persistence')
    
    axes[3].set_ylabel('Trend Strength', color='darkgreen')
    ax4_twin.set_ylabel('Trend Persistence', color='orange')
    axes[3].set_title('Trend Strength and Persistence')
    axes[3].grid(True, alpha=0.3)

plt.xlabel('Time')
plt.tight_layout()
plt.show()
```

# Example usage and parameter tuning guide

def example_usage():
“””
Example of how to use the Kalman filter for trading hours stock analysis
Replace this with your actual data loading
“””

```
# Example: Load your data - adjust this to your actual data loading
# df = pd.read_csv('your_stock_data.csv')
# df['timestamp'] = pd.to_datetime(df['timestamp'])
# df = df.set_index('timestamp')

# For demonstration, create sample trading hours data
np.random.seed(42)

# Create 5 days of trading hours data (8 AM to 6 PM)
trading_dates = pd.date_range('2024-01-01', periods=5, freq='D')
all_data = []

base_price = 100
for date in trading_dates:
    # Skip weekends for realism
    if date.weekday() < 5:  # Monday = 0, Sunday = 6
        trading_hours = pd.date_range(
            start=date.replace(hour=8), 
            end=date.replace(hour=17), 
            freq='H'
        )
        
        # Create intraday trend with some randomness
        daily_trend = np.random.randn() * 0.5
        hourly_changes = np.random.randn(len(trading_hours)) * 0.3 + daily_trend * 0.1
        
        for i, timestamp in enumerate(trading_hours):
            base_price += hourly_changes[i]
            noise = np.random.randn() * 0.2
            all_data.append({
                'timestamp': timestamp,
                'close': base_price + noise,
                'volume': np.random.randint(1000, 10000)
            })

df = pd.DataFrame(all_data)
df = df.set_index('timestamp')

print("Step 1: Apply Kalman Filter for Trading Hours")
# Apply Kalman filter with trading hours handling
df_filtered = apply_kalman_filter(df, 
                                 price_column='close',
                                 process_var=1e-4,
                                 obs_var=1e-2,
                                 trading_hours_only=True,
                                 market_open=8,
                                 market_close=18)

print("Step 2: Analyze Trading Hours Trends")
trend_analysis = analyze_trends(df_filtered, 
                               trend_threshold=0.01,
                               market_open=8,
                               market_close=18)

print("Step 3: Plot Results")
plot_kalman_analysis(df_filtered)

print("\nDaily Trend Summary:")
print(trend_analysis['daily_trends'])

print("\nGap Analysis (Overnight Changes):")
if not trend_analysis['gap_analysis'].empty:
    print(trend_analysis['gap_analysis'])

print("\nHourly Patterns:")
print(trend_analysis['hourly_patterns'])

return df_filtered, trend_analysis
```

# Parameter tuning guide

def tune_parameters(df, price_column=‘close’):
“””
Helper function to test different parameter combinations
“””
print(“Parameter Tuning Guide:”)
print(“1. process_var (1e-6 to 1e-3):”)
print(”   - Lower values = smoother, less responsive trends”)
print(”   - Higher values = more responsive, noisier trends”)
print(“2. obs_var (1e-4 to 1e-1):”)
print(”   - Lower values = trust price data more”)
print(”   - Higher values = more smoothing of price data”)
print(”\nRecommended starting points:”)
print(”- For noisy intraday data: process_var=1e-5, obs_var=1e-2”)
print(”- For cleaner daily data: process_var=1e-4, obs_var=1e-3”)
print(”- For very noisy tick data: process_var=1e-6, obs_var=1e-1”)

if **name** == “**main**”:
# Run example
df_result, analysis = example_usage()
tune_parameters(df_result)