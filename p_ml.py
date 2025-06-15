import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
from pandas.tseries.offsets import BDay  # For business days

# Set random seeds for reproducibility
np.random.seed(42)

# Page configuration
st.set_page_config(page_title="Multi-Model Forecasting", layout="wide")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Multi-Model Forecast", "Model Optimization", "Final Forecast"])

# Helper functions
def load_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    
    if data.empty:
        st.error("No data found for the ticker. Try a different symbol.")
        return None
    
    # Handle missing values - forward fill then backfill if needed
    data.ffill(inplace=True)
    data.bfill(inplace=True)
    return data

def calculate_rsi(series, window=14):
    """Calculate Relative Strength Index (RSI) using pandas"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bollinger_bands(series, window=20, num_std=2):
    """Calculate Bollinger Bands using pandas"""
    sma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return upper_band, sma, lower_band

def create_features(df):
    df = df.copy()
    
    # Lag features
    for i in [1, 2, 3, 5, 7]:
        df[f'lag_{i}'] = df['Close'].shift(i)
    
    # Technical indicators using pandas
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['RSI_14'] = calculate_rsi(df['Close'], window=14)
    
    # Bollinger Bands
    upper, middle, lower = calculate_bollinger_bands(df['Close'], window=20)
    df['Bollinger_Upper'] = upper
    df['Bollinger_Lower'] = lower
    
    # Momentum indicators
    df['Momentum_5'] = df['Close'] - df['Close'].shift(5)
    df['ROC_10'] = (df['Close'] / df['Close'].shift(10) - 1) * 100
    
    # Drop initial rows with NaN values from feature creation
    df.dropna(inplace=True)
    return df

def generate_evaluation_tables(actual, predicted, last_train_price):
    # Ensure inputs are 1D arrays
    actual = np.array(actual).flatten()
    predicted = np.array(predicted).flatten()
    
    # Create DataFrame for Table 1
    df_eval = pd.DataFrame({
        'Actual Price': actual,
        'Predicted Price': predicted
    })
    
    # Add previous actual price
    prev_actual = [last_train_price] + actual[:-1].tolist()
    df_eval['Previous Actual Price'] = prev_actual
    
    # Calculate directions using explicit iteration
    actual_directions = []
    predicted_directions = []
    
    for i in range(len(df_eval)):
        # Get values as floats to ensure proper comparison
        actual_price = float(df_eval['Actual Price'].iloc[i])
        prev_price = float(df_eval['Previous Actual Price'].iloc[i])
        predicted_price = float(df_eval['Predicted Price'].iloc[i])
        
        # Determine directions
        actual_dir = 'UP' if actual_price > prev_price else 'DOWN'
        pred_dir = 'UP' if predicted_price > prev_price else 'DOWN'
        
        actual_directions.append(actual_dir)
        predicted_directions.append(pred_dir)
    
    # Add directions to DataFrame
    df_eval['Actual Direction'] = actual_directions
    df_eval['Predicted Direction'] = predicted_directions
    
    # Add comments
    df_eval['Comment'] = np.where(
        df_eval['Actual Direction'] == df_eval['Predicted Direction'], '✅', '❌')
    
    # Generate confusion matrix
    actual_directions_numeric = [1 if d == 'UP' else 0 for d in actual_directions]
    predicted_directions_numeric = [1 if d == 'UP' else 0 for d in predicted_directions]
    
    cm = confusion_matrix(actual_directions_numeric, predicted_directions_numeric, labels=[1, 0])
    cm_df = pd.DataFrame(
        cm, 
        index=['Actual UP', 'Actual DOWN'], 
        columns=['Predicted UP', 'Predicted DOWN']
    )
    
    # Calculate metrics
    if len(np.unique(actual_directions_numeric)) > 1:  # Check if both classes exist
        precision = precision_score(actual_directions_numeric, predicted_directions_numeric)
        recall = recall_score(actual_directions_numeric, predicted_directions_numeric)
    else:
        precision = recall = 0.0  # Handle case with only one class
    
    accuracy = accuracy_score(actual_directions_numeric, predicted_directions_numeric)
    
    metrics_df = pd.DataFrame({
        'Metric': ['Precision', 'Recall', 'Accuracy'],
        'Value': [precision, recall, accuracy]
    })
    
    # Ensure all values are simple scalars (not Series or Timestamps)
    df_eval = df_eval.applymap(lambda x: x.item() if isinstance(x, (pd.Timestamp, pd.Series)) else x)
    
    return df_eval, cm_df, metrics_df

# Main application logic
if page == "Multi-Model Forecast":
    st.title("Multi-Model Time Series Forecasting")
    
    # Data input - default to TSLA
    ticker = st.text_input("Enter Stock Ticker", "TSLA").upper()
    
    if st.button("Run Analysis"):
        with st.spinner("Loading data and training models..."):
            # Load and prepare data
            data = load_data(ticker, '2022-12-15', datetime.today().strftime('%Y-%m-%d'))
            if data is None:
                st.stop()
            
            # Save ticker to session state
            st.session_state.ticker = ticker
            
            # Create features
            featured_data = create_features(data)
            
            # Split data
            train = featured_data.loc['2023-01-01':'2023-12-31']
            if train.empty:
                st.error("Training data for 2023 is empty. Try a different ticker.")
                st.stop()
                
            test_start = (train.index[-1] + timedelta(days=1)).strftime('%Y-%m-%d')
            test = featured_data.loc[test_start:]
            
            if len(test) < 15:
                st.error(f"Not enough future data available ({len(test)} days). Need at least 15 trading days after 2023.")
                st.stop()
            
            test = test.iloc[:15]  # Use first 15 trading days of 2024
            
            # Ensure last_train_price is a simple float
            last_train_price = float(train['Close'].iloc[-1])
            
            # Initialize results storage
            results = {}
            
            # Linear Regression
            try:
                lr = LinearRegression()
                X_train = train.drop(columns=['Open', 'High', 'Low', 'Close', 'Volume'])
                y_train = train['Close']
                lr.fit(X_train, y_train)
                
                # Recursive forecasting
                lr_preds = []
                current_features = X_train.iloc[-1].copy()
                
                for _ in range(15):
                    pred = lr.predict([current_features.values])[0]
                    lr_preds.append(float(pred))  # Ensure float
                    
                    # Update features for next prediction
                    current_features = current_features.shift(1)
                    current_features['lag_1'] = pred
            except Exception as e:
                st.error(f"Linear Regression failed: {str(e)}")
                lr_preds = [last_train_price] * 15
            
            # Prophet
            try:
                # Ensure we're using Series for y, not DataFrame
                prophet_train = train.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds'})
                prophet_train['y'] = prophet_train['Close']  # Create y as a separate Series
                prophet_model = Prophet()
                prophet_model.fit(prophet_train[['ds', 'y']])
                
                future = prophet_model.make_future_dataframe(periods=15)
                prophet_forecast = prophet_model.predict(future)
                prophet_preds = [float(x) for x in prophet_forecast['yhat'][-15:].values]  # Ensure floats
            except Exception as e:
                st.error(f"Prophet failed: {str(e)}")
                prophet_preds = [last_train_price] * 15
            
            # Simple Moving Average (SMA) Baseline
            try:
                sma_value = float(train['Close'].rolling(window=20).mean().iloc[-1])  # Ensure float
                sma_preds = [sma_value] * 15
            except Exception as e:
                st.error(f"SMA Baseline failed: {str(e)}")
                sma_preds = [last_train_price] * 15
            
            # Random Forest
            try:
                rf = RandomForestRegressor(n_estimators=100, random_state=42)
                X_train = train.drop(columns=['Open', 'High', 'Low', 'Close', 'Volume'])
                y_train = train['Close']
                rf.fit(X_train, y_train)
                
                # Recursive forecasting
                rf_preds = []
                current_features = X_train.iloc[-1].copy()
                
                for _ in range(15):
                    pred = rf.predict([current_features.values])[0]
                    rf_preds.append(float(pred))  # Ensure float
                    
                    # Update features for next prediction
                    current_features = current_features.shift(1)
                    current_features['lag_1'] = pred
            except Exception as e:
                st.error(f"Random Forest failed: {str(e)}")
                rf_preds = [last_train_price] * 15
            
            # XGBoost
            try:
                xgb = XGBRegressor(n_estimators=100, random_state=42)
                X_train = train.drop(columns=['Open', 'High', 'Low', 'Close', 'Volume'])
                y_train = train['Close']
                xgb.fit(X_train, y_train)
                
                # Recursive forecasting
                xgb_preds = []
                current_features = X_train.iloc[-1].copy()
                
                for _ in range(15):
                    pred = xgb.predict([current_features.values])[0]
                    xgb_preds.append(float(pred))  # Ensure float
                    
                    # Update features for next prediction
                    current_features = current_features.shift(1)
                    current_features['lag_1'] = pred
            except Exception as e:
                st.error(f"XGBoost failed: {str(e)}")
                xgb_preds = [last_train_price] * 15
            
            # Store results
            models = {
                'Linear Regression': lr_preds,
                'Prophet': prophet_preds,
                'SMA Baseline': sma_preds,
                'Random Forest': rf_preds,
                'XGBoost': xgb_preds
            }
            
            # Evaluate each model
            best_accuracy = 0
            best_model_name = None
            
            for model_name, preds in models.items():
                # Ensure actual_prices are simple floats
                actual_prices = [float(x) for x in test['Close'].values[:len(preds)]]
                df_eval, cm_df, metrics_df = generate_evaluation_tables(
                    actual_prices, preds, last_train_price
                )
                
                accuracy = metrics_df[metrics_df['Metric'] == 'Accuracy']['Value'].values[0]
                results[model_name] = {
                    'evaluation_table': df_eval,
                    'confusion_matrix': cm_df,
                    'metrics': metrics_df,
                    'accuracy': accuracy
                }
                
                # Track best model
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model_name = model_name
            
            # Save best model to session state
            st.session_state.best_model = best_model_name
            st.session_state.results = results
            
            # Display results
            st.success(f"Best Model: {best_model_name} (Accuracy: {best_accuracy:.2%})")
            
            for model_name, result in results.items():
                st.subheader(f"Model: {model_name}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Table 1: Evaluation Data**")
                    st.dataframe(result['evaluation_table'])
                
                with col2:
                    st.write("**Table 2: Confusion Matrix**")
                    st.dataframe(result['confusion_matrix'])
                
                st.write("**Table 3: Classification Metrics**")
                st.dataframe(result['metrics'])
                
                # Plot predictions vs actual
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(result['evaluation_table'].index, 
                        result['evaluation_table']['Actual Price'], 
                        label='Actual', marker='o')
                ax.plot(result['evaluation_table'].index, 
                        result['evaluation_table']['Predicted Price'], 
                        label='Predicted', marker='x')
                ax.set_title(f"{model_name} Forecast")
                ax.set_xlabel("Day Index")
                ax.set_ylabel("Price")
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)
                
                st.markdown("---")
    
elif page == "Model Optimization":
    st.title("Model Optimization Experiments")
    
    if 'best_model' not in st.session_state:
        st.warning("Please run the Multi-Model Forecast first to determine the best model")
        st.stop()
    
    best_model = st.session_state.best_model
    ticker = st.session_state.ticker
    
    st.header(f"Optimizing: {best_model}")
    
    # Load data
    data = load_data(ticker, '2022-12-15', datetime.today().strftime('%Y-%m-%d'))
    if data is None:
        st.stop()
    
    # Create features
    featured_data = create_features(data)
    
    # Experiment 1: Different Forecasting Horizons
    st.subheader("Experiment 1: Forecasting Horizons")
    
    horizons = [12, 9, 6]
    horizon_results = {}
    
    for horizon in horizons:
        # Prepare data
        train = featured_data.loc['2023-01-01':'2023-12-31']
        test_start = (train.index[-1] + timedelta(days=1)).strftime('%Y-%m-%d')
        test = featured_data.loc[test_start:].iloc[:horizon]
        
        if len(test) < horizon:
            st.warning(f"Not enough data for {horizon}-day forecast. Skipping...")
            continue
            
        # Ensure last_train_price is a simple float
        last_train_price = float(train['Close'].iloc[-1])
        
        # Train and predict with best model
        if best_model == 'Prophet':
            try:
                prophet_train = train.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds'})
                prophet_train['y'] = prophet_train['Close']
                prophet_model = Prophet()
                prophet_model.fit(prophet_train[['ds', 'y']])
                
                future = prophet_model.make_future_dataframe(periods=horizon)
                prophet_forecast = prophet_model.predict(future)
                preds = [float(x) for x in prophet_forecast['yhat'][-horizon:].values]  # Ensure floats
            except Exception as e:
                st.error(f"Prophet failed for {horizon}-day horizon: {str(e)}")
                preds = [last_train_price] * horizon
        elif best_model == 'SMA Baseline':
            try:
                sma_value = float(train['Close'].rolling(window=20).mean().iloc[-1])  # Ensure float
                preds = [sma_value] * horizon
            except Exception as e:
                st.error(f"SMA Baseline failed for {horizon}-day horizon: {str(e)}")
                preds = [last_train_price] * horizon
        else:
            try:
                X_train = train.drop(columns=['Open', 'High', 'Low', 'Close', 'Volume'])
                y_train = train['Close']
                
                if best_model == 'Linear Regression':
                    model = LinearRegression()
                elif best_model == 'Random Forest':
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                elif best_model == 'XGBoost':
                    model = XGBRegressor(n_estimators=100, random_state=42)
                
                model.fit(X_train, y_train)
                preds = []
                current_features = X_train.iloc[-1].copy()
                
                for _ in range(horizon):
                    pred = model.predict([current_features.values])[0]
                    preds.append(float(pred))  # Ensure float
                    current_features = current_features.shift(1)
                    current_features['lag_1'] = pred
            except Exception as e:
                st.error(f"{best_model} failed for {horizon}-day horizon: {str(e)}")
                preds = [last_train_price] * horizon
        
        # Evaluate
        actual_prices = [float(x) for x in test['Close'].values[:len(preds)]]  # Ensure floats
        _, _, metrics_df = generate_evaluation_tables(actual_prices, preds, last_train_price)
        accuracy = metrics_df[metrics_df['Metric'] == 'Accuracy']['Value'].values[0]
        horizon_results[horizon] = accuracy
    
    # Display horizon results
    st.write("**Accuracy by Forecast Horizon:**")
    if horizon_results:
        horizon_df = pd.DataFrame(list(horizon_results.items()), columns=['Horizon', 'Accuracy'])
        st.dataframe(horizon_df)
        
        fig, ax = plt.subplots()
        ax.bar(horizon_df['Horizon'].astype(str), horizon_df['Accuracy'])
        ax.set_title("Accuracy by Forecast Horizon")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1)
        st.pyplot(fig)
    else:
        st.warning("No valid horizon results available")
    
    # Experiment 2: Training Window Sizes
    st.subheader("Experiment 2: Training Window Sizes")
    
    window_sizes = [100, 200, 300]
    window_results = {}
    
    for window in window_sizes:
        # Prepare data
        train_end = '2023-12-31'
        train_start = (pd.to_datetime(train_end) - timedelta(days=window)).strftime('%Y-%m-%d')
        train = featured_data.loc[train_start:train_end]
        
        if train.empty:
            st.warning(f"Not enough data for {window}-day window. Skipping...")
            continue
            
        test_start = (train.index[-1] + timedelta(days=1)).strftime('%Y-%m-%d')
        test = featured_data.loc[test_start:].iloc[:15]
        
        if len(test) < 15:
            st.warning(f"Not enough test data for {window}-day window. Skipping...")
            continue
            
        # Ensure last_train_price is a simple float
        last_train_price = float(train['Close'].iloc[-1])
        
        # Train and predict with best model
        if best_model == 'Prophet':
            try:
                prophet_train = train.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds'})
                prophet_train['y'] = prophet_train['Close']
                prophet_model = Prophet()
                prophet_model.fit(prophet_train[['ds', 'y']])
                
                future = prophet_model.make_future_dataframe(periods=15)
                prophet_forecast = prophet_model.predict(future)
                preds = [float(x) for x in prophet_forecast['yhat'][-15:].values]  # Ensure floats
            except Exception as e:
                st.error(f"Prophet failed for {window}-day window: {str(e)}")
                preds = [last_train_price] * 15
        elif best_model == 'SMA Baseline':
            try:
                sma_value = float(train['Close'].rolling(window=20).mean().iloc[-1])  # Ensure float
                preds = [sma_value] * 15
            except Exception as e:
                st.error(f"SMA Baseline failed for {window}-day window: {str(e)}")
                preds = [last_train_price] * 15
        else:
            try:
                X_train = train.drop(columns=['Open', 'High', 'Low', 'Close', 'Volume'])
                y_train = train['Close']
                
                if best_model == 'Linear Regression':
                    model = LinearRegression()
                elif best_model == 'Random Forest':
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                elif best_model == 'XGBoost':
                    model = XGBRegressor(n_estimators=100, random_state=42)
                
                model.fit(X_train, y_train)
                preds = []
                current_features = X_train.iloc[-1].copy()
                
                for _ in range(15):
                    pred = model.predict([current_features.values])[0]
                    preds.append(float(pred))  # Ensure float
                    current_features = current_features.shift(1)
                    current_features['lag_1'] = pred
            except Exception as e:
                st.error(f"{best_model} failed for {window}-day window: {str(e)}")
                preds = [last_train_price] * 15
        
        # Evaluate
        actual_prices = [float(x) for x in test['Close'].values[:len(preds)]]  # Ensure floats
        _, _, metrics_df = generate_evaluation_tables(actual_prices, preds, last_train_price)
        accuracy = metrics_df[metrics_df['Metric'] == 'Accuracy']['Value'].values[0]
        window_results[window] = accuracy
    
    # Display window results
    st.write("**Accuracy by Training Window Size:**")
    if window_results:
        window_df = pd.DataFrame(list(window_results.items()), columns=['Window Size', 'Accuracy'])
        st.dataframe(window_df)
        
        fig, ax = plt.subplots()
        ax.bar(window_df['Window Size'].astype(str), window_df['Accuracy'])
        ax.set_title("Accuracy by Training Window Size")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1)
        st.pyplot(fig)
    else:
        st.warning("No valid window results available")
    
    # Determine optimal parameters
    optimal_horizon = max(horizon_results, key=horizon_results.get) if horizon_results else 15
    optimal_window = max(window_results, key=window_results.get) if window_results else 300
    
    st.success(f"Optimal Configuration: Horizon={optimal_horizon} days, Window={optimal_window} days")
    
    # Save optimal parameters to session state
    st.session_state.optimal_horizon = optimal_horizon
    st.session_state.optimal_window = optimal_window

elif page == "Final Forecast":
    st.title("Future Price Forecast")
    
    # Set default to TSLA if not already set
    ticker = st.session_state.get('ticker', 'TSLA')
    best_model = st.session_state.get('best_model', 'Prophet')  # Default to Prophet if not set
    
    st.header(f"6-Day Forecast for {ticker}")
    st.write(f"Using best model: {best_model}")
    
    # Calculate dates
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365)
    
    # Load data
    with st.spinner(f"Loading {ticker} data..."):
        data = load_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        if data is None:
            st.stop()
    
    # Create features
    with st.spinner("Creating features..."):
        featured_data = create_features(data)
    
    # Prepare training data (last 365 days)
    train = featured_data.copy()
    
    if len(train) < 100:
        st.error(f"Not enough data points ({len(train)}). Need at least 100 trading days.")
        st.stop()
    
    last_train_price = float(train['Close'].iloc[-1])
    last_date = train.index[-1]
    
    # Generate next 6 business days
    future_dates = [last_date + BDay(i) for i in range(1, 7)]
    
    # Make predictions
    with st.spinner("Generating forecast..."):
        if best_model == 'Prophet':
            try:
                prophet_train = train.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
                prophet_model = Prophet()
                prophet_model.fit(prophet_train)
                
                future_df = pd.DataFrame({'ds': future_dates})
                forecast = prophet_model.predict(future_df)
                preds = forecast['yhat'].values
            except Exception as e:
                st.error(f"Prophet failed: {str(e)}")
                preds = [last_train_price] * 6
        
        elif best_model == 'SMA Baseline':
            try:
                sma_value = float(train['Close'].rolling(window=20).mean().iloc[-1])
                preds = [sma_value] * 6
            except Exception as e:
                st.error(f"SMA Baseline failed: {str(e)}")
                preds = [last_train_price] * 6
        
        else:
            try:
                X_train = train.drop(columns=['Open', 'High', 'Low', 'Close', 'Volume'])
                y_train = train['Close']
                
                if best_model == 'Linear Regression':
                    model = LinearRegression()
                elif best_model == 'Random Forest':
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                elif best_model == 'XGBoost':
                    model = XGBRegressor(n_estimators=100, random_state=42)
                else:
                    model = LinearRegression()  # Fallback
                
                model.fit(X_train, y_train)
                
                preds = []
                current_features = X_train.iloc[-1].copy()
                
                for _ in range(6):
                    pred = model.predict([current_features.values])[0]
                    preds.append(float(pred))
                    current_features = current_features.shift(1)
                    current_features['lag_1'] = pred
            except Exception as e:
                st.error(f"{best_model} failed: {str(e)}")
                preds = [last_train_price] * 6
    
    # Create forecast DataFrame
    forecast_df = pd.DataFrame({
        'Date': [d.strftime('%Y-%m-%d') for d in future_dates],
        'Predicted Price': preds
    })
    
    # Calculate daily changes
    forecast_df['Change (%)'] = forecast_df['Predicted Price'].pct_change().fillna(0) * 100
    forecast_df['Change (%)'] = forecast_df['Change (%)'].apply(lambda x: f"{x:.2f}%")
    
    # Display forecast
    st.subheader("6-Day Price Forecast")
    st.dataframe(forecast_df.style.format({'Predicted Price': "{:.2f}"}))
    
    # Plot forecast
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Last 30 days of historical data
    history = train[['Close']].iloc[-30:]
    ax.plot(history.index, history['Close'], 'bo-', label='Historical Prices')
    
    # Forecasted prices
    ax.plot(future_dates, preds, 'ro--', label='Forecasted Prices')
    
    # Annotate forecast points
    for i, (date, price) in enumerate(zip(future_dates, preds)):
        ax.annotate(f"{price:.2f}", (date, price), 
                    textcoords="offset points", xytext=(0,10), ha='center')
    
    ax.set_title(f"{ticker} 6-Day Price Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True)
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    # Performance metrics
    current_price = last_train_price
    forecast_high = max(preds)
    forecast_low = min(preds)
    forecast_change = (preds[-1] - current_price) / current_price * 100
    
    st.subheader("Forecast Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Current Price", f"${current_price:.2f}")
    col2.metric("6-Day Forecast", f"${preds[-1]:.2f}", f"{forecast_change:.2f}%")
    col3.metric("Forecast Range", f"${forecast_low:.2f} - ${forecast_high:.2f}")

# Initialize session state variables
if 'best_model' not in st.session_state:
    st.session_state.best_model = None
if 'ticker' not in st.session_state:
    st.session_state.ticker = 'TSLA'  # Default to TSLA
if 'optimal_horizon' not in st.session_state:
    st.session_state.optimal_horizon = None
if 'optimal_window' not in st.session_state:
    st.session_state.optimal_window = None