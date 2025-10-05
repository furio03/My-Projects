import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

def calculate_cmf(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Calculate the Chaikin Money Flow (CMF).
    """
    mf_multiplier = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low']).replace(0, 1e-10)
    mf_volume = mf_multiplier * df['Volume']
    cmf = mf_volume.rolling(window=period).sum() / df['Volume'].rolling(window=period).sum()
    return cmf

def consecutive_up(series: pd.Series) -> pd.Series:
    """
    Count consecutive increases: increases by 1 for each rise, resets to 0 at the first correction or sideways movement.
    """
    consecutive = []
    count = 0
    prev_value = None
    for value in series:
        if prev_value is not None and value > prev_value:
            count += 1
        elif prev_value is not None:
            count = 0
        else:
            count = 0
        consecutive.append(count)
        prev_value = value
    return pd.Series(consecutive, index=series.index)

def consecutive_down(series: pd.Series) -> pd.Series:
    """
    Count consecutive decreases: increases by 1 for each drop, resets to 0 at the first correction or sideways movement.
    """
    consecutive = []
    count = 0
    prev_value = None
    for value in series:
        if prev_value is not None and value < prev_value:
            count += 1
        elif prev_value is not None:
            count = 0
        else:
            count = 0
        consecutive.append(count)
        prev_value = value
    return pd.Series(consecutive, index=series.index)

def find_best_k(df: pd.DataFrame, k_range=range(2, 10)):
    """
    Find the optimal value of k using the Silhouette Score.
    """
    feature_cols = [col for col in df.select_dtypes(include='number').columns if col != 'action']
    X = df[feature_cols].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    best_k = None
    best_score = -1
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=250)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        if score > best_score:
            best_score = score
            best_k = k
    return best_k

def add_kmeans(df: pd.DataFrame, k: int , model=None):
    """
    If model is None, find the optimal k, train KMeans on df and return the groups, the model and the chosen k.
    If model is provided, apply the model to the data and return the groups.
    """
    
    feature_cols = [col for col in df.select_dtypes(include='number').columns if col != 'action']
    X = df[feature_cols].dropna()
    scaler = StandardScaler()
    X[feature_cols] = scaler.fit_transform(X[feature_cols])
    if model is None:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=250)
        groups = pd.Series([pd.NA] * len(df), index=df.index)
        groups.loc[X.index] = kmeans.fit_predict(X)
        return groups, kmeans, k
    else:
        groups = pd.Series([pd.NA] * len(df), index=df.index)
        groups.loc[X.index] = model.predict(X)
        return groups


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate the Relative Strength Index (RSI) for a given price series.
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate various financial indicators for the given DataFrame.
    The indicator columns will include the window size in their names.
    """
    # Ensure the DataFrame has the required columns
    required_columns = ['Close','Volume']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"DataFrame must contain the following columns: {required_columns}")

    # Calculate indicators
    df['sma_10'] = df['Close'].rolling(window=10).mean()
    df['sma_30'] = df['Close'].rolling(window=30).mean()
    df['sma_50'] = df['Close'].rolling(window=50).mean()

    df['volume_sma_10'] = df['Volume'].rolling(window=10).mean()
    df['volume_sma_30'] = df['Volume'].rolling(window=30).mean()
    df['volume_sma_50'] = df['Volume'].rolling(window=50).mean()

    df['ema_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['ema_30'] = df['Close'].ewm(span=30, adjust=False).mean()
    df['ema_50'] = df['Close'].ewm(span=50, adjust=False).mean()

    df['ema_volume_10'] = df['Volume'].ewm(span=10, adjust=False).mean()
    df['ema_volume_30'] = df['Volume'].ewm(span=30, adjust=False).mean()
    df['ema_volume_50'] = df['Volume'].ewm(span=50, adjust=False).mean()

    df['macd'] = df['ema_10'] - df['ema_50']
    df['macd_signal'] = df['macd'].ewm(span=10, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    df['rsi_10'] = calculate_rsi(df['Close'], period=10)
    df['rsi_30'] = calculate_rsi(df['Close'], period=30)
    df['rsi_50'] = calculate_rsi(df['Close'], period=50)

    df['volume_rsi_10'] = calculate_rsi(df['Volume'], period=10)
    df['volume_rsi_30'] = calculate_rsi(df['Volume'], period=30)
    df['volume_rsi_50'] = calculate_rsi(df['Volume'], period=50)

    df['std_dev_10'] = df['Close'].rolling(window=10).std()
    df['std_dev_30'] = df['Close'].rolling(window=30).std()
    df['std_dev_50'] = df['Close'].rolling(window=50).std()

    df['std_dev_volume_10'] = df['Volume'].rolling(window=10).std()
    df['std_dev_volume_30'] = df['Volume'].rolling(window=30).std()
    df['std_dev_volume_50'] = df['Volume'].rolling(window=50).std()

    df['up_or_down_10_sma'] = df.apply(lambda row: 1 if row['Close'] > row['sma_10'] else -1, axis=1)
    df['up_or_down_30_sma'] = df.apply(lambda row: 1 if row['Close'] > row['sma_30'] else -1, axis=1)
    df['up_or_down_50_sma'] = df.apply(lambda row: 1 if row['Close'] > row['sma_50'] else -1, axis=1)

    df['up_or_down_10_volume_sma'] = df.apply(lambda row: 1 if row['Volume'] > row['volume_sma_10'] else -1, axis=1)
    df['up_or_down_30_volume_sma'] = df.apply(lambda row: 1 if row['Volume'] > row['volume_sma_30'] else -1, axis=1)
    df['up_or_down_50_volume_sma'] = df.apply(lambda row: 1 if row['Volume'] > row['volume_sma_50'] else -1, axis=1)

    df['up_or_down_10_ema'] = df.apply(lambda row: 1 if row['Close'] > row['ema_10'] else -1, axis=1)
    df['up_or_down_30_ema'] = df.apply(lambda row: 1 if row['Close'] > row['ema_30'] else -1, axis=1)
    df['up_or_down_50_ema'] = df.apply(lambda row: 1 if row['Close'] > row['ema_50'] else -1, axis=1)

    df['up_or_down_10_volume_ema'] = df.apply(lambda row: 1 if row['Volume'] > row['ema_volume_10'] else -1, axis=1)
    df['up_or_down_30_volume_ema'] = df.apply(lambda row: 1 if row['Volume'] > row['ema_volume_30'] else -1, axis=1)
    df['up_or_down_50_volume_ema'] = df.apply(lambda row: 1 if row['Volume'] > row['ema_volume_50'] else -1, axis=1)

    df['cci_10'] = (df['Close'] - df['sma_10']) / df['std_dev_10']
    df['cci_30'] = (df['Close'] - df['sma_30']) / df['std_dev_30']
    df['cci_50'] = (df['Close'] - df['sma_50']) / df['std_dev_50']

    df['cci_volume_10'] = (df['Volume'] - df['volume_sma_10']) / df['std_dev_volume_10']
    df['cci_volume_30'] = (df['Volume'] - df['volume_sma_30']) / df['std_dev_volume_30']
    df['cci_volume_50'] = (df['Volume'] - df['volume_sma_50']) / df['std_dev_volume_50']


    df['is_max_50'] = df.apply(
        lambda row: 1 if row['Close'] == df['Close'].rolling(window=50).max().loc[row.name] else -1, axis=1)
    df['is_max_30'] = df.apply(
        lambda row: 1 if row['Close'] == df['Close'].rolling(window=30).max().loc[row.name] else -1, axis=1)
    df['is_max_10'] = df.apply(
        lambda row: 1 if row['Close'] == df['Close'].rolling(window=10).max().loc[row.name] else -1, axis=1)
    

    df['is_min_50'] = df.apply(
        lambda row: 1 if row['Close'] == df['Close'].rolling(window=50).min().loc[row.name] else -1, axis=1)
    df['is_min_30'] = df.apply(
        lambda row: 1 if row['Close'] == df['Close'].rolling(window=30).min().loc[row.name] else -1, axis=1)
    df['is_min_10'] = df.apply(
        lambda row: 1 if row['Close'] == df['Close'].rolling(window=10).min().loc[row.name] else -1, axis=1)
    

    df['consecutive_up'] = consecutive_up(df['Close'])
    df['consecutive_down'] = consecutive_down(df['Close'])
    
    df['consecutive_up_volume'] = consecutive_up(df['Volume'])
    df['consecutive_down_volume'] = consecutive_down(df['Volume'])

    df['cmf_10'] = calculate_cmf(df, period=10)
    df['cmf_30'] = calculate_cmf(df, period=30)
    df['cmf_50'] = calculate_cmf(df, period=50)



    return df



