import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Union
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.decomposition import PCA, FastICA
from sklearn.feature_selection import SelectFromModel, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class AdvancedPreprocessor:
    def __init__(self):
        self.scalers = {}
        self.transformers = {}
        self.feature_selectors = {}
        
    def handle_missing_values(self, df: pd.DataFrame, 
                            method: str = 'interpolate') -> pd.DataFrame:
        """Handle missing values using advanced techniques"""
        if method == 'interpolate':
            # Time-based interpolation
            df = df.interpolate(method='time')
        elif method == 'knn':
            # KNN imputation
            from sklearn.impute import KNNImputer
            imputer = KNNImputer(n_neighbors=5)
            df = pd.DataFrame(imputer.fit_transform(df), 
                            columns=df.columns, 
                            index=df.index)
        elif method == 'mice':
            # Multiple Imputation by Chained Equations
            from sklearn.experimental import enable_iterative_imputer
            from sklearn.impute import IterativeImputer
            imputer = IterativeImputer(random_state=42)
            df = pd.DataFrame(imputer.fit_transform(df), 
                            columns=df.columns, 
                            index=df.index)
            
        return df
    
    def detect_outliers(self, df: pd.DataFrame, 
                       method: str = 'zscore') -> Dict[str, List[int]]:
        """Detect outliers using multiple methods"""
        outliers = {}
        
        if method == 'zscore':
            # Z-score method
            z_scores = np.abs(stats.zscore(df))
            outliers['zscore'] = np.where(z_scores > 3)[0]
            
        elif method == 'iqr':
            # IQR method
            Q1 = df.quantile(0.25)
            Q3 = df.quantile(0.75)
            IQR = Q3 - Q1
            outliers['iqr'] = np.where((df < (Q1 - 1.5 * IQR)) | 
                                     (df > (Q3 + 1.5 * IQR)))[0]
            
        elif method == 'isolation_forest':
            # Isolation Forest
            from sklearn.ensemble import IsolationForest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            preds = iso_forest.fit_predict(df)
            outliers['isolation_forest'] = np.where(preds == -1)[0]
            
        return outliers
    
    def handle_outliers(self, df: pd.DataFrame, 
                       method: str = 'winsorize') -> pd.DataFrame:
        """Handle outliers using advanced techniques"""
        if method == 'winsorize':
            # Winsorization
            from scipy.stats import mstats
            df = pd.DataFrame(mstats.winsorize(df, limits=[0.05, 0.05]), 
                            columns=df.columns, 
                            index=df.index)
            
        elif method == 'robust_scaler':
            # Robust scaling
            scaler = RobustScaler()
            df = pd.DataFrame(scaler.fit_transform(df), 
                            columns=df.columns, 
                            index=df.index)
            
        return df
    
    def scale_features(self, df: pd.DataFrame, 
                      method: str = 'standard') -> pd.DataFrame:
        """Scale features using advanced techniques"""
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        elif method == 'power':
            scaler = PowerTransformer(method='yeo-johnson')
            
        scaled_data = scaler.fit_transform(df)
        self.scalers[method] = scaler
        
        return pd.DataFrame(scaled_data, 
                          columns=df.columns, 
                          index=df.index)
    
    def transform_features(self, df: pd.DataFrame, 
                         method: str = 'pca') -> Tuple[pd.DataFrame, np.ndarray]:
        """Transform features using dimensionality reduction"""
        if method == 'pca':
            transformer = PCA(n_components=0.95)
        elif method == 'ica':
            transformer = FastICA(n_components=None, random_state=42)
            
        transformed_data = transformer.fit_transform(df)
        self.transformers[method] = transformer
        
        explained_variance = None
        if method == 'pca':
            explained_variance = transformer.explained_variance_ratio_
            
        return pd.DataFrame(transformed_data, 
                          index=df.index), explained_variance
    
    def select_features(self, df: pd.DataFrame, target: pd.Series,
                       method: str = 'mutual_info') -> Tuple[pd.DataFrame, List[str]]:
        """Select features using advanced techniques"""
        if method == 'mutual_info':
            # Mutual information
            scores = mutual_info_regression(df, target)
            selector = SelectFromModel(
                RandomForestRegressor(n_estimators=100, random_state=42),
                threshold='median'
            )
        elif method == 'lasso':
            # Lasso regularization
            from sklearn.linear_model import LassoCV
            selector = SelectFromModel(
                LassoCV(cv=5, random_state=42),
                threshold='median'
            )
            
        selector.fit(df, target)
        selected_features = df.columns[selector.get_support()]
        self.feature_selectors[method] = selector
        
        return df[selected_features], selected_features.tolist()
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between variables"""
        interactions = pd.DataFrame(index=df.index)
        
        # Create polynomial features
        for col in df.columns:
            interactions[f'{col}_squared'] = df[col] ** 2
            interactions[f'{col}_cubed'] = df[col] ** 3
            
        # Create interaction terms
        for i, col1 in enumerate(df.columns):
            for col2 in df.columns[i+1:]:
                interactions[f'{col1}_{col2}_interaction'] = df[col1] * df[col2]
                
        return pd.concat([df, interactions], axis=1)
    
    def create_cluster_features(self, df: pd.DataFrame, 
                              n_clusters: int = 5) -> pd.DataFrame:
        """Create cluster-based features"""
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(df)
        
        # Create cluster features
        cluster_features = pd.get_dummies(clusters, prefix='cluster')
        cluster_features.index = df.index
        
        return pd.concat([df, cluster_features], axis=1)
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        time_features = pd.DataFrame(index=df.index)
        
        # Time-based features
        time_features['hour'] = df.index.hour
        time_features['day_of_week'] = df.index.dayofweek
        time_features['day_of_month'] = df.index.day
        time_features['month'] = df.index.month
        time_features['quarter'] = df.index.quarter
        
        # Cyclical encoding
        time_features['hour_sin'] = np.sin(2 * np.pi * time_features['hour'] / 24)
        time_features['hour_cos'] = np.cos(2 * np.pi * time_features['hour'] / 24)
        time_features['day_sin'] = np.sin(2 * np.pi * time_features['day_of_week'] / 7)
        time_features['day_cos'] = np.cos(2 * np.pi * time_features['day_of_week'] / 7)
        
        return pd.concat([df, time_features], axis=1)
    
    def create_rolling_features(self, df: pd.DataFrame, 
                              windows: List[int] = [5, 10, 20, 50]) -> pd.DataFrame:
        """Create rolling window features"""
        rolling_features = pd.DataFrame(index=df.index)
        
        for window in windows:
            for col in df.columns:
                # Rolling statistics
                rolling_features[f'{col}_rolling_mean_{window}'] = df[col].rolling(window).mean()
                rolling_features[f'{col}_rolling_std_{window}'] = df[col].rolling(window).std()
                rolling_features[f'{col}_rolling_min_{window}'] = df[col].rolling(window).min()
                rolling_features[f'{col}_rolling_max_{window}'] = df[col].rolling(window).max()
                
                # Rolling changes
                rolling_features[f'{col}_rolling_change_{window}'] = df[col].pct_change(window)
                
        return pd.concat([df, rolling_features], axis=1)
    
    def create_lag_features(self, df: pd.DataFrame, 
                          lags: List[int] = [1, 2, 3, 5, 10]) -> pd.DataFrame:
        """Create lagged features"""
        lag_features = pd.DataFrame(index=df.index)
        
        for lag in lags:
            for col in df.columns:
                lag_features[f'{col}_lag_{lag}'] = df[col].shift(lag)
                
        return pd.concat([df, lag_features], axis=1) 