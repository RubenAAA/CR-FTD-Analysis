import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, roc_auc_score, roc_curve, 
                           precision_recall_curve, confusion_matrix, 
                           precision_score, recall_score, f1_score)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.utils import resample
import warnings
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from scipy.stats import chi2_contingency


warnings.filterwarnings('ignore')

# Focused FTD Feature Engineering
def create_ftd_features(df_ga4, df_postback):
    """
    Create features as specified in requirements:
    - Контентные: Тип первой страницы входа, последовательность N просмотренных типов страниц, количество просмотров бонусных страниц
    - Временные: Время на разных типах контента, общее время сессии, время суток, день недели (будний/выходной)  
    - Технические: Тип устройства (desktop/mobile), ОС, город
    - Поведенческие: Количество кликов, скроллов, взаимодействий с конкретными элементами (кнопки, баннеры, поиск)
    """
    
    print("🎯 Creating FTD prediction features...")
    
    # Group by user to get session-level features
    user_sessions = df_ga4.sort_values(['user_pseudo_id', 'event_timestamp']).groupby('user_pseudo_id')
    
    all_features = []
    
    for user_id, user_data in user_sessions:
        features = {'user_pseudo_id': user_id}
        
        # =================================================================
        # 1. КОНТЕНТНЫЕ ПРИЗНАКИ
        # =================================================================
        
        # Тип первой страницы входа (Type of first entry page)
        page_sequence = user_data['section_category'].tolist()
        features['first_page_type'] = page_sequence[0] if page_sequence else 'unknown'
        
        # Последовательность N просмотренных типов страниц (Sequence of N viewed page types)
        # Create sequence features for first 5 pages
        for i in range(5):
            if i < len(page_sequence):
                features[f'page_sequence_{i+1}'] = page_sequence[i]
            else:
                features[f'page_sequence_{i+1}'] = 'none'
        
        # Количество просмотров бонусных страниц (Number of bonus page views)
        features['bonus_page_views'] = sum(1 for page in page_sequence if page == 'bonuses')
        
        # Additional content counts
        features['bk_rating_page_views'] = sum(1 for page in page_sequence if page == 'bk_rating')
        features['predictions_page_views'] = sum(1 for page in page_sequence if page == 'predictions')
        features['news_page_views'] = sum(1 for page in page_sequence if page == 'news')
        features['total_page_views'] = len(page_sequence)
        
        # =================================================================
        # 2. ВРЕМЕННЫЕ ПРИЗНАКИ
        # =================================================================
        
        # Convert timestamps
        user_data_copy = user_data.copy()
        user_data_copy['event_datetime'] = pd.to_datetime(user_data_copy['event_timestamp'], unit='us')
        
        # Общее время сессии (Total session time)
        session_start = user_data_copy['event_datetime'].min()
        session_end = user_data_copy['event_datetime'].max()
        features['total_session_time_minutes'] = (session_end - session_start).total_seconds() / 60
        
        # Время суток (Time of day)
        features['session_hour'] = session_start.hour
        features['is_morning'] = 1 if 6 <= session_start.hour < 12 else 0
        features['is_afternoon'] = 1 if 12 <= session_start.hour < 18 else 0  
        features['is_evening'] = 1 if 18 <= session_start.hour < 24 else 0
        features['is_night'] = 1 if 0 <= session_start.hour < 6 else 0
        
        # День недели (будний/выходной) (Day of week - weekday/weekend)
        features['day_of_week'] = session_start.weekday()  # 0=Monday, 6=Sunday
        features['is_weekend'] = 1 if session_start.weekday() >= 5 else 0
        features['is_weekday'] = 1 - features['is_weekend']
        
        # Время на разных типах контента (Time on different content types)
        content_time = user_data_copy.groupby('section_category')['event_params_engagement_time_msec'].sum().fillna(0) / 1000  # Convert to seconds
        
        features['time_on_bonuses_seconds'] = content_time.get('bonuses', 0)
        features['time_on_bk_rating_seconds'] = content_time.get('bk_rating', 0)
        features['time_on_predictions_seconds'] = content_time.get('predictions', 0)
        features['time_on_news_seconds'] = content_time.get('news', 0)
        features['time_on_blog_seconds'] = content_time.get('blog', 0)
        features['time_on_home_seconds'] = content_time.get('home', 0)
        
        # Total engagement time
        features['total_engagement_time_seconds'] = user_data_copy['event_params_engagement_time_msec'].fillna(0).sum() / 1000
        
        # =================================================================
        # 3. ТЕХНИЧЕСКИЕ ПРИЗНАКИ
        # =================================================================
        
        # Take first row for session-level technical info
        first_row = user_data_copy.iloc[0]
        
        # Тип устройства (desktop/mobile) (Device type)
        device_category = first_row.get('device_category', 'unknown')
        features['device_type'] = device_category
        features['is_mobile'] = 1 if device_category == 'mobile' else 0
        features['is_desktop'] = 1 if device_category == 'desktop' else 0
        features['is_tablet'] = 1 if device_category == 'tablet' else 0
        
        # ОС (Operating System)
        features['operating_system'] = first_row.get('device_operating_system', 'unknown')
        
        # Город (City)
        features['city'] = first_row.get('geo_city', 'unknown')
        features['country'] = first_row.get('geo_country', 'unknown')
        
        # =================================================================
        # 4. ПОВЕДЕНЧЕСКИЕ ПРИЗНАКИ
        # =================================================================
        
        # Event counts
        event_counts = user_data_copy['event_name'].value_counts()
        
        # Количество кликов (Number of clicks)
        features['click_count'] = event_counts.get('click_on_bk_web', 0)
        
        # Скроллов (Scrolls) - if scroll events exist
        features['scroll_count'] = event_counts.get('scroll', 0)
        
        # Page views
        features['page_view_count'] = event_counts.get('page_view', 0)
        
        # Взаимодействий с конкретными элементами (Interactions with specific elements)
        
        # Search interactions
        features['search_interactions'] = event_counts.get('search', 0)
        
        # Button/Banner interactions (looking at event parameters)
        features['total_interactions'] = len(user_data_copy)
        
        # Look for specific element interactions in event_params_name_bk
        if 'event_params_name_bk' in user_data_copy.columns:
            element_interactions = user_data_copy['event_params_name_bk'].fillna('').str.lower()
            
            # Button interactions
            features['registration_button_interactions'] = sum(1 for name in element_interactions if 'regist' in str(name))
            features['deposit_button_interactions'] = sum(1 for name in element_interactions if any(word in str(name) for word in ['deposit', 'пополнить', 'депозит']))
            features['bonus_button_interactions'] = sum(1 for name in element_interactions if 'bonus' in str(name))
            features['login_button_interactions'] = sum(1 for name in element_interactions if any(word in str(name) for word in ['login', 'войти', 'вход']))
            
            # Banner interactions
            features['banner_interactions'] = sum(1 for name in element_interactions if any(word in str(name) for word in ['banner', 'баннер', 'promo']))
            
        else:
            # If no specific element data, set to 0
            features['registration_button_interactions'] = 0
            features['deposit_button_interactions'] = 0
            features['bonus_button_interactions'] = 0
            features['login_button_interactions'] = 0
            features['banner_interactions'] = 0
        
        # Behavioral ratios
        features['click_rate'] = features['click_count'] / max(features['page_view_count'], 1)
        features['interaction_rate'] = features['total_interactions'] / max(features['page_view_count'], 1)
        
        all_features.append(features)
    
    # Convert to DataFrame
    features_df = pd.DataFrame(all_features)
    
    # =================================================================
    # 5. ADD TARGET VARIABLE
    # =================================================================
    
    print("🎯 Adding target variable...")
    
    # Create target from postback data
    target = df_postback.groupby('meta_user_id')['et'].apply(
        lambda x: 1 if 'ftd' in x.values else 0
    ).reset_index()
    target.columns = ['user_pseudo_id', 'target_ftd']
    
    # Merge target
    features_df = features_df.merge(target, on='user_pseudo_id', how='left')
    features_df['target_ftd'] = features_df['target_ftd'].fillna(0)
    
    # =================================================================
    # 6. FINAL DATA PREPARATION
    # =================================================================
    
    # Fill missing values
    numeric_columns = features_df.select_dtypes(include=[np.number]).columns
    features_df[numeric_columns] = features_df[numeric_columns].fillna(0)
    
    categorical_columns = features_df.select_dtypes(include=['object']).columns
    features_df[categorical_columns] = features_df[categorical_columns].fillna('unknown')
    
    print("✅ Feature engineering complete!")
    print(f"   Total features: {len(features_df.columns)-2}")  # -2 for user_id and target
    print(f"   Total samples: {len(features_df)}")
    print(f"   FTD distribution: {features_df['target_ftd'].value_counts().to_dict()}")
    
    # Feature summary
    print("\n📊 FEATURE SUMMARY BY CATEGORY:")
    print("="*50)
    
    content_features = [col for col in features_df.columns if col in ['first_page_type'] + [f'page_sequence_{i}' for i in range(1,6)] + ['bonus_page_views', 'bk_rating_page_views', 'predictions_page_views', 'news_page_views', 'total_page_views']]
    temporal_features = [col for col in features_df.columns if 'time' in col or 'hour' in col or 'day' in col or 'weekend' in col or 'weekday' in col or 'morning' in col or 'afternoon' in col or 'evening' in col or 'night' in col]
    technical_features = [col for col in features_df.columns if col in ['device_type', 'is_mobile', 'is_desktop', 'is_tablet', 'operating_system', 'city', 'country']]
    behavioral_features = [col for col in features_df.columns if 'click' in col or 'scroll' in col or 'interaction' in col or 'button' in col or 'banner' in col or 'search' in col or 'rate' in col]
    
    print(f"🎯 Content Features ({len(content_features)}): {content_features}")
    print(f"⏰ Temporal Features ({len(temporal_features)}): {temporal_features}")  
    print(f"🔧 Technical Features ({len(technical_features)}): {technical_features}")
    print(f"🎭 Behavioral Features ({len(behavioral_features)}): {behavioral_features}")
    
    return features_df

# Usage
def prepare_model_data(features_df):
    """
    Prepare data for modeling
    """
    
    print("🛠️ Preparing data for modeling...")
    
    # Separate features and target
    X = features_df.drop(['user_pseudo_id', 'target_ftd'], axis=1)
    y = features_df['target_ftd']
    
    print(f"📊 Final dataset shape: {X.shape}")
    print(f"📈 Target distribution: {y.value_counts().to_dict()}")
    print(f"📉 Class imbalance ratio: {y.value_counts()[0] / y.value_counts()[1]:.1f}:1")
    
    return X, y



#################################################################################
# Preprocesing
#################################################################################


class FTDPreprocessor:
    """
    Complete preprocessing pipeline for FTD prediction model
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = None
        self.selected_features = None
        self.feature_scores = {}
        
    def analyze_correlations(self, X, y, threshold=0.95):
        """
        Analyze feature correlations and remove highly correlated features
        """
        
        print("🔗 CORRELATION ANALYSIS")
        print("="*40)
        
        # Calculate correlation matrix for numerical features only
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        X_numeric = X[numerical_cols]
        
        # Correlation with target
        target_corr = X_numeric.corrwith(y).abs().sort_values(ascending=False)
        
        print("📊 Top 15 features correlated with target:")
        print(target_corr.head(15).round(3))
        
        # Feature-to-feature correlations
        corr_matrix = X_numeric.corr().abs()
        
        # Find highly correlated feature pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > threshold:
                    feature1 = corr_matrix.columns[i]
                    feature2 = corr_matrix.columns[j]
                    corr_value = corr_matrix.iloc[i, j]
                    
                    # Keep feature with higher target correlation
                    if target_corr[feature1] >= target_corr[feature2]:
                        remove_feature = feature2
                        keep_feature = feature1
                    else:
                        remove_feature = feature1
                        keep_feature = feature2
                    
                    high_corr_pairs.append({
                        'feature1': feature1,
                        'feature2': feature2, 
                        'correlation': corr_value,
                        'remove': remove_feature,
                        'keep': keep_feature
                    })
        
        # Features to remove due to high correlation
        features_to_remove = list(set([pair['remove'] for pair in high_corr_pairs]))
        
        print(f"\n⚠️ Highly correlated feature pairs (>{threshold}):")
        for pair in high_corr_pairs[:10]:  # Show first 10
            print(f"  {pair['feature1']} ↔ {pair['feature2']}: {pair['correlation']:.3f} → Remove {pair['remove']}")
        
        if len(high_corr_pairs) > 10:
            print(f"  ... and {len(high_corr_pairs)-10} more pairs")
        
        print(f"\n🗑️ Features to remove due to high correlation: {len(features_to_remove)}")
        
        # Create correlation heatmap for top features
        top_features = target_corr.head(20).index.tolist()
        plt.figure(figsize=(12, 10))
        sns.heatmap(X_numeric[top_features].corr(), annot=True, cmap='coolwarm', center=0, 
                   square=True, linewidths=0.5)
        plt.title('Correlation Heatmap - Top 20 Features')
        plt.tight_layout()
        plt.show()
        
        return features_to_remove, target_corr
    
    def mutual_information_selection(self, X, y, k=50):
        """
        Feature selection using mutual information
        """
        
        print("\n🧠 MUTUAL INFORMATION FEATURE SELECTION")
        print("="*40)
        
        # Separate categorical and numerical features
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        print(f"📊 Analyzing {len(categorical_cols)} categorical + {len(numerical_cols)} numerical features")
        
        # Encode categorical features for mutual information calculation
        X_encoded = X.copy()
        temp_encoders = {}
        
        for col in categorical_cols:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
            temp_encoders[col] = le
        
        # Calculate mutual information scores
        mi_scores = mutual_info_classif(X_encoded, y, random_state=42)
        mi_scores = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
        
        print("🏆 Top 20 features by Mutual Information:")
        print(mi_scores.head(20).round(4))
        
        # Select top k features
        selected_features_mi = mi_scores.head(k).index.tolist()
        
        print(f"\n✅ Selected {len(selected_features_mi)} features using Mutual Information")
        
        # Plot MI scores
        plt.figure(figsize=(12, 8))
        mi_scores.head(30).plot(kind='barh')
        plt.title('Top 30 Features by Mutual Information Score')
        plt.xlabel('Mutual Information Score')
        plt.tight_layout()
        plt.show()
        
        return selected_features_mi, mi_scores
    
    def categorical_encoding(self, X_train, X_test=None):
        """
        Encode categorical variables with multiple strategies
        """
        
        print("\n🏷️ CATEGORICAL ENCODING")
        print("="*40)
        
        categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
        print(f"📋 Encoding {len(categorical_cols)} categorical features:")
        
        X_train_encoded = X_train.copy()
        X_test_encoded = X_test.copy() if X_test is not None else None
        
        for col in categorical_cols:
            print(f"  • {col}: {X_train[col].nunique()} unique values")
            
            # Strategy based on number of unique values
            n_unique = X_train[col].nunique()
            
            if n_unique <= 50:  # Label encoding for low cardinality
                le = LabelEncoder()
                X_train_encoded[col] = le.fit_transform(X_train[col].astype(str))
                self.label_encoders[col] = le
                
                if X_test_encoded is not None:
                    # Handle unseen categories in test set
                    test_values = X_test[col].astype(str)
                    known_values = set(le.classes_)
                    test_values_mapped = test_values.map(lambda x: x if x in known_values else 'unknown')
                    
                    # Add 'unknown' to encoder if needed
                    if 'unknown' not in known_values:
                        le.classes_ = np.append(le.classes_, 'unknown')
                    
                    X_test_encoded[col] = le.transform(test_values_mapped)
                
                print(f"    → Label Encoded (0-{n_unique-1})")
                
            else:  # High cardinality - use frequency encoding
                freq_encoding = X_train[col].value_counts(normalize=True).to_dict()
                X_train_encoded[col] = X_train[col].map(freq_encoding)
                
                if X_test_encoded is not None:
                    X_test_encoded[col] = X_test[col].map(freq_encoding).fillna(0)  # Unknown categories get 0
                
                print("    → Frequency Encoded (high cardinality)")
        
        print("\n✅ Categorical encoding complete!")
        
        if X_test_encoded is not None:
            return X_train_encoded, X_test_encoded
        else:
            return X_train_encoded
    
    def scale_features(self, X_train, X_test=None, method='standard'):
        """
        Scale numerical features
        """
        
        print(f"\n📏 FEATURE SCALING ({method.upper()})")
        print("="*40)
        
        numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        
        print(f"📊 Scaling {len(numerical_cols)} numerical features")
        
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy() if X_test is not None else None
        
        # Choose scaler
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError("Method must be 'standard' or 'robust'")
        
        # Fit and transform training data
        X_train_scaled[numerical_cols] = self.scaler.fit_transform(X_train[numerical_cols])
        
        # Transform test data if provided
        if X_test_scaled is not None:
            X_test_scaled[numerical_cols] = self.scaler.transform(X_test[numerical_cols])
        
        # Show scaling statistics
        print("📈 Scaling statistics:")
        if method == 'standard':
            print(f"  Mean: {X_train_scaled[numerical_cols].mean().mean():.3f}")
            print(f"  Std:  {X_train_scaled[numerical_cols].std().mean():.3f}")
        
        print("✅ Feature scaling complete!")
        
        if X_test_scaled is not None:
            return X_train_scaled, X_test_scaled
        else:
            return X_train_scaled
    
    def complete_preprocessing(self, X, y, test_size=0.2, correlation_threshold=0.95, 
                             n_features=50, encoding_strategy='auto', scaling_method='standard'):
        """
        Complete preprocessing pipeline
        """
        
        print("🚀 COMPLETE PREPROCESSING PIPELINE")
        print("="*50)
        
        # 1. Train-test split first
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"📊 Data split: Train {X_train.shape[0]}, Test {X_test.shape[0]}")
        
        # 2. Correlation analysis and removal
        features_to_remove_corr, target_corr = self.analyze_correlations(X_train, y_train, correlation_threshold)
        
        # Remove highly correlated features
        X_train = X_train.drop(columns=features_to_remove_corr, errors='ignore')
        X_test = X_test.drop(columns=features_to_remove_corr, errors='ignore')
        
        print(f"📉 Removed {len(features_to_remove_corr)} highly correlated features")
        print(f"📊 Remaining features: {X_train.shape[1]}")
        
        # 3. Mutual information feature selection
        selected_features_mi, mi_scores = self.mutual_information_selection(X_train, y_train, n_features)
        
        # Keep only selected features
        X_train = X_train[selected_features_mi]
        X_test = X_test[selected_features_mi]
        self.selected_features = selected_features_mi
        
        print(f"🎯 Selected {len(selected_features_mi)} features using MI")
        
        # 4. Categorical encoding
        X_train, X_test = self.categorical_encoding(X_train, X_test)
        
        # 5. Feature scaling
        X_train, X_test = self.scale_features(X_train, X_test, scaling_method)
        
        # Store feature importance scores
        self.feature_scores = {
            'mutual_information': mi_scores[selected_features_mi],
            'target_correlation': target_corr[target_corr.index.isin(selected_features_mi)]
        }
        
        print("\n🎉 PREPROCESSING COMPLETE!")
        print(f"   Final shape: {X_train.shape}")
        print(f"   Selected features: {len(self.selected_features)}")
        print(f"   Categorical encoders: {len(self.label_encoders)}")
        print(f"   Scaler fitted: {type(self.scaler).__name__}")
        
        return X_train, X_test, y_train, y_test
    
    def get_feature_importance_summary(self):
        """
        Get summary of feature importance from selection process
        """
        
        if not self.feature_scores:
            print("❌ No feature scores available. Run preprocessing first.")
            return None
        
        print("\n📊 FEATURE IMPORTANCE SUMMARY")
        print("="*40)
        
        # Combine scores
        mi_scores = self.feature_scores['mutual_information']
        corr_scores = self.feature_scores['target_correlation']
        
        # Create summary dataframe
        summary = pd.DataFrame({
            'feature': mi_scores.index,
            'mutual_information': mi_scores.values,
            'target_correlation': [corr_scores.get(feat, 0) for feat in mi_scores.index]
        })
        
        # Add rankings
        summary['mi_rank'] = summary['mutual_information'].rank(ascending=False)
        summary['corr_rank'] = summary['target_correlation'].abs().rank(ascending=False)
        summary['avg_rank'] = (summary['mi_rank'] + summary['corr_rank']) / 2
        
        summary = summary.sort_values('avg_rank')
        
        print("🏆 Top 20 Selected Features:")
        print(summary.head(20)[['feature', 'mutual_information', 'target_correlation', 'avg_rank']].round(4))
        
        return summary
    
    def transform_new_data(self, X_new):
        """
        Transform new data using fitted preprocessors
        """
        
        if not self.selected_features:
            raise ValueError("Preprocessor not fitted. Run complete_preprocessing first.")
        
        print("🔄 Transforming new data...")
        
        # Select features
        X_new = X_new[self.selected_features]
        
        # Encode categorical features
        categorical_cols = X_new.select_dtypes(include=['object']).columns.tolist()
        for col in categorical_cols:
            if col in self.label_encoders:
                le = self.label_encoders[col]
                # Handle unseen categories
                known_values = set(le.classes_)
                new_values = X_new[col].astype(str)
                new_values_mapped = new_values.map(lambda x: x if x in known_values else 'unknown')
                
                if 'unknown' not in known_values:
                    le.classes_ = np.append(le.classes_, 'unknown')
                
                X_new[col] = le.transform(new_values_mapped)
        
        # Scale numerical features
        numerical_cols = X_new.select_dtypes(include=[np.number]).columns.tolist()
        if self.scaler and numerical_cols:
            X_new[numerical_cols] = self.scaler.transform(X_new[numerical_cols])
        
        print(f"✅ New data transformed: {X_new.shape}")
        
        return X_new

# Usage example
def run_preprocessing_pipeline(features_df):
    """
    Main function to run complete preprocessing pipeline
    """
    
    # Prepare data
    X = features_df.drop(['user_pseudo_id', 'target_ftd'], axis=1)
    y = features_df['target_ftd']
    
    # Initialize preprocessor
    preprocessor = FTDPreprocessor()
    
    # Run complete preprocessing
    X_train, X_test, y_train, y_test = preprocessor.complete_preprocessing(
        X, y,
        test_size=0.2,
        correlation_threshold=0.9,  # Remove features with >90% correlation
        n_features=30,              # Select top 30 features
        scaling_method='standard'   # StandardScaler
    )
    
    # Get feature importance summary
    feature_summary = preprocessor.get_feature_importance_summary()
    
    return X_train, X_test, y_train, y_test, preprocessor, feature_summary

############################################################################################################
# Classifier
############################################################################################################
# Simple FTD Classification Models

class SimpleFTDClassifier:
    """
    Simple classification pipeline for FTD prediction using only sklearn
    """
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.feature_importance = {}
        
    def simple_undersample(self, X_train, y_train):
        """
        Simple undersampling using sklearn's resample
        """
        
        print("⚖️ SIMPLE UNDERSAMPLING")
        print("="*30)
        
        # Separate classes
        df_majority = X_train[y_train == 0]
        df_minority = X_train[y_train == 1]
        
        print(f"📊 Original: Majority={len(df_majority)}, Minority={len(df_minority)}")
        
        # Undersample majority class
        df_majority_downsampled = resample(df_majority, 
                                         replace=False,    
                                         n_samples=len(df_minority)*2,  # 2:1 ratio
                                         random_state=42)
        
        # Combine minority class with downsampled majority class
        X_balanced = pd.concat([df_majority_downsampled, df_minority])
        y_balanced = pd.concat([pd.Series([0]*len(df_majority_downsampled)), 
                               pd.Series([1]*len(df_minority))])
        
        print(f"📈 Balanced: Total={len(X_balanced)}, Ratio={len(df_majority_downsampled)}:{len(df_minority)}")
        
        return X_balanced, y_balanced
    
    def train_logistic_regression(self, X_train, y_train, balance_method='class_weight'):
        """
        Train Logistic Regression model
        """
        
        print("\n📈 TRAINING LOGISTIC REGRESSION")
        print("="*40)
        
        if balance_method == 'class_weight':
            # Use built-in class weights (RECOMMENDED)
            model = LogisticRegression(
                random_state=42,
                class_weight='balanced',
                max_iter=1000,
                solver='liblinear'
            )
            X_train_model, y_train_model = X_train, y_train
            print("📊 Using balanced class weights")
            
        elif balance_method == 'undersample':
            # Manual undersampling
            X_train_model, y_train_model = self.simple_undersample(X_train, y_train)
            model = LogisticRegression(
                random_state=42,
                max_iter=1000,
                solver='liblinear'
            )
            
        else:
            # No balancing
            model = LogisticRegression(
                random_state=42,
                max_iter=1000,
                solver='liblinear'
            )
            X_train_model, y_train_model = X_train, y_train
            print("📊 No class balancing")
        
        # Train model
        model.fit(X_train_model, y_train_model)
        self.models['logistic_regression'] = model
        
        # Feature importance (coefficients)
        if hasattr(X_train, 'columns'):
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'coefficient': model.coef_[0],
                'abs_coefficient': np.abs(model.coef_[0])
            }).sort_values('abs_coefficient', ascending=False)
            
            self.feature_importance['logistic_regression'] = feature_importance
            
            print("🏆 Top 10 most important features (by coefficient):")
            for _, row in feature_importance.head(10).iterrows():
                direction = "↑" if row['coefficient'] > 0 else "↓"
                print(f"  {direction} {row['feature']}: {row['coefficient']:.4f}")
        
        print("✅ Logistic Regression trained!")
        return model
    
    def train_random_forest(self, X_train, y_train, balance_method='class_weight'):
        """
        Train Random Forest model
        """
        
        print("\n🌲 TRAINING RANDOM FOREST")
        print("="*40)
        
        if balance_method == 'class_weight':
            # Use built-in class weights (RECOMMENDED)
            model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced',
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                n_jobs=-1
            )
            X_train_model, y_train_model = X_train, y_train
            print("📊 Using balanced class weights")
            
        elif balance_method == 'undersample':
            # Manual undersampling
            X_train_model, y_train_model = self.simple_undersample(X_train, y_train)
            model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                n_jobs=-1
            )
            
        else:
            # No balancing
            model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                n_jobs=-1
            )
            X_train_model, y_train_model = X_train, y_train
            print("📊 No class balancing")
        
        # Train model
        model.fit(X_train_model, y_train_model)
        self.models['random_forest'] = model
        
        # Feature importance
        if hasattr(X_train, 'columns'):
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            self.feature_importance['random_forest'] = feature_importance
            
            print("🏆 Top 10 most important features:")
            for _, row in feature_importance.head(10).iterrows():
                print(f"  • {row['feature']}: {row['importance']:.4f}")
        
        print("✅ Random Forest trained!")
        return model
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """
        Comprehensive model evaluation
        """
        
        print("\n📊 EVALUATING {model_name.upper()}")
        print("="*50)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate all required metrics
        auc_roc = roc_auc_score(y_test, y_pred_proba)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Store results
        results = {
            'model_name': model_name,
            'auc_roc': auc_roc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'y_true': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        self.results[model_name] = results
        
        # Print metrics with interpretation
        print("🎯 PERFORMANCE METRICS (Class Imbalance Considered):")
        print(f"   AUC-ROC:   {auc_roc:.4f}  {'🟢 Excellent' if auc_roc > 0.9 else '🟡 Good' if auc_roc > 0.8 else '🟠 Fair' if auc_roc > 0.7 else '🔴 Poor'}")
        print(f"   Precision: {precision:.4f}  (Of predicted FTDs, {precision:.1%} are correct)")
        print(f"   Recall:    {recall:.4f}  (Of actual FTDs, {recall:.1%} are caught)")
        print(f"   F1-Score:  {f1:.4f}  (Harmonic mean of precision & recall)")
        
        # Business interpretation
        cm = confusion_matrix(y_test, y_pred)
        total_actual_ftd = cm[1,0] + cm[1,1]  # True FTDs
        total_predicted_ftd = cm[0,1] + cm[1,1]  # Predicted FTDs
        
        if total_actual_ftd > 0 and total_predicted_ftd > 0:
            print("\n💼 BUSINESS INTERPRETATION:")
            print(f"   • Model catches {recall:.1%} of users who will make FTD")
            print(f"   • {precision:.1%} of users flagged as 'likely FTD' actually convert")
            print(f"   • Missing {(1-recall)*total_actual_ftd:.0f} potential FTD users")
            print(f"   • {(1-precision)*total_predicted_ftd:.0f} false alarms")
        
        # Detailed classification report
        print("\n📋 CLASSIFICATION REPORT:")
        print(classification_report(y_test, y_pred, target_names=['No FTD', 'FTD']))
        
        # Confusion Matrix
        print("\n🔍 CONFUSION MATRIX:")
        print("                 Predicted")
        print("               No FTD  FTD   Total")
        print(f"Actual No FTD    {cm[0,0]:4d}  {cm[0,1]:3d}   {cm[0,0]+cm[0,1]:5d}")
        print(f"       FTD       {cm[1,0]:4d}  {cm[1,1]:3d}   {cm[1,0]+cm[1,1]:5d}")
        print(f"Total           {cm[0,0]+cm[1,0]:4d}  {cm[0,1]+cm[1,1]:3d}   {cm.sum():5d}")
        
        return results
    
    def plot_comprehensive_results(self):
        """
        Create comprehensive plots for all models
        """
        
        if not self.results:
            print("❌ No results to plot")
            return
        
        n_models = len(self.results)
        fig, axes = plt.subplots(2, n_models + 1, figsize=(5*(n_models+1), 10))
        
        if n_models == 1:
            axes = axes.reshape(2, -1)
        
        model_colors = ['blue', 'green', 'red', 'purple']
        
        # Plot for each model
        for i, (model_name, results) in enumerate(self.results.items()):
            y_true = results['y_true']
            y_pred_proba = results['y_pred_proba']
            color = model_colors[i % len(model_colors)]
            
            # ROC Curve
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            axes[0, i].plot(fpr, tpr, color=color, linewidth=2, 
                           label=f'{model_name.replace("_", " ").title()} (AUC={results["auc_roc"]:.3f})')
            axes[0, i].plot([0, 1], [0, 1], 'k--', linewidth=1)
            axes[0, i].set_xlabel('False Positive Rate')
            axes[0, i].set_ylabel('True Positive Rate')
            axes[0, i].set_title(f'ROC Curve - {model_name.replace("_", " ").title()}')
            axes[0, i].legend()
            axes[0, i].grid(True, alpha=0.3)
            
            # Precision-Recall Curve
            precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_proba)
            axes[1, i].plot(recall_curve, precision_curve, color=color, linewidth=2,
                           label=f'F1={results["f1_score"]:.3f}')
            axes[1, i].set_xlabel('Recall')
            axes[1, i].set_ylabel('Precision')
            axes[1, i].set_title(f'Precision-Recall - {model_name.replace("_", " ").title()}')
            axes[1, i].legend()
            axes[1, i].grid(True, alpha=0.3)
        
        # Comparison plots in last column
        # ROC comparison
        for i, (model_name, results) in enumerate(self.results.items()):
            y_true = results['y_true']
            y_pred_proba = results['y_pred_proba']
            color = model_colors[i % len(model_colors)]
            
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            axes[0, -1].plot(fpr, tpr, color=color, linewidth=2,
                            label=f'{model_name.replace("_", " ").title()} (AUC={results["auc_roc"]:.3f})')
        
        axes[0, -1].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        axes[0, -1].set_xlabel('False Positive Rate')
        axes[0, -1].set_ylabel('True Positive Rate')
        axes[0, -1].set_title('ROC Comparison')
        axes[0, -1].legend()
        axes[0, -1].grid(True, alpha=0.3)
        
        # Metrics comparison
        metrics = ['AUC-ROC', 'Precision', 'Recall', 'F1-Score']
        model_names = [name.replace('_', ' ').title() for name in self.results.keys()]
        
        metric_values = []
        for results in self.results.values():
            metric_values.append([results['auc_roc'], results['precision'], 
                                results['recall'], results['f1_score']])
        
        x = np.arange(len(metrics))
        width = 0.35
        
        for i, (model_name, values) in enumerate(zip(model_names, metric_values)):
            color = model_colors[i % len(model_colors)]
            axes[1, -1].bar(x + i*width/len(model_names), values, width/len(model_names), 
                           label=model_name, color=color, alpha=0.7)
        
        axes[1, -1].set_xlabel('Metrics')
        axes[1, -1].set_ylabel('Score')
        axes[1, -1].set_title('Metrics Comparison')
        axes[1, -1].set_xticks(x)
        axes[1, -1].set_xticklabels(metrics)
        axes[1, -1].legend()
        axes[1, -1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def get_model_comparison(self):
        """
        Get model comparison table
        """
        
        if not self.results:
            print("❌ No results to compare")
            return None
        
        comparison_data = []
        for model_name, results in self.results.items():
            comparison_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'AUC-ROC': f"{results['auc_roc']:.4f}",
                'Precision': f"{results['precision']:.4f}",
                'Recall': f"{results['recall']:.4f}",
                'F1-Score': f"{results['f1_score']:.4f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print("\n🏆 MODEL COMPARISON TABLE")
        print("="*50)
        print(comparison_df.to_string(index=False))
        
        return comparison_df

def train_simple_models(X_train, X_test, y_train, y_test):
    """
    Simple function to train and evaluate models
    """
    
    print("🚀 TRAINING SIMPLE FTD PREDICTION MODELS")
    print("="*50)
    
    # Initialize classifier
    classifier = SimpleFTDClassifier()
    
    # Check class distribution
    print("📊 Training data distribution:")
    print(f"   No FTD: {(y_train == 0).sum():,} ({(y_train == 0).mean():.1%})")
    print(f"   FTD:    {(y_train == 1).sum():,} ({(y_train == 1).mean():.1%})")
    print(f"   Ratio:  {(y_train == 0).sum() / (y_train == 1).sum():.1f}:1")
    
    # Train models with class balancing
    lr_model = classifier.train_logistic_regression(X_train, y_train, balance_method='class_weight')
    rf_model = classifier.train_random_forest(X_train, y_train, balance_method='class_weight')
    
    # Evaluate models
    lr_results = classifier.evaluate_model(lr_model, X_test, y_test, 'logistic_regression')
    rf_results = classifier.evaluate_model(rf_model, X_test, y_test, 'random_forest')
    
    # Plot results
    classifier.plot_comprehensive_results()
    
    # Model comparison
    comparison_df = classifier.get_model_comparison()
    
    return classifier, comparison_df