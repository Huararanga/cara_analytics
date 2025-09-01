import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from itertools import groupby
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

import lib.hmm as hmm

# LOST DONORS REGRESSION
IS_LOST_THRESHOLD = 365 * 5; #below 4 makes results much worse
IS_LOST_REFERENCE_DATE = pd.Timestamp('2024-01-01')
STARTED_ZA = pd.Timestamp('2024-01-01')

# Lost donors prediction
def is_ZA(datetime):
    return datetime >= STARTED_ZA

def sanitize_for_xgboost(df):
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(df.median())
    
    # Optional sanity check
    if (df.abs() > 1e100).any().any():
        raise ValueError("Input contains extremely large values.")
    
    return df

def extract_person_features(person_data, modelZA, modelAA2):
    last_row = person_data.iloc[-1]
    
    model = modelZA if is_ZA(last_row["datetime"]) else modelAA2
    seq = person_data['obs'].values.reshape(-1, 1)
    hidden_states = model.predict(seq)
    state_counts = np.bincount(hidden_states, minlength=modelZA.n_components)
    state_durations = [len(list(g)) for _, g in groupby(hidden_states)]
    
    features = {
        'next_state': hidden_states[-1],
        'most_common_state': np.argmax(state_counts),
        'num_transitions': np.sum(np.diff(hidden_states) != 0),
        'mean_state_duration': np.mean(state_durations),
        'log_likelihood': model.score(seq),
        'seq_length': len(seq),
        
        # 'freq_last_month': last_row['freq_last_month'],
        'is_canceled': last_row.get('is_canceled', False),
        'age_years': last_row.get('age_years', np.nan),
        'person_sex': last_row.get('person_sex', 'unknown'),
        # 'id_clinic': last_row.get('id_clinic', 'unknown'),
        'clinic_name': last_row.get('clinic_name', 'unknown'),
        'payed_to_donor': last_row.get('payed_to_donor', 0),
        'proffesion_name': last_row.get('proffesion_name', 'unknown'),
        # 'previous_draw_days': (last_row["previous_draw_date"] - last_row["datetime"]).days if pd.notnull(last_row["previous_draw_date"]) else np.nan

        # face features
        # 'emotion_happy': last_row.get('emotion_happy', np.nan),
        # 'emotion_angry': last_row.get('emotion_angry', np.nan),
        # 'emotion_disgust': last_row.get('emotion_disgust', np.nan),
        # 'emotion_fear': last_row.get('emotion_fear', np.nan),
        # 'emotion_sad': last_row.get('emotion_sad', np.nan),
        # 'emotion_surprise': last_row.get('emotion_surprise', np.nan),
        'looks_older': last_row.get('looks_older', np.nan),
        'dominant_emotion': last_row.get('dominant_emotion', np.nan),
    }
    
    return features



def build_feature_matrix(data, ZA, AA2):
    # Prepare the feature matrix and labels using earlier steps
    model_ZA = ZA["model"]
    model_AA2 = AA2["model"]
    
    def encode_obs(row):
        if is_ZA(row["datetime"]):
            return int(hmm.ZA_encoding.get(row["draw_character"], -1))
        else:
            return int(hmm.AA2_encoding.get(row["deprecated_draw_character"], -1))

    data["obs"] = data.apply(encode_obs, axis=1)
    grouped = data.groupby('id_person')
    
    X = []
    y = []

    for person_id, group in grouped:
        if len(group) < 2:
            continue  # too short
        features = extract_person_features(group, model_ZA, model_AA2)

        # Define "lost donor" — e.g., no activity after a certain date or long inactivity
        last_date = group['datetime'].max()
        is_lost = (IS_LOST_REFERENCE_DATE - last_date).days > IS_LOST_THRESHOLD  # example threshold
        features['is_lost'] = int(is_lost)

        X.append(features)
        y.append(int(is_lost))

    df = pd.DataFrame(X)
    return df.drop(columns='is_lost'), df['is_lost']

def preprocess_features(X):
    X = X.copy()
    encoders = {}
    
    for col in X.select_dtypes(include='object').columns:
        X[col].fillna('unknown', inplace=True)
        # Use OrdinalEncoder instead of LabelEncoder to handle unseen values
        oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X[col] = oe.fit_transform(X[col].values.reshape(-1, 1)).flatten()
        encoders[col] = oe

    for col in X.select_dtypes(include='number').columns:
        X[col].fillna(X[col].median(), inplace=True)
    
    return X, encoders


def train_xgboost_model(X, y, test_size=0.2, random_state=42):
    """
    Train XGBoost model for lost donor prediction
    
    Args:
        X: Feature matrix
        y: Target variable
        test_size: Proportion of test set
        random_state: Random seed
        
    Returns:
        Dictionary containing model, test data, and predictions
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Sanitize data for XGBoost
    X_train = sanitize_for_xgboost(X_train)
    X_test = sanitize_for_xgboost(X_test)

    # XGBoost classifier
    clf = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=random_state,
        n_jobs=-1,
    )

    # Train model
    clf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = clf.predict(X_test)
    
    return {
        'model': clf,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred,
        'feature_names': X.columns
    }

def plot_feature_importance(model_results, figsize=(10, 6)):
    """
    Plot feature importance from trained XGBoost model
    
    Args:
        model_results: Dictionary from train_xgboost_model
        figsize: Figure size tuple
    """
    clf = model_results['model']
    feat_names = model_results['feature_names']
    
    importances = clf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]  # Descending order
    sorted_importances = importances[sorted_idx]
    sorted_feat_names = np.array(feat_names)[sorted_idx]

    # Plot
    plt.figure(figsize=figsize)
    plt.barh(sorted_feat_names, sorted_importances)
    plt.xlabel("Feature Importance")
    plt.title("XGBoost - Lost Donor Prediction")
    plt.gca().invert_yaxis()  # Highest importance at top
    plt.tight_layout()
    plt.show()

def predict_lost_by_person_id(
    person_id: int,
    history: pd.DataFrame,
    loaded_results: dict,
    clf: XGBClassifier,
    encoders: dict
):
    person_data = history[history["id_person"] == person_id].copy()

    last_datetime = person_data["datetime"].max()
    is_za = is_ZA(last_datetime)
    model = loaded_results["ZA"]["model"] if is_za else loaded_results["AA2"]["model"]
    encoding = hmm.ZA_encoding if is_za else hmm.AA2_encoding

    person_data["obs"] = person_data["draw_character"].map(encoding).fillna(-1).astype(int)

    features = extract_person_features(person_data, loaded_results["ZA"]["model"], loaded_results["AA2"]["model"])
    X_person = pd.DataFrame([features])

    # Reuse encoders - Updated for OrdinalEncoder
    for col in X_person.select_dtypes(include='object').columns:
        X_person[col].fillna('unknown', inplace=True)
        if col in encoders:
            oe = encoders[col]  # This is now an OrdinalEncoder
            # Use OrdinalEncoder transform method
            X_person[col] = oe.transform(X_person[col].values.reshape(-1, 1)).flatten()
        else:
            X_person[col] = -1

    for col in X_person.select_dtypes(include='number').columns:
        X_person[col].fillna(X_person[col].median(), inplace=True)

    pred = clf.predict(X_person)[0]
    proba = clf.predict_proba(X_person)[0][1]

    return {
        "person_id": person_id,
        "is_lost": bool(pred),
        "lost_probability": proba,
        "model_used": "ZA" if is_za else "AA2",
        "last_draw": last_datetime
    }

def plot_lost_donors(lost_predictions_df, figsize=(10, 6)):
    viz_df = lost_predictions_df.copy()
    # Set index to datetime
    viz_df = viz_df.set_index("last_draw_date")
    # lost_predictions_df = lost_predictions_df[lost_predictions_df.index > last_year]
    # Resample by week (W = weekly ending on Sunday)
    resampled_lost_counts = viz_df.resample("M")["probability_of_lost"].median()

    # Plot
    plt.figure(figsize=(12, 6))
    resampled_lost_counts.plot()
    plt.title("Predicted Lost Donors by Week")
    plt.xlabel("Week")
    plt.ylabel("Number of Lost Donors")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_profession_loss_probability(lost_predictions_df, min_count=25, figsize=(10, 18)):
    """
    Plot mean probability of loss by profession for common professions
    
    Args:
        lost_predictions_df: DataFrame with predictions and profession data
        min_count: Minimum count threshold for including a profession
        figsize: Figure size tuple
    """
    # Filter out rare professions
    common_profs = (
        lost_predictions_df['proffesion_name']
        .value_counts()
        .loc[lambda x: x >= min_count]
        .index
    )

    filtered_df = lost_predictions_df[lost_predictions_df['proffesion_name'].isin(common_profs)]

    # Compute mean probability
    df_mean = (
        filtered_df
        .groupby('proffesion_name', as_index=False)['probability_of_lost']
        .mean()
    )

    # Sort by mean probability, highest first
    df_sorted = df_mean.sort_values('probability_of_lost', ascending=False)

    # Plot
    plt.figure(figsize=figsize)
    sns.barplot(
        data=df_sorted,
        x='probability_of_lost',
        y='proffesion_name',
        palette='viridis'
    )

    plt.xlabel('Mean Probability of Lost')
    plt.ylabel('Profession')
    plt.title(f'Mean Probability of Loss by Profession (Min {min_count} people)')
    plt.show()
    
    return df_sorted