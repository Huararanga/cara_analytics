
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from hmmlearn.hmm import CategoricalHMM
import joblib
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_face_features(filepath: str = "./face_features.parquet") -> pd.DataFrame:
    """Load face features from parquet file"""
    return pd.read_parquet(filepath)

def join_face_features_to_history(history: pd.DataFrame, face_features: pd.DataFrame) -> pd.DataFrame:
    """
    Join last face_features (with highest id_photo) into history dataframe
    
    Args:
        history: History dataframe
        face_features: Face features dataframe
        
    Returns:
        DataFrame with face features joined to history
    """
    # Filter out rows where face_detected is False
    face_features_filtered = face_features[face_features.face_detected == True]
    
    # Get the last face_features for each person (highest id_photo)
    last_face_features = face_features_filtered.loc[
        face_features_filtered.groupby('id_person')['id_photo'].idxmax()
    ].copy()
    
    # Rename columns as specified
    last_face_features = last_face_features.rename(columns={
        'age': 'age_detected',
        'gender': 'detected_gender'
    })
    
    # Map gender values
    last_face_features['detected_gender'] = last_face_features['detected_gender'].map({
        'male': 'M',
        'female': 'Z'
    })
    
    # Select only the columns you want (excluding id_person, firstname, surname)
    columns_to_join = ['id_photo', 'face_detected', 'age_detected', 'detected_gender', 
                       'dominant_emotion', 'pose_yaw', 'pose_pitch', 'pose_roll',
                       'embedding', 'emotion_angry', 'emotion_disgust', 'emotion_fear',
                       'emotion_happy', 'emotion_sad', 'emotion_surprise', 'emotion_neutral']
    
    last_face_features_selected = last_face_features[columns_to_join]
    
    # Join with history dataframe
    history_with_face_features = history.merge(
        last_face_features_selected, 
        on='id_person', 
        how='left'
    )
    
    return history_with_face_features

def calculate_is_not_fine(history_with_face_features: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate is_not_fine column from emotions
    
    Args:
        history_with_face_features: DataFrame with emotion columns
        
    Returns:
        DataFrame with is_not_fine column added
    """
    # Method 1: Based on negative emotions (angry, disgust, fear, sad)
    history_with_face_features['is_not_fine'] = (
        (history_with_face_features['emotion_angry'] > 0.3) |
        (history_with_face_features['emotion_disgust'] > 0.3) |
        (history_with_face_features['emotion_fear'] > 0.3) |
        (history_with_face_features['emotion_sad'] > 0.3)
    )
    
    return history_with_face_features

def limit_data_to_complete_periods(historyFilt: pd.DataFrame, timeDelta: str, 
                                  min_period_quantile: float = 0.1) -> pd.DataFrame:
    """
    Limit historyFilt by whole timeDelta period to prevent start and end artefacts
    
    Args:
        historyFilt: Filtered history dataframe
        timeDelta: Time period column name
        min_period_quantile: Quantile threshold for minimum period size
        
    Returns:
        DataFrame limited to complete periods
    """
    # Get the complete timeDelta periods (excluding partial periods at start and end)
    complete_periods = historyFilt.groupby(timeDelta).size()
    min_period_size = complete_periods.quantile(min_period_quantile)
    
    # Filter to only include periods with sufficient data
    valid_periods = complete_periods[complete_periods >= min_period_size].index
    historyFilt_complete = historyFilt[historyFilt[timeDelta].isin(valid_periods)]
    
    return historyFilt_complete

# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_persons_by_period(data: pd.DataFrame, period_col: str, title_suffix: str = "", 
                          figsize: Tuple[int, int] = (10, 5), tick_interval: int = 1) -> pd.DataFrame:
    """
    Plot number of unique persons by time period
    
    Args:
        data: DataFrame with 'id_person' and period_col
        period_col: Column name for time period (e.g., 'year_period', 'month_period')
        title_suffix: Additional text for title
        figsize: Figure size tuple
        tick_interval: Show every nth tick label (1 for all, 3 for every 3rd, etc.)
        
    Returns:
        Aggregated data for further use
    """
    persons_by_period = (
        data.groupby(period_col)['id_person']
        .nunique()
        .reset_index(name='person_count')
    )
    
    plt.figure(figsize=figsize)
    sns.barplot(data=persons_by_period, x=period_col, y='person_count', color='skyblue')
    plt.title(f'Number of Persons with At Least One Draw {title_suffix}')
    plt.xlabel(period_col.replace('_', ' ').title())
    plt.ylabel('Number of Persons')
    
    # Make x-ticks less frequent to prevent overlap
    if tick_interval > 1:
        plt.xticks(range(0, len(persons_by_period), tick_interval), 
                   persons_by_period[period_col].iloc[::tick_interval], 
                   rotation=45)
    else:
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    return persons_by_period

def plot_emotion_by_clinic(historyFilt: pd.DataFrame) -> pd.DataFrame:
    """
    Create normalized emotion distribution table by clinic
    
    Args:
        historyFilt: Filtered history dataframe
        
    Returns:
        Normalized emotion distribution DataFrame
    """
    print("Dominant emotion per clinic")
    emotion_by_clinic = historyFilt.groupby(['clinic_name', 'dominant_emotion']).size().reset_index(name='count')
    emotion_by_clinic_pivot = emotion_by_clinic.pivot(index='clinic_name', columns='dominant_emotion', values='count').fillna(0)
    emotion_by_clinic_normalized = (emotion_by_clinic_pivot.div(emotion_by_clinic_pivot.sum(axis=1), axis=0) * 100).round(1)
    
    return emotion_by_clinic_normalized

def plot_profession_by_clinic(historyFilt: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Create normalized profession distribution table by clinic
    
    Args:
        historyFilt: Filtered history dataframe
        top_n: Number of top professions to include
        
    Returns:
        Normalized profession distribution DataFrame
    """
    print(f"Top {top_n} professions across all clinics")
    
    # Get profession counts by clinic
    profession_by_clinic = historyFilt.groupby(['clinic_name', 'proffesion_name']).size().reset_index(name='count')
    
    # Get top N professions across all clinics
    top_n_professions_all = historyFilt['proffesion_name'].value_counts().head(top_n).index
    
    # Filter data to only include top N professions
    profession_by_clinic_filtered = profession_by_clinic[
        profession_by_clinic['proffesion_name'].isin(top_n_professions_all)
    ]
    
    # Pivot to wide format
    profession_by_clinic_pivot = profession_by_clinic_filtered.pivot(
        index='clinic_name', 
        columns='proffesion_name', 
        values='count'
    ).fillna(0)
    
    # Normalize to percentages
    profession_by_clinic_normalized = (profession_by_clinic_pivot.div(profession_by_clinic_pivot.sum(axis=1), axis=0) * 100).round(1)
    
    return profession_by_clinic_normalized

def plot_face_features_analysis(history: pd.DataFrame) -> None:
    """
    Plot face features analysis
    
    Args:
        history: History dataframe with face features
    """
    history[['looks_older', 'probability_of_lost']].plot(
        figsize=(10, 5),
        secondary_y=['probability_of_lost'],
        grid=True
    )
    plt.title('Looks older vs Probability of lost')
    plt.show()

# ============================================================================
# STATISTICAL ANALYSIS FUNCTIONS
# ============================================================================

def linearFit(data: pd.DataFrame, xVar: str, yVar: str, 
              bounds: Tuple[int, int] = (0, 15000), p0: Tuple[int, int] = (1, 0)) -> Dict:
    """
    Perform linear fit on data
    
    Args:
        data: DataFrame with x and y variables
        xVar: Name of x variable column
        yVar: Name of y variable column
        bounds: Bounds for fitting
        p0: Initial parameters
        
    Returns:
        Dictionary with fit results
    """
    def linear_func(x, a, b):
        return a * x + b
    
    popt, pcov = curve_fit(linear_func, data[xVar], data[yVar], 
                          bounds=bounds, p0=p0)
    
    return {
        'popt': popt,
        'pcov': pcov,
        'a': popt[0],
        'b': popt[1]
    }

def exponentialFit(data: pd.DataFrame, xVar: str, yVar: str, 
                  bounds: Tuple[int, int] = (0, 15000), 
                  p0: Tuple[float, float] = (0.1, 0.001)) -> Dict:
    """
    Perform exponential fit on data
    
    Args:
        data: DataFrame with x and y variables
        xVar: Name of x variable column
        yVar: Name of y variable column
        bounds: Bounds for fitting
        p0: Initial parameters
        
    Returns:
        Dictionary with fit results
    """
    def exp_func(x, a, b):
        return a * np.exp(b * x)
    
    popt, pcov = curve_fit(exp_func, data[xVar], data[yVar], 
                          bounds=bounds, p0=p0)
    
    return {
        'popt': popt,
        'pcov': pcov,
        'a': popt[0],
        'b': popt[1]
    }

# ============================================================================
# HMM (Hidden Markov Model) FUNCTIONS
# ============================================================================

def plot_transition_matrix(transmat: np.ndarray, state_mapping: Dict, 
                          title: str = "State Transition Matrix", 
                          ax: Optional[plt.Axes] = None, 
                          title_x: float = -0.5) -> None:
    """Plot transition matrix heatmap"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(transmat, annot=True, fmt='.3f', cmap='Blues', ax=ax)
    ax.set_title(title, x=title_x)
    ax.set_xlabel('Next State')
    ax.set_ylabel('Current State')
    
    # Add state labels if provided
    if state_mapping:
        ax.set_xticklabels([state_mapping.get(i, i) for i in range(transmat.shape[1])])
        ax.set_yticklabels([state_mapping.get(i, i) for i in range(transmat.shape[0])])

def plot_emission_matrix(emission_probs: np.ndarray, state_mapping: Dict, obs_mapping: Dict) -> None:
    """Plot emission matrix heatmap"""
    plt.figure(figsize=(10, 6))
    sns.heatmap(emission_probs, annot=True, fmt='.3f', cmap='Greens')
    plt.title('Emission Probability Matrix')
    plt.xlabel('Observation')
    plt.ylabel('Hidden State')
    
    # Add labels if provided
    if obs_mapping and state_mapping:
        plt.xticks(range(len(obs_mapping)), [obs_mapping.get(i, i) for i in range(len(obs_mapping))])
        plt.yticks(range(len(state_mapping)), [state_mapping.get(i, i) for i in range(len(state_mapping))])

def save_hmm_results(filepath: str, results_dict: Dict) -> None:
    """Save HMM results to file"""
    joblib.dump(results_dict, filepath)

def load_hmm_results(filepath: str) -> Dict:
    """Load HMM results from file"""
    return joblib.load(filepath)

def predict_next_observation(model: CategoricalHMM, obs_seq: List[int]) -> Dict:
    """Predict next observation using HMM model"""
    if len(obs_seq) == 0:
        return {'next_obs_predicted': None, 'probabilities': None}
    
    # Get the last observation
    last_obs = obs_seq[-1]
    
    # Get transition probabilities from the last state
    # This is a simplified approach - you might need to adjust based on your model structure
    try:
        # Get emission probabilities for the last observation
        emission_probs = model.emissionprob_
        
        # Find the most likely state for the last observation
        state_probs = emission_probs[:, last_obs]
        most_likely_state = np.argmax(state_probs)
        
        # Get transition probabilities from the most likely state
        transition_probs = model.transmat_[most_likely_state, :]
        
        # Get emission probabilities for the next observation
        next_obs_probs = np.dot(transition_probs, emission_probs)
        
        # Predict the most likely next observation
        next_obs_predicted = np.argmax(next_obs_probs)
        
        return {
            'next_obs_predicted': next_obs_predicted,
            'probabilities': next_obs_probs
        }
    except Exception as e:
        return {'next_obs_predicted': None, 'probabilities': None, 'error': str(e)}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_photos_per_person(face_features: pd.DataFrame) -> pd.Series:
    """
    Get count of photos per person
    
    Args:
        face_features: Face features dataframe
        
    Returns:
        Series with person ID as index and photo count as values
    """
    return face_features['id_person'].value_counts()

def analyze_face_features_stats(face_features: pd.DataFrame) -> Dict:
    """
    Analyze face features statistics
    
    Args:
        face_features: Face features dataframe
        
    Returns:
        Dictionary with statistics
    """
    photos_per_person = get_photos_per_person(face_features)
    
    stats = {
        'total_people': len(photos_per_person),
        'total_photos': len(face_features),
        'avg_photos_per_person': len(face_features) / len(photos_per_person),
        'min_photos_per_person': photos_per_person.min(),
        'max_photos_per_person': photos_per_person.max(),
        'photos_distribution': photos_per_person.value_counts().sort_index().to_dict()
    }
    
    return stats
