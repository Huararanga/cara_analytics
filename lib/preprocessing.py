# use python 311, due deepface/pytorch deps
import numpy as np
import pandas as pd
import numpy
import scipy

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import seaborn as sns

from data_ingestion.data_sources.czech_cities.schema import CzechCities
from data_ingestion.data_sources.face_features.schema import PersonPhotoFaceFeatures


MODULE_FULL_PATH = '/<home>/<username>/my_project/my_project'

ZTS_LIST_FILE_PATH = "./competitors/list/zts_list.json"

# preprocessing and plot

def load_face_features(filepath: str = "./face_features.parquet") -> pd.DataFrame:
    """Load face features from parquet file"""
    return pd.read_parquet(filepath)

def load_face_features_from_db(engine) -> pd.DataFrame:
    """Load face features from db"""
    return pd.read_sql(PersonPhotoFaceFeatures.__table__.select(), engine)

def load_locations(engine) -> pd.DataFrame:
    locations = pd.read_sql(CzechCities.__table__.select(), engine)
    locations_unique = locations.drop_duplicates(subset=["postal_code"])

    return locations, locations_unique;

def join_longitute_latitude(history: pd.DataFrame, locations_unique: pd.DataFrame) -> pd.DataFrame:
    history = history.merge(
        locations_unique[["postal_code", "latitude", "longitude"]],
        on="postal_code",
        how="left"  # keeps all rows from history
    )
    history["latitude"] = history["latitude"].fillna(50)
    history["longitude"] = history["longitude"].fillna(19)
    return history;


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
    columns_to_join = ['id_person','id_photo', 'face_detected', 'age_detected', 'detected_gender', 
                    'dominant_emotion', 'pose_yaw', 'pose_pitch', 'pose_roll',
                    'emotion_angry', 'emotion_disgust', 'emotion_fear',
                    'emotion_happy', 'emotion_sad', 'emotion_surprise', 'emotion_neutral']

    last_face_features_selected = last_face_features[columns_to_join]

    # Join with history dataframe
    history_with_face_features = history.merge(
        last_face_features_selected, 
        on='id_person', 
        how='left'
    )
    
    return history_with_face_features

def calculate_is_not_fine(history: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate is_not_fine column from emotions
    
    Args:
        history_with_face_features: DataFrame with emotion columns
        
    Returns:
        DataFrame with is_not_fine column added
    """
    # Method 1: Based on negative emotions (angry, disgust, fear, sad)
    history['is_not_fine'] = (
        (history['emotion_angry'] > 0.25) |
        (history['emotion_disgust'] > 0.25) |
        (history['emotion_fear'] > 0.25) |
        (history['emotion_sad'] > 0.25) |
        (history['emotion_happy'] < 0.2)  # Low happiness
    )
    
    return history

def load_clinics(locations_unique, engine) -> pd.DataFrame:
    query = """
        SELECT
            postal_code,
            name
        FROM list.clinic as cl
        ;
        """
    clinics = pd.read_sql(query, engine)
    clinics = clinics.merge(
        locations_unique[["postal_code", "latitude", "longitude"]],
        on="postal_code",
        how="left"
    )
    return clinics

def limit_data_to_complete_periods(data: pd.DataFrame, timeDelta: str, 
                                  min_period_quantile: float = 0.1) -> pd.DataFrame:
    """
    Limit data by whole timeDelta period to prevent start and end artefacts
    
    Args:
        data: Filtered history dataframe
        timeDelta: Time period column name
        min_period_quantile: Quantile threshold for minimum period size
        
    Returns:
        DataFrame limited to complete periods
    """
    # Get the complete timeDelta periods (excluding partial periods at start and end)
    complete_periods = data.groupby(timeDelta).size()
    min_period_size = complete_periods.quantile(0.1)  # Or use a specific threshold

    # Filter to only include periods with sufficient data
    valid_periods = complete_periods[complete_periods >= min_period_size].index
    historyFilt_complete = data[data[timeDelta].isin(valid_periods)]
    
    return historyFilt_complete

# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_persons_by_period(data, period_col, title_suffix="", figsize=(10, 5), tick_interval=1):
    """
    Plot number of unique persons by time period
    
    Parameters:
    - data: DataFrame with 'id_person' and period_col
    - period_col: Column name for time period (e.g., 'year_period', 'month_period')
    - title_suffix: Additional text for title
    - figsize: Figure size tuple
    - tick_interval: Show every nth tick label (1 for all, 3 for every 3rd, etc.)
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

def plot_males_females_by_clinic(historyFilt: pd.DataFrame) -> pd.DataFrame:
    """
    Create normalized emotion distribution table by clinic
    
    Args:
        historyFilt: Filtered history dataframe
        
    Returns:
        Normalized emotion distribution DataFrame
    """
    print("Male vs Female per clinic")
    # Get the counts by clinic and sex
    sex_by_clinic = historyFilt.groupby(['clinic_name', 'person_sex']).size().reset_index(name='count')

    # Pivot to wide format (one line per clinic)
    sex_by_clinic_pivot = sex_by_clinic.pivot(index='clinic_name', columns='person_sex', values='count').fillna(0)

    # Calculate ratio (Male/Female)
    sex_by_clinic_pivot['ratio_male_female'] = sex_by_clinic_pivot['M'] / sex_by_clinic_pivot['Z']

    # Round ratio to 2 decimal places
    sex_by_clinic_pivot['ratio_male_female'] = sex_by_clinic_pivot['ratio_male_female'].round(2)

    return sex_by_clinic_pivot



def plot_draws_histogram_with_clinics(historyFilt, clinics, ax, bins=100, cmap="Reds", mode="hist"):
    """
    Plot draws histogram or KDE with clinic names.

    Args:
        historyFilt: DataFrame with longitude and latitude columns
        clinics: DataFrame with clinic names and coordinates
        ax: matplotlib Axes to plot on
        bins: number of bins for histogram
        cmap: colormap
        mode: "hist" (default) or "kde"
    """
    if mode == "hist":
        h = ax.hist2d(
            historyFilt["longitude"],
            historyFilt["latitude"],
            bins=bins,
            cmap=cmap,
            norm=LogNorm(vmin=1, vmax=historyFilt.shape[0])
        )
    elif mode == "kde":
        # seaborn kdeplot handles density estimation
        sns.kdeplot(
            x=historyFilt["longitude"],
            y=historyFilt["latitude"],
            fill=True,
            cmap=cmap,
            ax=ax,
            thresh=0.05,   # ignore extremely low densities
            levels=100
        )
    else:
        raise ValueError(f"Unknown mode: {mode}, use 'hist' or 'kde'")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # Plot clinic names
    for _, row in clinics.dropna(subset=["latitude", "longitude"]).iterrows():
        ax.text(
            row["longitude"], row["latitude"],
            row["name"],
            fontsize=6,
            ha='center', va='center',
            color='blue',
            alpha=0.7,
            rotation=0,
            clip_on=True  # ensures text is clipped to axes
        )

    

def calc_emotion_by_clinic(historyFilt: pd.DataFrame) -> pd.DataFrame:
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
    emotion_by_clinic_pivot
    emotion_by_clinic_normalized = (emotion_by_clinic_pivot.div(emotion_by_clinic_pivot.sum(axis=1), axis=0) * 100).round(1)
    emotion_by_clinic_normalized
    return emotion_by_clinic_normalized

def plot_age_groups(historyFilt: pd.DataFrame, start_date, cutoff_end) -> pd.DataFrame:
    g = sns.displot(
        data=historyFilt,
        x="datetime",
        hue="age_group",
        kind="kde",
        multiple="fill",
        bw_adjust=0.3,
        aspect=2,
        height=5
    )

    plt.xlim([start_date, cutoff_end])

    plt.title("Relative Draw Distribution Over Time by Age Group")
    plt.xlabel("Date")
    plt.ylabel("Proportion")
    plt.tight_layout()
    plt.show()
    # afer 5 years students will leave -> we need to acquire new ones!

    # sns.histplot(
    #     data=historyFilt,
    #     x="datetime",
    #     hue="age_group",
    #     multiple="stack",
    #     bins=60  # adjust for time granularity
    # )                         
    # plt.title("Absolute Count of Draws by Age Group Over Time")  

def plot_emotion_by_clinic(historyFilt: pd.DataFrame, start_date, cutoff_end) -> pd.DataFrame:
    g = sns.displot(
    data=historyFilt,
    x="datetime",
    hue="dominant_emotion",
    kind="kde",
    multiple="fill",
    bw_adjust=0.3,
    aspect=2,
    height=5
)

    plt.xlim([start_date, cutoff_end])

    plt.title("Relative Draw Distribution Over Time by Dominant Emotion")
    plt.xlabel("Date")
    plt.ylabel("Proportion")
    plt.tight_layout()
    plt.show()


def plot_draw_count_by_clinic(historyFilt: pd.DataFrame, start_date, cutoff_end) -> pd.DataFrame:
    g = sns.displot(
    data=historyFilt,
    x="datetime",
    hue="clinic_name",
    kind="kde",
    multiple="fill",
    bw_adjust=0.3,
    aspect=2,
    height=5
)

    plt.xlim([start_date, cutoff_end])

    plt.title("Relative Draw Distribution Over Time by Clinic Name")
    plt.xlabel("Date")
    plt.ylabel("Proportion")
    plt.tight_layout()
    plt.show()

def count_newbies_by_clinic(historyFilt: pd.DataFrame) -> pd.DataFrame:
    print("Newbies per clinic")
    # Absolute counts
    abs_counts = historyFilt.groupby(['clinic_name', 'is_newbie']).size().unstack(fill_value=0)

    # Relative counts (normalized per clinic)
    rel_counts = (abs_counts.div(abs_counts.sum(axis=1), axis=0) * 100).round(1)

    # Combine into one table with MultiIndex columns
    combined = pd.concat(
        {'absolute': abs_counts, 'relative': rel_counts},
        axis=1
    )

    # Order by relative percentage of newbies (True) in descending order
    combined_sorted = combined.sort_values(('relative', True), ascending=False)

    return combined_sorted

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

def plot_repeating_z(history: pd.DataFrame, last_year, last_two_years, cutoff_end) -> pd.DataFrame:
    g = sns.displot(
        data=history,
        x="datetime",
        hue="is_repeating_Z",
        kind="kde",
        multiple="fill",
        bw_adjust=0.3,
        aspect=2,
        height=5
    )

    plt.xlim([last_two_years, cutoff_end])

    plt.title("Relative amount of repeaed Z past two years")
    plt.xlabel("Date")
    plt.ylabel("Proportion")
    plt.tight_layout()
    plt.show()

    print("Repeated Z past year")
    print("Absolute")
    print(history[history["datetime"] > last_year]["is_repeating_Z"].value_counts())
    print("Relative")
    print(history[history["datetime"] > last_year]["is_repeating_Z"].value_counts(normalize=True)*100)

def calculate_monthly_churn_rate(historyFilt):
    """
    Calculate monthly churn rate from history data
    
    Args:
        historyFilt: Filtered history dataframe
        
    Returns:
        DataFrame with monthly churn rates
    """
    # Total unique persons per month
    persons_per_month = (
        historyFilt.groupby('month_period')['id_person']
        .nunique()
        .reset_index(name='persons_per_month')
    )

    # Unique persons with is_K == True per month
    persons_with_K = (
        historyFilt[historyFilt['is_K']].groupby('month_period')['id_person']
        .nunique()
        .reset_index(name='persons_per_month_is_K')
    )
    
    # Merge both aggregates on month_period
    churn_df = pd.merge(persons_per_month, persons_with_K, on='month_period', how='left')

    # Fill NaN (i.e. no one with is_K = True) with 0
    churn_df['persons_per_month_is_K'] = churn_df['persons_per_month_is_K'].fillna(0)

    # Compute churn rate
    churn_df['churn_rate'] = (
        (churn_df['persons_per_month'] - churn_df['persons_per_month_is_K']) /
        churn_df['persons_per_month']
    )
    
    return churn_df

def plot_monthly_churn_rate(churn_df, figsize=(12, 5), tick_interval=3):
    """
    Plot monthly churn rate
    
    Args:
        churn_df: DataFrame with churn rate data
        figsize: Figure size tuple
        tick_interval: Show every nth tick label
    """
    plt.figure(figsize=figsize)
    sns.lineplot(data=churn_df, x='month_period', y='churn_rate', marker='o')
    plt.title('Monthly Churn Rate')
    plt.xlabel('Month')
    plt.ylabel('Churn Rate')
    plt.ylim(0, 1)

    # Tick every nth month
    xticks = churn_df['month_period'].iloc[::tick_interval]
    plt.xticks(ticks=range(0, len(churn_df), tick_interval), labels=xticks, rotation=45, fontsize=8)

    plt.tight_layout()
    plt.show()

def calculate_monthly_newcomers_rate(historyFilt):
    """
    Calculate monthly newcomers rate from history data
    
    Args:
        historyFilt: Filtered history dataframe
        
    Returns:
        DataFrame with monthly newcomers rates
    """
    # Total unique persons per month
    persons_per_month = (
        historyFilt.groupby('month_period')['id_person']
        .nunique()
        .reset_index(name='persons_per_month')
    )

    # Unique persons with is_newbie == True per month
    persons_newbies = (
        historyFilt[historyFilt['is_newbie']].groupby('month_period')['id_person']
        .nunique()
        .reset_index(name='persons_per_month_newbies')
    )
    
    # Merge both aggregates on month_period
    newcomers_df = pd.merge(persons_per_month, persons_newbies, on='month_period', how='left')

    # Fill NaN (i.e. no newbies) with 0
    newcomers_df['persons_per_month_newbies'] = newcomers_df['persons_per_month_newbies'].fillna(0)

    # Compute newcomers rate
    newcomers_df['newcomers_rate'] = (
        newcomers_df['persons_per_month_newbies'] / newcomers_df['persons_per_month']
    )
    
    return newcomers_df

def plot_monthly_newcomers_rate(newcomers_df, figsize=(12, 5), tick_interval=3):
    """
    Plot monthly newcomers rate
    
    Args:
        newcomers_df: DataFrame with newcomers rate data
        figsize: Figure size tuple
        tick_interval: Show every nth tick label
    """
    plt.figure(figsize=figsize)
    sns.lineplot(data=newcomers_df, x='month_period', y='newcomers_rate', marker='o')
    plt.title('Monthly Newcomers Rate')
    plt.xlabel('Month')
    plt.ylabel('Newcomers Rate')
    plt.ylim(0, 1)

    # Tick every nth month
    xticks = newcomers_df['month_period'].iloc[::tick_interval]
    plt.xticks(ticks=range(0, len(newcomers_df), tick_interval), labels=xticks, rotation=45, fontsize=8)

    plt.tight_layout()
    plt.show()

def plot_persons_draw_count_per_year(historyFilt):
    per_person_draws = historyFilt.groupby(['year_period', 'id_person']).agg(
        draw_count=('id', 'count')
    ).reset_index()

    draw_distribution = per_person_draws.groupby(['year_period', 'draw_count']).agg(
        person_count=('id_person', 'count')
    ).reset_index()

    heatmap_data = draw_distribution.pivot(
        index='year_period', 
        columns='draw_count', 
        values='person_count'
    ).fillna(0)

    plt.figure(figsize=(12, 6))
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt='g',
        cmap='Blues',
        annot_kws={"size": 8}  # smaller font for counts
    )
    plt.title('Number of Persons per Draw Count per Year')
    plt.xlabel('Draw Count')
    plt.ylabel('Year')
    plt.tight_layout()
    plt.show()

# ============================================================================
# STATISTICAL ANALYSIS FUNCTIONS
# ============================================================================

def linearFit(data, xVar, yVar, bounds=(0, 15000), p0=(1, 0)):
    fitFn = lambda x, a, b: a * x + b
    params, _ = scipy.optimize.curve_fit(fitFn, data[xVar], data[yVar], p0=p0)
    
    xTest = np.linspace(bounds[0], bounds[1], 100)
    yTest = fitFn(xTest, params[0], params[1])
    
    plt.scatter(xTest, yTest, color='blue', label='Linear Fit')
    plt.scatter(data[xVar], data[yVar], label='Data')
    plt.xlabel(xVar)
    plt.ylabel(yVar)
    plt.legend()
    plt.show()
    
    print("Fitted parameters (slope, intercept):", params)

def exponentialFit(data, xVar, yVar, bounds=(0,15000), p0=(0.1, 0.001)):
    fitFn = lambda t,a,b: a*numpy.exp(b*t);
    params = scipy.optimize.curve_fit(fitFn, data[xVar],  data[yVar], p0=p0)
    xTest = np.linspace(bounds[0], bounds[1] , 100)
    yTest = fitFn(xTest, params[0][0], params[0][1])
    plt.scatter(xTest, yTest, color = 'blue')
    plt.scatter(data[xVar], data[yVar])
    #plt.annotate(r'\frac{-e^{i\pi}}{2^n}$!', (0,0))
    plt.xlabel(xVar)
    plt.ylabel(yVar)
    print(params)
    
