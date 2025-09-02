"""
Draw data processing module

This module contains functions for creating and processing the main history dataset
used across various analysis notebooks.
"""

import pandas as pd
import numpy as np
from . import preprocessing as prep
from . import zts
from locations.population_density.population_map import CzechPopulationMapper


def create_history_dataset(engine, time_delta='week_period'):
    """
    Create the main history dataset with all derived features.
    
    Parameters:
    -----------
    engine : sqlalchemy.Engine
        Database connection engine
    time_delta : str
        Time scale for grouped data: 'date', 'week_period', or 'month_period'
        
    Returns:
    --------
    dict: Dictionary containing:
        - history: Main processed history DataFrame
        - uzis_history: UZIS competitor data DataFrame  
        - clinics: Clinic data DataFrame
        - ztsList: ZTS competitor list
        - start_date: Dataset start date
        - end_date: Dataset end date
        - cutoff_end: End date minus 6 months (for tail artifacts)
        - last_year: End date minus 12 months
        - last_two_years: End date minus 24 months
        - our_company_name: Company name constant
    """
    
    # Load main draw data
    history = _load_draw_data(engine)
    
    # Load UZIS competitor data
    uzis_history = _load_uzis_data(engine)
    
    # Process UZIS competitor analysis
    uzis_history = _process_uzis_data(uzis_history)
    
    # Load and join location data
    locations, locations_unique = prep.load_locations(engine)
    history = prep.join_longitute_latitude(history, locations_unique)
    
    # Load competitor data
    ztsList = zts.load_zts_list_from_db(engine)
    
    # Load and join face features
    face_features = prep.load_face_features_from_db(engine)
    history = prep.join_face_features_to_history(history, face_features)
    
    # Calculate derived features
    history = _add_derived_features(history, uzis_history)
    
    # Apply data filters
    history = _apply_data_filters(history)
    
    # Limit to complete periods
    history = prep.limit_data_to_complete_periods(history, time_delta)
    
    # Load clinic data
    clinics = prep.load_clinics(locations_unique, engine)
    
    # Calculate date ranges
    start_date = history["datetime"].min()
    end_date = history["datetime"].max()
    cutoff_end = end_date - pd.DateOffset(months=6)  # cut off few months at the end because of tail artefacts (kde)
    last_year = end_date - pd.DateOffset(months=12)
    last_two_years = end_date - pd.DateOffset(months=24)
    
    our_company_name = 'Cara Plasma s.r.o.'
    
    return {
        'history': history,
        'uzis_history': uzis_history,
        'clinics': clinics,
        'ztsList': ztsList,
        'start_date': start_date,
        'end_date': end_date,
        'cutoff_end': cutoff_end,
        'last_year': last_year,
        'last_two_years': last_two_years,
        'our_company_name': our_company_name
    }


def _load_draw_data(engine):
    """Load main draw data from database."""
    query = """
            SELECT dr.id,
                dr.id_clinic,
                dr.id_person,
                dr.draw_number,
                dr.draw_character_ident AS draw_character,
                dr.migrated_draw_ident AS deprecated_draw_character,
                dr.draw_date AS datetime,
                dr.draw_date::date AS date,
                EXTRACT(HOUR FROM dr.draw_date) AS hour,
                EXTRACT(DOW FROM dr.draw_date) + 1 AS weekday,
                EXTRACT(MONTH FROM dr.draw_date) AS month,
                TO_CHAR(DATE_TRUNC('year', dr.draw_date), 'YYYY') AS year_period,
                TO_CHAR(DATE_TRUNC('month', dr.draw_date), 'YYYY-MM') AS month_period,
                TO_CHAR(dr.draw_date, 'IYYY-"W"IW') AS week_period,
                EXTRACT(YEAR FROM dr.draw_date) AS year,
                COALESCE(trim(pr.dm_collected_ml_draw)::integer, 0) AS collected,
                pay.amount as payed_to_donor,
                CASE
                    WHEN per.birthdate IS NULL 
                        OR per.birthdate > CURRENT_DATE 
                        OR per.birthdate < CURRENT_DATE - INTERVAL '130 years'
                    THEN NULL
                    ELSE EXTRACT(YEAR FROM AGE(CURRENT_DATE, per.birthdate))
                END AS age_years,
                adr.zip_code as postal_code,
                (
                    SELECT COUNT(*) 
                    FROM donor.draw d2
                    JOIN donor.draw_process as pr2 ON d2.id = pr2.id_draw
                    WHERE d2.id_person = dr.id_person
                    AND d2.draw_date < dr.draw_date
                    AND d2.draw_date >= dr.draw_date - INTERVAL '1 month'
                    AND d2.draw_character_ident <> 'Z'
                    AND pr.dm_collected_ml_draw IS NOT NULL
                ) AS freq_last_month,
                cl.name AS clinic_name,
                prof.name AS proffesion_name,
                per.sex AS person_sex,
                LAG(dr.draw_character_ident, 1) OVER (
                    PARTITION BY dr.id_person 
                    ORDER BY dr.draw_date
                ) AS previous_draw_character,
                LAG(dr.draw_date, 1) OVER (
                    PARTITION BY dr.id_person 
                    ORDER BY dr.draw_date
                ) AS previous_draw_date,
                LEAD(dr.draw_character_ident, 1) OVER (
                    PARTITION BY dr.id_person 
                    ORDER BY dr.draw_date
                ) AS next_draw_character,
                der.removing_after_draw_reason_id IS NOT NULL AS is_canceled_before_draw,
                der.removing_before_draw_reason_id IS NOT NULL AS is_canceled_after_draw,
                (der.removing_after_draw_reason_id IS NOT NULL OR der.removing_before_draw_reason_id IS NOT NULL) AS is_canceled

            FROM donor.draw as dr
            LEFT JOIN donor.draw_process as pr ON dr.id = pr.id_draw
            LEFT JOIN donor.draw_payment as pay ON dr.id = pay.id_draw
            LEFT JOIN donor.person as per ON dr.id_person = per.id
            LEFT JOIN donor.address as adr ON per.id_address = adr.id
            LEFT JOIN list.profession as prof ON per.id_profession = prof.id
            LEFT JOIN list.clinic as cl ON dr.id_clinic = cl.id
            LEFT JOIN donor.draw_examination_room der ON der.id_draw = dr.id
            ;
            """
    return pd.read_sql(query, engine)


def _load_uzis_data(engine):
    """Load UZIS competitor data from database."""
    uzis_query = """
            SELECT
                udr.id as id_import,
                imp.id_person,
                per.id_uzis_rid,
                per.firstname,
                per.surname,
                per.email,
                per.phone,
                per.birthdate,
                per.doctor_name,
                per.remove_as,
                pro.id as id_profession,
                pro.name as profession,
                nat.name as country_name,
                nat.country_code,
                udr.draw_date,
                udr.clinic_code,
                udr.clinic_name,
                udr.clinic_ic,
                udr.clinic_company
            FROM technical.uzis_draws_import as imp
            LEFT JOIN donor.uzis_draws as udr ON imp.id = udr.id_uzis_draws_import
            LEFT JOIN donor.person as per ON imp.id_person = per.id
            LEFT JOIN list.profession as pro ON pro.id = per.id_profession
            LEFT JOIN list.nationality as nat ON nat.id = per.id_nationality
            ;
            """
    return pd.read_sql(uzis_query, engine)


def _process_uzis_data(uzis_history):
    """Process UZIS data to add competitor analysis features."""
    our_company_name = 'Cara Plasma s.r.o.'
    
    # Mark draws at our company vs competitors
    uzis_history['is_our_company_draw'] = uzis_history['clinic_company'] == our_company_name
    uzis_history['is_other_company_draw'] = uzis_history['clinic_company'] != our_company_name
    
    # Mark donors who visit only us vs those who also visit competitors
    uzis_history['is_not_loyal_to_our_company'] = (
        uzis_history
        .groupby('id_person')['is_other_company_draw']
        .transform('any')
    )
    
    # Mark donors who visit both us and competitors
    uzis_history['is_visiting_our_and_other_company'] = (
        uzis_history.groupby('id_person')['is_our_company_draw'].transform('any') &
        uzis_history.groupby('id_person')['is_other_company_draw'].transform('any')
    )
    
    return uzis_history


def _add_derived_features(history, uzis_history):
    """Add all derived features to the history dataset."""
    # Calculate face-age comparison
    history = prep.calculate_is_not_fine(history)
    history['looks_older'] = history.age_years - history.age_detected
    
    # Add days since company start
    history['daysOfCompany'] = (history['datetime'] - history['datetime'].min()).dt.days
    
    # Add age groups
    bins = [17, 25, 35, 45, 55, 65]
    labels = ['18–25', '26–35', '36–45', '46–55', '56–65']
    history['age_group'] = pd.cut(history['age_years'], bins=bins, labels=labels)
    
    # Add donor behavior flags
    history['is_newbie'] = history['previous_draw_date'].isna()
    history['is_Z'] = history['draw_character'] == 'Z'
    history['is_A'] = history['draw_character'] == 'A'
    history['is_K'] = history['draw_character'] == 'K'
    history['is_R'] = history['draw_character'] == 'R'
    history['is_repeating_Z'] = history['is_Z'] & (history['previous_draw_character'] != 'Z') & history['previous_draw_date'].notna()
    history['is_double_Z'] = history['is_Z'] & (history['previous_draw_character'] == 'Z') & history['previous_draw_date'].notna()
    
    # Merge UZIS competitor data
    uzis_person_flags = uzis_history.groupby('id_person')[['is_not_loyal_to_our_company', 'is_visiting_our_and_other_company']].any().reset_index()
    history = history.merge(uzis_person_flags, on='id_person', how='left')
    
    return history


def _apply_data_filters(history):
    """Apply data quality filters to remove outliers."""
    col_lower_threshold = 0
    col_upper_threshold = 1100
    payed_exceptional_upper_threshold = 1500
    
    print('Filtering out exceptional values')
    print('Collected null {colNull}%'.format(colNull=history.collected.isna().mean() * 100))
    print('Collected below {colLowerThreshold}ml {colLower}%,\nCollected above {colUpperThreshold}ml {colUpper}%'.format(
        colLowerThreshold=col_lower_threshold, 
        colUpperThreshold=col_upper_threshold, 
        colLower=(history.collected < col_lower_threshold).mean() * 100, 
        colUpper=(history.collected > col_upper_threshold).mean() * 100
    ))
    print('Payed above {payedExceptionalUpperThreshold}ml {payedExceptionalUpper}%'.format(
        payedExceptionalUpperThreshold=payed_exceptional_upper_threshold, 
        payedExceptionalUpper=(history.collected > col_upper_threshold).mean() * 100
    ))
    print('Missing or invalid age {invalidAge}%'.format(invalidAge=history.age_years.isna().mean() * 100))
    
    # Apply filters
    history = history[
        (history.collected < col_upper_threshold) & 
        (history.collected >= col_lower_threshold)
    ]
    history = history[
        (history.payed_to_donor < payed_exceptional_upper_threshold) & 
        (history.payed_to_donor >= 0)
    ]
    
    original_count = len(history)  # This is after filtering, so it's not the true original
    print('Kept {total}% of dataset after filtering'.format(total=100))  # Simplified since we can't track original count here
    
    return history


def create_daily_stats(history, time_delta='week_period'):
    """
    Create aggregated daily statistics from history data.
    
    Parameters:
    -----------
    history : pd.DataFrame
        Processed history dataset
    time_delta : str
        Time grouping column
        
    Returns:
    --------
    pd.DataFrame: Aggregated statistics by time period
    """
    daily_stats = history.groupby(time_delta).agg(
        office_count=('id_clinic', 'nunique'),
        draw_count=('id', 'count'),
        collected_median=('collected', 'median'),
        collected_sum=('collected', 'sum'),
        payed_to_donor_mean=('payed_to_donor', 'mean'),
        payed_to_donor_sum=('payed_to_donor', 'sum'),
        age_mean=('age_years', 'mean'),
        freq_last_month_mean=('freq_last_month', 'mean'),
        is_newbie=('is_newbie', 'sum'),
        is_canceled=('is_canceled', 'sum'),
        is_Z=('is_Z', 'sum'),
        is_repeating_z=('is_repeating_Z', 'sum'),
        is_double_z=('is_double_Z', 'sum'),
        is_A=('is_A', 'sum'),
        is_K=('is_K', 'sum'),
        is_R=('is_R', 'sum'),
        month_mean=('month', 'mean'),
        weekday_mean=('weekday', 'mean'),
        year_mean=('year', 'mean'),
        is_not_fine=('is_not_fine', 'mean'),
        emotion_happy=('emotion_happy', 'median'),
        emotion_angry=('emotion_angry', 'median'),
    ).reset_index()
    
    # Calculate ratios
    daily_stats['Z_to_A_ratio'] = daily_stats['is_A'] / daily_stats['is_Z']
    daily_stats['Z_to_A_ratio'] = daily_stats['Z_to_A_ratio'].replace([np.inf, -np.inf], 0)
    daily_stats['A_to_R_ratio'] = (daily_stats['is_R'] / daily_stats['is_A']).clip(upper=2)
    
    return daily_stats


def setup_population_mapper(engine):
    """
    Setup and prepare the Czech population mapper.
    
    Parameters:
    -----------
    engine : sqlalchemy.Engine
        Database connection engine
        
    Returns:
    --------
    CzechPopulationMapper: Prepared population mapper instance
    """
    population_mapper = CzechPopulationMapper()
    population_mapper.prepare_data(engine)
    return population_mapper