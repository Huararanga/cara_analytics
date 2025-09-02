import marimo


__generated_with = "0.15.2"
app = marimo.App()


@app.cell
def _():
    # use python 311, due deepface/pytorch deps
    import numpy as np
    import pandas as pd
    from sklearn import preprocessing
    from dotenv import load_dotenv
    import os

    import matplotlib
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')
    matplotlib.rcParams['figure.figsize'] = (18, 6)

    # # produce vector inline graphics
    #from IPython.display import set_matplotlib_formats
    #set_matplotlib_formats('pdf', 'svg')
    #%config InlineBackend.figure_format = 'svg'

    import seaborn as sns

    from sklearn.metrics import classification_report

    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    load_dotenv()

    user = os.getenv("DB_USER") # should be readonly user
    password = os.getenv("DB_PASSWORD")
    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT")
    db = os.getenv("DB_NAME")

    from sqlalchemy import create_engine
    import psycopg2
    conn_str = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"
    engine = create_engine(conn_str)

    import lib.preprocessing as prep
    import lib.zts as zts
    import lib.hmm as hmm
    import lib.geo as geo
    import lib.lost_donors as lost_donors

    from locations.population_density.population_map import CzechPopulationMapper

    population_mapper = CzechPopulationMapper()
    population_mapper.prepare_data(engine)

    location_dimensions = {
        'prague': { 'longitude': (14,15), 'latitude': (49.7,50.3)}
    }

    ZTS_LIST_FILE_PATH = "./competitors/list/zts_list.json"

    import marimo as mo
    return (
        classification_report,
        engine,
        geo,
        hmm,
        location_dimensions,
        lost_donors,
        np,
        pd,
        plt,
        population_mapper,
        prep,
        preprocessing,
        sns,
        zts,
    )


@app.cell
def _():
    timeDelta = 'week_period' # time scale for grouped data: date | week_period | month_period

    grid_mode = "hist"  # Plot type for grouped data grid "hist" or "kde"
    return grid_mode, timeDelta


@app.cell
def _(engine, pd, prep, timeDelta, zts):
    # Load data into DataFrame
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
    history = pd.read_sql(query, engine)

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
    uzis_history = pd.read_sql(uzis_query, engine)

    # UZIS draws
    our_company_name = 'Cara Plasma s.r.o.'

    # odbery u nas
    uzis_history['is_our_company_draw'] = uzis_history['clinic_company'] == our_company_name

    # odbery u konkurence
    uzis_history['is_other_company_draw'] = uzis_history['clinic_company'] != our_company_name

    # odbery darcu kteri chodi jen k nam
    uzis_history['is_not_loyal_to_our_company'] = (
        uzis_history
        .groupby('id_person')['is_other_company_draw']
        .transform('any')
    )

    # odbery darcu kteri chodi k nam i ke konkurenci
    uzis_history['is_visiting_our_and_other_company'] = (
        uzis_history.groupby('id_person')['is_our_company_draw'].transform('any') &
        uzis_history.groupby('id_person')['is_other_company_draw'].transform('any')
    )

    our_company_name = 'Cara Plasma s.r.o.'

    # odbery u nas
    uzis_history['is_our_company_draw'] = uzis_history['clinic_company'] == our_company_name

    # odbery u konkurence
    uzis_history['is_other_company_draw'] = uzis_history['clinic_company'] != our_company_name

    # odbery darcu kteri chodi jen k nam
    uzis_history['is_not_loyal_to_our_company'] = (
        uzis_history
        .groupby('id_person')['is_other_company_draw']
        .transform('any')
    )

    # odbery darcu kteri chodi k nam i ke konkurenci
    uzis_history['is_visiting_our_and_other_company'] = (
        uzis_history.groupby('id_person')['is_our_company_draw'].transform('any') &
        uzis_history.groupby('id_person')['is_other_company_draw'].transform('any')
    )

    locations, locations_unique = prep.load_locations(engine)
    history = prep.join_longitute_latitude(history, locations_unique)
    ztsList = zts.load_zts_list_from_db(engine)
    face_features = prep.load_face_features_from_db(engine)
    history = prep.join_face_features_to_history(history, face_features)
    history = prep.calculate_is_not_fine(history)
    history['looks_older'] = history.age_years - history.age_detected
    clinics = prep.load_clinics(locations_unique, engine)
    history['daysOfCompany'] = (history['datetime'] - history['datetime'].min()).dt.days
    _bins = [17, 25, 35, 45, 55, 65]
    labels = ['18–25', '26–35', '36–45', '46–55', '56–65']
    history['age_group'] = pd.cut(history['age_years'], bins=_bins, labels=labels)
    history['is_newbie'] = history['previous_draw_date'].isna()
    history['is_Z'] = history['draw_character'] == 'Z'
    history['is_A'] = history['draw_character'] == 'A'
    history['is_K'] = history['draw_character'] == 'K'
    history['is_R'] = history['draw_character'] == 'R'
    history['is_repeating_Z'] = history['is_Z'] & (history['previous_draw_character'] != 'Z') & history['previous_draw_date'].notna()
    history['is_double_Z'] = history['is_Z'] & (history['previous_draw_character'] == 'Z') & history['previous_draw_date'].notna()
    history['is_K_collected'] = history['collected'].where(history['is_K'], 0)
    history['is_A_collected'] = history['collected'].where(history['is_A'], 0)
    history['is_R_collected'] = history['collected'].where(history['is_R'], 0)
    history = history.merge(uzis_history.groupby('id_person')[['is_not_loyal_to_our_company', 'is_visiting_our_and_other_company']].any().reset_index(), on='id_person', how='left')

    colLowerThreshold = 0
    colUpperThreshold = 1100
    payedExceptionalUpperThreshold = 1500
    print('Filtering out exceptional values')
    print('Collected null {colNull}%'.format(colNull=history.collected.isna().mean() * 100))
    print('Collected below {colLowerThreshold}ml {colLower}%,\nCollected above {colUpperThreshold}ml {colUpper}%'.format(colLowerThreshold=colLowerThreshold, colUpperThreshold=colUpperThreshold, colLower=(history.collected < colLowerThreshold).mean() * 100, colUpper=(history.collected > colUpperThreshold).mean() * 100))
    print('Payed above {payedExceptionalUpperThreshold}ml {payedExceptionalUpper}%'.format(payedExceptionalUpperThreshold=payedExceptionalUpperThreshold, payedExceptionalUpper=(history.collected > colUpperThreshold).mean() * 100))
    print('Missing or invalid age {invalidAge}%'.format(invalidAge=history.age_years.isna().mean() * 100))
    history = history[(history.collected < colUpperThreshold) & (history.collected >= colLowerThreshold)]
    history = history[(history.payed_to_donor < payedExceptionalUpperThreshold) & (history.payed_to_donor >= 0)]
    print('Keept {total}% of original dataset'.format(total=100 * history.shape[0] / history.shape[0]))

    history = prep.limit_data_to_complete_periods(history, timeDelta)

    #conf_vars

    start_date = history["datetime"].min()
    end_date = history["datetime"].max()
    cutoff_end = end_date - pd.DateOffset(months=6)  # cut of few months at the end because of tail artefacts (kde)
    last_year = end_date - pd.DateOffset(months=12)
    last_two_years = end_date - pd.DateOffset(months=24)
    return (
        clinics,
        cutoff_end,
        history,
        last_two_years,
        last_year,
        our_company_name,
        start_date,
        uzis_history,
        ztsList,
    )


@app.cell
def _(history):
    print(history['is_newbie'].value_counts())
    print(history['is_canceled'].value_counts())
    print(history['is_repeating_Z'].value_counts())
    print(history['is_double_Z'].value_counts())
    print(history['is_A'].value_counts())
    print(history['is_K'].value_counts())
    print(history['is_R'].value_counts())
    print(history['is_not_fine'].value_counts())
    return


@app.cell
def _(history):
    print("Collected mean - median are different {mean} vs {median}".format(mean=(history.collected.mean()), median=(history.collected.median())))
    print("Payed mean - median are different {mean} vs {median}".format(mean=(history.payed_to_donor.mean()), median=(history.payed_to_donor.median())))
    print("Age mean - median are different {mean} vs {median}".format(mean=(history.age_years.mean()), median=(history.age_years.median())))
    print("freq_last_month mean {mean} ".format(mean=(history.freq_last_month.mean())))
    return


@app.cell
def _(history, np, timeDelta):
    dailyStats = history.groupby(timeDelta).agg(
    # dailyStats = history[history["datetime"] > last_two_years].groupby(timeDelta).agg(
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
        is_A_collected=('is_A_collected', 'sum'),
        is_K_collected=('is_K_collected', 'sum'),
        is_R_collected=('is_R_collected', 'sum'),
        month_mean=('month', 'mean'),
        weekday_mean=('weekday', 'mean'),
        year_mean=('year', 'mean'),
        is_not_fine=('is_not_fine', 'mean'),
        emotion_happy=('emotion_happy', 'median'),
        emotion_angry=('emotion_angry', 'median'),
    ).reset_index()  # makes 'date' a column

    dailyStats['Z_to_A_ratio'] = dailyStats['is_A'] / dailyStats['is_Z']
    dailyStats['Z_to_A_ratio'] = dailyStats['Z_to_A_ratio'].replace([np.inf, -np.inf], 0)
    dailyStats['A_to_R_ratio'] = (
        dailyStats['is_R'] / dailyStats['is_A']
    ).clip(upper=2)
    return (dailyStats,)


@app.cell
def _(dailyStats, plt, timeDelta):
    dailyStats.plot(x=timeDelta, subplots=True, figsize=(15, 50));
    plt.show()
    return


@app.cell
def _(dailyStats, grid_mode, pd, preprocessing, sns, timeDelta):
    _bins = 30
    _n_levels = 30
    grid_data = dailyStats.drop(timeDelta, axis=1).dropna()
    _min_max_scaler = preprocessing.MinMaxScaler()
    _x_scaled = _min_max_scaler.fit_transform(grid_data.values)
    _scaledSample = pd.DataFrame(data=_x_scaled, columns=grid_data.columns.values)
    _cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=False)
    _g = sns.PairGrid(_scaledSample, diag_sharey=False)
    if grid_mode == 'kde':
        _g.map_lower(sns.kdeplot, n_levels=_n_levels, fill=True, cmap=_cmap)
        _g.map_upper(sns.kdeplot, n_levels=_n_levels, fill=True, cmap=_cmap)
        _g.map_diag(sns.kdeplot, fill=True)
    elif grid_mode == 'hist':
        _g.map_lower(sns.histplot, bins=_bins, pmax=0.8, cmap='Blues')
        _g.map_upper(sns.histplot, bins=_bins, pmax=0.8, cmap='Blues')
        _g.map_diag(sns.histplot, bins=_bins, kde=False)
    else:
        raise ValueError("mode must be either 'kde' or 'hist'")
    return (grid_data,)


@app.cell
def _(grid_data, plt, prep):
    prep.exponentialFit(grid_data, 'office_count', 'collected_sum', bounds=[0,20])
    plt.show()
    return


@app.cell
def _(cutoff_end, history, prep, start_date):
    prep.plot_age_groups(history, start_date, cutoff_end)
    return


@app.cell
def _(cutoff_end, history, prep, start_date):
    prep.plot_draw_count_by_clinic(history, start_date, cutoff_end)
    return


@app.cell
def _(zts, ztsList):
    zts.plot_cumulative_market_share(ztsList)
    return


@app.cell
def _(zts, ztsList):
    zts.plot_cumulative_market_share_top15(ztsList)
    return


@app.cell
def _(
    clinics,
    geo,
    history,
    last_two_years,
    plt,
    population_mapper,
    prep,
    zts,
    ztsList,
):
    ztsList_top_15_geo = zts.get_top_zts_geo_dataframe(ztsList)
    _fig, _axs = plt.subplots(1, 2, figsize=(14, 10))
    _ax = _axs[0]
    population_mapper.create_population_density_map_image(_ax)
    geo.plot_czech_boundaries(_ax)
    _ax = zts.plot_zts_map(ztsList_top_15_geo, _ax, legend=True)
    _ax.set_title('Clinic Distribution across Czech Republic by Company with population density')
    _ax = _axs[1]
    prep.plot_draws_histogram_with_clinics(history[history['datetime'] > last_two_years], clinics, _ax, mode='hist')
    geo.plot_czech_boundaries(_ax)
    plt.title('Cara draws map last two years')
    plt.show()
    return (ztsList_top_15_geo,)


@app.cell
def _(
    clinics,
    geo,
    history,
    last_two_years,
    location_dimensions,
    plt,
    population_mapper,
    prep,
    zts,
    ztsList_top_15_geo,
):
    prague_dims = location_dimensions['prague']
    _fig, _axs = plt.subplots(1, 2, figsize=(14, 10))
    _ax = _axs[0]
    population_mapper.create_population_density_map_image(_ax, prague_dims)
    geo.plot_czech_boundaries(_ax)
    _ax = zts.plot_zts_map(ztsList_top_15_geo, _ax, legend=False)
    _ax.set_title('Clinic Distribution across Prague by Company with population density')
    geo.zoom(_ax, prague_dims)
    _ax = _axs[1]
    plt.title('Cara draws map last two years in Prague')
    prep.plot_draws_histogram_with_clinics(geo.zoom_df(history[history['datetime'] > last_two_years], prague_dims), clinics, _ax, bins=20, mode='hist')
    _ax = zts.plot_zts_map(ztsList_top_15_geo, _ax)
    geo.zoom(_ax, prague_dims)
    plt.show()
    return


@app.cell
def _(cutoff_end, history, prep, start_date):
    prep.plot_emotion_by_clinic(history, start_date, cutoff_end)
    emotion_by_clinic = prep.calc_emotion_by_clinic(history)
    emotion_by_clinic
    #Uherský Brod	happy	47.9 - nice nurse here ;)
    return


@app.cell
def _(history):
    print("Proffesions")
    (history.proffesion_name.value_counts(normalize=True) * 100).head(20)
    return


@app.cell
def _(history, prep):
    sex_by_clinic_pivot = prep.plot_males_females_by_clinic(history);
    sex_by_clinic_pivot
    return


@app.cell
def _(history, prep):
    newbies_by_clinic = prep.count_newbies_by_clinic(history);
    newbies_by_clinic
    return


@app.cell
def _(cutoff_end, history, last_two_years, last_year, prep):
    prep.plot_repeating_z(history, last_year, last_two_years, cutoff_end);
    return


@app.cell
def _(history, last_year):
    print('Double Z past year')
    print('Absolute')
    print(history[history['datetime'] > last_year]['is_double_Z'].value_counts())
    print('Relative')
    print(history[history['datetime'] > last_year]['is_double_Z'].value_counts(normalize=True) * 100)
    return


@app.cell
def _(history, prep):
    # Usage examples:
    # For yearly data
    persons_per_year = prep.plot_persons_by_period(history, 'year_period', "Per Year")

    # For monthly data with less frequent ticks
    persons_per_month = prep.plot_persons_by_period(
        history, 
        'month_period', 
        "Per Month", 
        figsize=(12, 5), 
        tick_interval=3
    )
    return


@app.cell
def _(history, prep):
    churn_df = prep.calculate_monthly_churn_rate(history)
    prep.plot_monthly_churn_rate(churn_df)
    return


@app.cell
def _(history, prep):
    newcomers_df = prep.calculate_monthly_newcomers_rate(history)
    prep.plot_monthly_newcomers_rate(newcomers_df)
    return


@app.cell
def _(history, prep):
    prep.plot_persons_draw_count_per_year(history)
    return


@app.cell
def _(history, pd):
    ZA = history[history['datetime'] >= pd.Timestamp('2024-03-01')].copy()
    ZA_start_date = ZA['datetime'].min()
    ZA_end_date = ZA['datetime'].max()
    ZA_duration = ZA_end_date - ZA_start_date
    AA2_end_date = pd.Timestamp('2023-12-01')
    AA2_start_date = AA2_end_date - ZA_duration
    AA2 = history[(history['datetime'] <= AA2_end_date) & (history['datetime'] >= AA2_start_date)].copy()
    return AA2, ZA


@app.cell
def _():
    #hmm.train(ZA, AA2, hmm.ZA_encoding, hmm.AA2_encoding);
    return


@app.cell
def _(hmm):
    # Later or in another script: Load
    hmm_results = hmm.load_hmm_results("hmm_consequent_draw_processes_comparison_results.joblib")

    # Access elements
    #ZA_model = loaded_results["ZA"]["model"]
    #ZA_titles = loaded_results["ZA"]["state_titles"]

    #hmm.plot_transition_matrix(ZA_model.transmat_, ZA_titles)
    return (hmm_results,)


@app.cell
def _(AA2, ZA, hmm, hmm_results, plt):
    reordered_ZA_transmat, ZA_state_order = hmm.reorder_hmm_matrix_by_emissions(hmm_results['ZA']['model'].transmat_, hmm_results['ZA']['model'].emissionprob_, hmm.ZA_encoding)
    reordered_AA2_transmat, AA2_state_order = hmm.reorder_hmm_matrix_by_emissions(hmm_results['AA2']['model'].transmat_, hmm_results['AA2']['model'].emissionprob_, hmm.AA2_encoding)
    ZA_results_counted = hmm.compute_transition_matrix(data=ZA, encoding=hmm.ZA_encoding, character_col='draw_character')
    AA2_results_counted = hmm.compute_transition_matrix(data=AA2, encoding=hmm.AA2_encoding, character_col='deprecated_draw_character')
    _fig, _axs = plt.subplots(2, 2, figsize=(14, 10))
    hmm.plot_transition_matrix(ZA_results_counted, hmm_results['ZA']['result']['state_titles'], 'ZA - State Transition Matrix (Empirical)', ax=_axs[0][0])
    hmm.plot_transition_matrix(reordered_ZA_transmat, hmm_results['ZA']['result']['state_titles'], 'ZA - State Transition Matrix (Statistical HMM)', ax=_axs[0][1])
    hmm.plot_transition_matrix(AA2_results_counted, hmm_results['AA2']['result']['state_titles'], 'AA2 - State Transition Matrix (Empirical)', ax=_axs[1][0])
    hmm.plot_transition_matrix(reordered_AA2_transmat, hmm_results['AA2']['result']['state_titles'], 'AA2 - State Transition Matrix (Statistical HMM)', ax=_axs[1][1])
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(history, hmm, hmm_results):
    ### predict next draw character for person
    personDrawCharacters = history.sort_values("datetime").groupby("id_person")['draw_character'].apply(list)
    print(personDrawCharacters.index)

    lengths = personDrawCharacters.apply(len)
    length_distribution = lengths.value_counts().sort_index()
    #length_distribution.head(30)

    # Step 1: Get the sequence
    donor_sequence = personDrawCharacters.loc[100016]

    predicted_char = hmm.predict_next_draw_char(hmm_results["ZA"]["model"], hmm.ZA_encoding, donor_sequence)

    # Step 5: Display
    print("Original draw sequence:", donor_sequence)
    print("Next predicted draw (character):", predicted_char)
    return


@app.cell
def _(classification_report, history, hmm_results, lost_donors):
    _X, y = lost_donors.build_feature_matrix(history, hmm_results['ZA'], hmm_results['AA2'])
    X_preprocessed, encoders = lost_donors.preprocess_features(_X)
    model_results = lost_donors.train_xgboost_model(X_preprocessed, y)
    clf = model_results['model']
    print(classification_report(model_results['y_test'], model_results['y_pred']))
    lost_donors.plot_feature_importance(model_results)
    return clf, encoders


@app.cell
def _(clf, encoders, history, hmm_results, lost_donors):
    _result = lost_donors.predict_lost_by_person_id(person_id=305806, history=history, loaded_results=hmm_results, clf=clf, encoders=encoders)
    print(_result)
    return


@app.cell(hide_code=True)
def _(clf, encoders, history, hmm, hmm_results, lost_donors, pd):
    cutoff_date = pd.Timestamp.today() - pd.DateOffset(years=10)
    recent_ids = history[history['datetime'] >= cutoff_date]['id_person'].unique()
    history_grouped = history[history['id_person'].isin(recent_ids)].groupby('id_person', sort=False)

    def compute_features_for_group(gr):
        last_datetime = gr['datetime'].max()
        is_za = lost_donors.is_ZA(last_datetime)
        encoding = hmm.ZA_encoding if is_za else hmm.AA2_encoding
        gr = gr.copy()
        gr['obs'] = gr['draw_character'].map(encoding).fillna(-1).astype(int)
        features = lost_donors.extract_person_features(gr, hmm_results['ZA']['model'], hmm_results['AA2']['model'])
        return pd.DataFrame([features])
    print('Extracting features...')
    features_df = history_grouped.apply(compute_features_for_group).reset_index(level=0).reset_index(drop=True)
    print('Encoding...')
    for col in features_df.select_dtypes(include='object').columns:
        features_df[col] = features_df[col].fillna('unknown')
        if col in encoders:
            oe = encoders[col]
            features_df[col] = oe.transform(features_df[col].values.reshape(-1, 1)).flatten()
        else:
            features_df[col] = -1
    for col in features_df.select_dtypes(include='number').columns:
        features_df[col] = features_df[col].fillna(features_df[col].median())
    print('Predicting...')
    expected_features = clf.get_booster().feature_names
    _X = features_df.drop(columns=['id_person'], errors='ignore')
    for col in expected_features:
        if col not in _X.columns:
            _X[col] = 0
    _X = _X[expected_features]
    preds = clf.predict(_X)
    probas = clf.predict_proba(_X)[:, 1]

    def last_row_metadata(gr):
        last = gr.sort_values('datetime').iloc[-1]
        return pd.Series({'id_person': gr['id_person'].iloc[0], 'last_draw_date': last['datetime'], 'draw_count': len(gr), 'payed_to_donor': last.get('payed_to_donor', 0), 'proffesion_name': last.get('proffesion_name', 'unknown'), 'clinic_name': last.get('clinic_name', 'unknown'), 'person_sex': last.get('person_sex', 'unknown')})
    print('Collecting last-draw metadata...')
    last_draw_df = history_grouped.apply(last_row_metadata).reset_index(drop=True)
    encoded_cols = ['clinic_name', 'proffesion_name', 'person_sex', 'payed_to_donor']
    lost_predictions_df = features_df.drop(columns=encoded_cols, errors='ignore').assign(probability_of_lost=probas, is_lost=preds).merge(last_draw_df, on='id_person', how='left')
    print(f'Done. Generated {len(lost_predictions_df)} predictions.')
    return (lost_predictions_df,)


@app.cell
def _(lost_donors, lost_predictions_df):
    lost_donors.plot_lost_donors(lost_predictions_df)
    return


@app.cell
def _(lost_predictions_df):
    _median_probs = lost_predictions_df.groupby('proffesion_name').agg(probability_of_lost_median=('probability_of_lost', 'median'), count=('proffesion_name', 'size')).query('count >= 25').sort_values('count', ascending=False).reset_index()
    _median_probs
    return


@app.cell
def _(history, prep):
    profession_by_clinic_normalized = prep.plot_profession_by_clinic(history,10)
    # this isnt working looks like proffesion list is too wide
    profession_by_clinic_normalized
    return


@app.cell
def _(lost_donors, lost_predictions_df):
    profession_loss_data = lost_donors.plot_profession_loss_probability(lost_predictions_df)
    return


@app.cell
def _(lost_predictions_df):
    _median_probs = lost_predictions_df.groupby('payed_to_donor').agg(probability_of_lost_median=('probability_of_lost', 'median'), count=('payed_to_donor', 'size')).query('count >= 10').sort_values('probability_of_lost_median', ascending=False).reset_index()
    _median_probs
    return


@app.cell
def _(lost_predictions_df):
    _median_probs = lost_predictions_df.groupby('num_transitions').agg(probability_of_lost_median=('probability_of_lost', 'median'), count=('num_transitions', 'size')).sort_values('probability_of_lost_median', ascending=False).reset_index()
    _median_probs
    return


@app.cell
def _(lost_predictions_df):
    lost_predictions_df.log_likelihood.hist(bins=50)
    # how well can hmm predict, closer to 0 is better. More negative is usually connected with low draw count for person - low amount of data
    return


@app.cell
def _(lost_predictions_df):
    _median_probs = lost_predictions_df.groupby('next_state').agg(probability_of_lost_median=('probability_of_lost', 'median'), count=('next_state', 'size')).sort_values('probability_of_lost_median', ascending=False).reset_index()
    _median_probs
    return


@app.cell
def _(lost_predictions_df):
    _median_probs = lost_predictions_df.groupby('clinic_name').agg(probability_of_lost_median=('probability_of_lost', 'median'), count=('clinic_name', 'size')).sort_values('probability_of_lost_median', ascending=False).reset_index()
    _median_probs
    return


@app.cell
def _(history):
    history['looks_older'].hist(bins=50)
    return


@app.cell
def _(grid_mode, lost_predictions_df, pd, preprocessing, sns):
    _bins = 30
    _n_levels = 30
    _min_max_scaler = preprocessing.MinMaxScaler()
    cols = ['payed_to_donor', 'num_transitions', 'seq_length', 'mean_state_duration', 'looks_older', 'probability_of_lost']
    _x_scaled = _min_max_scaler.fit_transform(lost_predictions_df[cols].values)
    _scaledSample = pd.DataFrame(data=_x_scaled, columns=cols)
    _cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=False)
    _g = sns.PairGrid(_scaledSample, diag_sharey=False)
    if grid_mode == 'kde':
        _g.map_lower(sns.kdeplot, n_levels=_n_levels, fill=True, cmap=_cmap)
        _g.map_upper(sns.kdeplot, n_levels=_n_levels, fill=True, cmap=_cmap)
        _g.map_diag(sns.kdeplot, fill=True)
    elif grid_mode == 'hist':
        _g.map_lower(sns.histplot, bins=_bins, pmax=0.8, cmap='Blues')
        _g.map_upper(sns.histplot, bins=_bins, pmax=0.8, cmap='Blues')
        _g.map_diag(sns.histplot, bins=_bins, kde=False)
    else:
        raise ValueError("mode must be either 'kde' or 'hist'")
    return


@app.cell
def _(plt, uzis_history):
    topN = 5
    is_visiting_our_and_other_company_company_counts = (
        uzis_history
        .groupby('clinic_company')
        .size()                      # count rows per company
        .reset_index(name='count')   # turn into DataFrame
        .sort_values('count', ascending=False)
        .head(topN)
    )

    plt.figure(figsize=(12,6))
    plt.bar(is_visiting_our_and_other_company_company_counts['clinic_company'], is_visiting_our_and_other_company_company_counts['count'])
    plt.xticks(rotation=90)
    plt.xlabel("Clinic Company")
    plt.ylabel("Number of Draws")
    plt.title("Number of Draws for donors visiting ours and also other companies (this draws have good potential to be converted back to us by marketing)")
    plt.show()

    print(is_visiting_our_and_other_company_company_counts.head().to_markdown())
    return (is_visiting_our_and_other_company_company_counts,)


@app.cell
def _(
    is_visiting_our_and_other_company_company_counts,
    our_company_name,
    uzis_history,
):
    whithout_our = is_visiting_our_and_other_company_company_counts[is_visiting_our_and_other_company_company_counts['clinic_company'] != our_company_name]
    _result = uzis_history[uzis_history['clinic_company'].isin(whithout_our['clinic_company'])].groupby(['id_person', 'firstname', 'surname', 'email', 'phone', 'birthdate', 'doctor_name', 'remove_as', 'profession', 'country_code'], as_index=False).agg({'clinic_name': lambda x: list(set(x))})
    _result['clinic_count'] = _result['clinic_name'].str.len()
    _result = _result.sort_values('clinic_count', ascending=False)
    print('This people visiting our and also other companies(clinic_name), we should motivate them to come only to us')
    print(_result.head().to_markdown())
    print(_result['clinic_name'].count())
    return


@app.cell
def _(engine, pd):
    recom_query = """
            SELECT
                id_recommendation,
                id_person,
                id_person_recommendation,
                referral,
                success_closed,
                closed_date
            FROM external.person_recommendation as rec
            ;
            """
    recom = pd.read_sql(recom_query, engine)
    return (recom,)


@app.cell
def _(recom):
    recom
    return


@app.cell
def _(recom):
    top_referrers = recom.groupby("id_person_recommendation").agg(
        total_referrals=("id_person", "count"),
        successful=("success_closed", "sum")
    ).sort_values(by="total_referrals", ascending=False)
    top_referrers
    return (top_referrers,)


@app.cell
def _(top_referrers):
    top_referrers.total_referrals.value_counts(normalize=True)
    return


@app.cell
def _(recom):
    recom['closed_month'] = recom['closed_date'].dt.to_period('W')
    conversion_by_month = recom.groupby('closed_month').agg(
        referrals=('referral', 'sum'),
        successes=('success_closed', 'sum')
    )
    conversion_by_month['conversion_rate'] = conversion_by_month['successes'] / conversion_by_month['referrals']
    return (conversion_by_month,)


@app.cell
def _(conversion_by_month):
    conversion_by_month.plot()
    return


@app.cell
def _():
    # import networkx as nx

    # recommender_graph = nx.DiGraph()
    # # _edges = recomTopByCount[['id_person_recommendation', 'id_person']].head(500).dropna().values
    # recommender_graph.add_edges_from(recom[['id_person_recommendation', 'id_person']].head(500).dropna().values)

    # plt.figure(figsize=(10, 10))
    # nx.draw(recommender_graph, with_labels=False, node_size=20, arrows=True)
    # plt.show()
    return


@app.cell
def _(plt):
    import networkx as nx

    def plot_top_nodes(G, top_nodes, layout='spring', figsize=(10, 10), seed=42, label_top=True):
        """
        Plot a subgraph of top_nodes and the people they point to (successors).

        Parameters
        ----------
        G : nx.DiGraph
            The recommendation graph.
        top_nodes : list
            List of node IDs (e.g., from out-degree, PageRank, etc.).
        layout : str
            Layout type: 'spring', 'kamada_kawai', 'circular', 'shell', or 'random'.
        figsize : tuple
            Size of the matplotlib figure.
        seed : int
            Random seed for layout reproducibility.
        label_top : bool
            Whether to show labels for the top_nodes.
        """
        # Collect neighbors (recommended people)
        neighbors = set()
        for n in top_nodes:
            neighbors.update(G.successors(n))

        sub_nodes = set(top_nodes) | neighbors
        subG = G.subgraph(sub_nodes)

        # Choose layout
        if layout == 'spring':
            pos = nx.spring_layout(subG, seed=seed)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(subG)
        elif layout == 'circular':
            pos = nx.circular_layout(subG)
        elif layout == 'shell':
            pos = nx.shell_layout(subG)
        elif layout == 'random':
            pos = nx.random_layout(subG)
        else:
            raise ValueError(f"Unsupported layout: {layout}")

        # Plot
        plt.figure(figsize=figsize)
        nx.draw_networkx_nodes(subG, pos, nodelist=top_nodes,
                               node_color="red", node_size=200, alpha=0.8, label="Top nodes")
        nx.draw_networkx_nodes(subG, pos, nodelist=list(neighbors),
                               node_color="skyblue", node_size=50, alpha=0.6, label="Neighbors")
        nx.draw_networkx_edges(subG, pos, arrows=True, alpha=0.4)

        if label_top:
            nx.draw_networkx_labels(subG, pos,
                                    labels={n: str(n) for n in top_nodes},
                                    font_size=8, font_color="black")

        plt.legend()
        plt.axis("off")
        plt.title(f"Top Nodes and Their Referrals ({layout} layout)")
        plt.show()

        return subG

    return nx, plot_top_nodes


@app.cell
def _(nx, plot_top_nodes, recom):
    recommender_graph = nx.DiGraph()
    recommender_graph.add_edges_from(recom[['id_person_recommendation', 'id_person']].dropna().values)

    # Out-degree top recommenders - how many people each person recommended
    out_degrees = dict(recommender_graph.out_degree())
    top_out = [n for n, _ in sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)[:200]]
    plot_top_nodes(recommender_graph, top_out, layout='spring')


    return


if __name__ == "__main__":
    app.run()
