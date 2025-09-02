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
    central_db = os.getenv("CENTRAL_DB_URL")

    from sqlalchemy import create_engine
    import psycopg2
    conn_str = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"
    engine = create_engine(conn_str)
    central_engine = create_engine(central_db)

    import lib.preprocessing as prep
    import lib.zts as zts
    import lib.hmm as hmm
    import lib.geo as geo
    import lib.lost_donors as lost_donors
    import lib.draw as draw_module

    population_mapper = draw_module.setup_population_mapper(engine)

    location_dimensions = {
        'prague': { 'longitude': (14,15), 'latitude': (49.7,50.3)}
    }

    import networkx as nx
    import marimo as mo
    return central_engine, engine, mo, nx, pd, plt


@app.cell
def _():
    timeDelta = 'week_period' # time scale for grouped data: date | week_period | month_period

    grid_mode = "hist"  # Plot type for grouped data grid "hist" or "kde"
    return


@app.cell
def _(central_engine, mo):
    central_client = mo.sql(
        f"""
        SELECT * FROM clients
        """,
        engine=central_engine
    )
    return (central_client,)


@app.cell
def _(engine, mo):
    donors = mo.sql(
        f"""
        SELECT * FROM donor.person;
        """,
        engine=engine
    )
    return (donors,)


@app.cell
def _(central_client, donors, pd):
    # Filter central_client by user_registered <= maximum donor created_at
    max_created_at = donors["created_at"].max()
    # Ensure both columns are datetime
    filtered_central = central_client.copy()
    if not pd.api.types.is_datetime64_any_dtype(filtered_central["user_registered"]):
        filtered_central["user_registered"] = pd.to_datetime(filtered_central["user_registered"])

        filtered_central = filtered_central[filtered_central["user_registered"] <= max_created_at]

    # Merge on hemo_id (central_client) and id (donors)
    merged_df = pd.merge(
        filtered_central,
        donors,
        how="outer",
        left_on="hemo_id",
        right_on="id",
        suffixes=("_central", "_donor")
    )

    merged_df
    return filtered_central, merged_df


@app.cell
def _(donors, filtered_central, merged_df):
    # Analysis of missing records after outer join

    # Total rows from each source before merging
    central_total = filtered_central.shape[0]
    donor_total = donors.shape[0]
    merged_total = merged_df.shape[0]

    missing_in_central = merged_df['id'].isna().sum()
    missing_in_donor = merged_df['hemo_id'].isna().sum()

    # Compute percentages
    pct_missing_central = (missing_in_central / central_total) * 100
    pct_missing_donor = (missing_in_donor / donor_total) * 100

    print(f"Central records total: {central_total}")
    print(f"CIS records total: {donor_total}")
    print(f"Merged records total: {merged_total}\n")

    print(f"Records present in central but missing in donors: {missing_in_central}"
          f" ({pct_missing_central:.2f}% of central records)")
    print(f"Records present in donors but missing in central: {missing_in_donor}"
          f" ({pct_missing_donor:.2f}% of donor records)")
    return


@app.cell
def _(donors):
    print("Presence of donors without uzis id")
    print(donors["id_uzis_rid"].isna().value_counts())
    print(donors["id_uzis_rid"].isna().value_counts(normalize=True))
    return


@app.cell
def _(engine, pd):
    recom_query = """
            SELECT
                id_recommendation,
                id_person,
                id_person_recommendation,
                id_client,
                id_client_recommendation,
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
    print("Presence of clients without id_hemo, about 10% is not pairable with hemo/cis")
    print(recom[["id_person", "id_person_recommendation"]].isna().value_counts())
    print(recom[["id_person","id_person_recommendation"]].isna().value_counts(normalize=True))
    return


@app.cell
def _(recom):
    top_referrers = recom.groupby("id_client").agg(
            total_referrals=("id_client", "count"),
            successful=("success_closed", "sum")
        ).sort_values(by="total_referrals", ascending=False)
    top_referrers
    return (top_referrers,)


@app.cell
def _(top_referrers):
    top_referrers.total_referrals.value_counts().plot()
    return


@app.cell
def _(recom):
    recom['closed_week'] = recom['closed_date'].dt.to_period('W')
    conversion_by_month = recom.groupby('closed_week').agg(
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
def _(nx, plt):
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
    return (plot_top_nodes,)


@app.cell
def _(nx, recom):
    recommender_graph = nx.DiGraph()
    recommender_graph.add_edges_from(recom[['id_client', 'id_client_recommendation']].dropna().values)
    return (recommender_graph,)


@app.cell
def _(plot_top_nodes, recommender_graph):
    # Out-degree top recommenders - how many people each person recommended
    out_degrees = dict(recommender_graph.out_degree())
    top_out = [n for n, _ in sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)[:100]]
    plot_top_nodes(recommender_graph, top_out, layout='spring')
    return


@app.cell
def _(nx, plot_top_nodes, recommender_graph):
    # PageRank - Influence of each person in the recommendation network, considering both the quantity and quality of recommendations.
    page_rank = nx.pagerank(recommender_graph)
    top_page = [n for n, _ in sorted(page_rank.items(), key=lambda x: x[1], reverse=True)[:100]]
    plot_top_nodes(recommender_graph, top_page, layout='spring')
    return


if __name__ == "__main__":
    app.run()
