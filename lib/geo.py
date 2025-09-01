# geo
import geopandas as gpd
from locations.population_density.population_map import CzechPopulationMapper

_czech_borders = {}

## czech rep borders
## download here https://raw.githubusercontent.com/siwekm/czech-geojson/master/czech_republic.json
_czech_borders["data"] = gpd.read_file("./locations/czech_rep_map.json")\

def plot_czech_boundaries(ax):
    _czech_borders["data"].boundary.plot(ax=ax, edgecolor="black", linewidth=1)

def zoom(ax, what = None):
    if (what is None):
        print("Nothing to zoom")
        return;
    ax.set_xlim(what['longitude'])
    ax.set_ylim(what['latitude'])

def zoom_df(df, what, lon_col="longitude", lat_col="latitude"):
    """
    Filter a DataFrame to only include rows within the specified rectangle.

    Args:
        df: pandas DataFrame containing longitude and latitude columns
        what: dict with keys 'longitude' and 'latitude', each a [min, max] list
        lon_col: name of longitude column
        lat_col: name of latitude column

    Returns:
        Filtered DataFrame with only rows inside the rectangle
    """
    if what is None:
        return df.copy()  # no filtering

    filtered = df[
        (df[lon_col] >= what['longitude'][0]) &
        (df[lon_col] <= what['longitude'][1]) &
        (df[lat_col] >= what['latitude'][0]) &
        (df[lat_col] <= what['latitude'][1])
    ]
    return filtered


if __name__ == "__main__":
    pass