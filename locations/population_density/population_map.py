import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
import requests
import zipfile
import os

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("viridis")

class CzechPopulationMapper:
    def __init__(self):
        self.base_dir = os.path.dirname(__file__)
        self.data_dir = os.path.join(self.base_dir, "czech_population_data")
        os.makedirs(self.data_dir, exist_ok=True)
        self.df = None

    def download_worldpop_data(self):
        """Download WorldPop Czech Republic population density data (CSV inside ZIP)"""
        print("Downloading WorldPop Czech Republic data...")

        worldpop_url = "https://data.worldpop.org/GIS/Population_Density/Global_2000_2020_1km_UNadj/2020/CZE/cze_pd_2020_1km_UNadj_ASCII_XYZ.zip"

        try:
            response = requests.get(worldpop_url)
            if response.status_code == 200:
                zip_path = os.path.join(self.data_dir, "worldpop_czech.zip")
                with open(zip_path, "wb") as f:
                    f.write(response.content)

                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(self.data_dir)
                print("✅ WorldPop data downloaded and extracted successfully")
                return True
            else:
                print(f"❌ Failed to download WorldPop data: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Error downloading WorldPop data: {e}")
            return False

    def prepare_data(self):
        """Load Czech population CSV into DataFrame"""
        csv_files = [f for f in os.listdir(self.data_dir) if f.endswith(".csv")]
        if not csv_files:
            print("❌ No CSV file found in data directory")
            return

        csv_path = os.path.join(self.data_dir, csv_files[0])
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)
        df = df.rename(columns={"X": "longitude", "Y": "latitude", "Z": "density"})
        df = df[df["density"] > 0]

        self.df = df
        print(f"✅ Loaded DataFrame with {len(df)} rows")

    def create_population_density_map_grid(self, ax, what = None):
        """Plot WorldPop Czech population density on given matplotlib Axes using DataFrame"""

        if self.df is None or self.df.empty:
            print("❌ No data available, run prepare_data() first")
            return

        if (what is None):
            dataset = self.df;
        else:
            dataset = self.df[
                (self.df["longitude"] > what['longitude'][0]) &
                (self.df["longitude"] < what['longitude'][1]) &
                (self.df["latitude"] > what['latitude'][0]) &
                (self.df["latitude"] < what['latitude'][1])
            ]
        sc = ax.scatter(dataset["longitude"], dataset["latitude"], c=dataset["density"],
                        cmap='YlOrRd', norm=LogNorm(vmin=1, vmax=self.df["density"].max()),
                        s=2, alpha=0.8)
        # plt.colorbar(sc, ax=ax, shrink=0.6)

    def create_population_density_map_image(self, ax, what=None):
        if self.df is None or self.df.empty:
            print("❌ No data available, run prepare_data() first")
            return

        # Optionally filter
        if what is not None:
            dataset = self.df[
                (self.df["longitude"] > what['longitude'][0]) &
                (self.df["longitude"] < what['longitude'][1]) &
                (self.df["latitude"] > what['latitude'][0]) &
                (self.df["latitude"] < what['latitude'][1])
            ]
        else:
            dataset = self.df

        # Pivot to grid (lat, lon -> density matrix)
        pivoted = dataset.pivot(index="latitude", columns="longitude", values="density")
        lon = pivoted.columns.values
        lat = pivoted.index.values
        Z = pivoted.values

        # Show as image
        im = ax.pcolormesh(lon, lat, Z,
                        cmap="YlOrRd",
                        norm=LogNorm(vmin=1, vmax=self.df["density"].max()),
                        shading="auto")
        return im


# Usage example
if __name__ == "__main__":
    mapper = CzechPopulationMapper()

    # mapper.download_worldpop_data()  # run once
    mapper.prepare_data()

    fig, ax = plt.subplots(figsize=(10, 8))
    mapper.create_population_density_map(ax)
    plt.show()
