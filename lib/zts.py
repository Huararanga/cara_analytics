# ZTS list creation and maintenance, plots
import re
import pandas as pd
import ipywidgets as widgets
from IPython.display import display, clear_output

import re
import pandas as pd
import ipywidgets as widgets
from IPython.display import display, clear_output

import json
import re
import pandas as pd
import geopandas as gpd
from dotenv import load_dotenv
import os

import matplotlib.pyplot as plt
from sqlalchemy import create_engine

import seaborn as sns

import requests
from io import BytesIO

import ipywidgets as widgets
from IPython.display import display, clear_output

from data_ingestion.data_sources.zts_list.schema import ZTSList

class InteractiveLocationMatcher:
	def __init__(self, ztsList: pd.DataFrame, locations: pd.DataFrame):
		self.ztsList = ztsList.copy()
		self.locations = locations[['postal_code', 'place_name', 'latitude', 'longitude']].copy()
		self.current_index = 0

		# Matching state
		self.matches = {}  # index -> location dict or None
		self.company_matches = {}  # index -> company string or None
		self.company_options = set()

		# Widgets
		self.facility_label = widgets.HTML(value="")
		self.progress_label = widgets.HTML(value="")

		# Location widgets
		self.search_box = widgets.Text(
			description="Search:",
			placeholder="Type to search locations...",
			layout=widgets.Layout(width='800px')
		)
		self.location_dropdown = widgets.Dropdown(
			description="Location:",
			options=[],
			layout=widgets.Layout(width='800px')
		)

		# Company widgets
		self.company_search_box = widgets.Text(
			description="Company:",
			placeholder="Type to search or add...",
			layout=widgets.Layout(width='800px')
		)
		self.company_dropdown = widgets.Dropdown(
			description="Select:",
			options=[],
			layout=widgets.Layout(width='800px')
		)
		self.add_company_button = widgets.Button(description="Add Company")
		self.edit_companies_button = widgets.Button(description="Edit Companies")

		# Controls
		self.prev_button = widgets.Button(description="Previous")
		self.match_button = widgets.Button(description="Match")
		self.skip_button = widgets.Button(description="Skip")
		self.next_button = widgets.Button(description="Next")
		self.output_area = widgets.Output()

		# Wire events
		self.search_box.observe(self.on_search_change, names='value')
		self.company_search_box.observe(self.on_company_search_change, names='value')
		self.add_company_button.on_click(self.on_add_company)
		self.edit_companies_button.on_click(self.on_edit_companies)
		self.prev_button.on_click(self.on_prev)
		self.next_button.on_click(self.on_next)
		self.match_button.on_click(self.on_match)
		self.skip_button.on_click(self.on_skip)

		# Initialize state
		self._prime_company_options_from_df()
		self._load_existing_state_from_df()
		self.results_df = self._create_results_df()  # always available

		# Show first record
		self._update_display()

	def _prime_company_options_from_df(self):
		# Load company options from ztsList.company column
		if 'company' in self.ztsList.columns:
			values = self.ztsList['company'].dropna().astype(str).str.strip()
			self.company_options.update(v for v in values if v)
			print(f"📁 Loaded {len(self.company_options)} companies from ztsList.company")

	def _load_existing_state_from_df(self):
		# Rehydrate location matches from 'city' if available
		if 'city' in self.ztsList.columns:
			place_to_row = {str(r['place_name']).strip().lower(): r.to_dict() for _, r in self.locations.iterrows()}
			for idx, row in self.ztsList.iterrows():
				val = row['city']
				if pd.notna(val) and str(val).strip().lower() in place_to_row:
					self.matches[idx] = place_to_row[str(val).strip().lower()]

		# Rehydrate company matches from ztsList.company
		if 'company' in self.ztsList.columns:
			for idx, row in self.ztsList.iterrows():
				val = row['company']
				if pd.notna(val) and str(val).strip():
					name = str(val).strip()
					self.company_matches[idx] = name
					# Don't add to options here since _prime_company_options_from_df handles it
			print(f"�� Loaded {len([v for v in self.company_matches.values() if v])} existing company matches")

	def _create_results_df(self) -> pd.DataFrame:
		results = []
		for idx, row in self.ztsList.iterrows():
			loc = self.matches.get(idx)
			comp = self.company_matches.get(idx)
			results.append({
				'name': row['name'],
				'city': (loc or {}).get('place_name'),
				'city_postal_code': (loc or {}).get('postal_code'),
				'city_latitude': (loc or {}).get('latitude'),
				'city_longitude': (loc or {}).get('longitude'),
				'company': comp if comp else None
			})
		return pd.DataFrame(results)

	def _update_results_df(self):
		self.results_df = self._create_results_df()

	def on_search_change(self, change):
		query = (change['new'] or '').lower()
		if query:
			mask = self.locations['place_name'].str.lower().str.contains(query, na=False)
			filtered = self.locations[mask].head(200)
		else:
			filtered = self.locations.head(200)

		options = []
		for _, r in filtered.iterrows():
			label = f"{r['place_name']} ({r['postal_code']})  [{r['latitude']:.5f}, {r['longitude']:.5f}]"
			options.append((label, r.to_dict()))
		self.location_dropdown.options = options
		self.location_dropdown.value = options[0][1] if options else None

	def on_company_search_change(self, change):
		query = (change['new'] or '').lower()
		if query:
			cands = [c for c in self.company_options if query in c.lower()]
		else:
			cands = list(self.company_options)
		cands_sorted = sorted(set(cands))
		self.company_dropdown.options = [(c, c) for c in cands_sorted]
		self.company_dropdown.value = cands_sorted[0] if cands_sorted else None

	def on_add_company(self, _):
		text = self.company_search_box.value.strip()
		if text:
			self.company_options.add(text)
			self.company_matches[self.current_index] = text
			self.company_search_box.value = ""
			self._refresh_company_dropdown(preserve=text)

	def on_edit_companies(self, _):
		with self.output_area:
			clear_output()
			edit_widget = widgets.Textarea(
				value='\n'.join(sorted(self.company_options)),
				description="Companies:",
				layout=widgets.Layout(width='100%', height='220px')
			)
			save_button = widgets.Button(description="Save")

			def _save(_b):
				lines = [ln.strip() for ln in edit_widget.value.split('\n')]
				self.company_options = {ln for ln in lines if ln}
				self._refresh_company_dropdown()
				clear_output()

			save_button.on_click(_save)
			display(widgets.VBox([edit_widget, save_button]))

	def _refresh_company_dropdown(self, preserve: str | None = None):
		opts = sorted(self.company_options)
		self.company_dropdown.options = [(c, c) for c in opts]
		if preserve and preserve in self.company_options:
			self.company_dropdown.value = preserve
		else:
			self.company_dropdown.value = opts[0] if opts else None

	def on_match(self, _):
		# Save location
		if self.location_dropdown.value:
			self.matches[self.current_index] = self.location_dropdown.value
		# Save company
		if self.company_dropdown.value:
			self.company_matches[self.current_index] = self.company_dropdown.value

		self._update_results_df()
		self._next_or_finish()

	def on_skip(self, _):
		self.matches[self.current_index] = None
		self.company_matches[self.current_index] = None
		self._update_results_df()
		self._next_or_finish()

	def on_prev(self, _):
		if self.current_index > 0:
			self.current_index -= 1
			self._update_display()

	def on_next(self, _):
		if self.current_index < len(self.ztsList) - 1:
			self.current_index += 1
			self._update_display()

	def _next_or_finish(self):
		if self.current_index < len(self.ztsList) - 1:
			self.current_index += 1
			self._update_display()
		else:
			self._finish()

	def _update_display(self):
		row = self.ztsList.iloc[self.current_index]
		facility_name = str(row['name'])

		# Header and progress
		matched_loc = sum(v is not None for v in self.matches.values())
		matched_co = sum(v is not None for v in self.company_matches.values())
		skipped = sum(v is None for v in self.matches.values())
		total = len(self.ztsList)
		done = matched_loc + skipped
		self.facility_label.value = f"<h3>Facility {self.current_index + 1} / {total}</h3><p><b>{facility_name}</b></p>"
		self.progress_label.value = f"<p>Progress: {done}/{total} processed, {matched_loc} with location, {matched_co} with company</p>"

		# Prefill search with FULL facility name
		self.search_box.value = facility_name

		# Restore current selections if present
		if self.current_index in self.matches and self.matches[self.current_index]:
			loc = self.matches[self.current_index]
			label = f"{loc['place_name']} ({loc['postal_code']})  [{loc['latitude']:.5f}, {loc['longitude']:.5f}]"
			self.location_dropdown.options = [(label, loc)]
			self.location_dropdown.value = loc
		else:
			# Trigger search handler to populate options from full name
			self.on_search_change({'new': self.search_box.value})

		# Update company dropdown with all available options
		self._refresh_company_dropdown()
		
		# Restore current company selection if exists
		if self.current_index in self.company_matches and self.company_matches[self.current_index]:
			cur = self.company_matches[self.current_index]
			self.company_dropdown.value = cur
		else:
			# If no existing match, try to pre-select based on current facility's company
			current_company = row.get('company')
			if pd.notna(current_company) and str(current_company).strip():
				company_name = str(current_company).strip()
				if company_name in self.company_options:
					self.company_dropdown.value = company_name

	def _finish(self):
		with self.output_area:
			clear_output()
			print("Matching completed.")
			print(f"Total: {len(self.ztsList)}")
			print(f"With location: {sum(v is not None for v in self.matches.values())}")
			print(f"With company:  {sum(v is not None for v in self.company_matches.values())}")
			print("The latest results are available in matcher.results_df")

	def display(self):
		location_box = widgets.VBox([widgets.HTML("<b>Location</b>"), self.search_box, self.location_dropdown])
		company_box = widgets.VBox([widgets.HTML("<b>Company</b>"), self.company_search_box, self.company_dropdown, widgets.HBox([self.add_company_button, self.edit_companies_button])])

		ui = widgets.VBox([
			self.facility_label,
			self.progress_label,
			widgets.HBox([location_box, company_box]),
			widgets.HBox([self.prev_button, self.match_button, self.skip_button, self.next_button]),
			self.output_area
		])
		display(ui)
				
def save_zts_list_to_json(ztsList, filepath="./competitors/list/zts_list.json"):
    """
    Save ztsList DataFrame to JSON file
    
    Args:
        ztsList: pandas DataFrame to save
        filepath: path where to save the JSON file (default: ./competitors/list/zts_list.json)
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Convert DataFrame to JSON
        # Use orient='records' to create a list of dictionaries (one per row)
        json_data = ztsList.to_json(orient='records', force_ascii=False, indent=2)
        
        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(json_data)
        
        print(f"Successfully saved ztsList to {filepath}")
        print(f"File size: {os.path.getsize(filepath)} bytes")
        print(f"Number of records: {len(ztsList)}")
        
        return True
        
    except Exception as e:
        print(f"Error saving ztsList to JSON: {e}")
        return False
    
def load_zts_list_from_url(url="https://www.uzis.cz/res/file/registry/nrovdk/nrovdk-ciselnik-zts-sukl-kody.xlsx", 
                                   column_names=['name', 'ICO', 'id_SUKL', 'clinic_code'],
                                   timeout=30,
                                   verify_ssl=True):
    """
    Advanced function to download and load ZTS list from URL
    
    Args:
        url: URL of the Excel file to download
        column_names: list of column names to assign
        timeout: request timeout in seconds
        verify_ssl: whether to verify SSL certificates
        
    Returns:
        pandas DataFrame with ZTS list data or None if error
    """
    try:
        print(f"📥 Downloading Excel file from: {url}")
        response = requests.get(url, timeout=timeout, verify=verify_ssl)
        response.raise_for_status()
        
        print("�� Loading Excel file into DataFrame...")
        excel_data = BytesIO(response.content)
        ztsList = pd.read_excel(excel_data)
        
        # Set column names if provided
        if column_names and len(column_names) == len(ztsList.columns):
            ztsList = ztsList.set_axis(column_names, axis=1)
        
        print(f"✅ Successfully loaded DataFrame with shape: {ztsList.shape}")
        # print(f"�� Columns: {list(ztsList.columns)}")
        # print(f"�� First few rows:")
        # print(ztsList.head())
        
        return ztsList
        
    except requests.exceptions.Timeout:
        print(f"❌ Timeout error: Request took longer than {timeout} seconds")
        return None
    except requests.exceptions.RequestException as e:
        print(f"❌ Error downloading file: {e}")
        return None
    except Exception as e:
        print(f"❌ Error loading Excel file: {e}")
        return None
	
def load_zts_list_from_db(engine) -> pd.DataFrame:
    return pd.read_sql(ZTSList.__table__.select(), engine)
    
def load_and_verify_zts_list(json_filepath="./competitors/list/zts_list.json", 
                                     show_differences=True):
    """
    Detailed version with more comparison options
    
    Args:
        json_filepath: path to the JSON file
        show_differences: whether to show detailed differences in output
        
    Returns:
        pandas DataFrame from JSON if verification passes
    """
    try:
        # Load from JSON file
        print("📁 Loading ztsList from JSON file...")
        with open(json_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        json_ztsList = pd.DataFrame(data)
        
        # Load from URL
        print("🌐 Loading ztsList from URL for verification...")
        url_ztsList = load_zts_list_from_url()
        
        if url_ztsList is None:
            raise ValueError("Failed to load data from URL for verification")
        
        # Detailed comparison
        json_length = len(json_ztsList)
        url_length = len(url_ztsList)
        
        print(f"📊 Detailed comparison:")
        print(f"   JSON file: {json_length} records")
        print(f"   URL data:  {url_length} records")
        
        # Compare all columns if they exist
        json_names = set(json_ztsList['name'].str.strip())
        url_names = set(url_ztsList['name'].str.strip())
        
        only_in_json = json_names - url_names
        only_in_url = url_names - json_names
        common_names = json_names & url_names
        
        print(f"   Common names: {len(common_names)}")
        print(f"   Only in JSON: {len(only_in_json)}")
        print(f"   Only in URL: {len(only_in_url)}")
        
        # Check consistency
        if json_length != url_length or only_in_json or only_in_url:
            error_msg = "Data inconsistency detected:\n"
            
            if json_length != url_length:
                error_msg += f"Length mismatch: JSON={json_length}, URL={url_length}\n"
            
            if show_differences:
                if only_in_json:
                    error_msg += f"\nOnly in JSON ({len(only_in_json)}):\n"
                    for name in sorted(only_in_json):
                        error_msg += f"  - {name}\n"
                
                if only_in_url:
                    error_msg += f"\nOnly in URL ({len(only_in_url)}):\n"
                    for name in sorted(only_in_url):
                        error_msg += f"  - {name}\n"
            
            raise ValueError(error_msg)
        
        print("✅ Verification successful! Data is consistent.")
        return json_ztsList
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise


    
def export_zts_list(json_filepath="./competitors/list/zts_list.json", 
                           include_timestamp=False,
                           compression='snappy'):
    """
    Advanced export function with additional options
    
    Args:
        json_filepath: path to the JSON file
        include_timestamp: whether to include timestamp in filenames
        compression: parquet compression method ('snappy', 'gzip', 'brotli')
        
    Returns:
        tuple: (xlsx_path, parquet_path)
    """
    try:
        # Load verified ztsList
        print("📁 Loading and verifying ztsList...")
        ztsList = load_and_verify_zts_list(json_filepath)
        
        # Get base directory and filename
        base_dir = os.path.dirname(json_filepath)
        base_name = os.path.splitext(os.path.basename(json_filepath))[0]
        
        # Add timestamp if requested
        if include_timestamp:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"{base_name}_{timestamp}"
        
        # Define output paths
        xlsx_path = os.path.join(base_dir, f"{base_name}.xlsx")
        parquet_path = os.path.join(base_dir, f"{base_name}.parquet")
        
        # Create directory if it doesn't exist
        os.makedirs(base_dir, exist_ok=True)
        
        # Export to XLSX with formatting
        print(f"�� Exporting to XLSX: {xlsx_path}")
        with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
            ztsList.to_excel(writer, sheet_name='ZTS_List', index=False)
            
            # Auto-adjust column widths
            worksheet = writer.sheets['ZTS_List']
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width
        
        # Export to Parquet with compression
        print(f"📊 Exporting to Parquet: {parquet_path}")
        ztsList.to_parquet(parquet_path, index=False, compression=compression)
        
        # Verify file sizes
        xlsx_size = os.path.getsize(xlsx_path)
        parquet_size = os.path.getsize(parquet_path)
        
        print("✅ Export completed successfully!")
        print(f"📁 XLSX file: {xlsx_path} ({xlsx_size:,} bytes)")
        print(f"�� Parquet file: {parquet_path} ({parquet_size:,} bytes)")
        print(f"📊 Records exported: {len(ztsList):,}")
        print(f"�� Compression ratio: {xlsx_size/parquet_size:.2f}x smaller in Parquet")
        
        return xlsx_path, parquet_path
        
    except Exception as e:
        print(f"❌ Error during export: {e}")
        raise

def plot_cumulative_market_share(ztsList: pd.DataFrame):
    # Sort by clinic_code
    df = ztsList.copy()
    
    # Prepare cumulative counts
    companies = df["company"].unique()
    cumulative_counts = []

    company_totals = {c: 0 for c in companies}
    for _, row in df.iterrows():
        company_totals[row["company"]] += 1
        cumulative_counts.append(company_totals.copy())

    cum_df = pd.DataFrame(cumulative_counts)
    
    # Convert counts to proportions
    proportions = cum_df.div(cum_df.sum(axis=1), axis=0)
    proportions.index = df["clinic_code"]

    # Plot stacked area chart
    fig, ax = plt.subplots(figsize=(14, 6))
    proportions.plot(
        kind="area",
        stacked=True,
        ax=ax,
        cmap=sns.color_palette("tab20", as_cmap=True)
    )
    
    ax.set_ylabel("Proportion")
    ax.set_xlabel("Clinic Code")
    ax.set_title("Cumulative Market Share by Company")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

def plot_cumulative_market_share_top15(ztsList: pd.DataFrame):
    # Find top 15 companies by total clinic count
    top15 = (
        ztsList["company"]
        .value_counts()
        .nlargest(15)
        .index
    )
    
    # Replace non-top companies with "Other"
    df = ztsList.copy()
    df["company"] = df["company"].where(df["company"].isin(top15), "Other")
    
    # cannot sort due ZTS1001 in the middle of table...
    # Sort by clinic_code
    # df = df.sort_values("clinic_code").reset_index(drop=True)
    
    # Prepare cumulative counts
    companies = df["company"].unique()
    cumulative_counts = []
    company_totals = {c: 0 for c in companies}

    for _, row in df.iterrows():
        company_totals[row["company"]] += 1
        cumulative_counts.append(company_totals.copy())

    cum_df = pd.DataFrame(cumulative_counts)
    
    # Convert counts to proportions
    proportions = cum_df.div(cum_df.sum(axis=1), axis=0)
    proportions.index = df["clinic_code"]

    # Plot stacked area chart
    fig, ax = plt.subplots(figsize=(14, 6))
    proportions.plot(
        kind="area",
        stacked=True,
        ax=ax,
        color=sns.color_palette("tab20", n_colors=len(companies))
    )
    
    ax.set_ylabel("Proportion")
    ax.set_xlabel("Clinic Code")
    ax.set_title("Cumulative Market Share by Company (Top 15)")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

def get_top_zts_geo_dataframe(ztsList, n_top=15):
    topCompanies = ztsList["company"].value_counts().nlargest(n_top).index
    df = ztsList.copy()
    df["company_grouped"] = df["company"].where(df["company"].isin(topCompanies), "Other")

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["city_longitude"], df["city_latitude"]),
        crs="EPSG:4326"
    )
    return gdf

def plot_zts_map(gdf, ax, legend=True):
    # Derive company order from data
    topCompanies = [c for c in gdf["company_grouped"].unique() if c != "Other"]
    topCompanies.sort()  # optional, for consistent ordering

    # Choose a palette: distinct colors for top companies + grey for Other
    palette = sns.color_palette("tab20", n_colors=len(topCompanies))
    palette.append((0.5, 0.5, 0.5))  # append grey for "Other"
    company_order = topCompanies + ["Other"]
    color_map = dict(zip(company_order, palette))

    # Plot clinic points
    for company, group in gdf.groupby("company_grouped"):
        group.plot(
            ax=ax,
            markersize=50,
            marker="o",
            color=color_map[company],
            label=company,
            alpha=0.7
        )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    if legend == True:
        ax.legend(title="Company (top + Other)", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    # plt.show()
    return ax
# Usage examples:

# Create and display the interactive matcher,
# print("Creating interactive location matcher...")
# locations, locations_unique = load_locations()
# matcher = InteractiveLocationMatcher(ztsList, locations)
# matcher.display()

# print("\nInstructions:")
# print("1. For each facility name, search for the corresponding city in the search box, copy paste to google, to find district. Cleanup city to find coresponding locations")
# print("2. Select the correct location from the dropdown")
# print("3. Click 'Match' to save the selection and move to next facility")
# print("4. Click 'Skip' if no match is found")
# print("5. Use 'Previous' and 'Next' to navigate between facilities")
# print("6. When finished, the results will be available as 'matcher.results_df'")

# editor which loads current data from json, but 
# locations, locations_unique = load_locations()
# ztsList = load_and_verify_zts_list(ZTS_LIST_FILE_PATH)
# matcher = InteractiveLocationMatcher(ztsList, locations)
# matcher.display()

# this is probably no more needed
# ztsList = ztsList.drop(labels=['city', 'city_postal_code', 'city_latitude', 'city_longitude', 'company'], axis=1)
# ztsList = pd.concat([
#     ztsList, 
#     matcher.results_df[['city', 'company','city_postal_code', 'city_latitude', 'city_longitude']]
# ], axis=1)

# Basic export
# xlsx_path, parquet_path = export_zts_list(ZTS_LIST_FILE_PATH)

# Advanced export with timestamp
# xlsx_path, parquet_path = export_zts_list(
#     ZTS_LIST_FILE_PATH, 
#     include_timestamp=True, 
#     compression='gzip'
# )
