import os 
import json
import pandas as pd

# Directory
with open(os.path.join("../data/pond/directory.json"), "r") as f:
    paper_info = json.load(f)

registered_titles = [entry['title'] for entry in paper_info.values()]
registered_titles.sort()


pond_data = pd.read_csv("data/pond/pond_data.csv", encoding_errors='ignore')
pond_data = pond_data.loc[pond_data.title.isin(registered_titles)]
pond_data = pond_data.reset_index(drop=True)

pond_data = pond_data.loc[:,['author', 'title', 'pondname', 'location', 'author_term',
            'max_depth_m', 'mean_surfacearea_m2', 'macrophytes_percentcover', 'ph', 'tn_ugpl', 'tp_ugpl', 'chla_ugpl']]
pond_data.columns = ['author', 'title', 'name', 'location', 'ecosystem',
            'max_depth', 'surface_area', 'vegetation_cover', 'ph', 'tn', 'tp', 'chla']

# Split the dataframe's rows so that each measurement is in its own row
pond_data = pond_data.melt(id_vars=['author', 'title', 'name', 'location', 'ecosystem'], 
                       value_vars=['max_depth', 'surface_area', 'vegetation_cover', 'ph', 'tn', 'tp', 'chla'],
                       var_name='attribute', value_name='value')
pond_data = pond_data.dropna(subset=['value'])
pond_data = pond_data.reset_index(drop=True)
n_entries = pond_data.shape[0]

pond_data['date'] = None
pond_data['state'] = None

pond_data = pond_data.loc[:, ['author', 'title', 'name', 'location', 'ecosystem', 'date', 'state', 'attribute', 'value']]

pond_data.to_csv(os.path.join("data/pond/pond_data_cleaned.csv"), index=False)