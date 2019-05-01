# TODO: MissingValues
# TODO: Non-unique names
# TODO: Clean up graphs
# %%
from collections import Counter
import chardet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import chardet
import plotly.tools as tls
import plotly.offline as py
from plotly.offline import init_notebook_mode, iplot, plot
import plotly.graph_objs as go
from scipy.stats import ttest_ind

init_notebook_mode(connected=True)
sns.set(context='notebook', style='darkgrid', palette='deep',
        font='sans-serif', color_codes=False, rc={'savefig.facecolor': 'white', 'figure.figsize': (20, 20)})
# guessing encoding of data

# %%
with open("./kickstarter-projects/ks-projects-201801.csv", 'rb') as rawdata:
    encoding = chardet.detect(rawdata.read(20000))

encoding
# %%
projects = pd.read_csv(
    filepath_or_buffer='./kickstarter-projects/ks-projects-201801.csv')

projects.describe()

# %%
projects.nunique()

# %%
projects.isnull().sum()
# TODO: is it ok to just drop na
projects = projects.dropna()
# %%
unique_names = projects['name'].value_counts()
non_unique_names = unique_names[unique_names > 1]
non_unique_projects = projects.loc[projects['name'].isin(
    non_unique_names.index)]

# %%
projects_success_rates = round(
    projects['state'].value_counts()/len(projects['state'])*100, 2)
projects_success_rates = {
    'state': projects_success_rates.index, 'values': projects_success_rates.values}
projects_success_rates = pd.DataFrame(projects_success_rates)

# Bar Plot of state breakdown
projects_state_bar = sns.catplot(
    x='state', y='values', kind='bar', data=projects_success_rates)
projects_state_bar.savefig(
    './Figures/Exploration/projects_state_bar.jpg', dpi=300)
# 52.2% failed
# 35.4% success
# Ignore other values for classification


# %%
# Distribution of Main Categories
dist_mc_bar = sns.catplot(
    x='main_category', kind='count', data=projects,
    order=projects['main_category'].value_counts().index)
dist_mc_bar.set_xticklabels(rotation=90)
dist_mc_bar.savefig('./Figures/Exploration/MainCat_dist.jpg', dpi=300)
# %%
sns.set(font_scale=0.4)
dist_sc_bar = sns.catplot(x='category', kind='count', data=projects,
                          order=projects['category'].value_counts().nlargest(50).index)
dist_sc_bar.set_xticklabels(rotation=90)
dist_sc_bar.savefig('./Figures/Exploration/SubCat_dist.jpg', dpi=300)
# %%
# Missing Data
projects = projects.dropna(subset=['name'])
# %%
# Is there a differenc in the number of words used as the success of the project
# Calculate mean number of words used in successful projects
# split projects
projects['num_Words'] = projects['name'].apply(lambda x: len(x.split()))
successful_projects = projects[projects['state'] == 'successful']
failed_projects = projects[projects['state'] == 'failed']
# %%
mean_num_word_success = np.mean(successful_projects['num_Words'])
mean_num_word_failed = np.mean(failed_projects['num_Words'])
# %%
ttest_ind(failed_projects['num_Words'], successful_projects['num_Words'])
# %%
projects.to_hdf('store_projects.h5', 'table')
# %%
projects.groupby(['main_category', 'state']).size()

# %%
projects['launched'] = pd.to_datetime(
    projects['launched'], format='%Y-%m-%d %H:%M:%S')

# %%
projects['launched'].dt.day

# %%
projects = projects[(projects['state'] == 'failed') |
                    (projects['state'] == 'successful')]
out_side_range = projects[(projects['usd_goal_real'] < 1000) | (
    projects['usd_goal_real'] > 100000)]


# %%
# %%
projects['Launch_Quarter'] = projects['launched'].dt.quarter

# %%
projects['Launch_Quarter']

# %%
a = projects.drop('state', axis=1)

# %%
