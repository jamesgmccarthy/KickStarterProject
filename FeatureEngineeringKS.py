import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder


def prepare_data(data):
    """Reads in data,
    drops unneccessary columns
    selects only successful and failed df,
    drops missing data in name


    Arguments:
        data: csv file name

    Returns:
        df: Pandas DataFrame
    """
    # Read in data
    df = pd.read_csv(data)
    # TODO: Name
    # drop unneccessary columns (temp drop name)
    # 'usd_pledged_real',

    df = df.drop(
        ['pledged', 'usd pledged', 'goal'], axis=1)

    # Set ID as Index of dataset
    df = df.set_index("ID")
    # select only successful and failed df
    df = df[(df['state'] == 'failed') |
            (df['state'] == 'successful')]
    df['state'] = df['state'].map({
        'failed': 0,
        'successful': 1
    })
    # Drop missing data in 'name' column
    df = df.dropna(subset=['name'])
    return df


def create_features_date(df):
    """Feature engineering creating new features concerning dates of df
    Project duration ie. Deadline - Launched
    Launch Year
    Launch Quarter
    Launch Month
    Launch Week
    Launch Day

    Arguments:
        df: Pandas DataFrame
    """
    # Put Launched and Deadline into datetime format
    df['launched'] = pd.to_datetime(
        df['launched'], format='%Y-%m-%d %H:%M:%S')
    df['deadline'] = pd.to_datetime(
        df['deadline'], format='%Y-%m-%d %H:%M:%S')

    # Duration feature
    df['duration'] = df['deadline'].subtract(df['launched'])
    df['duration'] = df['duration'].astype('timedelta64[D]')

    # Launch Year
    df['launch_year'] = df['launched'].dt.year

    # Launch Quarter
    df['launch_quarter'] = df['launched'].dt.quarter

    # Launch Month
    df['launch_month'] = df['launched'].dt.month

    # Launch week
    df['launch_week'] = df['launched'].dt.week

    # Launch Day
    df['launch_day'] = df['launched'].dt.day

    return df

# TODO: Complete sentiment


def create_features_name(df):
    """Creates features concerning the name of the project
    name_len
    name_excl
    name_qm
    name_colons
    name_num_words
    name_num_words_upper
    name_num_words_upper_perc

    Arguments:
        df {[type]} -- [description]
    """
    # Code adapted from https://www.kaggle.com/shivamb/an-insightful-story-of-crowdfunding-projects
    df['name_len'] = df['name'].apply(lambda x: len(x))
    df['name_excl'] = df['name'].apply(lambda x: 1 if '!' in x else 0)
    df['name_qm'] = df['name'].apply(lambda x: 1 if '?' in x else 0)
    df['name_colons'] = df['name'].apply(lambda x: 1 if ':' or ';' in x else 0)
    df['name_num_words'] = df['name'].apply(lambda x: len(x.split()))
    df['name_num_words_upper'] = df['name'].apply(lambda x: len(
        [word for word in x.split() if word.isupper()]))
    df['name_num_words_upper_perc'] = df['name_num_words_upper'] / \
        df['name_num_words']
    return df


def random_split_data(df):
    """Split data into train test sets
    Done because some features a reliant on numeric values calculated within
    given dataset

    Arguments:
        df: DataFrame

    Returns:
        X_train, X_test, y_train, y_test: Dataframes
    """
    # Label
    y = df['state']

    X = df.drop('state', axis=1)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=0)
    return X_train, X_test, y_train, y_test


def create_buckets_features(X_train, X_test, y_train, y_test):
    """Creates bucket features for data concerning category and goal
    goal_bucket: based off category and goal
    num_proj_q
    num_proj_m
    num_proj_w

    Note: this should take place after train test split
    Arguments:
        df {DataFrame} -- [description]

    Returns:
        [type] -- [description]
    """
    # Code adapted from https://www.kaggle.com/srishti280992/kickstarter-project-classification-lgbm-70-3
    # Training set
    X_train['goal_bucket'] = X_train.groupby(['main_category'])['usd_goal_real'].transform(
        lambda x: pd.qcut(x, [0, 0.25, 0.5, 0.75, 1], labels=[1, 2, 3, 4]))
    # Number of projects per quarter
    project_per_quarter = X_train.groupby(['main_category',
                                           'launch_year', 'launch_quarter', 'goal_bucket', ]).count()
    project_per_quarter = project_per_quarter[['name']]
    project_per_quarter.reset_index(inplace=True)
    project_per_quarter.columns = [
        'main_category', 'launch_year', 'launch_quarter', 'goal_bucket', 'num_proj_q']

    # number of projects per month
    project_per_month = X_train.groupby(
        ['main_category', 'launch_year', 'launch_month', 'goal_bucket']).count()
    project_per_month = project_per_month[['name']]
    project_per_month.reset_index(inplace=True)
    project_per_month.columns = ['main_category', 'launch_year',
                                 'launch_month', 'goal_bucket', 'num_proj_m']

    # number of projects per week
    project_per_week = X_train.groupby(
        ['main_category', 'launch_year', 'launch_week', 'goal_bucket']).count()
    project_per_week = project_per_week[['name']]
    project_per_week.reset_index(inplace=True)
    project_per_week.columns = ['main_category',
                                'launch_year', 'launch_week', 'goal_bucket', 'num_proj_w']

    # Merge new df with original df
    X_train = pd.merge(X_train, project_per_quarter,
                       on=['main_category', 'launch_year',
                           'launch_quarter', 'goal_bucket'],
                       how='left').set_index(X_train.index)
    X_train = pd.merge(X_train, project_per_month,
                       on=['main_category', 'launch_year',
                           'launch_month', 'goal_bucket'],
                       how='left').set_index(X_train.index)
    X_train = pd.merge(X_train, project_per_week,
                       on=['main_category', 'launch_year',
                           'launch_week', 'goal_bucket'],
                       how='left').set_index(X_train.index)

    # Mean number of backers for successful projects based on main_cat, launch_year, launch_qtr, goal_bucket
    successful_proj = X_train.merge(
        y_train, how='left', right_index=True, left_index=True).set_index(X_train.index)
    successful_proj = successful_proj[successful_proj['state'] == 1]
    successful_proj['pledge_pb'] = successful_proj['usd_pledged_real'] / \
        successful_proj['backers']
    mean_pledge = pd.DataFrame(successful_proj.groupby(
        ['main_category', 'launch_year', 'launch_quarter', 'goal_bucket'])['pledge_pb'].mean())
    mean_pledge.columns = ['mean_pledge_pb']
    mean_pledge.reset_index(inplace=True)

    successful_proj = successful_proj.merge(mean_pledge, how='left',
                                            on=['main_category', 'launch_year',
                                                'launch_quarter', 'goal_bucket']).set_index(successful_proj.index)
    successful_proj['number_of_backers'] = successful_proj['usd_pledged_real'] / \
        successful_proj['mean_pledge_pb']
    successful_proj = successful_proj[[
        'main_category', 'launch_year', 'launch_quarter', 'goal_bucket', 'number_of_backers']]
    successful_proj = successful_proj.groupby(
        ['main_category', 'launch_year', 'launch_quarter', 'goal_bucket']).mean()
    successful_proj.reset_index(inplace=True)
    successful_proj.columns = ['main_category', 'launch_year',
                               'launch_quarter', 'goal_bucket', 'mean_number_of_backers']
    X_train = pd.merge(X_train, successful_proj, how='left',
                       on=['main_category', 'launch_year', 'launch_quarter', 'goal_bucket']).set_index(X_train.index)
    # Testing set
    X_test['goal_bucket'] = X_test.groupby(['main_category'])['usd_goal_real'].transform(
        lambda x: pd.qcut(x, [0, 0.25, 0.5, 0.75, 1], labels=[1, 2, 3, 4]))
    # Number of projects per quarter
    project_per_quarter = X_test.groupby(['main_category',
                                          'launch_quarter', 'launch_year', 'goal_bucket']).count()
    project_per_quarter = project_per_quarter[['name']]
    project_per_quarter.reset_index(inplace=True)
    project_per_quarter.columns = [
        'main_category', 'launch_quarter', 'launch_year', 'goal_bucket',  'num_proj_q']

    # Number of project per month
    project_per_month = X_test.groupby(
        ['main_category', 'launch_year', 'launch_month', 'goal_bucket']).count()
    project_per_month = project_per_month[['name']]
    project_per_month.reset_index(inplace=True)
    project_per_month.columns = ['main_category', 'launch_year',
                                 'launch_month', 'goal_bucket', 'num_proj_m']

    # number of projects per week
    project_per_week = X_test.groupby(
        ['main_category', 'launch_year', 'launch_week', 'goal_bucket']).count()
    project_per_week = project_per_week[['name']]
    project_per_week.reset_index(inplace=True)
    project_per_week.columns = ['main_category',
                                'launch_year', 'launch_week', 'goal_bucket', 'num_proj_w']

    X_test = pd.merge(X_test, project_per_quarter,
                      on=['main_category', 'launch_year',
                          'launch_quarter', 'goal_bucket'],
                      how='left').set_index(X_test.index)
    X_test = pd.merge(X_test, project_per_month,
                      on=['main_category', 'launch_year',
                          'launch_month', 'goal_bucket'],
                      how='left').set_index(X_test.index)
    X_test = pd.merge(X_test, project_per_week,
                      on=['main_category', 'launch_year',
                          'launch_week', 'goal_bucket'],
                      how='left').set_index(X_test.index)
    # Mean number of backers for successful projects based on main_cat, launch_year, launch_qtr, goal_bucket
    successful_proj = X_test.merge(
        y_test, how='left', right_index=True, left_index=True).set_index(X_test.index)
    successful_proj = successful_proj[successful_proj['state'] == 1]
    successful_proj['pledge_pb'] = successful_proj['usd_pledged_real'] / \
        successful_proj['backers']
    mean_pledge = pd.DataFrame(successful_proj.groupby(
        ['main_category', 'launch_year', 'launch_quarter', 'goal_bucket'])['pledge_pb'].mean())
    mean_pledge.columns = ['mean_pledge_pb']
    mean_pledge.reset_index(inplace=True)

    successful_proj = successful_proj.merge(mean_pledge, how='left',
                                            on=['main_category', 'launch_year',
                                                'launch_quarter', 'goal_bucket']).set_index(successful_proj.index)
    successful_proj['number_of_backers'] = successful_proj['usd_pledged_real'] / \
        successful_proj['mean_pledge_pb']
    successful_proj = successful_proj[[
        'main_category', 'launch_year', 'launch_quarter', 'goal_bucket', 'number_of_backers']]
    successful_proj = successful_proj.groupby(
        ['main_category', 'launch_year', 'launch_quarter', 'goal_bucket']).mean()
    successful_proj.reset_index(inplace=True)
    successful_proj.columns = ['main_category', 'launch_year',
                               'launch_quarter', 'goal_bucket', 'mean_number_of_backers']

    X_test = pd.merge(X_test, successful_proj, how='left',
                      on=['main_category', 'launch_year', 'launch_quarter', 'goal_bucket']).set_index(X_test.index)
    return X_train, X_test


def drop_features(X_train, X_test):
    # Drop Launched and Deadline date features
    X_train = X_train.drop(
        ['deadline', 'launched', 'name', 'backers', 'usd_pledged_real'], axis=1)
    X_test = X_test.drop(['deadline', 'launched', 'name',
                          'backers', 'usd_pledged_real'], axis=1)

    return X_train, X_test


def main(input, split='random', train=None, test=None):
    """Exectues all feature engineering functions of input data and saves output file

    Arguments:
        input: csv file
    """
    df = prepare_data(input)
    df = create_features_date(df)
    df = create_features_name(df)

    if split == 'random':
        X_train, X_test, y_train, y_test = random_split_data(df)
    elif (split == 'strat_k_fold') and (len(train) != 0 and len(test) != 0):
        y = df['state']
        X = df.drop('state', axis=1)
        X_train = X.iloc[train]
        X_test = X.iloc[test]
        y_train = y.iloc[train]
        y_test = y.iloc[test]
    X_train, X_test = create_buckets_features(X_train, X_test, y_train, y_test)
    X_train, X_test = drop_features(X_train, X_test)
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    main()
