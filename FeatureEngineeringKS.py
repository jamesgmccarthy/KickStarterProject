import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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
    #TODO: Name
    # drop unneccessary columns (temp drop name)
    df = df.drop(
        ['pledged',  'usd_pledged_real', 'usd pledged', 'goal', 'backers'], axis=1)

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
    name_upper_case
    name_sentiment

    Arguments:
        df {[type]} -- [description]
    """
    df['name_len'] = df['name'].apply(lambda x: len(x))
    df['name_excl'] = df['name'].apply(lambda x: 1 if '!' in x else 0)
    df['name_qm'] = df['name'].apply(lambda x: 1 if '?' in x else 0)
    df['name_colons'] = df['name'].apply(lambda x: 1 if ':' or ';' in x else 0)
    df['name_num_words'] = df['name'].apply(lambda x: len(x.split()))
    df['name_num_words'] = df['name'].apply(lambda x: len(
        [word for word in x.split() if word.isupper()]))
    # drop 'name'
    #df = df.drop('name', axis=1)
    return df


def create_label_encoding(df):
    lb_make = LabelEncoder()
    df['main_category'] = lb_make.fit_transform(df['main_category'])
    df['category'] = lb_make.fit_transform(df['category'])
    df['country'] = lb_make.fit_transform(df['country'])
    df['currency'] = lb_make.fit_transform(df['currency'])
    return df


def create_dummy_features(df):
    """Create dummy features for categorical features using one hot encoding:
    main_category
    category
    country
    currency
    (Launch variables??)
    Arguments:
        df {[type]} -- [description]
    """
    # Main categories
    df = pd.get_dummies(df, columns=['main_category'])

    # Sub Categories
    df = pd.get_dummies(df, columns=['category'])

    # Country
    df = pd.get_dummies(df, columns=['country'])

    # Currency
    df = pd.get_dummies(df, columns=['currency'])

    return df


def split_data(df):
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


def create_buckets_features(X_train, X_test):
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
    # Training set
    X_train['goal_bucket'] = X_train.groupby(['main_category'])['usd_goal_real'].transform(
        lambda x: pd.qcut(x, [0, 0.25, 0.5, 0.75, 1], labels=[1, 2, 3, 4]))
    project_per_quarter = X_train.groupby(['main_category', 'goal_bucket',
                                           'launch_year', 'launch_quarter']).count()
    project_per_quarter = project_per_quarter[['name']]
    project_per_quarter = project_per_quarter.reset_index(inplace=True)
    project_per_quarter.columns = [
        'goal_bucket', 'main_category', 'launch_quarter', 'launch_year', 'num_proj_q']
    X_train = pd.merge(X_train, project_per_quarter,
                       on=['goal_bucket', 'main_category',
                           'launch_quarter', 'lauch_year'],
                       how='left')

    # Testing set
    X_test['goal_buckets'] = X_test.groupby(['main_category'])['usd_goal_real'].transform(
        lambda x: pd.qcut(x, [0, 0.25, 0.5, 0.75, 1], labels=[1, 2, 3, 4]))
    project_per_quarter = X_test.groupby(['main_category', 'goal_bucket',
                                          'launch_year', 'launch_quarter']).count()
    project_per_quarter = project_per_quarter[['name']]
    project_per_quarter = project_per_quarter.reset_index(inplace=True)
    project_per_quarter.columns = [
        'goal_bucket', 'main_category', 'launch_quarter', 'launch_year', 'num_proj_q']
    X_test = pd.merge(X_test, project_per_quarter,
                      on=['goal_bucket', 'main_category',
                          'launch_quarter', 'lauch_year'],
                      how='left')

    return X_train, X_test


def drop_features(X_train, X_test):
    # Drop Launched and Deadline date features
    X_train = X_train.drop(['deadline', 'launched'], axis=1)
    X_test = X_test.drop(['deadline', 'launched'], axis=1)
    X_train = X_train.drop('name')
    X_test = X_test.drop('name')
    return X_train, X_test


def main(input, dummy=True):
    """Exectues all feature engineering functions of input data and saves output file

    Arguments:
        input: csv file
    """
    df = prepare_data(input)
    df = create_features_date(df)
    df = create_features_name(df)
    df_with_dummy = create_dummy_features(df)
    df_with_labels_encoding = create_label_encoding(df)
    X_train, X_test, y_train, y_test = split_data(df_with_labels_encoding)
    X_train_d, X_test_d, y_train_d, y_test_d = split_data(df_with_dummy)
    #X_train, X_test = create_buckets_features(X_train, X_test)
    #X_train_d, X_test_d = create_buckets_features
    #X_train, X_test = drop_features(X_train, X_test)
    #X_train_d, X_test_d = drop_features(X_train_d, X_test_d)
    if not dummy:
        return X_train, X_test, y_train, y_test
    else:  # Temporary
        return X_train_d, X_test_d, y_train_d, y_test_d


if __name__ == '__main__':
    main()
