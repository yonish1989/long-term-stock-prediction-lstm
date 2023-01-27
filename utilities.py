import nasdaqdatalink as quandl
import pandas as pd
import numpy as np
import logging
import random
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

def _get_logger(logger = None):
    if logger is None:
        logger = logging.getLogger('dummy')
        logger.addHandler(logging.NullHandler())
    return logger

def _drop_irrelevant_columns(df, logger = None):
    logger = _get_logger(logger)
    date_keys = df.select_dtypes(include=['datetime64']).columns.tolist()
    logger.debug(f"date keys found: {date_keys} and will be dropped")
    object_keys = df.select_dtypes(include=['object']).columns.tolist()
    logger.debug(f"object keys found: {object_keys} and will be dropped")
    columns_to_drop = date_keys + object_keys
    return df.drop(columns_to_drop, axis = 1)

def _get_dummy_columns_indeces(df, dummy_columns, logger = None):
    df_relevant = _drop_irrelevant_columns(df, logger)
    indeces = []
    for column in dummy_columns:
        indeces.append(df_relevant.columns.get_loc(column))
    return indeces

def build_logger(log_level):
    str_to_level_dict = {
        'debug' : logging.DEBUG,
        'info' : logging.INFO,
        'warning' : logging.WARNING,
        'error' : logging.ERROR,
        'critical' : logging.CRITICAL
    }
    logging.basicConfig(format='%(levelname)s: %(message)s')
    logger = logging.getLogger('main_logger')
    logger.setLevel(str_to_level_dict[log_level])
    ch = logging.StreamHandler()
    ch.setLevel(str_to_level_dict[log_level])
    logger.addHandler(ch)
    return logger

def handle_null_values(df, limit_null_percent, drop_null_rows = True, fill_value = 0, logger = None):
    logger = _get_logger(logger)
    logger.info('preprocessing the data')
    logger.info('dropping all rows where price is null')
    df = df[df['price'].notna()] 
    logger.info(f'dropping all columns that have more than {str(limit_null_percent * 100)}% null values')
    column_null_rates = (df.isnull().sum() / len(df)).sort_values()
    columns_to_drop = column_null_rates[column_null_rates > limit_null_percent].index.values.tolist()
    logger.debug(f"dropping columns: {columns_to_drop}")
    df = df.drop(columns_to_drop, axis = 1)
    if drop_null_rows == True:
        logger.info('dropping all rows that have null values')
        df = df.dropna()
    else:
        logger.info(f'filling null values with: {fill_value}')
        df = df.fillna(value = fill_value)
    return df

def add_meta_data(df, meta_df, meta_columns, rare_threshold = 0.1, logger = None):
    logger = _get_logger(logger)
    # meta columns are categorical variables
    logger.debug(f"adding columns: {meta_columns} from the meta DataFrame")
    meta_df_relevant = meta_df[meta_columns + ['ticker']]
    merged_df = df.merge(meta_df_relevant, on = 'ticker', validate = "many_to_one")
    non_dummy_columns = set(df.columns)
    kept_columns = []
    for column in meta_columns:
        count_values = merged_df[column].value_counts()
        common_values = count_values[(count_values / merged_df.shape[0]) > rare_threshold].index.tolist()
        if len(common_values) < 1:
            logger.info(f"dropping column: {column} because it has to many unique categories")
            merged_df = merged_df.drop([column], axis = 1)
        else:
            logger.debug(f"for column {column} common values are: {common_values}")
            merged_df[column] = merged_df[column].apply(lambda x : x if x in common_values else 'other')
            kept_columns.append(column)
    logger.debug(f"turning categories into dummies for columns: {kept_columns}")
    merged_df = pd.get_dummies(merged_df, prefix = kept_columns, columns = kept_columns)
    dummy_columns = list(set(merged_df.columns) - non_dummy_columns)
    dummy_columns_indeces = _get_dummy_columns_indeces(merged_df, dummy_columns, logger)
    return merged_df, dummy_columns_indeces


def bin_price_increase(price_increase):
    price_increase = 100 * (price_increase - 1)
    if price_increase >= 100:
        return([0,0,0,0,1])
    elif price_increase >= 50:
        return([0,0,0,1,0])
    elif price_increase >= 0:
        return([0,0,1,0,0])
    elif price_increase >= -50:
        return([0,1,0,0,0])
    else:
        return([1,0,0,0,0])

def get_category_indeces(labels):
    indeces = {
        1 : [],
        10 : [],
        100 : [],
        1000 : [],
        10000 : []
    }
    for idx, row in enumerate(labels):
        key = 0
        for i in range(5):
            key += row[i]*(10**i)
        indeces[key].append(idx)
    return indeces

def count_categories(labels):
    count = [0, 0, 0, 0, 0]
    for row in labels:
        count = [sum(x) for x in zip(count, row)]
    print(f"count of category [0,0,0,0,1] is: {count[4]}")
    print(f"count of category [0,0,0,1,0] is: {count[3]}")
    print(f"count of category [0,0,1,0,0] is: {count[2]}")
    print(f"count of category [0,1,0,0,0] is: {count[1]}")
    print(f"count of category [1,0,0,0,0] is: {count[0]}")

def build_features_labels_windows(df, min_date, max_date, years_interval = 5, leap_interval = 2, column_to_predict = "price", logger = None):
    logger = _get_logger(logger)
    X = []
    y = []
    ticker_list = df['ticker'].unique().tolist()
    for ticker in ticker_list:
        relevant_data = df[df['ticker'] == ticker].sort_values(by = ['calendardate'])
        features_df = relevant_data[(relevant_data['calendardate'] > min_date) & (relevant_data['calendardate'] < max_date)]
        start_index = 0
        end_index = start_index + years_interval - 1
        label_index = end_index + leap_interval
        while features_df.shape[0] > label_index:
            temp_df = _drop_irrelevant_columns(features_df, logger = logger)
            temp_X = temp_df.iloc[start_index:end_index].values
            if (features_df[column_to_predict].iloc[label_index] <= 0) or (features_df[column_to_predict].iloc[end_index] <= 0):
                start_index += 1
                end_index = start_index + years_interval - 1
                label_index = end_index + leap_interval
                continue
            label = features_df[column_to_predict].iloc[label_index] / features_df[column_to_predict].iloc[end_index]
            temp_y = bin_price_increase(label)
            X.append(temp_X)
            y.append(temp_y)
            start_index += 1
            end_index = start_index + years_interval - 1
            label_index = end_index + leap_interval
    return X, y

def drop_columns_with_one_dominant_value(df, one_value_threshold = 0.8, logger = None):
    logger = _get_logger(logger)
    columns_to_drop = []
    for column in df.columns:
        value_counts_column = df[column].value_counts()
        only_one_value = value_counts_column[(value_counts_column / df.shape[0]) > one_value_threshold].index.tolist()
        if len(only_one_value) > 0:
            logger.info(f"dropping column {column} because the value {only_one_value[0]} is in more then {str(one_value_threshold * 100)}% of the rows")
            columns_to_drop.append(column)
    if len(columns_to_drop) > 0:
        df = df.drop(columns_to_drop, axis = 1)
    return df
            
def load_data(quandl_api_key = None, logger = None):
    logger = _get_logger(logger)
    logger.info("loading the dataset")
    if (quandl_api_key is not None) and (quandl_api_key != "") and (quandl_api_key != '<Your API key>'):
        logger.info(f"downloading the data from the quandl database")
        quandl.ApiConfig.api_key = quandl_api_key
        return quandl.get_table('SHARADAR/SF1', dimension = 'MRY', paginate = True)
    else:
        raise FileNotFoundError("no quandl api key given, can't download data")

def train_data_augmentation(X_train, y_train, size, dummy_columns_indeces, duplicate_data_change_factor = 0.001):
    indeces = get_category_indeces(y_train)
    X_train_new = []
    y_train_new = []
    new_indeces = {}
    for key in indeces.keys():
        new_indeces[key] = np.random.choice(indeces[key], size = (size - len(indeces[key]))).tolist()
        new_indeces[key] += indeces[key]
        if len(np.unique(new_indeces[key])) != len(np.unique(indeces[key])):
            raise ValueError(f"size of unique index for key {key} is: {len(np.unique(new_indeces[key]))} and new is: {len(np.unique(indeces[key]))}")
    shuffled_indeces = []
    for key in new_indeces.keys():
        shuffled_indeces += new_indeces[key]
    np.random.shuffle(shuffled_indeces)
    for index in shuffled_indeces:
        X_train_new.append(X_train[index])
        for row in range(len(X_train_new[-1])):
            for col in range(len(X_train_new[-1][row])):
                if col not in dummy_columns_indeces:
                    X_train_new[-1][row][col] += X_train_new[-1][row][col] * duplicate_data_change_factor * random.randint(-5, 5)
        y_train_new.append(y_train[index])
    return X_train_new, y_train_new

def load_meta_data(quandl_api_key = None, ticker_list = None, logger = None):
    logger = _get_logger(logger)
    logger.info("loading the ticker dataset")
    if (quandl_api_key is not None) and (quandl_api_key != '') and (quandl_api_key != '<Your API key>'):
        logger.info(f"downloading the data from the quandl database")
        quandl.ApiConfig.api_key = quandl_api_key
        if ticker_list is not None:
            return quandl.get_table('SHARADAR/TICKERS', table = 'SF1', ticker = ticker_list, paginate = True)
        else:
            return quandl.get_table('SHARADAR/TICKERS', table = 'SF1', paginate = True)
    else:
        raise FileNotFoundError("no quandl api key given, can't download meta data")

def print_evaluation(evaluation_train, evaluation_test, evaluation_val = None, evaluation_train_aug = None):
    print("")
    print(f"loss on training set: {evaluation_train[0]}")
    print(f"accuracy on training set: {evaluation_train[1]}")
    print("")
    if evaluation_train_aug is not None:
        print(f"loss on augmented training set: {evaluation_train_aug[0]}")
        print(f"accuracy on augmented training set: {evaluation_train_aug[1]}")
        print("")
    if evaluation_val is not None:
        print(f"loss on validation set: {evaluation_val[0]}")
        print(f"accuracy on validation set: {evaluation_val[1]}")
        print("")
    print(f"loss on test set: {evaluation_test[0]}")
    print(f"accuracy on test set: {evaluation_test[1]}")

class MinMaxScaler3D(MinMaxScaler):
    def fit_transform(self, X, y = None):
        x = np.reshape(X, newshape = (X.shape[0] * X.shape[1], X.shape[2]))
        return np.reshape(super().fit_transform(x, y = y), newshape = X.shape)

class StandardScaler3D(StandardScaler):
    def fit_transform(self, X, y = None):
        x = np.reshape(X, newshape = (X.shape[0] * X.shape[1], X.shape[2]))
        return np.reshape(super().fit_transform(x, y = y), newshape = X.shape)

def apply3DScaler(X, scaler):
    X_new = X.copy()
    for idx, item in enumerate(X):
        X_new[idx] = scaler.transform(item)
    return X_new