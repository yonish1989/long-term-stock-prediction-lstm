import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import utilities
import config as cfg
import logging
import argparse
import os
import models.lstm_dropout_dense_droput as stockPredictorModule
import pickle

def get_model_path(name):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "models", name))

def parse_args():
    parser = argparse.ArgumentParser(description = "build and test the stock price prediction LSTM model")
    parser.add_argument("-q", "--quandl-api-key", dest = "quandl_api_key", default = cfg.quandl_api_key, type = str, help = "quandl key to use for")
    parser.add_argument("-f", "--full-data", dest = "full_data", action = "store_true", help = "use this flag if you only have access to the full sharadar dataset")
    parser.add_argument("-v", "--verbosity", type = str, choices = ['debug', 'info', 'warning', 'error', 'critical'], default = 'warning', help = "verbosity of the information printed")
    parser.add_argument("-t", "--train-model", dest = "train", action = "store_true", help = "train the model again, evaluate and save it, only applicable when having the full data")
    parser.add_argument("-s", "--save-model", dest = "save", action = "store_true", help = "save the train model in ./model/model.m")
    args = parser.parse_args()
    return args.quandl_api_key, args.full_data, args.verbosity, args.train, args.save

def train_model(X_train, X_val, X_test, y_train, y_val, y_test, dummy_columns_indeces, save, column_list, scale_data = True):
    print("training the model")
    stock_price_predictior = stockPredictorModule.stockPricePredictor(len(X_train[0]), len(X_train[0][0]))
    stock_price_predictior.compile()
    X_train_new, y_train_new = utilities.train_data_augmentation(X_train, y_train, cfg.augmented_number, dummy_columns_indeces)
    if scale_data:
        scaler = utilities.StandardScaler3D()
        X_train_new = scaler.fit_transform_3D(np.array(X_train_new))
        X_val = scaler.transform_3D(np.array(X_val))
        X_test = scaler.transform_3D(np.array(X_test))
    stock_price_predictior.fit(np.array(X_train_new), np.array(y_train_new), np.array(X_val), np.array(y_val))
    print("")
    print("model evalutations: ")
    print("")
    evaluation_train = stock_price_predictior.evaluate(np.array(X_train), np.array(y_train))
    evaluation_train_aug = stock_price_predictior.evaluate(np.array(X_train_new), np.array(y_train_new))
    evaluation_val = stock_price_predictior.evaluate(np.array(X_val), np.array(y_val))
    evaluation_test = stock_price_predictior.evaluate(np.array(X_test), np.array(y_test))
    utilities.print_evaluation(evaluation_train, evaluation_test, evaluation_val, evaluation_train_aug)
    if save:
        print("saving model")
        # save the model
        stock_price_predictior.model.save(get_model_path("model.m"))
        # save the scaler
        if scale_data:
            pickle.dump(scaler, open(get_model_path('scaler.pkl'), 'wb'))
        # save the column list
        pickle.dump(column_list, open(get_model_path('column_list.pkl'), 'wb'))
    return stock_price_predictior.model

def test_model(model, X_test, y_test, scale_data = True):
    if model is None:
        print("loading model")
        model = tf.keras.models.load_model(get_model_path("model.m"))
        if scale_data:
            scaler = pickle.load(open(get_model_path('scaler.pkl'), 'rb'))
            X_test = scaler.transform_3D(np.array(X_test))
    print("testing model")
    evaluation_test = model.evaluate(np.array(X_test), np.array(y_test))
    print(f"model accuracy on test set: {evaluation_test[1]}")
    print(f"model loss on test set: {evaluation_test[0]}")

def main():
    print("parsing args")
    quandl_api_key, full_data, verbosity, train, save = parse_args()
    logger = utilities.build_logger(log_level = verbosity)
    print("loading the data")
    df = utilities.load_data(quandl_api_key, logger = logger)
    df = df[(df['calendardate'] >= cfg.start_year) & (df['calendardate'] <= cfg.end_year)]
    print("preprocessing the data")
    column_list = None
    if train:
        df = utilities.handle_null_values(df, cfg.limit_null_percent, cfg.drop_null, cfg.fill_value, logger = logger)
        df = utilities.drop_columns_with_one_dominant_value(df, cfg.one_value_threshold)
    else:
        column_list = pickle.load(open(get_model_path('column_list.pkl'), 'rb'))
    print("adding meta data")
    ticker_list = None
    if not full_data:
        ticker_list = df['ticker'].unique().tolist()
    meta_df = utilities.load_meta_data(quandl_api_key, ticker_list, logger = logger)
    df, dummy_columns_indeces = utilities.add_meta_data(df, meta_df, cfg.meta_columns_to_use, cfg.rare_category_threshold, logger = logger)
    print("making time windows")
    if not train:
        df = df[df.columns.intersection(column_list)]
        df = df.fillna(value = cfg.fill_value)
    X, y = utilities.build_features_labels_windows(df, cfg.start_year, cfg.end_year, cfg.years_interval, cfg.leap_interval, logger = logger)
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size = cfg.test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size = cfg.validation_size)

    model = None
    if train:
        if not full_data:
            raise ValueError("can't train a model without the full sharadar dataset, exiting...")
        model = train_model(X_train, X_val, X_test, y_train, y_val, y_test, dummy_columns_indeces, save, df.columns.tolist())
    if not full_data:
        # in case we only have the small free smaple we want to test all of it.
        test_model(model, X, y)
    else:
        test_model(model, X_test, y_test)
    

if __name__ == "__main__":
    main()