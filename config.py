# get data only from this year forward, only relevant for when using the full dataset
start_year = '2010-12-31'
# get data up until this year, only relevant for when using the full dataset
end_year = '2019-12-31'
# max date for moving time window, only relevant for when using the full dataset
min_date = '2009-12-31'
# min date for moving time window, only relevant for when using the full dataset
max_date = '2020-12-31'
# drop rows that have null values
drop_null = True
# fill value in case we want to keep the rows with null values, will be ignored if drop_null is True 
fill_value = 0
# if column has more then (limit_null_percent * 100)% null values it will be dropped
limit_null_percent = 0.01
# if in a column a value has more then (one_value_threshold * 100)% of rows then the column will be dropped
one_value_threshold = 0.8
# if a category appears in less then (rare_category_threshold * 100)% of the rows it will be lumped as 'other' category with all the others. 
rare_category_threshold = 0.05
# Interval of years that will be given to the LSTM layer as input
years_interval = 5
# how many years into the future we would like to predict
leap_interval = 2
# size of the test set
test_size = 0.1
# size of the validation set from train set
validation_size = 0.11
# 10 - DEBUG, 20 - INFO, 30 - WARNING, 40 - ERROR, 50 - CRITICAL 
log_level = 20
# you need to input your quandl API Key here if you want to download the data from the quandl database
quandl_api_key = '<Your API key>'
# columns from the meta dataset we want to add
meta_columns_to_use = ['exchange', 'category', 'sicsector', 'sicindustry', 'sector', 'industry', 'scalemarketcap', 'scalerevenue', 'currency']
# number of records in each class after data augmentation must be above 6000
augmented_number = 6000
