# Long term stock price prediction - Technion Deep learning course final project

A project in Long term stock prediction.
Based on [Long Term Stock Prediction Based On Financial
Statements Paper](http://cs230.stanford.edu/projects_winter_2021/reports/70728801.pdf), [tradeX Github](https://github.com/saic-mdal/lama).


In this project we aim to extended [Original Paper](http://cs230.stanford.edu/projects_winter_2021/reports/70728801.pdf) goals. We aim to predict the stock price 2 years into the future based on a 5 years of historical financial Indicators

In our Project we use [Quandl Sharadar Dataset](https://data.nasdaq.com/databases/SF1/) to get the historical financial indicators instead of web scraping like the original paper. The Dataset is a **Premium Dataset** meaning you are going to need to pay for a license for using the dataset before continuing. 

Since we used a different dataset we had to compeletly refactor the code to fit the new dataset.

We were not able to get exactly the same financial indicators as the ones in the original paper so we used a different set. our set does not include some of the indicators the original set uses while the original paper does not include some of the indicators we used.

Written by Yoni Shamula and Tomer Lerner.

## Getting Started

```bash
git clone https://github.com/yonish1989/long-term-stock-prediction-lstm.git
cd long-term-stock-prediction-lstm
```

### Dataset

* [Sharadar Dataset](https://data.nasdaq.com/databases/SF1/) - Contains the sharadar Dataset link, you can either use the free API key which comes with creating an account and test the already existing model with the small sample data. or you can buy an access to the full dataset to completely retrain the model.

### Prerequisites

1. Setup conda 
    ```bash
    conda env create -f environment.yml
    conda activate stock_predictor_env
    pip install tensorflow
    ```
    This will create a working environment named 'stock_predictor_env'

## Running the Code
* command line arguments (first the code will try using the command line arguments then will use the config file for the defaults):
    * quandl-api-key - your Quandl API key, this **must** be available either throught here or defined in the config.py file.
    * full-data - use this flag if you have access to the full Sharadar Dataset through your API Key.
    * verbosity - verbosity of the information printed choices are : 'debug', 'info', 'warning', 'error', 'critical'
    * train-model - use this flag to train the model, will not work if you do not have access to the full data.
    * save-model - will save the trained model in ./models/model.m

### 1. Run a test on the model using the example sharadar data
  ```
  python ./main.py --quandl-api-key '<Your API Key>'
  ```
### 2. Train and save the model using the full dataset
  ```
  python ./main.py --quandl-api-key '<Your API Key>' --full-data --train-model --save-model
  ```

## Jupyter notebook
you can view our experiments and results by loading our notebook
```
  jupyter notebook
  <open the experiments.ipynb>
```

## Numerical Results - Best Network
| Network | Accuracy on train set | Accuracy on validation set | Accuracy on test set |
| :-------------: | :--------------: | :--------------------: | :--------------------: | 
| LSTM256(tanh) -> dropout -> Dense256(relu) -> dropout -> softmax        | 0.95      | 0.43     | 0.41 |


## License

This project is licensed under the APACHE 2.0 License - see the [LICENSE.md](LICENSE.md) file for details

## References
1. Original Paper : [Long Term Stock Prediction Based On Financial
Statements](http://cs230.stanford.edu/projects_winter_2021/reports/70728801.pdf) 
