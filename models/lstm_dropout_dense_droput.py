from . import lstm_dropout_dense_droput_cfg as cfg
import tensorflow as tf

class stockPricePredictor():
    activation_dict = {
        "tanh" : tf.keras.activations.tanh,
        "relu" : tf.keras.activations.relu,
        "sigmoid" : tf.keras.activations.sigmoid,
        "leaky_relu" : tf.keras.layers.LeakyReLU(alpha = 0.01)
    }
    reg_dict = {
        "L1" : tf.keras.regularizers.L1(l1=1e-5),
        "L2" : tf.keras.regularizers.L2(l2=1e-4)
    }
    init_dict = {
        "He" : tf.keras.initializers.HeUniform(seed = None)
    }
    def __init__(self, input_dim_first, input_dim_second, lstm_size = 256, dense_size = 256, activation_LSTM = None, activation_dense = None, reg_lstm = None, reg_dense = None, init_lstm = None, init_dense = None):
        self.input_dim_first = input_dim_first
        self.input_dim_second = input_dim_second
        activation_func_LSTM = self.activation_dict[activation_LSTM] if activation_LSTM is not None else self.activation_dict[cfg.activation_LSTM]
        activation_func_dense = self.activation_dict[activation_dense] if activation_dense is not None else self.activation_dict[cfg.activation_dense]
        reg_lstm_func = self.reg_dict[reg_lstm] if reg_lstm is not None else None
        reg_dense_func = self.reg_dict[reg_dense] if reg_dense is not None else None
        init_lstm_func = self.init_dict[init_lstm] if init_lstm is not None else "glorot_uniform"
        init_dense_func = self.init_dict[init_dense] if init_dense is not None else "glorot_uniform"
        self.model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape = (input_dim_first, input_dim_second)),
            tf.keras.layers.LSTM(lstm_size, activation = activation_func_LSTM, kernel_regularizer = reg_lstm_func, kernel_initializer = init_lstm_func,),
            tf.keras.layers.Dropout(rate = 0.1),
            tf.keras.layers.Dense(dense_size, activation = activation_func_dense, kernel_regularizer = reg_dense_func, kernel_initializer = init_dense_func),
            tf.keras.layers.Dropout(rate = 0.1),
            tf.keras.layers.Dense(5, activation = tf.keras.activations.softmax,),
        ])
    def reset_model(self):
        self.__init__(self.input_dim_first, self.input_dim_second)
    def compile(self):
        loss_fn = tf.keras.losses.CategoricalCrossentropy()
        self.model.compile(optimizer = 'adam', loss = loss_fn, metrics = [tf.keras.metrics.CategoricalAccuracy()])
    def fit(self, X_train, y_train, X_val = None, y_val = None, add_early_stopping = False):
        callbacks = []
        if add_early_stopping:
            es = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', mode = 'min', patience = cfg.patience)
            callbacks.append(es)
        callbacks = None if len(callbacks) < 1 else callbacks
        if X_val is None or y_val is None:
            return self.model.fit(X_train, y_train, epochs = cfg.epochs, batch_size = cfg.batch_size, validation_split = cfg.validation_split_size, callbacks = callbacks)
        else:
            return self.model.fit(X_train, y_train, epochs = cfg.epochs, batch_size = cfg.batch_size, validation_data = (X_val, y_val), callbacks = callbacks)
    def evaluate(self, X, y):
        return self.model.evaluate(X, y)

