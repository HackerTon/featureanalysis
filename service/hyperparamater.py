class Hyperparameter:
    def __init__(
        self,
        learning_rate,
        batch_size_test,
        batch_size_train,
        data_path,
    ):
        self.learning_rate = learning_rate
        self.batch_size_test = batch_size_test
        self.batch_size_train = batch_size_train
        self.data_path = data_path
