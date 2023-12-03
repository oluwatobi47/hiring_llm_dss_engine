class ModelTrainer:
    def __init__(self, model, training_data):
        self.training_data = training_data
        self.model = model

    def train(self):
        print("Training model", self.training_data)
