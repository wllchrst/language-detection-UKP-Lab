from torch import nn

class LangIDClassifier(nn.Module):
    """
    Neural Network classifier for language identification.
    """
    def __init__(self,
                 input_dimension: int,
                 hidden_dimension: int,
                 num_classes: int,
                 drop_out: float=0.2):
        super().__init__()
        self.fc = nn.Linear(input_dimension, hidden_dimension)
        self.last_fc = nn.Linear(input_dimension, num_classes)
        self.dropout = nn.Dropout(drop_out)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.last_fc(x)
        return x