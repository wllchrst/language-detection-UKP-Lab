from torch import nn

class LangIDClassifier(nn.Module):
    """
    Neural Network classifier for language identification.
    """
    def __init__(self,
                 input_dimension: int,
                 num_classes: int,
                 drop_out: float=0.2):
        super().__init__()
        self.fc = nn.Linear(input_dimension, 512)
        self.fc2 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(drop_out)
        self.activation = nn.ReLU()
        self.output_fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.fc(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.output_fc(x)
        return x