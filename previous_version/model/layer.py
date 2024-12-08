from torch import nn

class layerFC(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(layerFC, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(output_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        h = self.fc(x)
        h = self.relu(h)
        h = self.bn(h)
        h = self.dropout(h)
        return h
    

if __name__ == "__main__":
    import torch

    # random seed
    torch.manual_seed(42)

    # initialize the model
    layer = layerFC(input_dim=10, output_dim=10)
    print(layer)

    # generate random input data: batch size = 32
    input_data = torch.randn(32, 10)

    # test the forward pass
    output_data = layer(input_data)
    print(output_data[0])