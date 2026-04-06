import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, hidden_ch:(list[int]), in_dim:(int)=80, out_dim:(int)=3, dropout:(float)=0.05, leaky_slope:(float)=0.2):
        super().__init__()
        assert len(hidden_ch) <=7, 'Hidden channel list must have at most 7 elements'

        self.leaky = nn.LeakyReLU(negative_slope=leaky_slope)

        layer_list = [
            nn.Linear(in_features=in_dim, out_features=128), 
            nn.BatchNorm1d(128), self.leaky, 
            nn.Dropout1d(p=dropout),
            nn.Unflatten(dim=1, unflattened_size=(1, 128))
        ]

        # Conv layers increase the channel amounts, and reduce the sequence length
        layer_list.extend([
                nn.Conv1d(in_channels=1, out_channels=hidden_ch[0], kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(hidden_ch[0]),
                self.leaky,
                nn.Dropout(p=dropout),
                nn.MaxPool1d(kernel_size=2, stride=2)
            ])
        for i in range(len(hidden_ch)-1):

            layer_list.extend([
                nn.Conv1d(hidden_ch[i], hidden_ch[i+1], kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(hidden_ch[i+1]),
                self.leaky,
                nn.Dropout(p=dropout),
                nn.MaxPool1d(kernel_size=2, stride=2)
            ])
        
        # ConvTranspose layers decrease the channel amounts, and increase the sequence length
        reversed_hidden_ch = hidden_ch[::-1]
        for i in range(len(reversed_hidden_ch)-1):
            layer_list.extend([
                nn.ConvTranspose1d(reversed_hidden_ch[i], reversed_hidden_ch[i+1], kernel_size=4, stride=2, padding=1),
                nn.BatchNorm1d(reversed_hidden_ch[i+1]),
                self.leaky,
                nn.Conv1d(reversed_hidden_ch[i+1], reversed_hidden_ch[i+1], kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(reversed_hidden_ch[i+1]),
                self.leaky
            ])
        layer_list.extend([
            nn.ConvTranspose1d(reversed_hidden_ch[-1], 1, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm1d(1),
            self.leaky,
            nn.Flatten(start_dim=1),
            nn.Linear(65,out_dim)
        ])

        self.layers = nn.Sequential(*layer_list)
        # Apply manual weight init due to leaky relu
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
            # Kaiming Normal is best for LeakyReLU
            nn.init.kaiming_normal_(m.weight, mode='fan_out', a=0.2, nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
        elif isinstance(m, nn.BatchNorm1d):
            # BatchNorm should start with weights at 1 and bias at 0
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x:(torch.Tensor)) -> torch.Tensor:
        x = self.layers(x) # All layers applied
        x = torch.relu(x) # Applies relu, destroys negative vals
        return x / (torch.sum(x, dim=1, keepdim=True) + 1e-12) # Normalize to sum to 1 along dim 1 (ab dim)
