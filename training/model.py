import torch
import torch.nn as nn


class DualMLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        esmc_dim = int(config["esmc_dim"])
        chemberta_dim = int(config["chemberta_dim"])
        maccs_dim = int(config["maccs_dim"])
        total_dim = int(config["total_dim"])
        momentum = float(config["bn_momentum"])
        eps = float(config["bn_eps"])
        dropout = float(config["dropout"])
        hidden_dims = config["hidden_dims"]

        if total_dim != esmc_dim + chemberta_dim + maccs_dim:
            raise ValueError("Feature dimensions do not match total_dim")

        if config["activation"].lower() != "relu":
            raise ValueError("Only relu activation is supported")

        self.protein_bn = nn.BatchNorm1d(
            esmc_dim, momentum=momentum, eps=eps
        )
        self.chemberta_bn = nn.BatchNorm1d(
            chemberta_dim, momentum=momentum, eps=eps
        )

        layers = []
        in_dim = total_dim

        for hidden_dim in hidden_dims:
            hidden_dim = int(hidden_dim)
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(
                    hidden_dim, momentum=momentum, eps=eps
                ),
                nn.Dropout(dropout),
                nn.ReLU(),
            ])
            in_dim = hidden_dim

        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(in_dim, 1)

    def forward(self, esmc, chemberta, maccs):
        esmc = self.protein_bn(esmc)
        chemberta = self.chemberta_bn(chemberta)
        x = torch.cat([esmc, chemberta, maccs], dim=1)
        x = self.hidden(x)
        return self.output(x).squeeze(-1)
