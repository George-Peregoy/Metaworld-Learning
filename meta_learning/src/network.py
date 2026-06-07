import torch.nn as nn
import torch.optim as optim
import torch

class Network(nn.Module):
    """
    Generic MLP class.

    Attributes
    ----------
    network : torch.nn.Sequential
        Full network.
    loss_fn : nn.Module, optional
        Loss function class. Default is nn.MSELoss.
    optimizer : torch.optim.Optimizer
        Optimizer class. Default is torch.optim.Adam.

    Methods
    -------
    forward(X)
        Computes forward pass.
    update(X, y, loss)
        Computes forward and backward passes, updates network params.
    """

    def __init__(self, layer_sizes: list, 
                 lr: float, 
                 parameters=None,
                 hidden_activation=nn.ReLU, 
                 output_activation=None, 
                 loss_fn=nn.MSELoss, 
                 optimizer=optim.Adam):
        """
        Parameters
        ----------
        layer_sizes : list
            List of layer sizes. Example [17, 64, 32, 1].
        lr : float
            Learning rate for optimizer.
        parameters : nn.Parameter. Default None.
            Extra parameters for optimizer.
        activation_hidden : nn.Module. Default is nn.ReLU
            Activation function for hidden layers.
        activation_out : nn.Module. Default is None
            Output activation function, if None output is forward pass.
        loss_fn : nn.Module. Default is nn.MSELoss
            Loss function class.
        optimizer : torch.optim.Optimizer. Default is torch.optim.Adam.
            Optimizer class.
        """
        super().__init__()

        # build network
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            
            if i != len(layer_sizes) - 2:
                layers.append(hidden_activation())        

        if output_activation is not None:
            layers.append(output_activation())

        self.network = nn.Sequential(*layers)

        # init optim and loss
        self.loss_fn = loss_fn()
        if parameters is not None:
            self.optimizer = optimizer(params=list(self.network.parameters()) + [parameters], lr=self.lr)
        else:
            self.optimizer = optimizer(params = self.network.parameters(), lr=lr)

    def forward(self, X: torch.Tensor):
        """
        X : torch.Tensor
            Input features.
        
        Returns
        -------
        output : torch.Tensor
            Predicted output values.
        """
        return self.network(X)
    
    def update(self, X: torch.Tensor | None = None, y: torch.Tensor | None = None, loss: torch.Tensor | None = None, graph: bool | None = None):
        """
        Computes forward and backward passes, updates network params.

        Parameters
        ----------
        X : torch.Tensor
            Input features.
        y : torch.Tensor
            True output values.
        loss : torch.Tensor
            Externally computed loss. Used in critic. (default=None)
        
        Returns
        -------
        loss : float
            Computed loss as a float for comparison metrics.
        """
        
        # zero grad
        self.optimizer.zero_grad()

        # if no loss provided
        if X is not None and y is not None:
            output = self.network(X)
            loss = self.loss_fn(output, y)

        # backpass
        if graph:
            loss.backward(create_graph=True)
        else:
            loss.backward()

        # update network
        self.optimizer.step()

        # returns loss as python float
        return loss.item()