import torch
import dgl
from typing import Callable

class EGNNLayer(torch.nn.Module):
    """ Layer of E(n) Equivariant Graph Neural Networks.

    Parameters
    ----------
    in_features : int
        Input features.

    out_features : int
        Output features.

    edge_features : int
        Edge features.

    References
    ----------
    [1] Satorras, E.G. et al. "E(n) Equivariant Graph Neural Networks"
    https://arxiv.org/abs/2102.09844

    [2] https://github.com/vgsatorras/egnn

    """
    def __init__(
        self,
        in_features : int,
        hidden_features: int,
        out_features : int,
        activation : Callable=torch.nn.SiLU(),
        space_dimension : int=3,
    ):
        super(EGNNLayer, self).__init__()

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.activation = activation
        self.space_dimension = space_dimension

        self.coordinate_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_features, hidden_features),
            activation,
            torch.nn.Linear(hidden_features, 1)
        )

        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(
                in_features * 2 + 1,
                hidden_features
            ),
            activation,
            torch.nn.Linear(hidden_features, hidden_features),
            activation,
        )

        self.node_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_features + in_features, hidden_features),
            activation,
            torch.nn.Linear(hidden_features, out_features)
        )

        self.velocity_nlp = torch.nn.Sequential(
            torch.nn.Linear(in_features, hidden_features),
            activation,
            torch.nn.Linear(hidden_features, 1),
        )

    def _edge_model(self, edge):

        return {"h_e":
            self.edge_mlp(
                torch.cat(
                    [
                        edge.src["h_v"],
                        edge.dst["h_v"],
                        (edge.src["x"] - edge.dst["x"]).pow(2).sum(
                            dim=-1, keepdims=True
                        ),
                    ],
                    dim=-1
                )
            )
        }

    def _node_model(self, node):
        return {"h_v":
            self.node_mlp(
                torch.cat(
                    [
                        node.data["h_v"],
                        node.data["h_agg"],
                    ],
                    dim=-1,
                )
            )
        }

    def _coordinate_edge_model(self, edge):
        return {
            "x_e": (edge.src["x"] - edge.dst["x"])
            * self.coordinate_mlp(edge.data["h_e"])
        }

    def _coordinate_node_model(self, node):
        return {
            "x": node.data["x"] + node.data["x_agg"],
        }

    def _velocity_and_coordinate_node_model(self, node):
        v = self.velocity_nlp(node.data["h_v"]) * node.data["v"]\
            + node.data["x_agg"]

        return {
            "x": node.data["x"] + v,
            "v": v
        }

    def forward(self, graph, feat, coordinate, velocity=None):
        """ Forward pass.

        Parameters
        ----------
        graph : dgl.DGLGraph
            Input graph.

        feat : torch.Tensor
            Input features.

        coordinate : torch.Tensor
            Input coordinates.

        velocity : torch.Tensor
            Input velocity.

        Returns
        -------
        torch.Tensor : Output features.

        torch.Tensor : Output coordinates.
        """
        # get local copy of the graph
        graph = graph.local_var()

        # put features and coordinates into graph
        graph.ndata["h_v"], graph.ndata["x"] = feat, coordinate

        # apply representation update on edge
        # Eq. 3 in "E(n) Equivariant Graph Neural Networks"
        graph.apply_edges(func=self._edge_model)

        # apply coordinate update on edge
        graph.apply_edges(func=self._coordinate_edge_model)

        # aggregate coordinate update
        graph.update_all(
            dgl.function.copy_e("x_e", "x_msg"),
            dgl.function.sum("x_msg", "x_agg"),
        )

        # apply coordinate update on nodes
        if velocity is not None:
            graph.ndata["v"] = velocity
            graph.apply_nodes(func=self._velocity_and_coordinate_node_model)
        else:
            graph.apply_nodes(func=self._coordinate_node_model)

        ## aggregate representation update
        graph.update_all(
            dgl.function.copy_e("h_e", "h_msg"),
            dgl.function.sum("h_msg", "h_agg"),
        )

        # apply representation update on nodes
        graph.apply_nodes(func=self._node_model)

        # pull features
        feat = graph.ndata["h_v"]
        coordinate = graph.ndata["x"]

        return feat, coordinate

class EGNN(torch.nn.Module):
    """ E(n) Equivariant Graph Neural Networks.

    Parameters
    ----------

    References
    ----------
    [1] Satorras, E.G. et al. "E(n) Equivariant Graph Neural Networks"
    https://arxiv.org/abs/2102.09844

    """
    def __init__(
        self,
        in_features,
        hidden_features,
        out_features,
        depth=4,
        edge_features=0,
        activation=torch.nn.SiLU(),
    ):
        super(EGNN, self).__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.embedding_in = torch.nn.Linear(in_features, hidden_features)
        self.embedding_out = torch.nn.Linear(hidden_features, out_features)
        self.activation = activation
        self.depth = depth

        for idx in range(0, depth):
            self.add_module(
                "EGNNLayer_%s" % idx, EGNNLayer(
                    in_features=hidden_features,
                    hidden_features=hidden_features,
                    out_features=hidden_features,
                    activation=activation,
                )
            )

    def forward(self, graph, feat, coordinate):
        """ Forward pass.

        Parameters
        ----------
        graph : dgl.DGLGraph
            Input graph.

        feat : torch.Tensor
            Input features.

        coordinate : torch.Tensor
            Input coordinates.

        Returns
        -------
        torch.Tensor : Output features.

        torch.Tensor : Output coordinates.

        """
        graph = graph.local_var()
        feat = self.embedding_in(feat)
        for idx in range(self.depth):
            feat, coordinate = self._modules["EGNNLayer_%s" % idx](
                graph, feat, coordinate
            )
        feat = self.embedding_out(feat)
        return feat, coordinate