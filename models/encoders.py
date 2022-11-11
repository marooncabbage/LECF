import torch.nn as nn

import layers.hyp_layers as hyp_layers
import manifolds


class Encoder(nn.Module):
    def __init__(self, c):
        super(Encoder, self).__init__()
        self.c = c

    def encode(self, x, adj):
        if self.encode_graph:
            input = (x, adj)
            output, _ = self.layers.forward(input)  # output = h, adj
        else:
            output = self.layers.forward(x)
        return output


class LECF(Encoder):
    def __init__(self, c, args):
        super(LECF, self).__init__(c)
        self.manifold = getattr(manifolds, "Hyperboloid")()
        assert args.num_layers > 1

        hgc_layers = []
        in_dim = out_dim = args.embedding_dim
        hgc_layers.append(
            hyp_layers.HyperbolicGraphConvolution(
                self.manifold, in_dim, out_dim, self.c, args.network, args.num_layers
            )
        )
        self.layers = nn.Sequential(*hgc_layers) # https://realpython.com/python-kwargs-and-args/#unpacking-with-the-asterisk-operators
        self.encode_graph = True

    def encode(self, x, adj):
        x_hyp = self.manifold.proj(x, c=self.c)
        return super(LECF, self).encode(x_hyp, adj)
