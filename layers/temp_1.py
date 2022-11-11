import torch
import torch.nn as nn



class LECF_L(nn.Module):
    """the implementation of single layer of LECF
    Args:


    """
    def __init__(self,input_node_dim,output_node_dim,hidden_dim,act_fn=nn.ReLU(),recurrent=True,coords_weight=1.0):
        super(LECF_L,self).__init__()
        self.in_node_dim = input_node_dim
        self.out_node_dim = output_node_dim
        self.hidden_dim = hidden_dim
        self.act_fn = act_fn
        self.coords_weight = coords_weight
        self.recurrent = recurrent
        self.lorentz_radical_dim = 1

        self.edge_mlp=nn.Sequential(
            nn.Linear(self.in_node_dim * 2 + self.lorentz_radical_dim, self.hidden_dim),# dim: (3,8)
            self.act_fn,
            nn.Linear(self.hidden_dim, self.hidden_dim),
            self.act_fn)

        self.coord_mlp=nn.Sequential(
            nn.Linear(self.hidden_dim,self.hidden_dim),
            self.act_fn,
            nn.Linear(self.hidden_dim,1,bias=False)
        )
        torch.nn.init.xavier_uniform_(self.coord_mlp[3], gain=0.001) ## !!! test necessity

        self.node_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim + self.in_node_dim, self.hidden_dim),
            act_fn,
            nn.Linear(self.hidden_dim, self.out_node_dim))







































def unsorted_segment_sum(data, segment_ids, num_segments): #unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`.
        * torch.scatter_add_: the idx is the idx in the dim of src: self[idx][j]+= src[idx]
        https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_add_.html?highlight=scatter_add_#torch.Tensor.scatter_add_
    """
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)