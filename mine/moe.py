import torch
import torch.nn as nn

#TODO: Take norm and dropout outside of FF and have it be pre- and post- combination of experts

class MoELayer(nn.Module):
    def __init__(self, d_model, n_heads, n_experts):
        #encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads)
        self.MHA = MHALayer()
        self.ExpertFF = MoEFF()

    def forward(self, x):
        x = self.MHA(x)
        x = self.ExpertFF(x)

        return x

class MoEFF(nn.Module):
    def __init__(self, d_model, d_ff, n_experts, norm=False, project=False, rescale=False, dropout=0.0):
        self.router = Router(d_model, n_experts, norm, project, rescale)

        self.n_experts = n_experts
        self.experts = nn.ModuleList([FeedForward(d_model, d_ff, dropout) for _ in range(n_experts)])

    def forward(self, x, top_k=1):

        route_prob = self.router(x)
        route_prob_max, routes = torch.max(route_prob, dim=-1)

        indexes_list = [torch.eq(routes, i).nonzero(as_tuple=True)[0] for i in range(self.n_experts)]

        out = x.new_zeros(x.shape)

        capacity  = int(self.capacity_factor * len(x) / self.n_experts)
        counts = x.new_tensor([len(indexes_list[i]) for i in range(self.n_experts)])

        dropped = []
        # Only drop tokens if `drop_tokens` is `True`.
        if self.drop_tokens:
            # Drop tokens in each of the experts
            for i in range(self.n_experts):
                # Ignore if the expert is not over capacity
                if len(indexes_list[i]) <= capacity:
                    continue
                # Shuffle indexes before dropping
                indexes_list[i] = indexes_list[i][torch.randperm(len(indexes_list[i]))]
                # Collect the tokens over capacity as dropped tokens
                dropped.append(indexes_list[i][capacity:])
                # Keep only the tokens upto the capacity of the expert
                indexes_list[i] = indexes_list[i][:capacity]

        expert_output = [self.experts[i](x[indexes_list[i], :]) for i in range(self.n_experts)]
        for i in range(self.n_experts):
            out[indexes_list[i], :] = expert_output[i]

        if dropped:
            dropped = torch.cat(dropped)
            out[dropped, :] = x[dropped, :]

        if self.is_scale_prob:
            out = out * route_prob_max.view(-1,1)
        else:
            out = out * (route_prob_max/route_prob_max.detach()).view(-1,1)

        return out, counts, route_prob.sum(0), len(dropped), route_prob_max

class MHALayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0, layer_norm_eps=1e-5):
        self.W_key = nn.Linear(d_model,d_model)
        self.W_query = nn.Linear(d_model,d_model)
        self.W_value = nn.Linear(d_model,d_model)

        self.multihead_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout)

        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=True)
        #self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=True)
        self.dropout = nn.Dropout(dropout)
        #self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, causal_mask=None):
        K = self.W_key(x)
        Q = self.W_query(x)
        V = self.W_value(x)


        attn_output, _ = self.multihead_attn(K, Q, V, attn_mask=causal_mask) # attn_output_weights

        out = self.norm(x + self.dropout(attn_output))
        return out

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0, layer_norm_eps=1e-5):
        self.fc1 = nn.Linear(d_model, d_ff, bias=True)
        self.fc2 = nn.Linear(d_ff, d_model, bias=True)

        self.relu = nn.ReLU(dim=1)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=True)

    def forward(self, x):
        out = self.fc2(self.dropout1(self.relu(self.fc1(x))))
        return self.norm(x + self.dropout2(out))
    
class Router(nn.Module):
    def __init__(self, d_model, n_experts, norm=False, project=False, rescale=False):
        self.switch = nn.Linear(d_model,n_experts)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.switch(x))