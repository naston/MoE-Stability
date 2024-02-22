import torch
import torch.nn as nn

class PositionalEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.context_length = config.context_length

    def forward(self, x):
        pass

class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.decode = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, x):
        return self.decode(x)


class MoELayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.MHA = MHALayer(config)
        self.ExpertFF = MoEFF(config)

        self.norm1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps, bias=True)
        self.norm2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps, bias=True)

        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)

        self.norm_first = config.norm_first

    def forward(self, x, causal_mask):
        #causal_mask = None

        if self.norm_first:
            x = x + self.dropout1(self.MHA(self.norm1(x), causal_mask))
            x = x + self.dropout2(self.ExpertFF(self.norm2(x)))
        else:
            x = self.norm1(x + self.dropout1(self.MHA(x, causal_mask)))
            x = self.norm2(x + self.dropout2(self.ExpertFF(x)))

        return x


class MoEFF(nn.Module):
    def __init__(self, config, norm=False, project=False, rescale=False, ):
        super().__init__()

        self.router = Router(config, norm, project, rescale)

        self.n_experts = config.n_experts
        self.experts = nn.ModuleList([FeedForward(config) for _ in range(self.n_experts)])

        self.dropout = nn.Dropout(config.dropout)
        self.norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps, bias=True)

    def forward(self, x, top_k=1):
        #x = self.norm(x)
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

        #out = x + self.dropout(out)

        return out, counts, route_prob.sum(0), len(dropped), route_prob_max


class MHALayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.W_key = nn.Linear(config.d_model,config.d_model)
        self.W_query = nn.Linear(config.d_model,config.d_model)
        self.W_value = nn.Linear(config.d_model,config.d_model)

        self.multihead_attn = nn.MultiheadAttention(embed_dim=config.d_model, num_heads=config.n_heads, dropout=config.dropout)

    def forward(self, x, causal_mask=None):
        K = self.W_key(x)
        Q = self.W_query(x)
        V = self.W_value(x)


        out, _ = self.multihead_attn(K, Q, V, attn_mask=causal_mask) # attn_output_weights

        # out = self.norm(x + self.dropout(attn_output))
        return out


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.fc1 = nn.Linear(config.d_model, config.d_ff, bias=True)
        self.fc2 = nn.Linear(config.d_ff, config.d_model, bias=True)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        out = self.fc2(self.dropout(self.relu(self.fc1(x))))
        return out

 
class Router(nn.Module):
    def __init__(self, config, norm=False, project=False, rescale=False):
        super().__init__()

        self.switch = nn.Linear(config.d_model, config.n_experts)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m = nn.utils.weight_norm(self.switch, name='experts')
        # m.weight_g # magnitude
        # m.weight_v # direction

        return self.softmax(self.switch(x))