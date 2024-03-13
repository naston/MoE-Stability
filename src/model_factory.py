import torch
import torch.nn as nn
import torch.nn.functional as F

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
    
    def fedus_loss(self):
        return self.ExpertFF.fedus_loss()
    
    def loramoe_loss(self):
        return self.ExpertFF.loramoe_loss()


class MoEFF(nn.Module):
    def __init__(self, config, norm=False, project=False, rescale=False, top_k=1):
        super().__init__()

        self.router = Router(config, norm, project, rescale)

        self.n_experts = config.n_experts
        self.experts = nn.ModuleList([FeedForward(config) for _ in range(self.n_experts)])

        self.dropout = nn.Dropout(config.dropout)
        self.norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps, bias=True)

        self.top_k = top_k
        self.old_routed = torch.zeros(self.n_experts)

    def forward(self, x):
        #x = self.norm(x)
        route_prob = self.router(x)
        route_prob_topk, routes = torch.topk(route_prob, dim=-1, k=self.top_k)

        _, dummy_routes = torch.max(route_prob,dim=-1)

        indexes_list = [torch.eq(routes, i).nonzero(as_tuple=True)[0] for i in range(self.n_experts)] #TODO: verify this in top_k version
        # indexes_list should contain the list of data points for each expert

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
            out[indexes_list[i], :] += expert_output[i] * route_prob[indexes_list[i],i] #TODO: make sure route_prob is broadcasting fairly

        if dropped:
            raise NotImplementedError()
            dropped = torch.cat(dropped)
            out[dropped, :] = x[dropped, :]

        #if self.is_scale_prob:
        #    out = out * route_prob_max.view(-1,1)
        #else:
        #    out = out * (route_prob_max/route_prob_max.detach()).view(-1,1)

        #out = x + self.dropout(out)

        return out
    
    def shazeer_loss(self, inputs):
        # This is really shazeer loss as we are not restricting to top 1

        # Take only top_k routing probabilities for the batch
        route_prob = self.router(inputs)
        _, bottom_routes = torch.topk(route_prob, dim=-1, k=(self.n_experts-self.top_k),largest=False)
        route_prob[bottom_routes] = 0

        # sum routing proba (top k) across expert axis
        importance_scores = torch.sum(route_prob, dim=-1) #TODO: need to make sure this dim is correct
        
        # return sqaure of the coefficient of variation of the importance scores
        return torch.var(importance_scores) /  torch.square(torch.mean(importance_scores))

    def loramoe_loss(self,inputs):
        raise NotImplementedError
        # Honestly may not be usable?
        route_prob = self.router(inputs)
        Q = torch.sum(route_prob, dim=-1)
        I = None # This is what idk if I can calculate as it is preset for each batch depending on task type in LoRA-MoE
        Z = I * Q
        return torch.var(Z) / torch.mean(Z)

    def gating_loss(self,inputs):
        # Based on MoLE: Mixture of LoRA Experts
        route_prob = self.router(inputs)
        importance_scores = torch.sum(route_prob, dim=-1)
        return -1 * torch.log(torch.prod(importance_scores))
    
    def fedus_loss(self,inputs):
        route_prob = self.router(inputs)
        r = self.routed(inputs)
        f = r / torch.sum(r)

        W = route_prob * f
        return torch.sum(W.view(1,-1)) * self.n_experts / inputs.shape[0]

    def routed(self,inputs):
        # get the number of tokens routed to each expert as a vector
        route_prob = self.router(inputs)
        _, routes = torch.topk(route_prob, dim=-1, k=self.top_k)
        route_counts = torch.bincount(routes.view(1,-1))
        return F.pad(route_counts, (0,(self.n_experts-route_counts.size)), "constant", 0)
    
    def CV(self,inputs):
        # Take only top_k routing probabilities for the batch
        route_prob = self.router(inputs)
        _, bottom_routes = torch.topk(route_prob, dim=-1, k=(self.n_experts-self.top_k),largest=False) #TODO: Do I 0 out these values?
        route_prob[bottom_routes] = 0

        # sum routing proba (top k) across expert axis
        importance_scores = torch.sum(route_prob, dim=-1) #TODO: need to make sure this dim is correct
        
        # return sqaure of the coefficient of variation of the importance scores
        return torch.std(importance_scores) /  torch.mean(importance_scores)
    

    def RC(self,inputs):
        # https://stats.stackexchange.com/questions/123490/what-is-the-correct-formula-for-between-class-scatter-matrix-in-lda
        route_prob = self.router(inputs)
        _, routes = torch.topk(route_prob, dim=-1, k=self.top_k)
        route_counts = torch.bincount(routes.view(1,-1))

        indexes_list = [torch.eq(routes, i).nonzero(as_tuple=True)[0] for i in range(self.n_experts)] #TODO: verify this in top_k version
        universal_mean = torch.mean(inputs,dim=-1)
        class_means = torch.zeros(self.n_experts)

        W = torch.zeros((inputs.shape[1],inputs.shape[1]))
        for i,data_subset in enumerate(indexes_list):
            class_means[i] = torch.mean(inputs[data_subset, :],dim=-1)
            W += torch.cov(inputs[data_subset, :])
        class_means -= universal_mean #maybe this works?

        B = torch.zeros((inputs.shape[1],inputs.shape[1]))
        for i,class_vect in enumerate(class_means):
            #temp = class_vect-universal_mean
            B += route_counts[i] * torch.outer(class_vect,class_vect)

        return torch.trace(torch.matmul(W,torch.linalg.pinv(B)))
        

    def RC_bal(self,inputs):
        # https://stats.stackexchange.com/questions/123490/what-is-the-correct-formula-for-between-class-scatter-matrix-in-lda
        route_prob = self.router(inputs)
        _, routes = torch.topk(route_prob, dim=-1, k=self.top_k)

        indexes_list = [torch.eq(routes, i).nonzero(as_tuple=True)[0] for i in range(self.n_experts)] #TODO: verify this in top_k version
        #universal_mean = torch.mean(inputs,dim=-1)
        class_means = torch.zeros(self.n_experts)

        W = torch.zeros((inputs.shape[1],inputs.shape[1]))
        for i,data_subset in enumerate(indexes_list):
            class_means[i] = torch.mean(inputs[data_subset, :],dim=-1)
            W += torch.cov(inputs[data_subset, :])

        universal_mean = torch.mean(class_means, dim=-1)
        class_means -= universal_mean #maybe this works?

        B = torch.zeros((inputs.shape[1],inputs.shape[1]))
        for i,class_vect in enumerate(class_means):
            #temp = class_vect-universal_mean
            B += torch.outer(class_vect,class_vect)
        B *= inputs.shape[0]/self.n_experts

        return torch.trace(torch.matmul(W,torch.linalg.pinv(B)))
        

    def IC(self, inputs):
        raise NotImplementedError
        # This method would require a whole load of runs which are averaged together it seems
        new_routed = self.routed(inputs)
        cen_new_routed = new_routed - torch.mean(new_routed)
        
        cen_old_routed = self.old_routed - torch.mean(self.old_routed)

        numer = torch.matmul(cen_new_routed, cen_old_routed)
        denom = torch.sqrt(torch.square(torch.sum(cen_new_routed)) * torch.square(torch.sum(cen_old_routed)))

        self.old_routed = new_routed
        
        return numer / denom
    




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