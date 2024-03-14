import torch.nn as nn
from model_factory import MoELayer, Decoder, PositionalEncoder
from transformers import PreTrainedModel, PretrainedConfig

class MoEConfig(PretrainedConfig):
    model_type = 'MoE'
    def __init__(self, config ,**kwargs):
        super().__init__(**kwargs)

        self.d_model = config.d_model
        self.d_ff = config.d_ff
        self.n_layers = config.n_layers

        self.n_heads = config.n_heads
        self.n_experts = config.n_experts

        self.vocab_size = config.vocab_size
        self.context_length = config.context_length

        self.dropout = config.dropout
        self.layer_norm_eps = config.layer_norm_eps
        self.norm_first = config.norm_first


class MoE(PreTrainedModel):
    config_class = MoEConfig
    def __init__(self, config):
        super().__init__(config)
        self.causal_mask = None

        self.pos_enc = PositionalEncoder(config)
        self.layers = nn.ModuleList([MoELayer(config) for _ in range(config.n_layers)])
        self.decoder = Decoder(config)

    def forward(self, x):
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, self.causal_mask)
        return self.decoder(x)
    
    def fedus_loss(self,inputs):
        loss = 0
        for layer in self.layers:
            loss+=layer.fedus_loss(inputs)
            inputs = self.layer(inputs, self.causal_mask)
        return loss

    def shazeer_loss(self,inputs):
        loss = 0
        for layer in self.layers:
            loss+=layer.shazeer_loss(inputs)
            inputs = self.layer(inputs, self.causal_mask)
        return loss
    
    def loramoe_loss(self,inputs):
        raise NotImplementedError()
        loss = 0
        for layer in self.layers:
            loss+=layer.loramoe_loss(inputs)
        return loss

    def CV(self,inputs):
        CV = 0
        for layer in self.layers:
            CV+=layer.CV(inputs)
        return CV

    def RC(self,inputs):
        RC = 0
        for layer in self.layers:
            RC+=layer.RC(inputs)
        return RC

    def RC_bal(self,inputs):
        RC = 0
        for layer in self.layers:
            RC+=layer.RC_bal(inputs)
        return RC