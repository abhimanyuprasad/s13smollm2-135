import torch
import torch.nn as nn

class LlamaForCausalLM(nn.Module):
    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__()
        
        # Model architecture as per model_architecture.txt
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config['vocab_size'], config['hidden_size'], bias=False)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        logits = self.lm_head(outputs)
        return logits

class LlamaModel(nn.Module):
    def __init__(self, config):
        super(LlamaModel, self).__init__()
        self.embed_tokens = nn.Embedding(config['vocab_size'], config['hidden_size'])
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config['num_hidden_layers'])])
        self.norm = nn.LayerNorm(config['hidden_size'])

    def forward(self, input_ids, attention_mask=None):
        # Forward pass through the model
        embeddings = self.embed_tokens(input_ids)
        for layer in self.layers:
            embeddings = layer(embeddings, attention_mask)
        return self.norm(embeddings)

class LlamaDecoderLayer(nn.Module):
    def __init__(self, config):
        super(LlamaDecoderLayer, self).__init__()
        self.self_attn = LlamaAttention(config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = nn.LayerNorm(config['hidden_size'])
        self.post_attention_layernorm = nn.LayerNorm(config['hidden_size'])

    def forward(self, x, attention_mask=None):
        # Forward pass through the decoder layer
        attn_output = self.self_attn(x, attention_mask)
        x = self.input_layernorm(x + attn_output)
        mlp_output = self.mlp(x)
        return self.post_attention_layernorm(x + mlp_output)

class LlamaAttention(nn.Module):
    def __init__(self, config):
        super(LlamaAttention, self).__init__()
        self.q_proj = nn.Linear(config['hidden_size'], config['hidden_size'], bias=False)
        self.k_proj = nn.Linear(config['hidden_size'], config['hidden_size'] // 3, bias=False)
        self.v_proj = nn.Linear(config['hidden_size'], config['hidden_size'] // 3, bias=False)
        self.o_proj = nn.Linear(config['hidden_size'], config['hidden_size'], bias=False)

    def forward(self, x, attention_mask=None):
        # Attention mechanism
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # Implement attention logic here
        return self.o_proj(v)

class LlamaMLP(nn.Module):
    def __init__(self, config):
        super(LlamaMLP, self).__init__()
        self.gate_proj = nn.Linear(config['hidden_size'], config['intermediate_size'], bias=False)
        self.up_proj = nn.Linear(config['hidden_size'], config['intermediate_size'], bias=False)
        self.down_proj = nn.Linear(config['intermediate_size'], config['hidden_size'], bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        # MLP forward pass
        return self.down_proj(self.act_fn(self.gate_proj(x)))

# Configuration loading from custom_smollm2.yaml
def load_config():
    import yaml
    with open('custom_smollm2.yaml', 'r') as file:
        config = yaml.safe_load(file)
    return config['model']['model_config']

if __name__ == "__main__":
    config = load_config()
    model = LlamaForCausalLM(config)
    print(model) 