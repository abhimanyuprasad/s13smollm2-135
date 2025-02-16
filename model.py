import torch
import torch.nn as nn

class LlamaForCausalLM(nn.Module):
    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__()
        
        # Model architecture as per model_architecture.txt
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config['hidden_size'], config['vocab_size'], bias=False)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        logits = self.lm_head(outputs)
        return logits

class LlamaModel(nn.Module):
    def __init__(self, config):
        super(LlamaModel, self).__init__()
        self.embed_tokens = nn.Embedding(config['vocab_size'], config['hidden_size'])
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config['num_hidden_layers'])])
        self.norm = nn.LayerNorm(config['hidden_size'],elementwise_affine=False)
        self.rotary_emb = LlamaRotaryEmbedding(
            config["hidden_size"] // config["num_attention_heads"],
            max_position_embeddings=config["max_position_embeddings"],
        )

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
        self.input_layernorm = nn.LayerNorm(config['hidden_size'], elementwise_affine=False)
        self.post_attention_layernorm = nn.LayerNorm(config['hidden_size'], elementwise_affine=False)

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
        self.k_proj = nn.Linear(config['hidden_size'], config['hidden_size'], bias=False)
        self.v_proj = nn.Linear(config['hidden_size'], config['hidden_size'], bias=False)
        self.o_proj = nn.Linear(config['hidden_size'], config['hidden_size'], bias=False)
        self.num_heads = config['num_attention_heads']
        self.head_dim = config['hidden_size'] // self.num_heads
        self.hidden_size = config['hidden_size']

    def forward(self, x, attention_mask=None):
        batch_size, seq_length, _ = x.size()

        # Project inputs to query, key, and value
        q = self.q_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_length, head_dim)
        k = self.k_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_length, head_dim)
        v = self.v_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_length, head_dim)

        # Compute attention scores
        attn_scores = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim ** 0.5))  # Scaled dot-product attention

        # Apply attention mask if provided
        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(attention_mask == 0, float('-inf'))

        # Compute attention weights
        attn_weights = attn_scores.softmax(dim=-1)  # (batch_size, num_heads, seq_length, seq_length)

        # Compute the output
        attn_output = (attn_weights @ v).transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_size)  # (batch_size, seq_length, hidden_size)

        # Apply the output projection
        output = self.o_proj(attn_output)

        return output

class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Create inverse frequency bands
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Create position embeddings cache
        self._set_cos_sin_cache(seq_len=max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=torch.float)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

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