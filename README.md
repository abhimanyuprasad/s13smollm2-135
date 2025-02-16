# Llama Model Training

This project implements a training pipeline for a Llama-based language model using PyTorch. The model is designed to read input data from a text file and perform training with options for mixed precision and gradient accumulation.

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [License](#license)

## Requirements

- Python 3.6 or higher
- PyTorch 1.7 or higher
- CUDA (if using GPU)
- Additional libraries:
  - `torchsummary`
  - `yaml`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/llama-model-training.git
   cd llama-model-training
   ```

2. Install the required packages:
   ```bash
   pip install torch torchvision torchaudio
   pip install torchsummary
   ```

3. Ensure you have a compatible CUDA version installed if you plan to use GPU acceleration.

## Usage

1. Prepare your input data in a text file named `input.txt`. Each line should contain space-separated token IDs.

2. Configure the model parameters in `custom_smollm2.yaml`.

3. Run the training script:
   ```bash
   python train.py
   ```

4. The model will be trained for a specified number of steps, and checkpoints will be saved in the `checkpoints` directory.

## Model Architecture

The model architecture is defined in `model.py`, which includes:
- `LlamaForCausalLM`: The main model class.
- `LlamaModel`: The core model structure.
- `LlamaDecoderLayer`: The decoder layer implementation.
- `LlamaAttention`: The attention mechanism.
- `LlamaMLP`: The feedforward network.

## Training

- The training process includes options for mixed precision training to optimize memory usage.
- The batch size can be adjusted in the `train.py` file.
- Gradient accumulation is implemented to simulate larger batch sizes without increasing memory usage.

## Model Summary

LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(49152, 576)
    (layers): ModuleList(
      (0-29): 30 x LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=576, out_features=576, bias=False)
          (k_proj): Linear(in_features=576, out_features=576, bias=False)
          (v_proj): Linear(in_features=576, out_features=576, bias=False)
          (o_proj): Linear(in_features=576, out_features=576, bias=False)
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=576, out_features=1536, bias=False)
          (up_proj): Linear(in_features=576, out_features=1536, bias=False)
          (down_proj): Linear(in_features=1536, out_features=576, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LayerNorm((576,), eps=1e-05, elementwise_affine=False)
        (post_attention_layernorm): LayerNorm((576,), eps=1e-05, elementwise_affine=False)      )
    )
    (norm): LayerNorm((576,), eps=1e-05, elementwise_affine=False)
    (rotary_emb): LlamaRotaryEmbedding()
  )
  (lm_head): Linear(in_features=576, out_features=49152, bias=False)
)


# Sample training log

Total number of parameters in the model: 176062464
Input IDs shape: torch.Size([4, 1024])
/content/s13smollm2-135/train.py:72: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast():  # Mixed precision context
Step 0: Loss = 10.9609375
Predicted IDs at step 0: tensor([[1415, 1415, 1415,  ..., 1415, 1415, 1415],
        [1415, 1415, 1415,  ..., 1415, 1415, 1415],
        [1415, 1415, 1415,  ..., 1415, 1415, 1415],
        [1415, 1415, 1415,  ..., 1415, 1415, 1415]], device='cuda:0')
Input IDs shape: torch.Size([4, 1024])
Step 0: Loss = 10.9609375
Predicted IDs at step 0: tensor([[1415, 1415, 1415,  ..., 1415, 1415, 1415],
        [1415, 1415, 1415,  ..., 1415, 1415, 1415],
        [1415, 1415, 1415,  ..., 1415, 1415, 1415],
        [1415, 1415, 1415,  ..., 1415, 1415, 1415]], device='cuda:0')
Input IDs shape: torch.Size([4, 1024])