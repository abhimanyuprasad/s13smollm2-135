import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import yaml
import os
import time
from torch.nn.utils.rnn import pad_sequence

# Import the model class
from model import LlamaForCausalLM, load_config

# Custom dataset class to read from input.txt
class TextDataset(Dataset):
    def __init__(self, file_path, vocab_size):
        with open(file_path, 'r') as file:
            self.lines = file.readlines()
        self.vocab_size = vocab_size

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        original_idx = idx
        while True:
            line = self.lines[idx].strip()
            tokens = line.split()
            
            input_ids = []
            for token in tokens:
                try:
                    token_id = int(token)
                    if token_id < self.vocab_size:
                        input_ids.append(token_id)
                except ValueError:
                    continue  # Skip non-integer tokens

            input_ids = torch.tensor(input_ids, dtype=torch.long)

            # Pad input_ids to a fixed length (e.g., 2048)
            max_length = 2048
            if len(input_ids) < max_length:
                input_ids = F.pad(input_ids, (0, max_length - len(input_ids)), value=0)  # Pad with 0s
            else:
                input_ids = input_ids[:max_length]  # Truncate to max_length

            attention_mask = torch.ones_like(input_ids)  # Simple attention mask
            return input_ids, attention_mask

            # Move to the next index
            idx = (idx + 1) % len(self.lines)
            if idx == original_idx:  # If we've looped through all lines
                raise IndexError("No valid input IDs found in the dataset.")

def train_model(model, dataloader, optimizer, num_steps, checkpoint_path):
    model.train()
    for step in range(num_steps):
        for input_ids, attention_mask in dataloader:
            optimizer.zero_grad()
            
            # Check the shape of input_ids
            print(f"Input IDs shape: {input_ids.shape}")
            
            outputs = model(input_ids)
            loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), input_ids.view(-1))
            loss.backward()
            optimizer.step()

            if step % 500 == 0:
                print(f"Step {step}: Loss = {loss.item()}")
                # Save checkpoint
                torch.save(model.state_dict(), checkpoint_path)

def main():
    # Load configuration
    config = load_config()
    
    # Initialize model
    model = LlamaForCausalLM(config)
    #model.train()

    # Create a dataset and dataloader using input.txt
    dataset = TextDataset(file_path='input.txt', vocab_size=config['vocab_size'])
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Set up optimizer
    optimizer = optim.AdamW(model.parameters(), lr=0.003)

    # Define checkpoint path
    checkpoint_path = 'checkpoints/model_checkpoint.pth'
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    # Train for 5000 steps
    train_model(model, dataloader, optimizer, num_steps=5000, checkpoint_path=checkpoint_path)

    # Load the checkpoint and continue training for 50 more steps
    model.load_state_dict(torch.load(checkpoint_path))
    print("Loaded checkpoint. Continuing training for 50 more steps.")
    train_model(model, dataloader, optimizer, num_steps=50, checkpoint_path=checkpoint_path)

if __name__ == "__main__":
    main() 