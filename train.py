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

            # Pad input_ids to a fixed length (e.g., 1024)
            max_length = 1024
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

def count_parameters(model):
    """Counts the total number of parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_model(model, dataloader, optimizer, num_steps, checkpoint_path, device, accumulation_steps=4):
    model.train()
    for step in range(num_steps):
        for i, (input_ids, attention_mask) in enumerate(dataloader):
            # Move inputs to the same device as the model
            input_ids = input_ids.to(device)  # Ensure device is defined
            attention_mask = attention_mask.to(device)  # Ensure device is defined
            
            optimizer.zero_grad()
            
            # Check the shape of input_ids
            print(f"Input IDs shape: {input_ids.shape}")
            
            with torch.cuda.amp.autocast():  # Mixed precision context
                outputs = model(input_ids)
                loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), input_ids.view(-1))
            
            loss.backward()

            # Perform optimization step every accumulation_steps
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()

            if step % 500 == 0:
                print(f"Step {step}: Loss = {loss.item()}")
                
                # Generate predictions
                with torch.no_grad():
                    predictions = model(input_ids)
                    predicted_ids = predictions.argmax(dim=-1)  # Get the predicted token IDs
                    print(f"Predicted IDs at step {step}: {predicted_ids}")

                # Save checkpoint
                torch.save(model.state_dict(), checkpoint_path)

def main():
    # Load configuration
    config = load_config()
    
    # Initialize model
    model = LlamaForCausalLM(config)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Print total number of parameters
    total_params = count_parameters(model)
    print(f"Total number of parameters in the model: {total_params}")

    # Create a dataset and dataloader using input.txt
    dataset = TextDataset(file_path='input.txt', vocab_size=config['vocab_size'])
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)  # Further reduced batch size

    # Set up optimizer
    optimizer = optim.AdamW(model.parameters(), lr=0.003)

    # Define checkpoint path
    checkpoint_path = 'checkpoints/model_checkpoint.pth'
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    # Train for 5000 steps
    train_model(model, dataloader, optimizer, num_steps=5000, checkpoint_path=checkpoint_path, device=device)

    # Load the checkpoint and continue training for 50 more steps
    model.load_state_dict(torch.load(checkpoint_path))
    print("Loaded checkpoint. Continuing training for 50 more steps.")
    train_model(model, dataloader, optimizer, num_steps=50, checkpoint_path=checkpoint_path, device=device)

if __name__ == "__main__":
    main() 