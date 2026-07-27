import torch
import torch.nn as nn
import torch.optim as optim
from ml_predictor import GrokGQA_Transformer

def test_overfit():
    print("Testing if model can overfit a single batch...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model
    model = GrokGQA_Transformer(
        input_dim=11, seq_len=32, embed_dim=128, 
        num_layers=4, num_q_heads=8, num_kv_heads=2, dropout=0.0
    ).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    # Create a single batch of random data and random labels
    xb = torch.randn(64, 32, 11).to(device)
    yb = torch.randint(0, 2, (64,)).float().to(device)
    
    for step in range(100):
        optimizer.zero_grad()
        pred = model(xb).squeeze(1)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        
        if (step + 1) % 20 == 0:
            acc = ((pred > 0.0) == yb.bool()).float().mean().item() * 100
            print(f"Step {step+1}: Loss = {loss.item():.4f}, Acc = {acc:.1f}%")
            
    final_acc = ((pred > 0.0) == yb.bool()).float().mean().item() * 100
    if final_acc == 100.0:
        print("✅ SUCCESS: Model can overfit a single batch. Architecture is capable of learning.")
    else:
        print("❌ FAILED: Model cannot overfit. There is a bug in the forward pass or gradients.")

if __name__ == "__main__":
    test_overfit()
