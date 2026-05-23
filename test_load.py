import torch

try:
    # Try to load the file we just generated
    data = torch.load('synthetic_aegis_graph.pt', weights_only=False)
    
    print("✅ Successfully loaded the graph!")
    print(f"Number of accounts: {data['account'].num_nodes}")
    print(f"Number of transactions: {data['account', 'transacts', 'account'].num_edges}")
    
except Exception as e:
    print(f"❌ Failed to load graph. Error: {e}")