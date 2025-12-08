import torch

with open('data/dim(3,3)_sf10.ds', 'rb') as f:
    dataset = torch.load(f, weights_only=False)

print("Is hierarchical?", dataset.hierarchical)  
print("Number of concepts:", len(dataset.concepts))

# Get ALL unique fixed vectors
unique_fixed = set([concept[1] for concept in dataset.concepts])
print(f"\nFound {len(unique_fixed)} unique fixed vectors:")
for fixed in sorted(unique_fixed):
    print("  ", fixed)