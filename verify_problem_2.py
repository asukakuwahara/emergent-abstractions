import torch

with open('data/dim(3,3)_sf10.ds', 'rb') as f:
    dataset = torch.load(f, weights_only=False)

print("Is hierarchical?", dataset.hierarchical)  
print("Number of concepts:", len(dataset.concepts))

# Get ALL unique fixed vectors
unique_fixed_3_3 = set([concept[1] for concept in dataset.concepts])
print(f"\nFound {len(unique_fixed_3_3)} unique fixed vectors:")
for fixed in sorted(unique_fixed_3_3):
    print("  ", fixed)

with open('data/dim(3,4)_sf10.ds', 'rb') as f:
    dataset = torch.load(f, weights_only=False)

print("Is hierarchical?", dataset.hierarchical)  
print("Number of concepts:", len(dataset.concepts))

# Get ALL unique fixed vectors
unique_fixed_3_4 = set([concept[1] for concept in dataset.concepts])
print(f"\nFound {len(unique_fixed_3_4)} unique fixed vectors:")
for fixed in sorted(unique_fixed_3_4):
    print("  ", fixed)
    
with open('data/dim(4,4)_sf10.ds', 'rb') as f:
    dataset = torch.load(f, weights_only=False)

print("Is hierarchical?", dataset.hierarchical)  
print("Number of concepts:", len(dataset.concepts))

# Get ALL unique fixed vectors
unique_fixed_4_4 = set([concept[1] for concept in dataset.concepts])
print(f"\nFound {len(unique_fixed_4_4)} unique fixed vectors:")
for fixed in sorted(unique_fixed_4_4):
    print("  ", fixed)

with open('data/dim(3,8)_sf10.ds', 'rb') as f:
    dataset = torch.load(f, weights_only=False)

print("Is hierarchical?", dataset.hierarchical)  
print("Number of concepts:", len(dataset.concepts))

# Get ALL unique fixed vectors
unique_fixed_8_3= set([concept[1] for concept in dataset.concepts])
print(f"\nFound {len(unique_fixed_8_3)} unique fixed vectors:")
for fixed in sorted(unique_fixed_8_3):
    print("  ", fixed)