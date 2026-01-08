# Installing Dependencies

## from argparse import Namespace:
Used to read YAML configuration files and convert arguments into an easily accessible Python object (e.g., `args.lr`, `args.batch_size`).

!!YAML is a human-readable data serialization language often used for writing configuration files.

## import egg.core as core:
Core module of the "Emergent Language at Google" (EGG) library. This library provides ready-made tools for communication games and multi-agent systems (training loops, game definitions, etc.).

## import os:
Used for operating system operations (creating file paths, directory management, etc.).

## import itertools:
Used to create data combinations and permutations (can be used when generating datasets).

## import yaml:
Used to read configuration files in YAML format (for loading hyperparameters from an external file).

## from language_analysis_local import *:
Likely a local module containing custom functions for analyzing the emergent language (e.g., extracting message statistics).

## opts = core.init(params=['--random_seed=7', # will initialize numpy, torch, and python RNGs
                         '--lr=1e-3',   # sets the learning rate for the selected optimizer 
                         '--batch_size=32',
                         '--optimizer=adam']):

Initializes all EGG training-related settings (random seed, learning rate, batch size, optimizer) in a single line. Uses EGG's own argument parser.
* Specifies the size of a batch. During training, the model processes 32 samples simultaneously.
* optimizer='man' specifies the learning/algorithm optimization method of the model. Adam (Adaptive Moment Estimation) is one of the most widely used optimization algorithms in deep learning.

What it does:

Updates model parameters: Adjusts weights to reduce the model's prediction error (loss).

Adaptive learning rate: It uses different learning rates for each parameter, providing faster and more stable convergence.

## def load_config(path: str) -> Namespace:
    """Read a yaml file and return an argparse.Namespace."""
    with open(path, 'r') as f:
        cfg_dict = yaml.safe_load(f)          # → plain dict
    return Namespace(**cfg_dict):
This function reads a YAML file and converts the settings inside into a Namespace object (better for code reusability when reading settings from a file).

---------- What I know  -----------


Purpose of the game: You correctly understood that it is a Sender–Receiver communication game where the Receiver tries to guess the target based on the Sender's message. ✅

train.py: Will be used for training. ✅

Dataset: It is understood that data will be generated/created. ✅

torch: Correct, used for PyTorch. ✅

archs import: Used to import the Sender and Receiver models from the `archs.py` file. ✅

device: Use CUDA (GPU) if available, otherwise use CPU. ✅

SPLIT: %60 train, %20 validation, %20 test split (the last part is typically for validation). ✅

pickle: Used to save outputs (models, results, etc.) in a specific format. ✅

------------------------------------------------------------------

# Configurations

# config.yml:
Contains all settings (hyperparameters) of the game

opts = load_config('config.yml') 

# vocab size and message length
opts.dimensions = list(itertools.repeat(opts.values, opts.attributes))
vocab_size = opts.vocab_size_user
print("vocab size", vocab_size)

opts.values: How many different values ​​each attribute has (ex: 4 options for color)

opts.attributes: How many different attributes are there (ex: 3 → color, shape, size)

Result: opts.dimensions = [4, 4, 4] → 3 attributes, 4 options each 

# allow user to specify a maximum message length
if opts.max_mess_len > 0:
    max_len = opts.max_mess_len
# default: number of attributes
else:
    max_len = len(opts.dimensions)
print("message length", max_len)

-------------------------------------------------------------------------

# Preparations
FILE SYSTEM SETUP

AIM:
Recording test results regularly.

1. CREATE BASIC FOLDERS
#1) 'data' folder - For raw data
#2) 'results' folder - For results

2. DETERMINE THE FOLDER NAME

 Example: "(3,4)_game_size_128"
 Meaning: 3 features, 4 options in each feature, 128 games

3. DETERMINE GAME TYPE
# There are two options:
#1) 'context_aware' - Receiver sees ALL options
#2) 'context_unaware' - Receiver sees ONLY the target

Context Aware: "Give me the BLUE one of the 3 books on the table."
Context Unaware: "Give me the BLUE book" (you don't see the others)

4. CREATE FULL FOLDER PATH
# Example path:
"results/(3,4)_game_size_128/context_aware/"

5. SAVING PROCESS
# If saving is on (opts.save = True):
#1) Create a new "run" folder (0, 1, 2...)
#2) Save all settings in 'params.pkl' file

FINAL FILE STRUCTURE:
results/
└── (3,4)_game_size_128/ # Game features
    └── context_aware/ # Game type
        ├── 0/ # 1st attempt
        │ └── params.pkl # Settings record
        ├── 1/ # 2nd attempt
        ├── 2/ # 3rd try
        └── ...

*We create a separate file for each experiment and a separate folder for each setting, which:

*no confusion
*Make it easy to find later
*Let's compare different experiments

----------------------------------------------------------------------

# Dataset
# generate if not given 
if not opts.load_dataset:
    data_set = dataset.DataSet(opts.dimensions,
                               game_size=opts.game_size,
                               scaling_factor=opts.scaling_factor,
                               device=device)
else:
    data_set = torch.load(opts.path + 'data/' + opts.load_dataset)
    print('data loaded from: ' + 'data/' + opts.load_dataset)
        
train, val, test = data_set

dimensions = train.dimensions

train = torch.utils.data.DataLoader(train, batch_size=opts.batch_size, shuffle=True)
val = torch.utils.data.DataLoader(val, batch_size=opts.batch_size, shuffle=False, drop_last=True)
test = torch.utils.data.DataLoader(test, batch_size=opts.batch_size, shuffle=False)

# drop_last=True:
100 examples:
[1-32] [33-64] [65-96] [97-100]
  ✓ ✓ ✓ ✗ (only 4 examples!)
Full batch Full batch Full batch MISSING batch

------------------------------------------------------------------------------

# Game

Summary: "Sender produces the message, Receiver interprets it, loss is calculated, the model learns to predict correctly!

1. LOSS FUNCTION:
loss_fn = nn.BCEWithLogitsLoss()
BCEWithLogitsLoss = Binary Cross Entropy with Sigmoid

Binary: 0 or 1 for each object (target or not?)
Sigmoid: Squeezes the output between 0-1

Example:
receiver_output: [0.8, -0.2, 0.1, -0.5] # 4 objects
labels: [1.0, 0.0, 0.0, 0.0] # Only 1st object target

# After sigmoid:
[0.69, 0.45, 0.52, 0.38] > 0.5 = [1, 0, 1, 0]
# Error: 3rd object is wrong!

2. ACCURACY ACCOUNT:
receiver_pred = (receiver_output > 0).float()
# 1 if greater than 0, 0 otherwise

3. IDENTIFICATION OF SENDER AND RECEIVER:

4. GAME WRAPPERS:
python
core.RnnSenderGS # Sender with Gumbel-Softmax (GS kısmı)
core.RnnReceiverGS # Receiver with Gumbel-Softmax  
core.SenderReceiverRnnGS # Combine the two

# What is Gumbel-Softmax? 🤔
Normally the message selection is "sharp" (1, 0, 0)
Gumbel-Softmax softens: (0.7, 0.2, 0.1)
Advantage: Provides gradient flow, training becomes easier

# temperature=opts.temperature # THIS IS VERY IMPORTANT!
temperature HIGH → Softer: [0.25, 0.25, 0.25, 0.25]
temperature LOW → Harder: [0.1, 0.1, 0.7, 0.1]
temperature → 0 → Full hard: [0, 0, 1, 0]

# output:
RnnSenderGS(
  (agent): Sender(
    (fc1): Linear(in_features=120, out_features=128, bias=True)
    (fc2): Linear(in_features=120, out_features=128, bias=True)
    (fc3): Linear(in_features=256, out_features=128, bias=True)
  )
  (hidden_to_output): Linear(in_features=128, out_features=16, bias=True)
  (embedding): Linear(in_features=16, out_features=64, bias=True)
  (cell): GRUCell(64, 128)
)

✨ EASY TO REMEMBER:
text
120 → ANIMAL CARD (how much detail)
128 → ROBOT BRAIN (how smart)  
16 → PAINT BOX (how many colors are there)
64 → COLOR CARD (how well the color is described)
"The robot sees 120 details, thinks with 128 brains, chooses from 16 colors, and describes colors with 64 features!" 🎨

-------------------------------------------------------------------
# Training

callbacks = [
    SavingConsoleLogger(...), # Logging and recording
    TemperatureUpdater(...), # Gumbel-Softmax temperature setting
    InteractionSaver(...), # Save agent interactions
    CheckpointSaver(...) # Model checkpoints


trainer = core.Trainer(
    game=game, # Sender-Receiver game
    optimizer=optimizer, # Optimization algorithm
    train_data=train, 60% training set
    validation_data=val, 20% validation set
    callbacks=callbacks, # Monitoring and recording mechanisms
    device=device # GPU/CPU

# Epoch 1-3 (Adaptation Phase):
text
Train Acc: 53% → 79%      (+26%)
Test Acc:  64% → 79%      (+15%)
Length:    2.6 → 3.5      (+0.9)

Observation: Fast learning, message length increases

# Epoch 4-7 (Consolidation Phase):
text
Train Acc: 79% → 87% (+8%)
Test Acc: 79% → 82% (+3%)
Length: 3.5 → 4.0 (+0.5)

Observation: Learning slows down, risk of overfitting

# Epoch 8-10 (Saturation Phase):
text
Train Acc: 87% → 90% (+3%)
Test Acc: 82% → 86% (+4%)
Length: 4.0 → 4.0 (fixed)

Observation: Maximum message length reached, performance plateaus

------------------------------------------------------------------------

# Results and analysis

RESULTING FILE STRUCTURE:
text
results/
└── (3,4)_game_size_128/ # Dataset configuration
    └── context_aware/ # Game type
        └── 0/ # Run #0 (trial #1)
            ├── params.pkl # 1. SETTINGS FILE
            ├── loss_and_metrics.pkl # 2. METRICS FILE
            ├── final.tar # 3. TRAINED MODEL
            └── interactions/ # 4. INTERACTION RECORDS
                ├── train_interactions.pkl
                ├── val_interactions.pkl
                └── test_interactions.pkl

# 1. PARAMS.PKL - CONFIGURATION RECORD

{
    'random_seed': 7, # Randomness seed
    'learning_rate': 0.001, # Learning rate
    'batch_size': 32, # Batch size
    'hidden_size': 128, # Model hidden layer size
    'vocab_size': 16, # Number of symbols
    'max_mess_len': 4, # Maximum message length
    'game_size': 10, # Number of objects per game
    'n_epochs': 10, # Number of training rounds
    'context_unaware': False, # Game type
    'dimensions': [4, 4, 4], # Dataset dimensions
    'save_path': 'results/(3,4)_game_size_128/context_aware/0'

WHY IS IT IMPORTANT?

Reproducibility: Ability to run again with the same settings
Comparison: Compare different configurations
Debugging: Tracking problems

# 2. LOSS_AND_METRICS.PKL - PERFORMANCE DATA
CONTAINS:
python
{
    # Metrics (lists) by epoch:
    'train_loss': [0.689, 0.572, 0.467, ...], #10 epoch
    'train_acc': [0.534, 0.704, 0.790, ...], #10 epoch
    'train_length': [2.65, 2.96, 3.68, ...], #10 epoch
    
    'val_loss': [0.653, 0.515, 0.496, ...], #10 epoch  
    'val_acc': [0.641, 0.771, 0.790, ...], #10 epoch
    'val_length': [2.34, 3.53, 4.00, ...], #10 epoch
    
    # Final test performance:
    'final_test_loss': 0.3537483513355255,
    'final_test_acc': 0.8634548187255859,
    
    # Additional metrics:
    'epoch_times': [32.1, 31.8, 31.5, ...], # Epoch times
    'best_epoch': 8 # Best performance
}

## USE FOR ANALYSIS: ##
python
import pickle
import matplotlib.pyplot as plt

# Load data
with open('loss_and_metrics.pkl', 'rb') as f:
    metrics = pickle.load(f)

# Loss chart
plt.plot(metrics['train_loss'], label='Train')
plt.plot(metrics['val_loss'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Accuracy chart
plt.plot(metrics['train_acc'], label='Train')
plt.plot(metrics['val_acc'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# 3. FINAL.TAR - TRAINED MODEL
CONTAINS:
python
{
    'sender_state_dict': {...}, # Sender model parameters
    'receiver_state_dict': {...}, # Receiver model parameters
    'optimizer_state_dict': {...}, # Optimizer state
    'epoch': 10, # Last epoch
    'loss': 0.283, # Last loss
    'accuracy': 0.886, # Latest accuracy
    'config': {...} # Model configuration
}

## LOADING THE MODEL: ##
python
import torch

# Load model
checkpoint = torch.load('final.tar', map_location=device)

# Recreate Sender and Receiver
sender.load_state_dict(checkpoint['sender_state_dict'])
receiver.load_state_dict(checkpoint['receiver_state_dict'])

# To continue training:
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# 4. INTERACTIONS/ - INTERACTION RECORDS
CONTAINS (train_interactions.pkl):
python
{
    'sender_input': tensor(...), # Input objects [batch, features]
    'message': tensor(...), # Generated messages [batch, seq_len]
    'receiver_input': tensor(...), # Receiver input
    'receiver_output': tensor(...), # Receiver output [batch, n_objects]
    'labels': tensor(...), # Actual targets [batch, n_objects]
    'aux': {'acc': tensor(...)}, # Accuracy values
    'length': tensor(...) # Message lengths
}

## USES FOR SCIENTIFIC ANALYSIS: ##
1. Language Features Analysis:
Compositionality: Can messages be broken down into components?
Systematicity: Similar entries → similar messages?
Efficiency: Message length vs accuracy trade-off

2. Model Behavior:
Emergent Protocols: Communication protocols developed by robots
Error Patterns: In what situations does it make errors?
Generalization: How does it perform on data it has not seen?


View Results:
bash
ls -la results/(3,4)_game_size_128/context_aware/0/
# params.pkl, loss_and_metrics.pkl, final.tar, interactions/

SUMMARY TABLE:
File                    Content                              Purpose of Use
params.pkl               All configurations                 Reproducibility, comparison
loss_and_metrics.pkl     Training metrics                   Performance analysis, charting
final.tar                Model checkpoint                   Model deploy, continue training
interactions/            Interaction records                Language analysis, error analysis

---------------------------------------------------------------------------

# Model performance

1. INSTALL REQUIRED LIBRARIES:
python
import pandas as pd # For data analysis
from matplotlib import pyplot as plt # For plotting
plt.style.use('default') # Default graphic style
import random # For randomness
import seaborn # For statistical graphs

# Special helper functions:
from utils.load_results import * # Loading results
from utils.plot_helpers import * # Plot helpers  
from utils.analysis_from_interaction import * # Interaction analysis

2. LOAD RESULTS:
python
paths = [folder_name] # 'results/(3,4)_game_size_10'
# folder_name: The folder path we created before

all_accuracies = load_accuracies(
    paths, # Results in which folder
    n_runs=1, # How many different runs
    n_epochs=opts.n_epochs, # How many epochs (10)
    val_steps=1, # How often was validation done
    context_unaware=False # In context-aware mode
)

3. DRAW THE GRAP:
python
plot_training_trajectory(
    all_accuracies['train_acc'], # Training accuracies
    all_accuracies['val_acc'], # Validation accuracies
    ylim=(0.5, 1), # Y axis limits (%50-100%)
    steps=(1, 1), # 1 train, 1 val per epoch
    n_epochs=10, # Total number of epochs
    plot_indices=(1,) # Plot which runes (run 1)
)

We can also plot how the message length varies during training.

plot_training_trajectory(all_accuracies['train_acc'], all_accuracies['val_acc'], message_length_train=all_accuracies['train_message_lengths'], message_length_val=all_accuracies['val_message_lengths'], steps=(1, 1), n_epochs=10, plot_indices=(1,), message_length_plot=True, train_only=True)

------------------------------------------------------------------------

# Interactions

