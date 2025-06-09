"""
Trains a PyTorch image classification model using device-agnostic code.
"""
import os
import torch
from torchvision import transforms
from timeit import default_timer as timer 
import data_setup, engine, utils, model_builder
import argparse

# hyperparameters
NUM_EPOCHS = 5
BATCH_SIZE = 32
HIDDEN_UNITS = 10 
LEARNING_RATE = 0.001
train_dir = "data/pizza_steak_sushi/train"
test_dir = "data/pizza_steak_sushi/test"

# CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=NUM_EPOCHS, help="Number of epochs (default: 5)")
parser.add_argument("--batch", type=int, default=BATCH_SIZE, help="Batch size (default: 32)")
parser.add_argument("--hu", type=int, default=HIDDEN_UNITS, help="Hidden units (default: 10)")
parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate (default: 0.001)")
parser.add_argument("--traindir", type=str, default=train_dir, help="Training data directory")
parser.add_argument("--testdir", type=str, default=test_dir, help="Testing data directory")

# Parse arguments
args = parser.parse_args()

# Override defaults with CLI arguments
NUM_EPOCHS = args.epoch
BATCH_SIZE = args.batch
HIDDEN_UNITS = args.hu
LEARNING_RATE = args.lr
train_dir = args.traindir
test_dir = args.testdir

print(f"Config: Epochs={NUM_EPOCHS}, Batch={BATCH_SIZE}, HU={HIDDEN_UNITS}, LR={LEARNING_RATE}")
print(f"Directories: Train={train_dir} | Test={test_dir}")


device = "cuda" if torch.cuda.is_available() else "cpu"

data_transform = transforms.Compose([
    transforms.Resize((64, 64)), 
    transforms.ToTensor()
])

train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir, 
                                                                               test_dir, 
                                                                               data_transform,
                                                                               BATCH_SIZE)
model = model_builder.TinyVGG(3, HIDDEN_UNITS, len(class_names)).to(device)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

start_time = timer()

engine.train(model=model,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader, 
            loss_fn=loss_fn, 
            optimizer=optimizer, 
            epochs=NUM_EPOCHS, 
            device=device)

end_time = timer()
print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")


utils.save_model(model=model, 
                target_dir="models", 
                model_name="modular_script_model.pth")
