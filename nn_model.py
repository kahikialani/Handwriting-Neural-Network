import pandas as pd
import os
from PIL import Image
from tqdm import tqdm
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class CRNN(nn.Module):
    def __init__(self, img_height, img_width, num_classes, hidden_size = 256):
        super(CRNN, self).__init__()

        self.img_height = img_height
        self.img_width = img_width
        self.num_classes = num_classes

        self.cnn = nn.Sequential(
            
            nn.Conv2d(1, 32, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = (2,2)),

            nn.Conv2d(32, 64, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = (2,2)),

            nn.Conv2d(64, 128, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = (2,1)),

            nn.Conv2d(128, 256, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = (2,1)),
        )
        self.rnn_input_size = 256 * 3
        self.rnn = nn.LSTM(
            input_size = self.rnn_input_size,
            hidden_size = hidden_size,
            num_layers = 2,
            bidirectional = True,
            batch_first = True
        )

        self.classifier = nn.Linear(hidden_size * 2, num_classes)
    
    def forward(self, x):
        conv_features = self.cnn(x)

        batch_size, channels, height, width = conv_features.size()
        conv_features = conv_features.permute(0, 3, 1, 2)
        conv_features = conv_features.contiguous().view(batch_size, width, channels * height)
        
        rnn_out, _ = self.rnn(conv_features)

        output = self.classifier(rnn_out)
        return output

class HandwritingDataset(Dataset):
    def __init__(self, csv_file, img_dir, char_to_idx, transform=None, skip_range=None):
        self.data = csv_file.dropna().reset_index(drop=True)
        self.img_dir = img_dir
        self.char_to_idx = char_to_idx
        self.transform = transform

        if skip_range:
            start_idx, end_idx = skip_range
            self.data = pd.concat([
                self.data.iloc[:start_idx],
                self.data.iloc[end_idx:]
            ]).reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_name = row['FILENAME']
        label = row['IDENTITY']
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        label_indicies = [self.char_to_idx[char] for char in label]
        return image, torch.tensor(label_indicies, dtype = torch.long), label
    
# Clean the data to strip lowercase letters and transition them to all upper.
def clean_labels(df):
    df_clean = df.copy()
    df_clean['IDENTITY'] = df_clean['IDENTITY'].str.upper()
    return df_clean

def collate_fn(batch):
    images, label_indices, label_texts = zip(*batch)
    images = torch.stack(images)
    input_lengths = torch.full((len(images),), 71, dtype=torch.int32)
    target_lengths = torch.tensor([len(label) for label in label_indices], dtype=torch.int32)
    targets = torch.cat(label_indices)
    return images, targets, input_lengths, target_lengths, label_texts

def train_epoch(model, train_loader, optimizer, ctc_loss, device):
    model.train()
    total_loss = 0
    pbar = tqdm(train_loader, desc = "Training")
    for batch_idx, (images, targets, input_lengths, target_lengths, _) in enumerate(pbar):
        #move images to device
        images = images.to(device)
        targets = targets.to(device)
        #forward pass
        outputs = model(images)
        log_probs = F.log_softmax(outputs, dim=2).transpose(0,1)

        # compute the loss
        loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)

        #backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0) # gradient clipping modifier
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss':f'{loss.item():.4f}'})

        if batch_idx > 0 and batch_idx % print_every == 0:
            avg_loss = total_loss / (batch_idx +1)
            print(f"Batch {batch_idx}, Average Loss: {avg_loss:.4f}")
        if batch_idx % 2000 == 0 and batch_idx > 0:
            gc.collect()
    return total_loss / len(train_loader)

def validate(model, val_loader, ctc_loss, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for images, targets, input_lengths, target_lengths, _ in tqdm(val_loader, desc = "Validating"):
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            log_probs = F.log_softmax(outputs, dim = 2).transpose(0,1)
            loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
            total_loss += loss.item()
    return total_loss / len(val_loader)

def decode_predictions(outputs, idx_to_char):
    predictions = []
    for output in outputs:
        pred_indices = torch.argmax(output, dim = 1)
        decoded = []
        prev_char = None
        for idx in pred_indices:
            char_idx = idx.item()
            if char_idx != 0 and char_idx != prev_char:
                decoded.append(idx_to_char[char_idx])
            prev_char = char_idx
        predictions.append(''.join(decoded))
    return predictions

train_df = pd.read_csv('data/written_name_train_v2.csv')
test_df = pd.read_csv('data/written_name_test_v2.csv')
val_df = pd.read_csv('data/written_name_validation_v2.csv')
train_image_dir = 'data/train_v2/train/'

train_df_clean = clean_labels(train_df)
test_df_clean = clean_labels(test_df)
val_df_clean = clean_labels(val_df)
all_train_chars = set(''.join(train_df_clean['IDENTITY'].dropna().tolist()))
all_test_chars = set(''.join(test_df_clean['IDENTITY'].dropna().tolist()))
all_val_chars = set(''.join(val_df_clean['IDENTITY'].dropna().tolist()))
vocab_chars = sorted(list(all_train_chars | all_test_chars | all_val_chars))

# Adds a CTC blankspace for model requirements.
vocab_chars = [' '] + vocab_chars
# Creates an index for character id's and the inverse index
char_to_idx = {char: idx for idx, char in enumerate(vocab_chars)}
idx_to_char = {idx: char for char, idx in char_to_idx.items()}

#
IMG_HEIGHT = 50
IMG_WIDTH = 284
NUM_CLASSES = len(vocab_chars)

# Transform IMAGES to grayscale
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),                       
    transforms.Normalize(mean=[0.5], std=[0.5])   
])

# Calls model structure
model = CRNN(
    img_height=IMG_HEIGHT,
    img_width=IMG_WIDTH,
    num_classes=NUM_CLASSES,
    hidden_size=256
)

# Used to skip failing corrupt data in the data set.
skip_start = 24800 * 16
skip_end = 25500 * 16
# Creates training dataset
train_dataset = HandwritingDataset(
    csv_file = train_df_clean,
    img_dir = 'data/train_v2/train/',
    char_to_idx = char_to_idx,
    transform = transform,
    skip_range = (skip_start, skip_end)
)

# Creates validation dataset
val_dataset = HandwritingDataset(
    csv_file = val_df_clean,
    img_dir = 'data/validation_v2/validation',
    char_to_idx = char_to_idx,
    transform = transform
)

# Creates training and validation data loaders as well as setting the batch_size
batch_size = 16
train_loader = DataLoader(
    train_dataset,
    batch_size = batch_size,
    shuffle = True,
    collate_fn = collate_fn,
    num_workers = 0
)
val_loader = DataLoader(
    val_dataset,
    batch_size = batch_size,
    shuffle = False,
    collate_fn = collate_fn,
    num_workers = 0
)

# Sets u CTC loss
ctc_loss = nn.CTCLoss(blank = 0, reduction = "mean", zero_infinity = True)

# Assigning the utilizing device, as well as the optimizer / scheduler, and modifiable attributes (3 epochs, progress print ever 1000 batches, etc)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr = 0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = 2, factor = 0.5)
num_epochs = 3
model.train()
test_branches = 3
print_every = 1000

# Training Setup

for batch_idx, (images, targets, input_lengths, target_lengths, texts) in enumerate(train_loader):
    if batch_idx >= test_branches:
        break
    images = images.to(device)
    targets = targets.to(device)
    outputs = model(images)
    log_probs = F.log_softmax(outputs, dim=2).transpose(0,1)
    loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
    print(f"Batch {batch_idx +1} loss: {loss.item():.4f}")
print("Training setup test completed successfully")

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Training batches per epoch: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")


# Taking the training setup and implementing the model to actually train / validate it.

train_losses = []
val_losses = []

print(f"Starting training for {num_epochs} epochs...")
print(f"Training batches per epoch: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")

for epoch in range(num_epochs):
    print(f"\n{'='*50}")
    print(f"Epoch {epoch + 1}/{num_epochs}")
    print(f"{'='*50}")
    
    # Training
    train_loss = train_epoch(model, train_loader, optimizer, ctc_loss, device)
    train_losses.append(train_loss)
    
    # Validation
    val_loss = validate(model, val_loader, ctc_loss, device)
    val_losses.append(val_loss)
    
    # Learning rate scheduling
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    
    print(f"\nEpoch {epoch + 1} Results:")
    print(f"  Train Loss: {train_loss:.4f}")
    print(f"  Val Loss: {val_loss:.4f}")
    print(f"  Learning Rate: {current_lr:.6f}")
    
    # Show sample predictions
    print("Sample of Predictions:")
    model.eval()
    with torch.no_grad():
        sample_batch = next(iter(val_loader))
        images, targets, input_lengths, target_lengths, true_texts = sample_batch
        images = images.to(device)
        
        outputs = model(images)
        predictions = decode_predictions(outputs[:5], idx_to_char)
        
        for i, (pred, true) in enumerate(zip(predictions, true_texts[:5])):
            accuracy = "O" if pred == true else "X"
            print(f"  {i+1}. {accuracy} Predicted: '{pred}' | Actual: '{true}'")
    
    # Save checkpoint after each epoch
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),

        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'char_to_idx': char_to_idx,
        'idx_to_char': idx_to_char,
        'vocab_chars': vocab_chars,
        'config': {
            'IMG_HEIGHT': IMG_HEIGHT,
            'IMG_WIDTH': IMG_WIDTH,
            'NUM_CLASSES': NUM_CLASSES,
            'hidden_size': 256
        }
    }
    
    torch.save(checkpoint, f'crnn_epoch_{epoch + 1}.pth')
    print(f"Checkpoint saved: crnn_epoch_{epoch + 1}.pth")

print("\nTraining completed!")

# Save final model
torch.save(checkpoint, 'crnn_model_final.pth')
print("Final model saved as: crnn_model_final.pth")
