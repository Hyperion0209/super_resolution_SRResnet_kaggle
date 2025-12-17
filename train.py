import os
import math
import time
import argparse # Can be used later for command-line args
import random
import base64 # Needed for submission
import pandas as pd # Needed for submission
import cv2 # Needed for submission encoding/saving

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image # Useful for saving intermediate test outputs
from PIL import Image
from tqdm import tqdm # Progress bar
from skimage.metrics import peak_signal_noise_ratio as calculate_psnr
# from skimage.metrics import structural_similarity as calculate_ssim # Optional: Uncomment if you want SSIM

import matplotlib.pyplot as plt # For plotting

# --- Configuration ---
# TODO: Adjust these paths and hyperparameters
DATA_DIR = 'train-kaggle'   # Path to the main dataset directory containing lr/hr folders
# ! IMPORTANT: Set the correct path to your test images
TEST_DATA_DIR = 'lr'   # Path to the low-resolution test images folder
OUTPUT_DIR = 'checkpoints_and_outputs_last' # Directory for model checkpoints, logs, plots, test outputs
TEST_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'test_predictions') # Subdir for saving predicted HR test images
SUBMISSION_CSV_FILE = os.path.join(OUTPUT_DIR, 'submission.csv') # Path for the final submission file
PLOT_FILE = os.path.join(OUTPUT_DIR, 'training_plots.png') # Path to save plots
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, 'best_srresnet_split.pth')
LOG_FILE = os.path.join(OUTPUT_DIR, 'training_log_split.txt')

# Data Splitting
VAL_SPLIT_RATIO = 0.15
RANDOM_SEED = 42

# Model hyperparameters
UPSCALE_FACTOR = 4
NUM_RES_BLOCKS = 16
IMG_CHANNELS = 3

# Training hyperparameters
NUM_EPOCHS = 100
BATCH_SIZE = 8 # Adjust based on GPU memory
LEARNING_RATE = 1e-3
NUM_WORKERS = 2 # Adjust based on your system CPU cores

# --- Ensure Output Directories Exist ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEST_OUTPUT_DIR, exist_ok=True) # Create test output dir

# --- SRDataset Definition (No changes needed) ---
class SRDataset(Dataset):
    """
    Custom PyTorch Dataset for Super-Resolution tasks.
    Modified to accept a list of filenames for a specific split.
    """
    def __init__(self, root_dir, filenames, lr_transform=None, hr_transform=None):
        self.root_dir = root_dir
        self.lr_dir = os.path.join(root_dir, 'lr')
        self.hr_dir = os.path.join(root_dir, 'hr')
        self.lr_transform = lr_transform
        self.hr_transform = hr_transform
        self.image_filenames = filenames

        if not os.path.isdir(self.lr_dir) or not os.path.isdir(self.hr_dir):
            raise FileNotFoundError(f"LR or HR directory not found in {root_dir}")
        if not self.image_filenames:
            print(f"Warning: No filenames provided for this dataset split.")

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        if idx >= len(self.image_filenames):
            raise IndexError("Index out of bounds")

        img_name = self.image_filenames[idx]
        lr_img_path = os.path.join(self.lr_dir, img_name)
        hr_img_path = os.path.join(self.hr_dir, img_name)

        try:
            if not os.path.exists(hr_img_path):
                print(f"Warning: HR file missing for {img_name} at {hr_img_path}. Skipping.")
                return None, None # Filtered by collate_fn

            lr_image = Image.open(lr_img_path).convert('RGB')
            hr_image = Image.open(hr_img_path).convert('RGB')
        except FileNotFoundError:
            print(f"Error loading image: {img_name}. Check paths: {lr_img_path}, {hr_img_path}")
            return None, None
        except Exception as e:
            print(f"Error opening image {img_name}: {e}")
            return None, None

        orig_lr, orig_hr = lr_image, hr_image
        if self.lr_transform:
            lr_image = self.lr_transform(lr_image)
        if self.hr_transform:
            hr_image = self.hr_transform(hr_image)

        if not isinstance(lr_image, torch.Tensor) or not isinstance(hr_image, torch.Tensor):
            print(f"Warning: Transforms did not return tensors for {img_name}. Check transform definitions.")
            return None, None

        return lr_image, hr_image

# --- SRResNet Model Definition (No changes needed) ---
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        return out

class SRResNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, feature_channels=64, num_res_blocks=16, upscale_factor=4):
        super(SRResNet, self).__init__()
        self.conv_input = nn.Conv2d(in_channels, feature_channels, kernel_size=9, padding=4)
        self.relu_input = nn.ReLU(inplace=True)
        res_blocks = [ResidualBlock(feature_channels) for _ in range(num_res_blocks)]
        self.residual_blocks = nn.Sequential(*res_blocks)
        self.conv_mid = nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1)
        self.bn_mid = nn.BatchNorm2d(feature_channels)
        num_upsample_blocks = int(math.log2(upscale_factor))
        upsample_blocks = []
        for _ in range(num_upsample_blocks):
            upsample_blocks += [
                nn.Conv2d(feature_channels, feature_channels * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.ReLU(inplace=True)
            ]
        self.upsampling = nn.Sequential(*upsample_blocks)
        self.conv_output = nn.Conv2d(feature_channels, out_channels, kernel_size=9, padding=4)

    def forward(self, x):
        out_input = self.conv_input(x)
        out_input_relu = self.relu_input(out_input)
        # residual = out_input_relu # Original SRResNet doesn't use this as input to blocks
        out_res = self.residual_blocks(out_input_relu) # Pass input relu output to blocks
        out_mid = self.conv_mid(out_res)
        out_mid = self.bn_mid(out_mid)
        out_mid_skip = out_mid + out_input_relu # Skip connection *after* BN
        out_up = self.upsampling(out_mid_skip)
        out = self.conv_output(out_up)
        return out


# --- Helper Functions ---
def log_message(message):
    print(message)
    with open(LOG_FILE, 'a') as f:
        f.write(message + '\n')

def calculate_metrics(output, target):
    # Ensure tensors are on CPU, converted to numpy, and clipped
    output_np = output.detach().cpu().numpy().clip(0, 1)
    target_np = target.detach().cpu().numpy().clip(0, 1)

    # Transpose from [B, C, H, W] to [B, H, W, C] for skimage
    # Note: skimage PSNR/SSIM often expect channel-last format or handle channel-first
    # Check skimage version documentation if issues arise. Assuming channel-first works here.
    # If not, uncomment the transpose lines:
    # output_np = np.transpose(output_np, (0, 2, 3, 1))
    # target_np = np.transpose(target_np, (0, 2, 3, 1))

    batch_psnr = 0
    # batch_ssim = 0 # Optional
    valid_images = 0
    for i in range(output_np.shape[0]):
        try:
            # Ensure target and output have same shape for metrics
            # This shouldn't happen if data loading is correct, but a safety check
            if target_np[i].shape != output_np[i].shape:
                 print(f"Warning: Shape mismatch between target {target_np[i].shape} and output {output_np[i].shape} for image {i}. Skipping metrics.")
                 continue

            psnr = calculate_psnr(target_np[i], output_np[i], data_range=1.0)
            # ssim = calculate_ssim(target_np[i], output_np[i], data_range=1.0, channel_axis=0, win_size=7) # Adjust channel_axis if needed, win_size might need tuning

            if math.isinf(psnr): # Handle potential inf values if images are identical
                 # PSNR is infinite if images are identical, often capped at a high value like 100dB
                 # Or you can just acknowledge it happened. skimage might return a large number anyway.
                 # Let's just add it, but be aware it can happen.
                 pass

            batch_psnr += psnr
            # batch_ssim += ssim # Optional
            valid_images += 1
        except ValueError as e:
            print(f"Warning: Metric calculation error: {e}. Skipping image {i}.")
            # This might happen if dimensions are wrong or data range issues occur
            pass

    avg_psnr = batch_psnr / valid_images if valid_images > 0 else 0
    # avg_ssim = batch_ssim / valid_images if valid_images > 0 else 0 # Optional
    return avg_psnr #, avg_ssim

# --- Function for Test Set Prediction ---
def generate_test_predictions(model, test_data_dir, output_dir, device):
    log_message("\n--- Generating Test Set Predictions ---")
    model.eval() # Set model to evaluation mode
    transform = transforms.Compose([transforms.ToTensor()]) # Basic transform for test images

    test_filenames = sorted([f for f in os.listdir(test_data_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if not test_filenames:
        log_message(f"Error: No test images found in {test_data_dir}")
        return

    log_message(f"Found {len(test_filenames)} test images.")

    with torch.no_grad():
        for filename in tqdm(test_filenames, desc="Predicting Test"):
            lr_img_path = os.path.join(test_data_dir, filename)
            try:
                lr_image_pil = Image.open(lr_img_path).convert('RGB')
                lr_tensor = transform(lr_image_pil).unsqueeze(0).to(device) # Add batch dim and send to device

                sr_tensor = model(lr_tensor)

                # Save the generated SR image
                output_filename = os.path.join(output_dir, filename)
                # Clamp output, remove batch dim, save directly using torchvision
                # save_image handles the conversion from [0,1] tensor to image file
                save_image(sr_tensor.clamp(0.0, 1.0).squeeze(0), output_filename)

                # Alternative saving using cv2 (if you prefer specific control)
                # sr_img_np = sr_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0) # C,H,W -> H,W,C
                # sr_img_np = (sr_img_np.clip(0, 1) * 255).astype(np.uint8) # Scale to 0-255
                # sr_img_bgr = cv2.cvtColor(sr_img_np, cv2.COLOR_RGB2BGR) # Convert RGB to BGR for cv2
                # cv2.imwrite(output_filename, sr_img_bgr)

            except FileNotFoundError:
                log_message(f"Error: Test image not found at {lr_img_path}. Skipping.")
            except Exception as e:
                log_message(f"Error processing test image {filename}: {e}")

    log_message(f"Test predictions saved to: {output_dir}")


# --- Function for Submission File Generation ---
def encode_image_to_base64(image_path):
    """Encodes an image file to base64 string."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image {image_path} with cv2. Skipping encoding.")
            return None
        _, buffer = cv2.imencode('.png', image) # Encode as PNG
        return base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None

def create_submission_file(predicted_images_dir, output_csv_file):
    """Generates the submission CSV file."""
    log_message("\n--- Creating Submission File ---")
    encoded_images_data = []
    filenames = sorted(os.listdir(predicted_images_dir)) # Get filenames from predicted outputs

    for file_name in tqdm(filenames, desc="Encoding Images"):
        if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue # Skip non-image files if any

        file_path = os.path.join(predicted_images_dir, file_name)
        encoded_image = encode_image_to_base64(file_path)

        if encoded_image: # Only add if encoding was successful
            encoded_images_data.append({'id': file_name, 'Encoded_Image': encoded_image})
        else:
             log_message(f"Skipping {file_name} due to encoding error.")

    if not encoded_images_data:
        log_message("Error: No images were successfully encoded. Submission file not created.")
        return

    # Create DataFrame from list of dictionaries
    df_encoded = pd.DataFrame(encoded_images_data)
    try:
        df_encoded.to_csv(output_csv_file, index=False)
        log_message(f"Submission file saved to: {output_csv_file}")
    except Exception as e:
        log_message(f"Error saving submission CSV to {output_csv_file}: {e}")


# --- Main Training and Evaluation Script ---
if __name__ == '__main__':

    # --- Setup Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_message(f"Using device: {device}")
    log_message(f"--- Configuration ---")
    log_message(f"Train Data Dir: {DATA_DIR}")
    log_message(f"Test Data Dir: {TEST_DATA_DIR}")
    log_message(f"Output Dir: {OUTPUT_DIR}")
    log_message(f"Validation Split Ratio: {VAL_SPLIT_RATIO}")
    log_message(f"Random Seed: {RANDOM_SEED}")
    log_message(f"Upscale Factor: {UPSCALE_FACTOR}")
    log_message(f"Num Residual Blocks: {NUM_RES_BLOCKS}")
    log_message(f"Epochs: {NUM_EPOCHS}")
    log_message(f"Batch Size: {BATCH_SIZE}")
    log_message(f"Learning Rate: {LEARNING_RATE}")
    log_message(f"---------------------\n")

    # --- Data Splitting Logic ---
    # (Same as before, seems robust)
    try:
        lr_data_dir = os.path.join(DATA_DIR, 'lr')
        hr_data_dir = os.path.join(DATA_DIR, 'hr')
        if not os.path.isdir(lr_data_dir): raise FileNotFoundError(f"LR directory not found at {lr_data_dir}")
        if not os.path.isdir(hr_data_dir): raise FileNotFoundError(f"HR directory not found at {hr_data_dir}")

        all_filenames = sorted([f for f in os.listdir(lr_data_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        valid_filenames = [fname for fname in all_filenames if os.path.exists(os.path.join(hr_data_dir, fname))]
        missing_hr = len(all_filenames) - len(valid_filenames)
        if missing_hr > 0: log_message(f"Warning: Skipped {missing_hr} LR images due to missing HR counterparts.")

        if not valid_filenames: raise ValueError("No valid image pairs found in the dataset directory.")

        random.seed(RANDOM_SEED)
        random.shuffle(valid_filenames)
        split_index = int(len(valid_filenames) * (1 - VAL_SPLIT_RATIO))
        train_filenames = valid_filenames[:split_index]
        val_filenames = valid_filenames[split_index:]

        if not train_filenames: raise ValueError("Training split resulted in zero files. Check VAL_SPLIT_RATIO.")
        if not val_filenames: log_message("Warning: Validation split resulted in zero files. Proceeding without validation.")

        log_message(f"Total valid image pairs found: {len(valid_filenames)}")
        log_message(f"Splitting into {len(train_filenames)} training and {len(val_filenames)} validation pairs.")

    except Exception as e:
        log_message(f"Error during data preparation: {e}")
        exit(1)


    # --- Datasets and DataLoaders ---
    # Consider adding augmentations to train_transform if needed
    train_transform = transforms.Compose([
        # transforms.RandomHorizontalFlip(), # Optional augmentation
        transforms.ToTensor()
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None and x[0] is not None and x[1] is not None, batch))
        if not batch: return None, None
        return torch.utils.data.dataloader.default_collate(batch)

    try:
        train_dataset = SRDataset(root_dir=DATA_DIR, filenames=train_filenames, lr_transform=train_transform, hr_transform=val_transform) # Use same ToTensor for HR
        val_dataset = None
        val_loader = None
        if val_filenames:
            val_dataset = SRDataset(root_dir=DATA_DIR, filenames=val_filenames, lr_transform=val_transform, hr_transform=val_transform)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, # Can use larger batch for val if memory allows
                                    num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn)
        else:
            log_message("Skipping validation loader creation.")

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn)

    except Exception as e:
        log_message(f"Error creating Datasets or DataLoaders: {e}")
        exit(1)

    log_message(f"Training DataLoader ready with {len(train_loader)} batches.")
    if val_loader: log_message(f"Validation DataLoader ready with {len(val_loader)} batches.")


    # --- Model, Loss, Optimizer ---
    model = SRResNet(in_channels=IMG_CHANNELS, out_channels=IMG_CHANNELS, feature_channels=64,
                   num_res_blocks=NUM_RES_BLOCKS, upscale_factor=UPSCALE_FACTOR).to(device)

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True) # Example scheduler based on Val Loss

    # --- Training Loop ---
    best_val_psnr = 0.0
    train_losses, val_losses, val_psnrs = [], [], [] # For plotting
    log_message("\n--- Starting Training ---")

    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        model.train()
        epoch_train_loss = 0.0

        pbar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
        for lr_imgs, hr_imgs in pbar_train:
            if lr_imgs is None or hr_imgs is None: continue

            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)

            optimizer.zero_grad()
            sr_imgs = model(lr_imgs)
            loss = criterion(sr_imgs, hr_imgs)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            pbar_train.set_postfix({'Loss': loss.item()})

        avg_train_loss = epoch_train_loss / len(train_loader) if len(train_loader) > 0 else 0
        train_losses.append(avg_train_loss)

        # --- Validation Loop ---
        epoch_val_loss = 0.0
        epoch_val_psnr = 0.0
        if val_loader:
            model.eval()
            valid_batches = 0
            with torch.no_grad():
                pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]")
                for lr_imgs, hr_imgs in pbar_val:
                    if lr_imgs is None or hr_imgs is None: continue

                    lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
                    sr_imgs = model(lr_imgs)
                    loss = criterion(sr_imgs, hr_imgs)
                    epoch_val_loss += loss.item()

                    batch_psnr = calculate_metrics(sr_imgs, hr_imgs) # Add ssim here if using
                    epoch_val_psnr += batch_psnr
                    valid_batches += 1
                    pbar_val.set_postfix({'PSNR': batch_psnr}) # Add SSIM if using

            avg_val_loss = epoch_val_loss / valid_batches if valid_batches > 0 else 0
            avg_val_psnr = epoch_val_psnr / valid_batches if valid_batches > 0 else 0
            val_losses.append(avg_val_loss)
            val_psnrs.append(avg_val_psnr)

            # Checkpointing
            if avg_val_psnr > best_val_psnr:
                best_val_psnr = avg_val_psnr
                torch.save(model.state_dict(), BEST_MODEL_PATH)
                log_message(f"*** Epoch {epoch+1}: New best model saved with Val PSNR: {best_val_psnr:.3f} dB ***")

            # if scheduler: # Example for ReduceLROnPlateau
            #     scheduler.step(avg_val_loss)

        else: # No validation
            val_losses.append(None) # Append placeholder if no validation
            val_psnrs.append(None)

        epoch_time = time.time() - start_time
        log_message(f"Epoch {epoch+1}/{NUM_EPOCHS} | Time: {epoch_time:.2f}s | "
                    f"Train Loss: {avg_train_loss:.4f} | "
                    f"Val Loss: {avg_val_loss:.4f} | " # Will be 0.0 if no val_loader
                    f"Val PSNR: {avg_val_psnr:.3f} dB") # Will be 0.0 if no val_loader

    log_message("\n--- Training Finished ---")
    if val_loader:
        log_message(f"Best Validation PSNR achieved: {best_val_psnr:.3f} dB")
        log_message(f"Best model saved to: {BEST_MODEL_PATH}")
    else:
        log_message("Training finished without validation. Saving final model state.")
        final_model_path = os.path.join(OUTPUT_DIR, 'final_srresnet_split.pth')
        torch.save(model.state_dict(), final_model_path)
        log_message(f"Final model state saved to: {final_model_path}")
        # Use final model for testing if no best model exists
        if not os.path.exists(BEST_MODEL_PATH):
             BEST_MODEL_PATH = final_model_path


    # --- Plotting ---
    try:
        epochs_range = range(1, NUM_EPOCHS + 1)
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, train_losses, label='Training Loss')
        # Only plot validation loss if it exists
        valid_val_losses = [l for l in val_losses if l is not None]
        valid_epochs_loss = [e for e, l in zip(epochs_range, val_losses) if l is not None]
        if valid_val_losses:
            plt.plot(valid_epochs_loss, valid_val_losses, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss (L1)')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        # Only plot validation PSNR if it exists
        valid_val_psnrs = [p for p in val_psnrs if p is not None]
        valid_epochs_psnr = [e for e, p in zip(epochs_range, val_psnrs) if p is not None]
        if valid_val_psnrs:
            plt.plot(valid_epochs_psnr, valid_val_psnrs, label='Validation PSNR (dB)')
            plt.title('Validation PSNR')
            plt.xlabel('Epochs')
            plt.ylabel('PSNR (dB)')
            plt.legend()
            plt.grid(True)
        else:
             plt.text(0.5, 0.5, 'No Validation PSNR Data', horizontalalignment='center', verticalalignment='center')
             plt.title('Validation PSNR')


        plt.tight_layout()
        plt.savefig(PLOT_FILE)
        log_message(f"\nTraining plots saved to: {PLOT_FILE}")
        # plt.show() # Optionally display the plot

    except Exception as e:
        log_message(f"Error generating plots: {e}")


    # --- Generate Test Predictions using the Best Model ---
    if os.path.exists(BEST_MODEL_PATH):
        log_message(f"\nLoading best model from {BEST_MODEL_PATH} for test predictions.")
        try:
            # Need to instantiate model again before loading state dict
            test_model = SRResNet(in_channels=IMG_CHANNELS, out_channels=IMG_CHANNELS, feature_channels=64,
                                  num_res_blocks=NUM_RES_BLOCKS, upscale_factor=UPSCALE_FACTOR).to(device)
            test_model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
            generate_test_predictions(test_model, TEST_DATA_DIR, TEST_OUTPUT_DIR, device)

            # --- Create Submission File ---
            create_submission_file(TEST_OUTPUT_DIR, SUBMISSION_CSV_FILE)

        except Exception as e:
             log_message(f"Error during test prediction or submission file creation: {e}")

    else:
        log_message("\nError: Best model file not found. Cannot generate test predictions.")

    log_message("\n--- Script Finished ---")