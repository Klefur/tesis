from __future__ import annotations

from pathlib import Path
import os
import random

import numpy as np
import pandas as pd
import pydicom
import torch
from skimage.transform import resize
from torch.utils.data import DataLoader, Dataset

### Data functions ###
def ct_normalization(img, meta):
    """
    Normalize HU to [0, 1].
    Adjust from -1024 (air) to 3072 (dense tissue) to cover a wide range
    to highlight anatomical details.
    """
    def scale(img, meta, slope=1.0, intersept=-1024):
        MIN_B = -1024
        MAX_B = 3072

        slope = float(meta.get("Rescale Slope", slope))
        intersept = float(meta.get("Rescale Intercept", intersept))
        
        img = np.asarray(img) * slope + intersept  # Apply the scale and offset transformation
        img = resize(img, (256, 256), preserve_range=True)  # Ensure all images have the same size
        img = (img - MIN_B) / (MAX_B - MIN_B)  # Scale to [0, 1]
        if img.ndim == 2:
            img = np.expand_dims(img, axis=0)
        return img.astype(np.float32)
    
    return scale(img, meta)


def _read_dicom(path: Path) -> np.ndarray:
    
    ds = pydicom.dcmread(path)  # Read the DICOM file and convert to numpy array
    metadata = {}
    for element in ds:
            if getattr(element, "name", None) == "Pixel Data":
                continue
            try:
                name = str(element.name)
                value = element.value
                if value is None:
                    continue
                metadata[name] = str(value)
            except Exception:
                continue

    return ct_normalization(ds.pixel_array, metadata)


class LDCTDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        model_type: str = "model",
    ) -> None:
        """
        Args:
            df: DataFrame with either:
            transform: Optional transform to be applied on a sample.
            model_type: "model" or "vae", used to determine how to process the data.
        """
        self.data = df.dropna()
        self.model_type = model_type
        self.cache_saving = False

        self._preprocess()

    def _preprocess(self) -> pd.DataFrame:
        rows = []

        unwrap_dir = lambda path: sorted([str(path / fname) for fname in os.listdir(path) if (path / fname).is_file()])

        idx = 0
        for row in self.data.itertuples(index=False):
            sdct_dir = Path(row.SDCT)
            sdct_files = unwrap_dir(sdct_dir)
            
            if self.model_type == "model":
                ldct_dir = Path(row.LDCT)
                ldct_files = unwrap_dir(ldct_dir)
                rows.extend([
                    {'Case': row.Case, 'SDCT': s, 'LDCT': l, 'split_count': len(ldct_files), 'split_idx': idxS} 
                    for idxS, (s, l) in enumerate(zip(sdct_files, ldct_files))
                ])
            
            elif self.model_type == "vae":
                rows.extend([
                    {'Case': row.Case, 'SDCT': s, 'split_count': len(sdct_files), 'split_idx': idxS} 
                    for idxS, s in enumerate(sdct_files)
                ])
            idx += 1
                    
        df = pd.DataFrame(rows)
        self.data = df.to_dict("records")
        self.size = len(self.data)
    
    def load_dicom(self, idx: int, modality: str = "SDCT") -> dict:
        row = self.data[idx]
        cache_path = "./cache/tensors"
        img = _read_dicom(Path(row[modality]))
        tensor = torch.as_tensor(img).float().contiguous()  # Shape: (1, H, W)
        return tensor
    
    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict:
        row = self.data[idx]
        sample = {'Case': row['Case']}
        sample['split_idx'] = row['split_idx']
        sample['size'] = (256, 256)
        if self.model_type == "model":
            sample['SDCT'] = self.load_dicom(idx, modality='SDCT')
            sample['LDCT'] = self.load_dicom(idx, modality='LDCT')
        elif self.model_type == "vae":
            sample['SDCT'] = self.load_dicom(idx, modality='SDCT')
        
        return sample
    

def save_checkpoint(state: dict, path: Path) -> None:
    if torch is None:
        raise RuntimeError("save_checkpoint requires PyTorch to be installed.")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)

def save_epoch_metrics(epoch: int, avg_loss: float, path: Path) -> None:
    import csv
    metrics_file = path / "metrics.csv"
    file_exists = metrics_file.exists()
    
    with open(metrics_file, mode='a', newline='') as csvfile:
        fieldnames = ['epoch', 'avg_loss']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow({'epoch': epoch, 'avg_loss': avg_loss})

def verify_and_create_folder(path: Path) -> Path:
    import glob
    
    # Find all existing folders matching pattern monai_*
    existing_folders = glob.glob(str(path / "test_monai_*"))
    existing_numbers = []
    
    for folder in existing_folders:
        try:
            number = int(folder[34:])
            existing_numbers.append(number)
        except:
            pass

    # Get the last number or start at 0
    last_number = max(existing_numbers) if existing_numbers else 0
    new_number = last_number + 1
    
    # Create new folder
    new_folder = path / f"test_monai_{new_number}"
    Path(new_folder).mkdir(parents=True, exist_ok=True)
    return new_folder

SEED = 42
DATA_FRACTION = 0.1
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

train = True
dataset = pd.read_csv(os.path.join("Dataset", 'train.txt' if train else 'test.txt'), sep='\t', names=('Case','SDCT','LDCT',))
dataset = dataset.dropna(subset=["Case", "SDCT", "LDCT"])
dataset = dataset.sample(frac=DATA_FRACTION, random_state=SEED).reset_index(drop=True)
print("Dataset loaded successfully.")
ldct_dataset = LDCTDataset(dataset, model_type="model")
print("Dataset object created successfully.")
loader = DataLoader(ldct_dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available())

train = False
val_dataset = pd.read_csv(os.path.join("Dataset", 'train.txt' if train else 'test.txt'), sep='\t', names=('Case','SDCT','LDCT',))
print("Validation Dataset loaded successfully.")
val_dataset = val_dataset.dropna(subset=["Case", "SDCT", "LDCT"])
val_dataset = val_dataset.sample(frac=DATA_FRACTION, random_state=SEED).reset_index(drop=True)
val_ldct_dataset = LDCTDataset(val_dataset, model_type="model")
print("Validation Dataset object created successfully.")

loader = DataLoader(ldct_dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available())
val_loader = DataLoader(val_ldct_dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available())

from diffusers import UNet2DModel
import torch
from monai.networks.nets import AutoencoderKL

# Load model from checkpoints/
state_dict = torch.load("./checkpoints/VaeMonai/monai_7/vae_best.pt", map_location=device)
# --- 1. Hardware and model configuration ---
torch.cuda.empty_cache()

vae = AutoencoderKL(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    channels=(128, 256, 512, 512),
    latent_channels=4, 
    num_res_blocks=2,
    attention_levels=(False, False, True, True), 
).to(device)

vae.load_state_dict(state_dict["model"])
vae.eval()
vae.requires_grad_(False)

# Initialize a UNet2D model for flow matching
model = UNet2DModel(
    sample_size=32,
    in_channels=4,
    out_channels=4,
    layers_per_block=2,       # Layers per residual block
    block_out_channels=(128, 256, 512, 512),  # Channel progression
    down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
    up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
).to(device)

import torch.nn.functional as F
from tqdm import tqdm

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

num_epochs = 100
batch_size = 4
best_metric = float('inf')

path = Path("./checkpoints/VaeMonaiFM")
ckpt_path = verify_and_create_folder(path)

for epoch in range(1, num_epochs + 1):
    model.train()
    epoch_loss = 0
    loop = tqdm(loader, desc=f"Train {epoch}/{num_epochs}")

    for batch in loop:
        ldct = batch["LDCT"].to(device) # Entrada ruidosa
        sdct = batch["SDCT"].to(device) # Objetivo limpio

        # 2. Latent space encoding
        # with torch.no_grad():
        #     z_0, _ = vae.encode(ldct) # Latente inicial (LDCT)
        #     z_1, _ = vae.encode(sdct) # Latente objetivo (SDCT)
        # 2.1 Original Image (Without VAE)
        z_0 = ldct
        z_1 = sdct

        # 3. Time sampling
        batch_size = z_0.shape[0]
        t = torch.rand(batch_size, device=device)
        t_reshaped = t.view(-1, 1, 1, 1)

        # 4. Latent interpolation
        # x_t = (1 - t) * z_0 + t * z_1
        x_t = (1 - t_reshaped) * z_0 + t_reshaped * z_1

        # 5. Define target velocity
        # v = z_1 - z_0
        target_velocity = z_1 - z_0

        # 6. Model prediction
        pred_velocity = model(x_t, t).sample
        loss = F.mse_loss(pred_velocity, target_velocity)
        # loss = F.huber_loss(pred_velocity, target_velocity)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        loop.set_postfix({"loss": loss.item()})

    print(f"Epoch {epoch} completed. Avg Loss: {epoch_loss/len(loader):.6f}")

    val_loop = tqdm(val_loader, desc=f"Validating {epoch}/{num_epochs}")
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loop:
            ldct = batch["LDCT"].to(device)
            sdct = batch["SDCT"].to(device)

            # with torch.no_grad():
            #     z_0, _ = vae.encode(ldct)
            #     z_1, _ = vae.encode(sdct)
            z_0 = ldct
            z_1 = sdct

            batch_size = z_0.shape[0]
            t = torch.rand(batch_size, device=device)
            t_reshaped = t.view(-1, 1, 1, 1)

            x_t = (1 - t_reshaped) * z_0 + t_reshaped * z_1
            target_velocity = z_1 - z_0
            pred_velocity = model(x_t, t).sample
            
            val_loss += F.huber_loss(pred_velocity, target_velocity).item()
    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch} validated. Avg Loss: {avg_val_loss:.6f}")
    
    avg_loss = epoch_loss / len(loader) if val_loader is None else avg_val_loss
    
    save_epoch_metrics(epoch, avg_loss, ckpt_path)

    state = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': loss.item(),
    }

    current_metric = avg_loss
    if current_metric < best_metric:
        best_metric = current_metric
        save_checkpoint(state, ckpt_path / "vaeFM_best.pt")

    should_save = epoch % 10 == 0 or epoch == 100
    if should_save:
        save_checkpoint(state, ckpt_path / f"monaiFM_{epoch}.pt")