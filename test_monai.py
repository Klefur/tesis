from __future__ import annotations

from pathlib import Path
import os

import numpy as np
import pandas as pd
import pydicom
import torch
from skimage.transform import resize
from torch.utils.data import DataLoader, Dataset

### Data functions ###


def ct_normalization(img, meta):
    """
    Normaliza el rango HU a [0, 1].
    Ajustamos de -1024 (aire) a 3072 (tejido denso) para cubrir un rango amplio
    para resaltar detalles anatómicos.
    """
    def scale(img, meta, slope=1.0, intersept=-1024):
        MIN_B = -1024
        MAX_B = 3072

        slope = float(meta.get("Rescale Slope", slope))
        intersept = float(meta.get("Rescale Intercept", intersept))
        
        img = np.asarray(img) * slope + intersept  # Aplica la transformación de escala y desplazamiento
        img = resize(img, (256, 256), preserve_range=True)  # Asegura que todas las imágenes tengan el mismo tamaño
        img = (img - MIN_B) / (MAX_B - MIN_B)  # Escala a [0, 1]
        if img.ndim == 2:
            img = np.expand_dims(img, axis=0)
        return img.astype(np.float32)
    
    return scale(img, meta)


def _read_dicom(path: Path) -> np.ndarray:
    
    ds = pydicom.dcmread(path)  # Lee el archivo DICOM y convierte a numpy array
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

def save_tensor_cache(tensor, cache_path: Path) -> None:
        if cache_path is None:
            return
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        torch.save(tensor, tmp_path)
        try:
            with open(tmp_path, "rb+") as handle:
                os.fsync(handle.fileno())
        except OSError:
            pass
        os.replace(tmp_path, cache_path)


class LDCTDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        dim: int = 2,
        model_type: str = "model",
    ) -> None:
        """
        Args:
            df: DataFrame with either:
                - dim == 2: columns ['Case', 'SDCT', 'LDCT'] where SDCT/LDCT are file paths to DICOM images
                - dim == 3: columns ['Case', 'SDCT', 'LDCT'] where SDCT/LDCT are volume directories
            transform: Optional transform to be applied on a sample.
            model_type: "model" or "vae", used to determine how to process the data.
        """
        self.dim = dim
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
                
                if self.dim == 3:
                    rows.append({
                        'Case': row.Case, 
                        'SDCT': sdct_files, 
                        'LDCT': ldct_files,
                        'split_count': len(ldct_files),
                        'split_idx': idx
                    })
                else:
                    rows.extend([
                        {'Case': row.Case, 'SDCT': s, 'LDCT': l, 'split_count': len(ldct_files), 'split_idx': idxS} 
                        for idxS, (s, l) in enumerate(zip(sdct_files, ldct_files))
                    ])
            
            elif self.model_type == "vae":
                if self.dim == 3:
                    rows.append({
                        'Case': row.Case, 
                        'SDCT': sdct_files,
                        'split_count': len(sdct_files),
                        'split_idx': idx
                    })
                else:
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

        check_cache = lambda modality: True if Path(f"./cache/tensors/{self.data[idx]['Case']}_{modality}.pt").exists() else False

        if self.dim == 3:
            if check_cache(modality):
                cache_file = Path(f"./cache/tensors/{self.data[idx]['Case']}_{modality}.pt")
                return torch.load(cache_file)
            volume = np.array([_read_dicom(f) for f in sorted(row[modality])]) # Shape: (D, H, W)
            tensor = torch.as_tensor(volume).float().contiguous()  # Shape: (1, D, H, W)
            
            if self.cache_saving:
                save_tensor_cache(tensor, cache_path=Path(f"{cache_path}/{row['Case']}_{modality}").with_suffix('.pt'))
                
            return tensor
        else:
            if check_cache(modality):
                cache_file = Path(f"./cache/tensors/{self.data[idx]['Case']}_{modality}.pt")
                return torch.load(cache_file)
            img = _read_dicom(Path(row[modality]))
            tensor = torch.as_tensor(img).float().contiguous()  # Shape: (1, H, W)
            if self.cache_saving:
                save_tensor_cache(tensor, Path(f"{cache_path}/{row['split_idx']}_{row['Case']}_{modality}").with_suffix('.pt'))
            return tensor
    
    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict:
        row = self.data[idx]
        sample = {'Case': row['Case']}
        sample['split_idx'] = row['split_idx']
        sample['size'] = (256, 256) if self.dim == 2 else (len(row['SDCT']), 256, 256)
        if self.model_type == "model":
            # sample['SDCT'] = row['SDCT']
            sample['SDCT'] = self.load_dicom(idx, modality='SDCT')
            # sample['LDCT'] = row['LDCT']
            sample['LDCT'] = self.load_dicom(idx, modality='LDCT')
        elif self.model_type == "vae":
            # sample['SDCT'] = row['SDCT']
            sample['SDCT'] = self.load_dicom(idx, modality='SDCT')
        
        return sample
    

import torch.nn.functional as F

## Eval Functions ##

def save_checkpoint(state: dict, path: Path) -> None:
    if torch is None:
        raise RuntimeError("save_checkpoint requires PyTorch to be installed.")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def verify_and_create_folder(path: Path) -> Path:
    import glob
    
    # Find all existing folders matching pattern test_monai_*
    existing_folders = glob.glob(str(path / "test_monai_*"))
    existing_numbers = []
    
    for folder in existing_folders:
        try:
            number = int(folder[32:])
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

def save_model_config(config: dict, path: Path) -> None:
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path / "config.json", "w") as f:
        json.dump(config, f, indent=4)

def save_epoch_metrics(epoch: int, recon_loss: float, kl_loss: float, avg_loss: float, perceptual_loss: float, csv_path: Path) -> None:
    """
    Save epoch metrics to a CSV file.
    
    Args:
        epoch: Current epoch number
        recon_loss: Reconstruction loss value
        kl_loss: KL divergence loss value
        avg_loss: Average loss for the epoch
        perceptual_loss: Perceptual loss value
        csv_path: Path to save the CSV file
    """
    import csv
    csv_path = csv_path / "metrics.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if file exists to determine if we need to write headers
    file_exists = csv_path.exists()
    
    with open(csv_path, 'a', newline='') as csvfile:
        fieldnames = ['epoch', 'recon_loss','perceptual_loss', 'kl_loss', 'avg_loss']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header only if file is new
        if not file_exists:
            writer.writeheader()
        
        writer.writerow({
            'epoch': epoch,
            'recon_loss': f"{recon_loss:.4f}",
            'perceptual_loss': f"{perceptual_loss:.4f}",
            'kl_loss': f"{kl_loss:.4f}",
            'avg_loss': f"{avg_loss:.4f}"
        }) 

def focal_loss(logits: torch.Tensor, targets: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean") -> torch.Tensor:
    """
    Simple focal loss on logits (binary).
    """
    prob = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * (1 - p_t).pow(gamma) * ce
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss

def bce_focal_loss(logits: torch.Tensor, targets: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean") -> torch.Tensor:
    """
    Combined BCE + focal loss on logits. No extra weighting; simple sum.
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction=reduction)
    foc = focal_loss(logits, targets, alpha=alpha, gamma=gamma, reduction=reduction)
    return bce + foc

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train = True
dataset = pd.read_csv(os.path.join("Dataset", 'train.txt' if train else 'test.txt'), sep='\t', names=('Case','SDCT','LDCT',))
print("Dataset loaded successfully.")
# Recortar el dataset al 40% para pruebas
new_length = int(len(dataset.SDCT) * .1)

dataset["Case"] = dataset["Case"][: new_length]
dataset["SDCT"] = dataset["SDCT"][: new_length]
dataset["LDCT"] = dataset["LDCT"][: new_length]
dataset = dataset.dropna(subset=["SDCT"])

# train = True
# dataset = pd.read_csv(os.path.join("Dataset", 'train.txt' if train else 'test.txt'), sep='\t', names=('Case','SDCT','LDCT',))
# print("Dataset loaded successfully.")
ldct_dataset = LDCTDataset(dataset, dim=2, model_type="vae")
print("Dataset object created successfully.")

train = False
val_dataset = pd.read_csv(os.path.join("Dataset", 'train.txt' if train else 'test.txt'), sep='\t', names=('Case','SDCT','LDCT',))
print("Validation Dataset loaded successfully.")
val_ldct_dataset = LDCTDataset(val_dataset, dim=2, model_type="vae")
print("Validation Dataset object created successfully.")

loader = DataLoader(ldct_dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available())
val_loader = DataLoader(val_ldct_dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available())

import torch
from monai.networks.nets import AutoencoderKL
from monai.losses import PerceptualLoss
from tqdm import tqdm

# --- 1. Configuración de Hardware y Modelo ---
torch.cuda.empty_cache()
# device = torch.device("cuda") # GPU seleccionada

model_params = {
    'spatial_dims': 2,
    'in_channels': 1,
    'out_channels': 1,
    'channels': (128, 256, 512, 512),
    'latent_channels': 4,
    'num_res_blocks': 2,
    'attention_levels': (False, False, True, True),
}

model = AutoencoderKL(
    **model_params
).to(device)

perceptual = PerceptualLoss(
    spatial_dims=2, 
    network_type="vgg", 
    is_fake_3d=False, 
    pretrained=True
).to(device)


optimizer = torch.optim.AdamW(model.parameters(), lr=1e-04)

# --- 2. Parámetros de Entrenamiento ---
kl_weight =  1e-06 # Peso de la KL
perceptual_weight = 1.0
num_epochs = 100
batch_size = 4
best_metric = float('inf')

path = Path("./checkpoints/VaeMonai")
ckpt_path = verify_and_create_folder(path)
save_model_config(model_params, ckpt_path)

for epoch in range(1, num_epochs+1):
    model.train()
    num_samples = 0
    loop = tqdm(loader, desc=f"Train [{epoch}/{num_epochs}]")
    totals = {'loss': 0.0, 'recon_loss': 0.0, 'kl_loss': 0.0, 'perceptual_loss': 0.0}

    for batch in loop:

        optimizer.zero_grad()

        inputs = batch["SDCT"].to(device)
        # Asegurar dimensiones [B, 1, 256, 256]
        if inputs.ndim == 3:
            inputs = inputs.unsqueeze(1)
        
        bs = inputs.size(0)
        chunks = inputs.split(batch_size)
        accumulation_steps = len(chunks)

        for chunk in chunks:
            # 1. Forward Pass
            recon, mu, sigma = model(chunk)
            logvar = 2 * torch.log(sigma + 1e-8)

            recon = torch.sigmoid(recon)  # Asegurar que los valores estén en [0, 1] para la pérdida de reconstrucción

            # 2. Cálculo de Pérdidas
            recon_loss = F.l1_loss(recon, chunk)
            # focal requiere sigmoid al final para asegurar que los valores estén en [0, 1]
            # recon_loss = F.binary_cross_entropy_with_logits(recon, chunk, reduction='mean')
            # recon_loss = bce_focal_loss(recon, chunk)

            if perceptual:
                recon_p = recon.repeat(1, 3, 1, 1)
                # Aplicar sigmoid antes de la pérdida perceptual para asegurar que los valores estén en [0, 1]
                # recon_p = torch.sigmoid(recon).repeat(1, 3, 1, 1)
                chunk_p = chunk.repeat(1, 3, 1, 1)
                # 3. Perdida perceltual
                perceptual_loss = perceptual(recon_p, chunk_p)

            # KL Divergence 
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            kl_loss = kl_div.mean()

            # 3. Total Loss con Microbatching
            # if epoch <= 5:
            #     loss = recon_loss
            # else:
            loss = (
                recon_loss 
                + (kl_weight * kl_loss)
                + (perceptual_weight * perceptual_loss)
            )


            (loss / accumulation_steps).backward()
            
            chunk_bs = chunk.size(0)
            totals['loss'] += loss.detach().cpu() * chunk_bs
            totals['recon_loss'] += recon_loss.detach().cpu() * chunk_bs
            totals['kl_loss'] += kl_loss.detach().cpu() * chunk_bs
            totals['perceptual_loss'] += perceptual_loss.detach().cpu() * chunk_bs
            
        num_samples += bs
        # Actualizar barra de progreso (usando .item() para evitar leaks de memoria)
        postfix = {
            'loss': f"{totals['loss']/max(1, num_samples):.4f}",
            'recon_loss': f"{totals['recon_loss']/max(1, num_samples):.4f}",
            'kl_loss': f"{totals['kl_loss']/max(1, num_samples):.4f}",
            'perceptual_loss': f"{totals['perceptual_loss']/max(1, num_samples):.4f}"
        }
        loop.set_postfix(**postfix)

        optimizer.step()

    avg_totals = {
        k: v/max(1, num_samples) for k, v in totals.items()
    }

    if val_loader is not None:
        val_num_samples = 0
        val_totals = {k: 0.0 for k in totals}

        model.eval()
        with torch.no_grad():
            val_loop = tqdm(val_loader, desc=f"Val [{epoch}/{num_epochs}]")
            
            for batch in val_loop:
                inputs = batch["SDCT"].to(device)
                # Asegurar dimensiones [B, 1, 256, 256]
                if inputs.ndim == 3:
                    inputs = inputs.unsqueeze(1)
                
                bs = inputs.size(0)
                chunks = inputs.split(batch_size)
                accumulation_steps = len(chunks)

                for chunk in chunks:
                    # 1. Forward Pass
                    recon, mu, sigma = model(chunk)
                    logvar = 2 * torch.log(sigma + 1e-8)

                    recon = torch.sigmoid(recon)  # Asegurar que los valores estén en [0, 1] para la pérdida de reconstrucción

                    # 2. Cálculo de Pérdidas
                    recon_loss = F.l1_loss(recon, chunk)
                    # recon_loss = F.binary_cross_entropy_with_logits(recon, chunk, reduction='mean')
                    # focal requiere sigmoid al final para asegurar que los valores estén en [0, 1]
                    # recon_loss = bce_focal_loss(recon, chunk)

                    if perceptual:
                        recon_p = recon.repeat(1, 3, 1, 1)
                        chunk_p = chunk.repeat(1, 3, 1, 1)
                        # 3. Perdida perceltual
                        perceptual_loss = perceptual(recon_p, chunk_p)

                    # KL Divergence 
                    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
                    kl_loss = kl_div.mean()

                    # 3. Total Loss con Microbatching
                    loss = (
                        recon_loss 
                        + kl_weight * kl_loss
                        + perceptual_weight * perceptual_loss
                    )
                
                    chunk_bs = chunk.size(0)
                    val_totals['loss'] += loss.detach().cpu() * chunk_bs
                    val_totals['recon_loss'] += recon_loss.detach().cpu() * chunk_bs
                    val_totals['kl_loss'] += kl_loss.detach().cpu() * chunk_bs
                    val_totals['perceptual_loss'] += perceptual_loss.detach().cpu() * chunk_bs
                    
                val_num_samples += bs
                # Actualizar barra de progreso (usando .item() para evitar leaks de memoria)
                postfix = {
                    'loss': f"{val_totals['loss']/max(1, val_num_samples):.4f}",
                    'recon_loss': f"{val_totals['recon_loss']/max(1, val_num_samples):.4f}",
                    'kl_loss': f"{val_totals['kl_loss']/max(1, val_num_samples):.4f}",
                    'perceptual_loss': f"{val_totals['perceptual_loss']/max(1, val_num_samples):.4f}"
                }
                val_loop.set_postfix(**postfix)

        val_avg_totals = {
            k: v/max(1, val_num_samples) for k, v in val_totals.items()
        }

    l, rec, kl, perc = [v for _, v in avg_totals.items()] if val_loader is not None else [v for _, v in val_avg_totals.items()]
    print(f"--- Epoch {epoch} Finalizada. Avg Loss: {l:.4f} Avg: KL loss: {kl:.4f} Avg Recon loss: {rec:.4f}, Avg Perceptual loss: {perc:.4f}---")
    
    save_epoch_metrics(epoch, rec, kl, l, perc, ckpt_path)

    state = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': loss.item(),
    }

    current_metric = l
    if current_metric < best_metric:
        best_metric = current_metric
        save_checkpoint(state, ckpt_path / "vae_best.pt")

    should_save = epoch % 10 == 0 or epoch == 100
    if should_save:
        state = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': loss.item(),
        }

        save_checkpoint(state, ckpt_path / f"monai_{epoch}.pt")
        