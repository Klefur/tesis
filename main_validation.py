from __future__ import annotations
from pathlib import Path

import os
import argparse
import numpy as np
import pandas as pd
import pydicom
import matplotlib.pyplot as plt
import time
import torch

from torch.utils.data import DataLoader, Dataset
from skimage.transform import resize
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
from diffusers import UNet2DModel
from monai.networks.nets import AutoencoderKL


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

### Data functions ###
def ct_normalization(img, meta):
    """
    Normalizes the HU range to [0, 1].
    We adjust from -1024 (air) to 3072 (dense tissue) to cover a wide range
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
    
    ds = pydicom.dcmread(path)  # Read DICOM file and convert to a numpy array
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


from skimage import exposure

def equalize_images_to_common_contrast(image, reference):
    image = np.asarray(image, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)

    if image.ndim == 2:
        target = exposure.equalize_adapthist(reference)
        image_eq = exposure.match_histograms(image, target)
        reference_eq = exposure.match_histograms(reference, target)

    return np.clip(image_eq, 0.0, 1.0), np.clip(reference_eq, 0.0, 1.0)

@torch.no_grad()
def plot_flow_inference_vae(dataset, vae, unet, batch, device, num_steps=20, show_plots=True):
    vae.eval()
    unet.eval()
    
    # 1. Preparar datos
    ldct = batch["LDCT"][:1].to(device)
    sdct = batch["SDCT"][:1].to(device)
    
    z_t, _ = vae.encode(ldct)
    dt = 1.0 / num_steps
    
    for i in range(num_steps):
        t = torch.ones((1,), device=device) * (i / num_steps)
        v_pred = unet(z_t, t).sample
        z_t = z_t + v_pred * dt
        
    recon_sdct_flow = vae.decode(z_t) # Resultado del Flow Matching
    recon_sdct_flow = torch.sigmoid(recon_sdct_flow)

    ldct_img = ldct[0,0].cpu().numpy()
    sdct_img = sdct[0,0].cpu().numpy()
    flow_img = recon_sdct_flow[0,0].cpu().numpy()

    if dataset == 'clrm':
        flow_img, sdct_img = equalize_images_to_common_contrast(flow_img, sdct_img)

    data_range = 1.0 
    psnr_val = compute_psnr(sdct_img, flow_img, data_range=data_range)
    ssim_val = compute_ssim(sdct_img, flow_img, data_range=data_range)

    if not show_plots:
        return psnr_val, ssim_val
    
    print(f"PSNR: {psnr_val:.2f} dB, SSIM: {ssim_val:.4f}")
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(ldct_img, cmap='gray')
    axes[0].set_title("Input (LDCT)")
    axes[0].axis('off')
    
    axes[1].imshow(sdct_img, cmap='gray')
    axes[1].set_title("Target (SDCT)")
    axes[1].axis('off')
    
    axes[2].imshow(flow_img, cmap='gray')
    axes[2].set_title(f"Flow Result (Steps: {num_steps})")
    axes[2].axis('off')
    
    error_map = np.abs(sdct_img - flow_img)
    im = axes[3].imshow(error_map, cmap='hot')
    axes[3].set_title("Absolute Error")
    axes[3].axis('off')
    plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

    plt.suptitle(f"Inferencia de Flow Matching - Con VAE", fontsize=16)
    plt.tight_layout()
    plt.show()

@torch.no_grad()
def plot_flow_inference_model(unet, batch, device, num_steps=20, show_plots=True):
    unet.eval()
    
    ldct = batch["LDCT"][:1].to(device)
    sdct = batch["SDCT"][:1].to(device)
    
    z_t = ldct    
    dt = 1.0 / num_steps
    
    for i in range(num_steps):
        t = torch.ones((1,), device=device) * (i / num_steps)
        v_pred = unet(z_t, t).sample
        z_t = z_t + v_pred * dt
        
    recon_sdct_flow = z_t # Resultado del Flow Matching

    ldct_img = ldct[0,0].cpu().numpy()
    sdct_img = sdct[0,0].cpu().numpy()
    flow_img = recon_sdct_flow[0,0].cpu().numpy()

    data_range = 1.0 

    psnr_val = compute_psnr(sdct_img, flow_img, data_range=data_range)
    ssim_val = compute_ssim(sdct_img, flow_img, data_range=data_range)

    if not show_plots:
        return psnr_val, ssim_val
    
    print(f"PSNR: {psnr_val:.2f} dB, SSIM: {ssim_val:.4f}")

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(ldct_img, cmap='gray')
    axes[0].set_title("Input (LDCT)")
    axes[0].axis('off')
    
    axes[1].imshow(sdct_img, cmap='gray')
    axes[1].set_title("Target (SDCT)")
    axes[1].axis('off')
    
    axes[2].imshow(flow_img, cmap='gray')
    axes[2].set_title(f"Flow Result (Steps: {num_steps})")
    axes[2].axis('off')
    
    error_map = np.abs(sdct_img - flow_img)
    im = axes[3].imshow(error_map, cmap='hot')
    axes[3].set_title("Absolute Error")
    axes[3].axis('off')
    plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

    plt.suptitle(f"Inferencia de Flow Matching - Sin VAE", fontsize=16)
    plt.tight_layout()
    plt.show()

def print_results(model, pasos, psnr_values, ssim_values, start, end):
    elapsed = end - start
    minutes = int(elapsed // 60)
    seconds = elapsed % 60
    print(f"Resultados Modelo {model} con {pasos} pasos")
    print(f"Tiempo: {minutes} min {seconds:.2f} s\n")
    # print(f"Promedio PSNR: {np.mean(psnr_values):.2f} dB, SSIM: {np.mean(ssim_values):.4f}")

def save_results_to_csv(dataset, model, pasos, psnr_values, ssim_values, start, end, filename="results.csv"):
    elapsed = end - start
    minutes = int(elapsed // 60)
    seconds = elapsed % 60
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)

    results_df = pd.DataFrame({
        "Dataset": [dataset],
        "Model": [model],
        "Steps": [pasos],
        # "PSNR Values": [psnr_values],
        # "SSIM Values": [ssim_values],
        # "Average PSNR": [avg_psnr],
        # "Average SSIM": [avg_ssim],
        "Time (min)": [minutes],
        "Time (s)": [seconds]
    })

    if not os.path.isfile(filename):
        results_df.to_csv(filename, index=False)
    else:
        results_df.to_csv(filename, mode='a', header=False, index=False)

def mean_metrics(dataset, vae, models, val_data, device, steps):
    psnr_values = []
    ssim_values = []

    file_name = f"time_results_vae7fm4_fm3.csv"

    start = time.time()
    for batch in val_data:
        psnr, ssim = plot_flow_inference_vae(dataset, vae, models[0], batch, device, num_steps=steps, show_plots=False)
        psnr_values.append(psnr)
        ssim_values.append(ssim)
    end = time.time()

    print_results("con vae", steps, psnr_values, ssim_values, start, end)
    save_results_to_csv(dataset, "con vae", steps, psnr_values, ssim_values, start, end, filename=file_name)

    psnr_values = []
    ssim_values = []
    
    start = time.time()
    for batch in val_data:
        psnr, ssim = plot_flow_inference_model(models[1], batch, device, num_steps=steps, show_plots=False)
        psnr_values.append(psnr)
        ssim_values.append(ssim)
    end = time.time()

    print_results("sin vae", steps, psnr_values, ssim_values, start, end)
    save_results_to_csv(dataset,"sin vae", steps, psnr_values, ssim_values, start, end, filename=file_name)

def get_validation_loader(dataset_name: str):
    path_map = {
        "clrm": os.path.join("Dataset", "clrm_test.txt"),
        "mgc": os.path.join("Dataset", "mgc_test.txt"),
        "test": os.path.join("Dataset", "test.txt")
    }

    if dataset_name == 'mgc':
        mgc_dataset=pd.read_csv(os.path.join("Dataset", "mgc_test.txt"), sep='\t', names=('Case','SDCT','LDCT',))
        mgc_dataset = mgc_dataset.dropna(subset=["SDCT"])
        val_ldct_dataset = LDCTDataset(mgc_dataset, model_type="model")

    if dataset_name == 'clrm':
        clrm_dataset=pd.read_csv(os.path.join("Dataset", "clrm_test.txt"), sep='\t', names=('Case','SDCT','LDCT',))
        clrm_dataset = clrm_dataset.dropna(subset=["SDCT"])
        # Select random 50% to train and 50% to validation
        train_indices = clrm_dataset.sample(frac=0.5, random_state=42).index
        val_indices = clrm_dataset.drop(train_indices).index

        val_ldct_dataset = LDCTDataset(clrm_dataset.iloc[val_indices], model_type="model")
    else: 
        val_dataset = pd.read_csv(path_map[dataset_name], sep='\t', names=('Case','SDCT','LDCT',))
        val_ldct_dataset = LDCTDataset(val_dataset, model_type="model")
    return DataLoader(
        val_ldct_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

def main():
    parser = argparse.ArgumentParser(description="Validate models on a selected dataset.")
    parser.add_argument(
        "--dataset",
        "-d",
        choices=["test", "clrm", "mgc"],
        default=["test", "clrm", "mgc"],
        help="Dataset to use",
    )
    parser.add_argument(
        "--steps",
        "-s",
        type=int,
        nargs="+",
        default=[1, 10, 50],
        help="Steps for evaluation",
    )
    args = parser.parse_args()

    vae_state_dict = torch.load("./checkpoints/VaeMonai/monai_7/vae_best.pt")
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

    vae.load_state_dict(vae_state_dict["model"])
    vae.eval()
    vae.requires_grad_(False)

    model = UNet2DModel(
        sample_size=32,
        in_channels=4,
        out_channels=4,
        layers_per_block=2,       # Layers per residual block
        block_out_channels=(128, 256, 512, 512),  # Channel progression
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
    ).to(device)

    state_dict = torch.load("./checkpoints/VaeMonaiFM/monai_4/vaeFM_best.pt", map_location=device)
    model.load_state_dict(state_dict["model"])

    model2 = UNet2DModel(
        sample_size=256,
        in_channels=1,
        out_channels=1,
        layers_per_block=2,       # Layers per residual block
        block_out_channels=(128, 256, 512, 512),  # Channel progression
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
        # [block_out_channels=(128, 256, 256, 512, 512),  # Channel progression
        # down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        # up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),]
    ).to(device)
    state_dict = torch.load("./checkpoints/VaeMonaiFM/monai_3/vaeFM_best.pt", map_location=device)
    model2.load_state_dict(state_dict["model"])

    for dataset in args.dataset:
        val_loader = get_validation_loader(dataset)
        print(f"Using dataset: {dataset}")
        for steps in args.steps:
            mean_metrics(dataset, vae, [model, model2], val_loader, device, steps)

if __name__ == "__main__":
    main()
