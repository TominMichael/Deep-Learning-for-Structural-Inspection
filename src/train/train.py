
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import os
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report
import warnings

warnings.filterwarnings('ignore')

plt.switch_backend('Agg')


os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'


sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (16, 8)


   
    LOCAL_INPUT_DIR = Path("C:/Users/User/crack/data")
    METADATA_PATH = LOCAL_INPUT_DIR / "metadata.csv"
    MODEL_SAVE_DIR = Path("C:/Users/User/crack/models/")
    LOGS_DIR = Path("C:/Users/User/crack/logs/")
    ANALYSIS_DIR = Path("C:/Users/User/crack/analysis/")

    MODEL_SAVE_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)
    ANALYSIS_DIR.mkdir(exist_ok=True)

    MODEL_ARCHITECTURE = "unetplusplus"
    ENCODER_BACKBONE = "efficientnet-b7"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    LEARNING_RATE = 1e-5
    BATCH_SIZE = 32 
    NUM_EPOCHS = 300  
    IMAGE_SIZE = 512  
    NUM_WORKERS = 0  

    LOSS_ALPHA = 0.7
    LOSS_BETA = 0.3
    LOSS_GAMMA = 2.0

    print(f"\nConfiguration:")
    print(f"   • Device: {DEVICE}")
    print(f"   • Model: {MODEL_ARCHITECTURE} + {ENCODER_BACKBONE}")
    print(f"   • Image Size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"   • Batch Size: {BATCH_SIZE}")
    print(f"   • Learning Rate: {LEARNING_RATE}")
    print(f"   • Epochs: {NUM_EPOCHS}")


    print(f"\n🔧 Setting up components...")

    
    train_transforms = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(scale=(0.9, 1.1), translate_percent=(-0.0625, 0.0625),
                 rotate=(-25, 25), p=0.3),
        A.RandomBrightnessContrast(p=0.3),

        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    val_test_transforms = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

 
    class CrackDataset(Dataset):
        def __init__(self, dataframe, base_path, transforms=None):
            self.df = dataframe
            self.base_path = base_path
            self.transforms = transforms

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]

            if '\\' in row['image_path']:
                image_filename = row['image_path'].split('\\')[-1]
                mask_filename = row['mask_path'].split('\\')[-1]
            else:
                image_filename = row['image_path'].split('/')[-1]
                mask_filename = row['mask_path'].split('/')[-1]

            image_path = self.base_path / row['split'] / 'images' / image_filename
            mask_path = self.base_path / row['split'] / 'masks' / mask_filename

            image = cv2.imread(str(image_path))
            if image is None:
                return torch.zeros((3, IMAGE_SIZE, IMAGE_SIZE)), torch.zeros((1, IMAGE_SIZE, IMAGE_SIZE))

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

            if mask is None:
                return torch.zeros((3, IMAGE_SIZE, IMAGE_SIZE)), torch.zeros((1, IMAGE_SIZE, IMAGE_SIZE))

            mask = mask.astype(np.float32) / 255.0

            if self.transforms:
                augmented = self.transforms(image=image, mask=mask)
                image, mask = augmented['image'], augmented['mask']

            return image, mask.unsqueeze(0).float()

    
    class FocalTverskyLoss(nn.Module):
        def __init__(self, alpha, beta, gamma, smooth=1e-6):
            super().__init__()
            self.alpha, self.beta, self.gamma, self.smooth = alpha, beta, gamma, smooth

        def forward(self, y_pred, y_true):
            y_pred = torch.sigmoid(y_pred)
            y_pred, y_true = y_pred.view(-1), y_true.view(-1)
            true_pos = (y_pred * y_true).sum()
            false_neg = ((1 - y_pred) * y_true).sum()
            false_pos = (y_pred * (1 - y_true)).sum()
            tversky_index = (true_pos + self.smooth) / (
                    true_pos + self.alpha * false_neg + self.beta * false_pos + self.smooth)
            return (1 - tversky_index) ** self.gamma

  
    def calculate_comprehensive_metrics(y_pred, y_true, threshold=0.5):
        """Calculate all segmentation and classification metrics"""
        y_pred_sigmoid = torch.sigmoid(y_pred)
        y_pred_binary = (y_pred_sigmoid > threshold).float()

      
        intersection = (y_pred_binary * y_true).sum()
        dice = (2. * intersection + 1e-6) / (y_pred_binary.sum() + y_true.sum() + 1e-6)

        union = y_pred_binary.sum() + y_true.sum() - intersection
        iou = (intersection + 1e-6) / (union + 1e-6)

        y_pred_flat = y_pred_binary.view(-1).cpu().numpy()
        y_true_flat = y_true.view(-1).cpu().numpy()

        accuracy = accuracy_score(y_true_flat, y_pred_flat)
        precision = precision_score(y_true_flat, y_pred_flat, average='binary', zero_division=0)
        recall = recall_score(y_true_flat, y_pred_flat, average='binary', zero_division=0)
        f1 = f1_score(y_true_flat, y_pred_flat, average='binary', zero_division=0)

       
        specificity = 0
        if len(np.unique(y_true_flat)) > 1:
            tn, fp, fn, tp = confusion_matrix(y_true_flat, y_pred_flat).ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        return {
            'dice': dice.item(),
            'iou': iou.item(),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'specificity': specificity
        }

    def dice_score(y_pred, y_true, smooth=1e-6):
        y_pred = torch.sigmoid(y_pred)
        y_pred = (y_pred > 0.5).float().view(-1)
        y_true = y_true.view(-1)
        intersection = (y_pred * y_true).sum()
        return (2. * intersection + smooth) / (y_pred.sum() + y_true.sum() + smooth)

  
    def train_one_epoch(loader, model, optimizer, loss_fn, scaler, device, epoch):
        model.train()
        loop = tqdm(loader, desc=f"Training Epoch {epoch}")
        running_loss = 0.0
        batch_losses = []

        for batch_idx, (data, targets) in enumerate(loop):
            data, targets = data.to(device), targets.to(device)

            if device == "cuda" and scaler is not None:
                with torch.cuda.amp.autocast():
                    predictions = model(data)
                    loss = loss_fn(predictions, targets)
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                predictions = model(data)
                loss = loss_fn(predictions, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch_loss = loss.item()
            running_loss += batch_loss
            batch_losses.append(batch_loss)
            loop.set_postfix(loss=batch_loss, avg_loss=running_loss / (batch_idx + 1))

        return running_loss / len(loader), batch_losses

   
    def validate_one_epoch(loader, model, loss_fn, device):
        model.eval()
        loop = tqdm(loader, desc="Validating")
        val_loss = 0.0
        all_dice_scores = []

        with torch.no_grad():
            for data, targets in loop:
                data, targets = data.to(device), targets.to(device)
                predictions = model(data)
                val_loss += loss_fn(predictions, targets).item()
                all_dice_scores.append(dice_score(predictions, targets).cpu().numpy())

        avg_val_loss = val_loss / len(loader)
        avg_dice = np.mean(all_dice_scores)

        return avg_val_loss, avg_dice


    def comprehensive_validation(model, val_loader, loss_fn, device, epoch):
        """Enhanced validation with detailed metrics"""
        model.eval()

        all_metrics = []
        all_losses = []

        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(tqdm(val_loader, desc="Detailed Validation")):
                data, targets = data.to(device), targets.to(device)
                predictions = model(data)
                loss = loss_fn(predictions, targets).item()
                all_losses.append(loss)

         
                for i in range(data.shape[0]):
                    pred_sample = predictions[i:i + 1]
                    target_sample = targets[i:i + 1]

                    metrics = calculate_comprehensive_metrics(pred_sample, target_sample)
                    all_metrics.append(metrics)

        metrics_df = pd.DataFrame(all_metrics)
        avg_metrics = metrics_df.mean()
        std_metrics = metrics_df.std()

        return avg_metrics, std_metrics, metrics_df, all_losses

   
    def plot_training_progress(history_df, save_path):
        """Plot comprehensive training progress - SAVE ONLY"""
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))

 
        axes[0, 0].plot(history_df['epoch'], history_df['train_loss'], 'b-', label='Training Loss', linewidth=2)
        axes[0, 0].plot(history_df['epoch'], history_df['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        axes[0, 0].set_title('Training and Validation Loss', fontsize=16, weight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

      
        axes[0, 1].plot(history_df['epoch'], history_df['dice_score'], 'g-', label='Validation Dice Score', linewidth=2)
        axes[0, 1].set_title('Validation Dice Score Progress', fontsize=16, weight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Dice Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

      
        if len(history_df) > 5:
            recent_epochs = history_df.tail(10)
            axes[1, 0].plot(recent_epochs['epoch'], recent_epochs['dice_score'], 'g-', linewidth=2, label='Recent Dice')
            axes[1, 0].set_title('Recent Performance (Last 10 Epochs)', fontsize=16, weight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Dice Score')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

     
        if len(history_df) > 1:
            axes[1, 1].plot(history_df['epoch'], history_df['val_loss'], 'r-', linewidth=2, label='Validation Loss')
            axes[1, 1].set_title('Validation Loss Trend', fontsize=16, weight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  
        print(f"Training progress graph saved to: {save_path}")

    def plot_metrics_summary(metrics_df, epoch, save_path):
        """Plot metrics summary - SAVE ONLY"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Validation Metrics Summary - Epoch {epoch}', fontsize=16, weight='bold')

        metrics = ['dice', 'iou', 'accuracy', 'precision', 'recall', 'f1']
        for i, metric in enumerate(metrics):
            row = i // 3
            col = i % 3

            axes[row, col].hist(metrics_df[metric], bins=30, alpha=0.7, color=f'C{i}')
            axes[row, col].axvline(metrics_df[metric].mean(), color='red', linestyle='--',
                                   label=f'Mean: {metrics_df[metric].mean():.3f}')
            axes[row, col].set_title(f'{metric.upper()} Distribution')
            axes[row, col].set_xlabel(metric.upper())
            axes[row, col].set_ylabel('Frequency')
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  
        print(f" Metrics summary saved to: {save_path}")

  
    def save_enhanced_checkpoint(model, optimizer, epoch, train_loss, val_loss, dice_score, history, filepath):
        """Save complete model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'dice_score': dice_score,
            'training_history': history,
            'model_config': {
                'architecture': MODEL_ARCHITECTURE,
                'backbone': ENCODER_BACKBONE,
                'image_size': IMAGE_SIZE,
                'batch_size': BATCH_SIZE,
                'learning_rate': LEARNING_RATE
            },
            'training_info': {
                'user': 'Ashish480',
                'datetime': '2025-07-09 03:15:14',
                'hardware': 'Dual Quadro GV100',
                'dataset_size': '21700 images'
            }
        }
        torch.save(checkpoint, filepath)
        print(f" Enhanced checkpoint saved: {filepath}")

    print(" All enhanced components ready.")

    print(f"\nStarting Continuous Training (50 Epochs)...")


    if DEVICE == "cuda":
        gpu_count = torch.cuda.device_count()
        print(f" Found {gpu_count} GPU(s)")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        print(" WARNING: No GPU found.")

    
    if not METADATA_PATH.exists():
        print(f" ERROR: Metadata file not found at {METADATA_PATH}")
        

    df = pd.read_csv(METADATA_PATH)
    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    val_df = df[df['split'] == 'validation'].reset_index(drop=True)

    print(f" Dataset: {len(train_df)} train, {len(val_df)} validation")

    
    train_dataset = CrackDataset(train_df, base_path=LOCAL_INPUT_DIR, transforms=train_transforms)
    val_dataset = CrackDataset(val_df, base_path=LOCAL_INPUT_DIR, transforms=val_test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                              pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                            pin_memory=True, shuffle=False)

    print(f"{len(train_loader)} train batches, {len(val_loader)} validation batches")

    
    try:
        import segmentation_models_pytorch as smp
    except ImportError:
        print("Install: pip install segmentation-models-pytorch")
      

    model = smp.UnetPlusPlus(encoder_name=ENCODER_BACKBONE, encoder_weights="imagenet",
                             in_channels=3, classes=1).to(DEVICE)

   
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)

    total_params = sum(p.numel() for p in model.parameters())
    print(f" Model: {total_params:,} parameters")

    loss_fn = FocalTverskyLoss(alpha=LOSS_ALPHA, beta=LOSS_BETA, gamma=LOSS_GAMMA)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler('cuda') if DEVICE == "cuda" else None

   
    best_val_loss = float('inf')
    best_dice_score = 0.0
    history = {'train_loss': [], 'val_loss': [], 'dice_score': [], 'epoch': []}

    print(f"\n Starting Continuous 50-Epoch Training...")
    training_start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        current_epoch = epoch + 1

        print(f"\n{'=' * 100}")
        print(f"🚀 EPOCH {current_epoch}/{NUM_EPOCHS}")
        print(f"{'=' * 100}")

        train_loss, batch_losses = train_one_epoch(train_loader, model, optimizer, loss_fn, scaler, DEVICE,
                                                   current_epoch)

    
        val_loss, dice = validate_one_epoch(val_loader, model, loss_fn, DEVICE)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['dice_score'].append(dice)
        history['epoch'].append(current_epoch)

        epoch_time = time.time() - epoch_start

        print(f"\n EPOCH {current_epoch} SUMMARY:")
        print(f"Time: {epoch_time / 60:.1f} minutes")
        print(f"Train Loss: {train_loss:.4f}")
        print(f" Val Loss: {val_loss:.4f}")
        print(f" Dice Score: {dice:.4f}")

       
        if dice > best_dice_score:
            best_dice_score = dice
            best_val_loss = val_loss
            best_model_path = MODEL_SAVE_DIR / f"{MODEL_ARCHITECTURE}_{ENCODER_BACKBONE}_best_model.pth"
            if hasattr(model, 'module'):
                torch.save(model.module.state_dict(), best_model_path)
            else:
                torch.save(model.state_dict(), best_model_path)
            print(f" NEW BEST MODEL: Dice {best_dice_score:.4f}")

    
        if current_epoch % 5 == 0:
            print(f"\nSaving checkpoint and analysis for epoch {current_epoch}...")

          
            checkpoint_path = MODEL_SAVE_DIR / f"checkpoint_epoch_{current_epoch}.pth"
            save_enhanced_checkpoint(model, optimizer, current_epoch, train_loss, val_loss, dice, history,
                                     checkpoint_path)

          
            avg_metrics, std_metrics, metrics_df, val_losses = comprehensive_validation(
                model, val_loader, loss_fn, DEVICE, current_epoch)

            print(f"    Comprehensive Metrics:")
            print(f"      • Dice: {avg_metrics['dice']:.4f} ± {std_metrics['dice']:.4f}")
            print(f"      • IoU: {avg_metrics['iou']:.4f}")
            print(f"      • Precision: {avg_metrics['precision']:.4f}")
            print(f"      • Recall: {avg_metrics['recall']:.4f}")
            print(f"      • F1-Score: {avg_metrics['f1']:.4f}")

            
            metrics_df.to_csv(ANALYSIS_DIR / f"validation_metrics_epoch_{current_epoch}.csv", index=False)

          
            history_df = pd.DataFrame(history)
            plot_training_progress(history_df, LOGS_DIR / f"training_progress_epoch_{current_epoch}.png")
            plot_metrics_summary(metrics_df, current_epoch, ANALYSIS_DIR / f"metrics_summary_epoch_{current_epoch}.png")

        
        history_df = pd.DataFrame(history)
        history_df.to_csv(LOGS_DIR / "training_history.csv", index=False)

       
        elapsed_time = time.time() - training_start_time
        avg_epoch_time = elapsed_time / current_epoch
        remaining_epochs = NUM_EPOCHS - current_epoch
        estimated_remaining = remaining_epochs * avg_epoch_time

        print(f"  Estimated remaining: {estimated_remaining / 3600:.1f} hours")
        print(
            f"  Expected completion: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() + estimated_remaining))}")

    total_training_time = time.time() - training_start_time

    


  
    print(f"\n Final Test Set Evaluation...")
    test_df = df[df['split'] == 'test'].reset_index(drop=True)
    test_dataset = CrackDataset(test_df, base_path=LOCAL_INPUT_DIR, transforms=val_test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    
    best_model_path = MODEL_SAVE_DIR / f"{MODEL_ARCHITECTURE}_{ENCODER_BACKBONE}_best_model.pth"
    if best_model_path.exists():
        if hasattr(model, 'module'):
            model.module.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
        else:
            model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
        print("Best model loaded for final evaluation")

    
    final_avg_metrics, final_std_metrics, final_metrics_df, _ = comprehensive_validation(
        model, test_loader, loss_fn, DEVICE, "FINAL")

    print(f"\n FINAL TEST RESULTS:")
    print(f" Dice Score: {final_avg_metrics['dice']:.4f} ± {final_std_metrics['dice']:.4f}")
    print(f" IoU: {final_avg_metrics['iou']:.4f}")
    print(f" Precision: {final_avg_metrics['precision']:.4f}")
    print(f" Recall: {final_avg_metrics['recall']:.4f}")
    print(f" F1-Score: {final_avg_metrics['f1']:.4f}")

    
    final_metrics_df.to_csv(ANALYSIS_DIR / "final_test_metrics.csv", index=False)

   
    final_history_df = pd.DataFrame(history)
    plot_training_progress(final_history_df, LOGS_DIR / "final_training_progress.png")

 
    final_report = {
        
            'total_epochs': NUM_EPOCHS,
            'training_time_hours': total_training_time / 3600,
            'best_dice_score': best_dice_score,
            'best_val_loss': best_val_loss,
            'hardware': 'Dual Quadro GV100'
        },
        'final_test_metrics': final_avg_metrics.to_dict(),
        'model_config': {
            'architecture': MODEL_ARCHITECTURE,
            'backbone': ENCODER_BACKBONE,
            'image_size': IMAGE_SIZE,
            'batch_size': BATCH_SIZE
        }
    }

    import json
    with open(ANALYSIS_DIR / "final_training_report.json", 'w') as f:
        json.dump(final_report, f, indent=2)

    print(f"\nAll results saved to:")
    print(f"Models: {MODEL_SAVE_DIR}")
    print(f"Logs: {LOGS_DIR}")
    print(f"Analysis: {ANALYSIS_DIR}")



if __name__ == '__main__':
    import multiprocessing

    multiprocessing.freeze_support()
    main()