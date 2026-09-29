"""
train.py - Transformer Dil Modeli Eğitim Betiği
Bu betik, decoder-only GPT modelini hazirlanan veri seti uzerinde egitir.
Ozellikler:
- AdamW Optimizer ve Cosine Annealing LR Scheduler
- Otomatik Karisik Duyarlilik (Automatic Mixed Precision - AMP) ile hizlandirma
- Gradyan Kirpma (Gradient Clipping) ile kararlilik
- Erken Durdurma (Early Stopping) ve En Iyi Modeli Kaydetme (Checkpointing)
"""

import os
import time
import argparse
from typing import Dict
import torch

from config import ModelConfig, TrainingConfig
from model import GPT
from dataset import MemmapDataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Decoder LLM Model Eğitimi")
    
    # Model mimari argümanları
    parser.add_argument("--n_embd", type=int, default=600, help="Embedding boyutu")
    parser.add_argument("--n_head", type=int, default=10, help="Attention başlık sayısı")
    parser.add_argument("--n_layer", type=int, default=10, help="Transformer katman sayısı")
    parser.add_argument("--block_size", type=int, default=1024, help="Context uzunluğu")
    parser.add_argument("--dropout", type=float, default=0.15, help="Dropout oranı")
    
    # Eğitim argümanları
    parser.add_argument("--batch_size", type=int, default=72, help="Batch boyutu")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Öğrenme oranı")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="AdamW weight decay")
    parser.add_argument("--max_iters", type=int, default=20000, help="Maksimum adım sayısı")
    parser.add_argument("--eval_interval", type=int, default=250, help="Kaç adımda bir eval yapılacağı")
    parser.add_argument("--eval_iters", type=int, default=50, help="Eval adım sayısı")
    parser.add_argument("--patience", type=int, default=4, help="Early stopping sabır sayısı")
    
    # Dizinler ve donanım
    parser.add_argument("--data_dir", type=str, default="data", help="Veri dizini (.bin dosyaları)")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Model kayıt dizini")
    parser.add_argument("--device", type=str, default=None, help="Cihaz (cuda, mps, cpu)")
    parser.add_argument("--no_amp", action="store_true", help="Mixed precision'ı devre dışı bırak")
    
    return parser.parse_args()


@torch.no_grad()
def estimate_loss(
    model: GPT,
    dataloader: MemmapDataLoader,
    eval_iters: int,
    batch_size: int,
    block_size: int,
    device: str
) -> Dict[str, float]:
    """Train ve validation veri setlerinde ortalama loss değerini hesaplar."""
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = dataloader.get_batch(split, batch_size, block_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    args = parse_args()
    
    # Konfigürasyonları oluştur
    model_cfg = ModelConfig(
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        block_size=args.block_size,
        dropout=args.dropout
    )
    
    train_cfg = TrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_iters=args.max_iters,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        patience=args.patience,
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        use_amp=not args.no_amp
    )

    device = args.device or train_cfg.device
    print(f"==================================================")
    print(f"Decoder-Only Transformer Model Eğitimi")
    print(f"Cihaz: {device}")
    print(f"Model Yapısı: {model_cfg.n_layer} katman, {model_cfg.n_head} başlık, {model_cfg.n_embd} embedding")
    print(f"Context Boyutu: {model_cfg.block_size} token")
    print(f"==================================================")

    # Dizinleri hazırla
    os.makedirs(train_cfg.checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(train_cfg.checkpoint_dir, train_cfg.best_model_file)
    final_model_path = os.path.join(train_cfg.checkpoint_dir, train_cfg.final_model_file)

    # Veri yükleyiciyi başlat
    dataloader = MemmapDataLoader(data_dir=train_cfg.data_dir)
    print("Veri setleri başarıyla belleğe eşlendi (memmap).")

    # Modeli oluştur
    model = GPT(model_cfg).to(device)
    num_params = model.get_num_params()
    print(f"Toplam Parametre Sayısı: {num_params / 1e6:.2f} Milyon")

    # Optimizer ve Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=train_cfg.max_iters,
        eta_min=train_cfg.min_lr
    )

    # AMP Scaler (CUDA için FP16 hızlandırması)
    is_cuda = "cuda" in device
    scaler = torch.cuda.amp.GradScaler(enabled=(train_cfg.use_amp and is_cuda))

    # Early stopping takipçileri
    best_val_loss = float("inf")
    patience_counter = 0

    print("\nEğitim Başlıyor...")
    start_time = time.time()

    for step in range(train_cfg.max_iters):
        t0 = time.time()

        # Belirli aralıklarla loss hesapla ve kontrol et
        if step % train_cfg.eval_interval == 0 or step == train_cfg.max_iters - 1:
            losses = estimate_loss(
                model=model,
                dataloader=dataloader,
                eval_iters=train_cfg.eval_iters,
                batch_size=train_cfg.batch_size,
                block_size=model_cfg.block_size,
                device=device
            )
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"[Adım {step:5d}/{train_cfg.max_iters}] "
                f"Train Loss: {losses['train']:.4f} | "
                f"Val Loss: {losses['val']:.4f} | "
                f"LR: {current_lr:.6f}"
            )

            # Model iyileşti mi?
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                patience_counter = 0
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "iter": step,
                    "val_loss": best_val_loss,
                    "config": {
                        "vocab_size": model_cfg.vocab_size,
                        "n_embd": model_cfg.n_embd,
                        "n_head": model_cfg.n_head,
                        "n_layer": model_cfg.n_layer,
                        "block_size": model_cfg.block_size,
                        "dropout": model_cfg.dropout,
                    }
                }
                torch.save(checkpoint, best_model_path)
                print(f"  ★ Yeni en iyi model kaydedildi -> {best_model_path} (Val Loss: {best_val_loss:.4f})")
            else:
                patience_counter += 1
                print(f"  → İyileşme yok ({patience_counter}/{train_cfg.patience})")
                if patience_counter >= train_cfg.patience:
                    print(f"\n[!] Early Stopping tetiklendi! Adım {step}'te durduruldu.")
                    print(f"En iyi Validation Loss: {best_val_loss:.4f}")
                    break

        # Batch al
        xb, yb = dataloader.get_batch("train", train_cfg.batch_size, model_cfg.block_size, device)
        optimizer.zero_grad(set_to_none=True)

        # Forward pass (Mixed Precision ile)
        device_type = "cuda" if is_cuda else "cpu"
        with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=(train_cfg.use_amp and is_cuda)):
            _, loss = model(xb, yb)

        # Backward pass ve gradyan kırpma
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=train_cfg.grad_clip)

        # Optimizasyon adımı
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

    total_time = time.time() - start_time
    print(f"\nEğitim tamamlandı! Toplam süre: {total_time / 60:.2f} dakika.")

    # Son model durumunu kaydet
    torch.save(model.state_dict(), final_model_path)
    print(f"Son model ağırlıkları kaydedildi -> {final_model_path}")


if __name__ == "__main__":
    main()
