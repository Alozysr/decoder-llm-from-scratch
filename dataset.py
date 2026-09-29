"""
dataset.py - Memory-Mapped Veri Yükleyici (DataLoader)
Bellek dostu numpy memmap yapısıyla büyük boyutlu .bin dosyalarından
hızlıca eğitim ve doğrulama batch'leri çeker.
"""

import os
from typing import Tuple
import numpy as np
import torch


class MemmapDataLoader:
    """
    Disk üzerindeki .bin dosyalarından RAM'i tüketmeden
    rastgele girdi (x) ve hedef (y) tensörleri çeker.
    """
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.train_path = os.path.join(data_dir, "train.bin")
        self.val_path = os.path.join(data_dir, "val.bin")
        
        # Eğer val.bin yoksa validation.bin kontrol et (uyumluluk için)
        if not os.path.exists(self.val_path):
            alt_val = os.path.join(data_dir, "validation.bin")
            if os.path.exists(alt_val):
                self.val_path = alt_val

        self._check_files()
        
        self.train_data = np.memmap(self.train_path, dtype=np.uint16, mode="r")
        self.val_data = np.memmap(self.val_path, dtype=np.uint16, mode="r")

    def _check_files(self):
        if not os.path.exists(self.train_path):
            raise FileNotFoundError(
                f"Eğitim veri dosyası bulunamadı: {self.train_path}\n"
                f"Lütfen önce 'python prepare_data.py' komutunu çalıştırarak verileri hazırlayın."
            )
        if not os.path.exists(self.val_path):
            raise FileNotFoundError(
                f"Doğrulama veri dosyası bulunamadı: {self.val_path}\n"
                f"Lütfen önce 'python prepare_data.py' komutunu çalıştırarak verileri hazırlayın."
            )

    def get_batch(
        self,
        split: str,
        batch_size: int,
        block_size: int,
        device: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        split: 'train' veya 'val'
        Belirtilen split'ten rastgele batch_size kadar dizi çeker.
        Her örnek block_size uzunluğundadır; y dizisi x'in bir token sağa ötelenmiş halidir.
        """
        data = self.train_data if split == "train" else self.val_data
        
        if len(data) <= block_size:
            raise ValueError(
                f"{split} verisi toplam {len(data)} token içeriyor, "
                f"bu değer block_size ({block_size}) değerinden büyük olmalıdır!"
            )

        # Rastgele başlangıç indisleri seç
        ix = torch.randint(len(data) - block_size, (batch_size,))
        
        # CPU üzerinde tensörleri yığınla
        x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix])

        # Hedef cihaza aktar
        if "cuda" in device:
            # pin_memory ve non_blocking transferi GPU'ya aktarımı hızlandırır
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)

        return x, y
