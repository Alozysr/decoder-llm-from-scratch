"""
config.py - Model, Eğitim ve Veri Parametreleri
Bu dosya modelin mimarisini ve eğitim sürecini belirleyen tüm hiperparametreleri içerir.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import torch


@dataclass
class ModelConfig:
    """Transformer Dil Modeli Mimari Parametreleri"""
    vocab_size: int = 50257       # GPT-2 BPE sözlük boyutu
    block_size: int = 1024        # Maksimum bağlam uzunluğu (context length)
    n_embd: int = 600             # Embedding (vektör) boyutu
    n_head: int = 10              # Çoklu dikkat (multi-head attention) başlık sayısı
    n_layer: int = 10             # Transformer blok (katman) sayısı
    dropout: float = 0.15         # Dropout düzenlileştirme oranı
    bias: bool = True             # LayerNorm ve Lineer katmanlarda bias kullanımı

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0, (
            f"n_embd ({self.n_embd}) n_head ({self.n_head}) değerine tam bölünmelidir!"
        )


@dataclass
class TrainingConfig:
    """Model Eğitimi ve Optimizasyon Hiperparametreleri"""
    # Cihaz ayarı (CUDA / Apple Silicon MPS / CPU)
    device: str = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Batch ve iterasyon
    batch_size: int = 72          # Her adımda işlenecek örnek sayısı (GPU belleğinize göre ayarlayın)
    max_iters: int = 20000        # Toplam eğitim adım sayısı
    eval_interval: int = 250      # Kaç adımda bir validation loss hesaplanacağı
    eval_iters: int = 50          # Değerlendirmede ortalaması alınacak batch sayısı
    
    # Optimizasyon
    learning_rate: float = 3e-4   # Başlangıç öğrenme oranı
    weight_decay: float = 0.1     # Ağırlık azaltma katsayısı (AdamW)
    grad_clip: float = 1.0        # Gradyan patlamasını önlemek için kırpma eşiği
    min_lr: float = 1e-5          # Cosine Annealing minimum öğrenme oranı
    
    # Erken Durdurma (Early Stopping)
    patience: int = 4             # İyileşme olmazsa kaç eval periyodu sonra durdurulacağı
    
    # Donanım hızlandırma
    use_amp: bool = True          # Mixed precision (FP16) kullanımı (CUDA için)
    
    # Kayıt yolları
    data_dir: str = "data"
    checkpoint_dir: str = "checkpoints"
    best_model_file: str = "best_model.pt"
    final_model_file: str = "final_model.pt"


@dataclass
class DataConfig:
    """Veri Seti Hazırlama ve Ön İşleme Parametreleri"""
    dataset_name: str = "sedthh/gutenberg_english"
    test_size: float = 0.05       # Validation seti oranı (%5)
    seed: int = 42
    tokenizer_name: str = "gpt2"
    output_dir: str = "data"
    
    # Roman / Kurgu filtreleme anahtar kelimeleri
    fiction_keywords: List[str] = field(default_factory=lambda: [
        "fiction", "novel", "romance", "adventure", "mystery", "gothic"
    ])


@dataclass
class GenerationConfig:
    """Metin Üretim (Inference) Parametreleri"""
    max_new_tokens: int = 750     # Üretilecek maksimum yeni token sayısı
    temperature: float = 0.6      # Sıcaklık (düşük = daha tutarlı, yüksek = daha yaratıcı)
    top_p: Optional[float] = 0.9  # Nucleus sampling (top_p)
    top_k: Optional[int] = None   # Top-k sampling
    repetition_penalty: float = 1.2  # Tekrar eden kelimeleri engelleme cezası
