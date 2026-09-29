# 🧠 Decoder LLM From Scratch

PyTorch kullanılarak sıfırdan (*from scratch*) geliştirilmiş, **Decoder-Only (GPT tarzı)** bir dil modeli projesi. 

Gutenberg Kütüphanesi'ndeki klasik İngilizce romanlar ve kurgu eserleri üzerinde eğitilerek, verilen başlangıç metinlerini roman üslubunda akıcı ve tutarlı bir şekilde devam ettirmeyi amaçlar.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Alozysr/decoder-llm-from-scratch/blob/main/dil_modeli_denemesi.ipynb)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🌟 Öne Çıkan Özellikler

- **Saf PyTorch Mimarisi:** Harici hazır model kütüphaneleri (HuggingFace Transformers vb.) kullanılmadan, Transformer blokları ve dikkat mekanizmaları matematiksel temelleriyle sıfırdan kodlanmıştır.
- **Nedensel Çok Başlıklı Öz-Dikkat (Causal Multi-Head Self-Attention):** Gelecekteki token'ları gizlemek için alt üçgensel maskeleme matrisi (`tril mask`) kullanır.
- **Ağırlık Paylaşımı (Weight Tying):** `token_emb` ve `lm_head` ağırlıkları paylaştırılarak hem model boyutu optimize edilmiş hem de öğrenme performansı artırılmıştır.
- **BPE Tokenizasyon:** OpenAI GPT-2 standartlarındaki `tiktoken` (50,257 kelime boyutu) kullanılmıştır.
- **Hafıza Dostu Veri Akışı (Memory-Mapped I/O):** Milyonlarca token'dan oluşan veri setleri doğrudan RAM'e yüklenmek yerine `numpy.memmap` ile diskten anlık çekilir.
- **Gelişmiş Eğitim Döngüsü:**
  - `AdamW` optimizasyonu ve `CosineAnnealingLR` öğrenme oranı çizelgeleyici
  - Donanım hızlandırma için Otomatik Karışık Duyarlılık (`torch.amp.autocast` / `GradScaler`)
  - Gradyan patlamasını önlemek için `Gradient Clipping`
  - Aşırı öğrenmeyi önlemek için Doğrulama Kaybı (Val Loss) takipli `Early Stopping`
- **Zengin Örnekleme (Inference) Stratejileri:**
  - `Temperature` ölçeklendirme
  - `Top-p (Nucleus)` ve `Top-k` filtreleme
  - `Repetition Penalty` (aynı kelime veya ifadelerin tekrar etmesini önleme)

---

## 📁 Proje Dizin Yapısı

Proje, GitHub üzerinde kolay okunabilmesi ve genişletilebilmesi için modüler parçalara ayrılmıştır:

```text
decoder-llm-from-scratch/
│
├── config.py                 # Model, eğitim, veri ve metin üretim parametreleri (Dataclasses)
├── model.py                  # CausalSelfAttention, Block ve GPT mimarisi
├── prepare_data.py           # Veri setini indirme, filtreleme, BPE tokenization ve .bin oluşturma
├── dataset.py                # Disk üzerindeki .bin dosyalarından bellek dostu (memmap) batch yükleyici
├── train.py                  # Eğitim döngüsü, AMP, LR Scheduler, Early Stopping ve Checkpoint yönetimi
├── generate.py               # Eğitilmiş checkpoint üzerinden metin üretimi (Inference)
│
├── dil_modeli_denemesi.ipynb # Google Colab üzerinde geliştirilen orijinal etkileşimli notebook
├── requirements.txt          # Gerekli Python bağımlılıkları
├── .gitignore                # Git tarafında takip edilmeyecek büyük dosyalar (.bin, .pt vb.)
└── README.md                 # Proje dokümantasyonu
```

---

## ⚙️ Model ve Eğitim Parametreleri

Tüm hiperparametreler [`config.py`](file:///Users/aliozyasar/Programlama/V%C4%B1sual%20Stud%C4%B1o%20Code/Python/Eski%20Romanlardan%20Metin%20%C3%9Cretme%20Projesi/decoder-llm-from-scratch/config.py) dosyası içerisinde merkezi olarak yönetilir:

### Model Parametreleri (`ModelConfig`)
| Parametre | Değer | Açıklama |
| :--- | :--- | :--- |
| `vocab_size` | 50,257 | GPT-2 BPE sözlük boyutu |
| `block_size` | 1024 | Maksimum bağlam uzunluğu (context length) |
| `n_embd` | 600 | Embedding / Gizli katman vektör boyutu |
| `n_head` | 10 | Çoklu dikkat başlık sayısı (`600 // 10 = 60` başlık boyutu) |
| `n_layer` | 10 | Ardışık Transformer Blok (Decoder) katman sayısı |
| `dropout` | 0.15 | Düzenlileştirme (regularization) oranı |

### Eğitim Parametreleri (`TrainingConfig`)
| Parametre | Değer | Açıklama |
| :--- | :--- | :--- |
| `batch_size` | 72 | Her adımda işlenen örnek sayısı |
| `learning_rate`| 3e-4 | Maksimum başlangıç öğrenme oranı |
| `weight_decay` | 0.1 | AdamW ağırlık azaltma |
| `max_iters` | 20,000 | Toplam eğitim iterasyon hedefi |
| `eval_interval`| 250 | Kaç adımda bir doğrulama loss'unun ölçüleceği |
| `patience` | 4 | Early stopping sabır periyodu |
| `use_amp` | True | FP16 Karışık Duyarlılık (GPU hızlandırıcı) |

---

## 🚀 Hızlı Başlangıç

### 1. Kurulum

Öncelikle depoyu klonlayın ve gerekli bağımlılıkları yükleyin:

```bash
git clone https://github.com/Alozysr/decoder-llm-from-scratch.git
cd decoder-llm-from-scratch
pip install -r requirements.txt
```

### 2. Veri Setini Hazırlama

Gutenberg veri setini indirmek, kurgu/roman eserlerini filtrelemek ve binary dosyalara dönüştürmek için:

```bash
python prepare_data.py --output_dir data
```
Bu işlem sonucunda `data/` klasörü altında `train.bin` ve `val.bin` dosyaları oluşturulur.

### 3. Modeli Eğitme

Eğitimi başlatmak için:

```bash
python train.py --batch_size 72 --max_iters 20000
```

> **İpucu:** Kendi donanımınıza göre parametreleri değiştirebilirsiniz (örneğin GPU belleğiniz sınırlıysa `--batch_size 32` veya `--batch_size 16` kullanabilirsiniz).

Eğitim sırasında en düşük validation loss değerine ulaşıldığında model otomatik olarak `checkpoints/best_model.pt` dosyasına kaydedilir.

### 4. Metin Üretme (Inference)

Eğitilen en iyi checkpoint'i kullanarak roman devamı yazdırmak için:

```bash
python generate.py \
  --checkpoint checkpoints/best_model.pt \
  --prompt "The old carriage stopped in front of the dark mansion. Mr. Darcy stepped out," \
  --temperature 0.6 \
  --top_p 0.9 \
  --repetition_penalty 1.2 \
  --max_new_tokens 500
```

---

## 📖 Google Colab ile Kullanım

Proje doğrudan Google Colab üzerinde de çalıştırılabilir:
1. Depodaki [Open In Colab](https://colab.research.google.com/github/Alozysr/decoder-llm-from-scratch/blob/main/dil_modeli_denemesi.ipynb) butonuna tıklayın.
2. GPU çalışma zamanını etkinleştirin (`Runtime` -> `Change runtime type` -> `T4 GPU`).
3. Hücreleri sırasıyla çalıştırarak modeli eğitin veya hazır ağırlıklarla test edin.

---

## 📜 Lisans

Bu proje [MIT Lisansı](LICENSE) kapsamında açık kaynak olarak sunulmuştur. Öğrenme, araştırma ve geliştirme amaçlı dilediğiniz gibi kullanabilirsiniz.
