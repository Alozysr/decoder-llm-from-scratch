"""
prepare_data.py - Veri Seti İndirme, Filtreleme ve Tokenization
sedthh/gutenberg_english veri setini indirir, roman/kurgu eserlerini filtreler,
tiktoken (GPT-2 BPE) ile tokenize eder ve hızlı eğitim için memmap uyumlu
train.bin ve val.bin dosyalarını oluşturur.
"""

import os
import json
import argparse
import numpy as np
import tiktoken
from datasets import load_dataset

from config import DataConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Veri seti indirme ve tokenization")
    parser.add_argument("--output_dir", type=str, default="data", help="İşlenmiş .bin dosyalarının kaydedileceği dizin")
    parser.add_argument("--dataset_name", type=str, default="sedthh/gutenberg_english", help="Hugging Face veri seti adı")
    parser.add_argument("--test_size", type=float, default=0.05, help="Validation seti ayrılma oranı (varsayılan: 0.05)")
    parser.add_argument("--max_books", type=int, default=None, help="İşlenecek maksimum kitap sayısı (isteğe bağlı)")
    parser.add_argument("--seed", type=int, default=42, help="Rastgelelik tohumu")
    return parser.parse_args()


def filter_fiction(dataset, keywords):
    """Metadata içindeki konu ve kategori bilgilerine göre roman/kurgu eserlerini filtreler."""
    def is_fiction(example):
        try:
            meta = json.loads(example["METADATA"])
            text = (meta.get("bookshelves", "") + " " + meta.get("subjects", "")).lower()
            return any(kw in text for kw in keywords)
        except (json.JSONDecodeError, TypeError, KeyError):
            return False

    print("Roman ve kurgu kitapları filtreleniyor...")
    filtered = dataset.filter(is_fiction)
    print(f"Filtreleme tamamlandı: {len(filtered)} adet kurgu kitabı bulundu.")
    return filtered


def main():
    args = parse_args()
    config = DataConfig(
        dataset_name=args.dataset_name,
        test_size=args.test_size,
        seed=args.seed,
        output_dir=args.output_dir
    )

    os.makedirs(config.output_dir, exist_ok=True)

    print(f"Hugging Face veri seti indiriliyor: {config.dataset_name}")
    raw_dataset = load_dataset(config.dataset_name)
    train_split = raw_dataset["train"]
    print(f"Toplam ham kitap sayısı: {len(train_split)}")

    # Kurgu/Roman filtreleme
    fiction_split = filter_fiction(train_split, config.fiction_keywords)

    if args.max_books is not None and args.max_books < len(fiction_split):
        print(f"Seçilen kitap sayısı sınırlandırılıyor: {args.max_books}")
        fiction_split = fiction_split.shuffle(seed=config.seed).select(range(args.max_books))

    # Train / Validation bölümü
    print(f"Veri seti bölünüyor (Validation oranı: {config.test_size})...")
    splits = fiction_split.train_test_split(test_size=config.test_size, seed=config.seed)
    dataset = {
        "train": splits["train"],
        "val": splits["test"]
    }
    print(f"Eğitim seti: {len(dataset['train'])} kitap | Doğrulama seti: {len(dataset['val'])} kitap")

    # Tokenizer yükleme (GPT-2 BPE)
    print("GPT-2 BPE Tokenizer (tiktoken) başlatılıyor...")
    enc = tiktoken.get_encoding(config.tokenizer_name)

    def process(example):
        ids = enc.encode_ordinary(example["TEXT"])
        ids.append(enc.eot_token)  # Metin sonu token'ı ekle
        return {"ids": ids, "len": len(ids)}

    # Tokenize ve .bin dosyalarına yazma
    cpu_count = os.cpu_count() or 1
    for split_name, dset in dataset.items():
        print(f"\n[{split_name.upper()}] seti tokenize ediliyor...")
        tokenized = dset.map(
            process,
            remove_columns=dset.column_names,
            desc=f"{split_name} tokenize ediliyor",
            num_proc=max(1, cpu_count // 2),
            writer_batch_size=800,
        )

        total_tokens = int(np.sum(tokenized["len"], dtype=np.uint64))
        bin_path = os.path.join(config.output_dir, f"{split_name}.bin")
        print(f"Toplam token sayısı ({split_name}): {total_tokens:,}")
        print(f"Binary dosyaya yazılıyor: {bin_path}")

        # GPT-2 kelime dağarcığı 50257 < 65535 olduğu için uint16 bellek açısından en verimlisidir
        dtype = np.uint16
        arr = np.memmap(bin_path, dtype=dtype, mode="w+", shape=(total_tokens,))

        idx = 0
        for example in tokenized:
            length = example["len"]
            arr[idx : idx + length] = example["ids"]
            idx += length
        arr.flush()
        print(f"Tamamlandı: {bin_path} ({os.path.getsize(bin_path) / (1024 * 1024):.2f} MB)")

    print("\nTüm veri hazırlığı başarıyla tamamlandı!")


if __name__ == "__main__":
    main()
