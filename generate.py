"""
generate.py - Eğitilmiş Modelden Metin Üretme (Inference)
Eğitilen model checkpoint'ini yükler ve verilen başlangıç metnine (prompt) göre
temperature, top-p, top-k ve repetition penalty ayarlarıyla yeni metin üretir.
"""

import os
import argparse
import torch
import tiktoken

from config import ModelConfig, GenerationConfig
from model import GPT


def parse_args():
    parser = argparse.ArgumentParser(description="Decoder LLM Metin Üretimi (Inference)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Yüklenecek model checkpoint yolu")
    parser.add_argument(
        "--prompt",
        type=str,
        default="The old carriage stopped in front of the dark mansion. Mr. Darcy stepped out, looking at the broken windows",
        help="Modelin devam ettireceği başlangıç metni"
    )
    parser.add_argument("--max_new_tokens", type=int, default=750, help="Üretilecek yeni token sayısı")
    parser.add_argument("--temperature", type=float, default=0.6, help="Yaratıcılık / rastgelelik derecesi (0.1 - 1.5)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling eşiği (0.0 - 1.0)")
    parser.add_argument("--top_k", type=int, default=None, help="Top-K örnekleme filtresi")
    parser.add_argument("--repetition_penalty", type=float, default=1.2, help="Tekrar eden kelimeler için ceza katsayısı")
    parser.add_argument("--device", type=str, default=None, help="Cihaz (cuda, mps, cpu)")
    return parser.parse_args()


def load_model(checkpoint_path: str, device: str) -> GPT:
    """Checkpoint dosyasından konfigürasyonu ve model ağırlıklarını yükler."""
    if not os.path.exists(checkpoint_path):
        # Alternatif yollar dene (örneğin doğrudan ana dizinde best_model.pt)
        alt_path = os.path.basename(checkpoint_path)
        if os.path.exists(alt_path):
            checkpoint_path = alt_path
        else:
            raise FileNotFoundError(f"Model dosyası bulunamadı: {checkpoint_path}")

    print(f"Model yükleniyor: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Checkpoint içindeki model yapılandırmasını al
    if isinstance(checkpoint, dict) and "config" in checkpoint:
        cfg_dict = checkpoint["config"]
        model_config = ModelConfig(
            vocab_size=cfg_dict.get("vocab_size", 50257),
            n_embd=cfg_dict.get("n_embd", 600),
            n_head=cfg_dict.get("n_head", 10),
            n_layer=cfg_dict.get("n_layer", 10),
            block_size=cfg_dict.get("block_size", 1024),
            dropout=cfg_dict.get("dropout", 0.0),
        )
    else:
        print("Uyarı: Checkpoint'te config bulunamadı, varsayılan ModelConfig kullanılıyor.")
        model_config = ModelConfig()

    model = GPT(model_config).to(device)

    # Ağırlıkları yükle
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        if "val_loss" in checkpoint:
            print(f"Model Checkpoint Val Loss: {checkpoint['val_loss']:.4f}")
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print("Model başarıyla yüklendi ve değerlendirme moduna alındı.\n")
    return model


def main():
    args = parse_args()
    
    # Cihaz belirle
    device = args.device or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Kullanılan cihaz: {device}")

    # Modeli yükle
    model = load_model(args.checkpoint, device)

    # Tokenizer yükle (GPT-2)
    enc = tiktoken.get_encoding("gpt2")

    # Prompt'ı hazırla
    prompt = args.prompt
    print(f"--- Başlangıç Metni (Prompt) ---")
    print(prompt)
    print("--------------------------------\n")
    print("Metin üretiliyor...")

    prompt_tokens = enc.encode(prompt)
    context = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)

    # Metin üret
    generated_tokens = model.generate(
        idx=context,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        top_k=args.top_k,
        top_p=args.top_p
    )

    # Token'ları metne dönüştür
    output_text = enc.decode(generated_tokens[0].tolist())

    print("\n================== OLUŞTURULAN METİN ==================")
    print(output_text)
    print("=======================================================")


if __name__ == "__main__":
    main()
