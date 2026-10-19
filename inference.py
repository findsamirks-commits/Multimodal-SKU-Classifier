"""Inference module for Multimodal Product Classification.
Provides a clean, production-ready ProductPredictor class for single-item
and batch predictions given a product title and image.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from PIL import Image, ImageOps
import torch

from dataset import SimpleVocab, get_image_transforms
from model import MultimodalProductClassifier


class ProductPredictor:
    """Production predictor for scoring product titles and images."""

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        vocab_path: Optional[Union[str, Path]] = None,
        label_map_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found at: {self.checkpoint_path}"
            )

        self.device = (
            device
            if device
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.cfg_dict = checkpoint["config"]

        # 1. Label Mapping
        if label_map_path and Path(label_map_path).exists():
            with open(label_map_path, "r", encoding="utf-8") as f:
                self.label_to_id = json.load(f)
        else:
            self.label_to_id = checkpoint.get("label_to_id", {})
        self.id_to_label = {int(v): k for k, v in self.label_to_id.items()}

        # 2. Vocabulary
        if vocab_path and Path(vocab_path).exists():
            self.vocab = SimpleVocab.load(vocab_path)
        else:
            default_vocab_path = self.checkpoint_path.parent / "vocab.json"
            if default_vocab_path.exists():
                self.vocab = SimpleVocab.load(default_vocab_path)
            else:
                raise FileNotFoundError(
                    "Vocabulary file not found. Please provide vocab_path."
                )

        # 3. Model Architecture & Weights
        self.model = MultimodalProductClassifier(
            num_classes=self.cfg_dict["num_classes"],
            vocab_size=self.cfg_dict["vocab_size"],
            image_backbone=self.cfg_dict["image_backbone"],
            pretrained_image=False,
            text_encoder_type=self.cfg_dict["text_encoder_type"],
            text_embed_dim=self.cfg_dict["text_embed_dim"],
            dropout=self.cfg_dict.get("dropout", 0.0),
            use_batch_norm=self.cfg_dict.get("use_batch_norm", True),
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        # 4. Transform Pipeline
        image_size = self.cfg_dict.get("image_size", 224)
        self.transform = get_image_transforms(is_train=False, image_size=image_size)

    def _prepare_image(self, image_input: Union[str, Path, Image.Image]) -> torch.Tensor:
        """Convert input image (path or PIL Image) to normalized RGB tensor."""
        if isinstance(image_input, (str, Path)):
            with Image.open(image_input) as img:
                img = ImageOps.exif_transpose(img)
                pil_img = img.convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image_input)}")

        return self.transform(pil_img)

    @torch.no_grad()
    def predict(
        self,
        title: str,
        image: Union[str, Path, Image.Image],
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """Classify a single product title and image.

        Returns top predicted class, confidence, and top-k probability
        distribution.
        """
        # Process image
        img_tensor = self._prepare_image(image).unsqueeze(0).to(self.device)

        # Process text
        token_ids = self.vocab.encode(title)
        flat_tokens = torch.tensor(token_ids, dtype=torch.long, device=self.device)
        offsets = torch.tensor([0], dtype=torch.long, device=self.device)

        logits = self.model(images=img_tensor, tokens=flat_tokens, offsets=offsets)
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        sorted_indices = np.argsort(probs)[::-1]
        top_k = min(top_k, len(sorted_indices))

        top_predictions = [
            {
                "category": self.id_to_label.get(idx, f"Class_{idx}"),
                "probability": float(probs[idx]),
            }
            for idx in sorted_indices[:top_k]
        ]

        best_idx = int(sorted_indices[0])
        return {
            "title": title,
            "predicted_category": self.id_to_label.get(best_idx, f"Class_{best_idx}"),
            "confidence": float(probs[best_idx]),
            "top_k": top_predictions,
            "all_probabilities": {
                self.id_to_label.get(i, f"Class_{i}"): float(probs[i])
                for i in range(len(probs))
            },
        }

    @torch.no_grad()
    def predict_batch(
        self,
        titles: List[str],
        images: List[Union[str, Path, Image.Image]],
        top_k: int = 1,
    ) -> List[Dict[str, Any]]:
        """Run batch inference for multiple products."""
        if len(titles) != len(images):
            raise ValueError(
                f"Titles ({len(titles)}) and images ({len(images)}) must have matching lengths."
            )

        return [
            self.predict(title=t, image=img, top_k=top_k)
            for t, img in zip(titles, images)
        ]


if __name__ == "__main__":
    from config import default_config

    if default_config.CHECKPOINT_PATH.exists():
        predictor = ProductPredictor(default_config.CHECKPOINT_PATH)
        print("ProductPredictor initialized successfully.\n")

        # Test single-item prediction
        sample_img_dir = default_config.DATA_DIR / "images"
        sample_images = list(sample_img_dir.glob("*.jpg")) if sample_img_dir.exists() else []

        if sample_images:
            test_img = sample_images[0]
            test_title = "Wireless Bluetooth Noise-Cancelling Headphones with Microphone"
            print(f"Testing Inference on sample:")
            print(f"  Title: '{test_title}'")
            print(f"  Image: {test_img.name}")

            res = predictor.predict(title=test_title, image=test_img, top_k=3)
            print(f"\n[Result]")
            print(f"  Predicted Category: {res['predicted_category']}")
            print(f"  Confidence: {res['confidence'] * 100:.2f}%")
            print(f"  Top 3 Ranking:")
            for rank, item in enumerate(res["top_k"], 1):
                print(f"    {rank}. {item['category']}: {item['probability'] * 100:.2f}%")
        else:
            print("No sample images found in data/images. Run generate_sample_data.py first.")
    else:
        print(f"No checkpoint found at: {default_config.CHECKPOINT_PATH}")

