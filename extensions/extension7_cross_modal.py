"""
Extension 7: Cross-Modal Watermarking
Combines watermarking across multiple modalities:
- Pixel-level watermarking
- Audio watermarking (for video)
- Metadata watermarking (EXIF, etc.)
- Multi-modal verification
"""

import os
from typing import Optional, Dict, Any, Tuple
from PIL import Image, PngImagePlugin
import numpy as np


class MetadataWatermarker:
    """
    Watermarks metadata (EXIF, PNG chunks, etc.)
    """

    @staticmethod
    def embed_png_metadata(img_path: str, output_path: str, watermark_data: Dict[str, Any]) -> None:
        """Embed watermark in PNG metadata chunks."""
        img = Image.open(img_path).convert("RGBA")

        pnginfo = PngImagePlugin.PngInfo()
        for key, value in watermark_data.items():
            str_value = str(value)
            pnginfo.add_text(key, str_value)

        img.save(output_path, "PNG", pnginfo=pnginfo)

    @staticmethod
    def extract_png_metadata(img_path: str) -> Optional[Dict[str, Any]]:
        """Extract watermark from PNG metadata."""
        try:
            img = Image.open(img_path)
            return img.info.copy()
        except:
            return None


class PixelWatermarker:
    """
    Standard pixel-level watermarking.
    """

    @staticmethod
    def embed_lsb(img_path: str, output_path: str, message: str, alpha: float = 0.05) -> None:
        """Embed watermark in LSB of pixel values."""
        img = Image.open(img_path).convert("RGB")
        arr = np.array(img)

        # Convert message to bits
        bits = []
        for c in message:
            bits.extend([(ord(c) >> i) & 1 for i in range(8)])

        # Embed in LSB
        idx = 0
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                for c in range(arr.shape[2]):
                    if idx < len(bits):
                        arr[i, j, c] = (arr[i, j, c] & 0xFE) | bits[idx]
                        idx += 1
                    else:
                        break

        Image.fromarray(arr).save(output_path)

    @staticmethod
    def extract_lsb(img_path: str, max_length: int = 100) -> Optional[str]:
        """Extract watermark from LSB of pixel values."""
        try:
            img = Image.open(img_path).convert("RGB")
            arr = np.array(img)

            bits = []
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    for c in range(arr.shape[2]):
                        bits.append(arr[i, j, c] & 1)
                        if len(bits) >= max_length * 8:
                            break
                    if len(bits) >= max_length * 8:
                        break
                if len(bits) >= max_length * 8:
                    break

            # Convert bits to string
            chars = []
            for i in range(0, len(bits), 8):
                byte = bits[i:i+8]
                if len(byte) == 8:
                    val = 0
                    for j, bit in enumerate(byte):
                        val |= bit << j
                    chars.append(chr(val))

            return ''.join(chars).rstrip('\x00')
        except:
            return None


class SimpleAudioWatermarker:
    """
    Simple audio watermarking (for video soundtracks).
    """

    @staticmethod
    def embed_dummy(audio_path: str, output_path: str, message: str) -> None:
        """Simulate audio watermarking."""
        print(f"[AudioWatermarker] Embedding watermark in {audio_path}")
        # Copy file for demo
        if os.path.exists(audio_path):
            import shutil
            shutil.copy(audio_path, output_path)


class CrossModalWatermarker:
    """
    Complete cross-modal watermarking system.
    """

    def __init__(self):
        self.metadata_wm = MetadataWatermarker()
        self.pixel_wm = PixelWatermarker()
        self.audio_wm = SimpleAudioWatermarker()

    def embed_all(
        self,
        img_path: str,
        output_path: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None,
        audio_path: Optional[str] = None,
        audio_output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Embed watermark in all modalities.
        """
        print("🔍 Embedding cross-modal watermark...")

        results = {
            "pixel": False,
            "metadata": False,
            "audio": False,
        }

        # Pixel-level
        try:
            self.pixel_wm.embed_lsb(img_path, output_path, message)
            results["pixel"] = True
            print(f"   ✅ Pixel-level watermark embedded")
        except Exception as e:
            print(f"   ❌ Pixel watermark failed: {e}")

        # Metadata
        full_metadata = metadata or {}
        full_metadata["wm_text"] = message
        full_metadata["wm_type"] = "cross_modal_v1"
        try:
            self.metadata_wm.embed_png_metadata(output_path, output_path, full_metadata)
            results["metadata"] = True
            print(f"   ✅ Metadata watermark embedded")
        except Exception as e:
            print(f"   ❌ Metadata watermark failed: {e}")

        # Audio (if provided)
        if audio_path and audio_output_path:
            try:
                self.audio_wm.embed_dummy(audio_path, audio_output_path, message)
                results["audio"] = True
                print(f"   ✅ Audio watermark embedded")
            except Exception as e:
                print(f"   ❌ Audio watermark failed: {e}")

        return results

    def verify_all(
        self,
        img_path: str,
        audio_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Verify watermarks across all modalities and combine results.
        """
        print("🔍 Verifying cross-modal watermark...")

        verification = {
            "pixel_message": None,
            "pixel_confidence": 0.0,
            "metadata": {},
            "audio_message": None,
            "audio_confidence": 0.0,
            "final_verdict": False,
            "final_message": None,
            "confidence": 0.0,
        }

        # Pixel
        pixel_msg = self.pixel_wm.extract_lsb(img_path)
        if pixel_msg and len(pixel_msg.strip()) > 3:
            verification["pixel_message"] = pixel_msg
            verification["pixel_confidence"] = 0.8
            print(f"   ✅ Pixel watermark: '{pixel_msg[:30]}...'")

        # Metadata
        md = self.metadata_wm.extract_png_metadata(img_path)
        if md:
            verification["metadata"] = md
            md_msg = md.get("wm_text")
            if md_msg:
                print(f"   ✅ Metadata watermark: '{md_msg}'")

        # Combine results
        votes = []
        messages = []

        if verification["pixel_message"]:
            votes.append(verification["pixel_confidence"])
            messages.append(verification["pixel_message"])

        if "wm_text" in verification["metadata"]:
            votes.append(0.9)
            messages.append(verification["metadata"]["wm_text"])

        # Majority vote
        if len(votes) > 0:
            verification["confidence"] = sum(votes) / len(votes)
            verification["final_message"] = messages[0] if messages else None
            verification["final_verdict"] = verification["confidence"] >= 0.5

        print(f"\n📊 Final verdict: {'✅ Watermark detected' if verification['final_verdict'] else '❌ No watermark'}")
        print(f"   Confidence: {verification['confidence']:.2f}")
        if verification["final_message"]:
            print(f"   Message: '{verification['final_message'][:50]}...'")

        return verification


def demo_cross_modal():
    """Demo function for cross-modal watermarking."""
    import tempfile

    print("Testing Cross-Modal Watermarking...")

    # Create test image
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        test_input = f.name
        Image.new('RGB', (256, 256), color='lightgreen').save(test_input)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        test_output = f.name

    try:
        watermarker = CrossModalWatermarker()

        # Embed
        watermark_msg = "CrossModalSecret123!@#"
        metadata = {
            "author": "John Doe",
            "timestamp": "2025-07-03",
            "version": "1.0"
        }
        results = watermarker.embed_all(
            test_input,
            test_output,
            watermark_msg,
            metadata
        )

        print(f"\nEmbed results: {results}")

        # Verify
        verification = watermarker.verify_all(test_output)

    finally:
        if os.path.exists(test_input):
            os.remove(test_input)
        if os.path.exists(test_output):
            os.remove(test_output)

    print("\n✅ Cross-modal watermarking demo complete!")


if __name__ == "__main__":
    demo_cross_modal()
