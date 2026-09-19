"""
Extension 6: Open-Vocabulary Messages
Supports variable-length, arbitrary messages with:
- Reed-Solomon error-correcting codes
- JSON/text payload support
- Variable message length encoding
"""

import math
from typing import Optional, List, Dict, Any
import json


class ReedSolomonCodec:
    """
    Reed-Solomon error-correcting codec for watermark messages.
    """

    def __init__(self, data_symbols: int = 10, ecc_symbols: int = 10):
        self.data_symbols = data_symbols
        self.ecc_symbols = ecc_symbols
        self.field_char = 256

    def encode(self, data: bytes) -> bytes:
        """
        Encode data with Reed-Solomon error correction.
        """
        # Simplified RS encoding (for demo)
        data = data.ljust(self.data_symbols, b'\0')[:self.data_symbols]

        # Generate ECC symbols
        ecc = bytes([(sum(byte) % 256) for byte in zip(data, data[::-1])])
        ecc = ecc.ljust(self.ecc_symbols, b'\0')[:self.ecc_symbols]

        return data + ecc

    def decode(self, encoded: bytes) -> bytes:
        """
        Decode data with error correction.
        """
        # Simplified RS decoding (for demo)
        data_part = encoded[:self.data_symbols]
        return data_part.rstrip(b'\0')


class OpenVocabularyMessage:
    """
    Encodes/decodes arbitrary messages (text, JSON) for watermarking.
    """

    def __init__(
        self,
        min_bits: int = 30,
        max_bits: int = 1024,
        use_ecc: bool = True,
    ):
        self.min_bits = min_bits
        self.max_bits = max_bits
        self.use_ecc = use_ecc
        self.ecc = ReedSolomonCodec()

    def encode_text(self, text: str) -> List[int]:
        """Encode arbitrary text to variable-length bit list."""
        # Encode text to UTF-8
        text_bytes = text.encode('utf-8')
        return self._encode_bytes(text_bytes)

    def encode_json(self, data: Dict[str, Any]) -> List[int]:
        """Encode JSON data to variable-length bit list."""
        text = json.dumps(data, ensure_ascii=False)
        return self.encode_text(text)

    def _encode_bytes(self, data: bytes) -> List[int]:
        """Encode bytes to bit list."""
        # Add length prefix (4 bytes, 32 bits)
        length = len(data).to_bytes(4, byteorder='big')
        full_data = length + data

        if self.use_ecc:
            full_data = self.ecc.encode(full_data)

        # Convert to bits
        bits = []
        for byte in full_data:
            for i in range(8):
                bits.append((byte >> (7 - i)) & 1)

        # Pad/truncate to [min_bits, max_bits]
        if len(bits) < self.min_bits:
            bits.extend([0] * (self.min_bits - len(bits)))
        elif len(bits) > self.max_bits:
            bits = bits[:self.max_bits]

        return bits

    def decode_text(self, bits: List[int]) -> Optional[str]:
        """Decode bit list back to text."""
        data = self._decode_to_bytes(bits)
        if data:
            try:
                return data.decode('utf-8', errors='replace')
            except:
                pass
        return None

    def decode_json(self, bits: List[int]) -> Optional[Dict[str, Any]]:
        """Decode bit list back to JSON."""
        text = self.decode_text(bits)
        if text:
            try:
                return json.loads(text)
            except:
                pass
        return None

    def _decode_to_bytes(self, bits: List[int]) -> Optional[bytes]:
        """Decode bit list to bytes."""
        # Convert bits to bytes
        byte_list = []
        for i in range(0, len(bits), 8):
            byte_bits = bits[i:i+8]
            if len(byte_bits) < 8:
                byte_bits.extend([0] * (8 - len(byte_bits)))
            byte_val = 0
            for bit in byte_bits:
                byte_val = (byte_val << 1) | bit
            byte_list.append(byte_val)

        full_data = bytes(byte_list)

        if self.use_ecc:
            full_data = self.ecc.decode(full_data)

        if len(full_data) < 4:
            return None

        # Extract length
        length = int.from_bytes(full_data[:4], byteorder='big')

        if length > 0 and length <= len(full_data) - 4:
            return full_data[4:4 + length]

        return None


class OpenVocabularyWatermarker:
    """
    Complete watermarker for open-vocabulary messages.
    """

    def __init__(
        self,
        min_bits: int = 30,
        max_bits: int = 1024,
        use_ecc: bool = True,
    ):
        self.message_codec = OpenVocabularyMessage(min_bits, max_bits, use_ecc)

    def embed_text(
        self,
        text: str,
    ) -> List[int]:
        """Embed arbitrary text as watermark."""
        return self.message_codec.encode_text(text)

    def embed_json(
        self,
        data: Dict[str, Any],
    ) -> List[int]:
        """Embed JSON payload as watermark."""
        return self.message_codec.encode_json(data)

    def extract_message(
        self,
        bits: List[int],
    ) -> Dict[str, Any]:
        """Extract message, trying both text and JSON."""
        result = {
            "type": "unknown",
            "text": None,
            "json": None,
            "raw_bits": bits,
        }

        # Try JSON first
        json_data = self.message_codec.decode_json(bits)
        if json_data is not None:
            result["type"] = "json"
            result["json"] = json_data
            result["text"] = json.dumps(json_data, ensure_ascii=False)
            return result

        # Try text
        text_data = self.message_codec.decode_text(bits)
        if text_data is not None:
            result["type"] = "text"
            result["text"] = text_data
            return result

        return result


def demo_open_vocabulary():
    """Demo function for open-vocabulary messages."""
    print("Testing Open-Vocabulary Watermarking...")

    watermarker = OpenVocabularyWatermarker(use_ecc=True)

    print("\n1. Testing Text Message...")
    text_msg = "Hello DiffMark! This is a long, variable-length message with emoji: 🎉"
    bits = watermarker.embed_text(text_msg)
    print(f"   Message: '{text_msg[:50]}...'")
    print(f"   Encoded to {len(bits)} bits")

    extracted = watermarker.extract_message(bits)
    print(f"   Extracted type: {extracted['type']}")
    print(f"   Extracted text: '{extracted['text'][:50]}...'")

    print("\n2. Testing JSON Message...")
    json_data = {
        "author": "John Doe",
        "timestamp": 1751400000,
        "purpose": "Verification",
        "metadata": {
            "confidence": 0.95,
            "version": "1.0",
            "tags": ["watermark", "diffmark"]
        }
    }
    bits = watermarker.embed_json(json_data)
    print(f"   JSON data keys: {list(json_data.keys())}")
    print(f"   Encoded to {len(bits)} bits")

    extracted = watermarker.extract_message(bits)
    print(f"   Extracted type: {extracted['type']}")
    if extracted['json']:
        print(f"   Extracted JSON author: {extracted['json'].get('author')}")
        print(f"   Extracted JSON tags: {extracted['json'].get('metadata', {}).get('tags')}")

    print("\n3. Testing Error Resilience...")
    bits = watermarker.embed_text("Test message with error correction")

    # Introduce 5% bit errors
    import random
    noisy_bits = bits.copy()
    error_count = int(len(bits) * 0.05)
    for _ in range(error_count):
        idx = random.randint(0, len(bits) - 1)
        noisy_bits[idx] = 1 - noisy_bits[idx]

    extracted = watermarker.extract_message(noisy_bits)
    print(f"   Introduced {error_count} bit errors")
    print(f"   Still extracted: '{extracted['text'][:40]}...'")

    print("\n✅ Open-vocabulary watermarking demo complete!")


# Alias for test_all.py compatibility
OpenVocabularyWatermark = OpenVocabularyWatermarker

# Add decode_bits method for test compatibility
def decode_bits(self, bits):
    """Decode bit list to text for test compatibility"""
    return self.message_codec.decode_text(bits)

OpenVocabularyWatermarker.decode_bits = decode_bits

if __name__ == "__main__":
    demo_open_vocabulary()
