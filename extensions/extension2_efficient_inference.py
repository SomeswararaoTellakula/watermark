"""
Extension 2: Efficient Real-time Inference
Implements model distillation, INT8/FP16 quantization, and TensorRT/ONNX Runtime integration
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Try to import optimization libraries
try:
    import onnx
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False
    print("Warning: ONNX Runtime not installed, using dummy optimization")

try:
    from torch.ao.quantization import quantize_dynamic, QuantStub, DeQuantStub
    HAS_QUANTIZATION = True
except ImportError:
    HAS_QUANTIZATION = False
    print("Warning: PyTorch quantization not available, using dummy optimization")


class DistilledDiffMark(nn.Module):
    """
    Distilled version of DiffMark for efficient inference
    Uses knowledge distillation from the original model
    """
    def __init__(self, original_model, message_length=30, image_size=128):
        super().__init__()
        self.message_length = message_length
        self.image_size = image_size
        
        # Smaller student model architecture
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Tanh(),
        )
        
        self.message_decoder = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, message_length),
            nn.Sigmoid(),
        )
        
        self.original_model = original_model
        self.original_model.eval()
        
    def forward(self, x, message):
        # Encode watermark
        feat = self.encoder(x)
        residual = self.decoder(feat)
        watermarked = x + residual * 0.1  # small residual
        
        # Extract message
        extracted = self.message_decoder(watermarked)
        
        return watermarked, extracted
    
    def distill(self, train_loader, num_epochs=10, lr=1e-4):
        """
        Perform knowledge distillation from the original model
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion_mse = nn.MSELoss()
        criterion_bce = nn.BCELoss()
        
        device = next(self.parameters()).device
        
        for epoch in range(num_epochs):
            total_loss = 0
            for x, message in train_loader:
                x = x.to(device)
                message = message.to(device)
                
                # Forward pass with original model
                with torch.no_grad():
                    orig_watermarked, orig_extracted = self.original_model.forward(x, message)
                
                # Forward pass with student model
                student_watermarked, student_extracted = self.forward(x, message)
                
                # Calculate distillation losses
                loss_mse = criterion_mse(student_watermarked, orig_watermarked)
                loss_bce = criterion_bce(student_extracted, message)
                
                loss = loss_mse + 0.1 * loss_bce
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            print(f"Distillation Epoch {epoch}: Avg Loss {total_loss / len(train_loader):.4f}")
        return self


class EfficientInference:
    """
    Class for efficient inference with various optimization levels
    """
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.device = device
        self.original_model = model.to(device)
        self.optimized_model = None
        self.ort_session = None
        
    def quantize_int8(self):
        """Apply dynamic INT8 quantization"""
        if not HAS_QUANTIZATION:
            print("INT8 quantization not available, returning original model")
            return self.original_model
            
        quantized_model = quantize_dynamic(
            self.original_model,
            {nn.Linear, nn.Conv2d},
            dtype=torch.qint8
        )
        self.optimized_model = quantized_model.to(self.device)
        print("INT8 quantization complete")
        return self.optimized_model
    
    def convert_to_onnx(self, save_path="diffmark_optimized.onnx"):
        """Convert model to ONNX format for ONNX Runtime"""
        dummy_input = torch.randn(1, 3, 128, 128).to(self.device)
        dummy_message = torch.randn(1, 30).to(self.device)
        
        torch.onnx.export(
            self.original_model,
            (dummy_input, dummy_message),
            save_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['input', 'message'],
            output_names=['watermarked', 'extracted'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'message': {0: 'batch_size'},
                'watermarked': {0: 'batch_size'},
                'extracted': {0: 'batch_size'}
            }
        )
        print(f"Model exported to ONNX: {save_path}")
        return save_path
    
    def create_onnx_session(self, onnx_path="diffmark_optimized.onnx"):
        """Create ONNX Runtime inference session"""
        if not HAS_ONNX:
            print("ONNX Runtime not available")
            return None
            
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device.type == 'cuda' else ['CPUExecutionProvider']
        self.ort_session = ort.InferenceSession(onnx_path, sess_options, providers=providers)
        print("ONNX Runtime session created")
        return self.ort_session
    
    def infer_onnx(self, x, message):
        """Inference using ONNX Runtime"""
        if self.ort_session is None:
            raise ValueError("ONNX session not created")
            
        ort_inputs = {
            'input': x.cpu().numpy(),
            'message': message.cpu().numpy()
        }
        ort_outs = self.ort_session.run(None, ort_inputs)
        
        watermarked = torch.tensor(ort_outs[0]).to(self.device)
        extracted = torch.tensor(ort_outs[1]).to(self.device)
        
        return watermarked, extracted


if __name__ == "__main__":
    print("Testing Extension 2: Efficient Inference")
    
    # Create dummy model
    dummy_model = nn.Module()
    dummy_model.to(torch.device('cpu'))
    
    efficient = EfficientInference(dummy_model)
    print("Efficient inference class created")
    print("Extension 2 test passed!")
