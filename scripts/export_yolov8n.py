import torch
import torch.nn as nn
import onnx
import os

def export_model():
    from ultralytics import YOLO

    # Load YOLOv8n pretrained model
    print("Loading YOLOv8n model...")
    model = YOLO("yolov8n.pt")

    # Extract backbone (layers 0-9, up to SPPF)
    # Output: [batch, 256, 7, 7] for 224x224 input
    backbone_layers = model.model.model[:10]

    class YOLOv8nBackbone(nn.Module):
        def __init__(self, layers):
            super().__init__()
            self.layers = nn.Sequential(*layers)

        def forward(self, x):
            return self.layers(x)

    backbone = YOLOv8nBackbone(backbone_layers)
    backbone.eval()

    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224)

    # Verify output shape
    with torch.no_grad():
        output = backbone(dummy_input)
        print(f"Output shape: {output.shape}")  # [1, 256, 7, 7]

    # Define output path
    output_path = "app/src/main/assets/yolov8n_feature_extractor.onnx"

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Exporting model to {output_path}...")
    torch.onnx.export(
        backbone,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print("Model exported successfully.")

    # Downgrade IR version for compatibility with ONNX Runtime 1.17.0
    print("Downgrading IR version to 9...")
    onnx_model = onnx.load(output_path)
    onnx_model.ir_version = 9

    # Save as a single file without external data
    onnx.save(onnx_model, output_path)

    # If an external data file was created, reload and re-save without it
    data_file = output_path + ".data"
    if os.path.exists(data_file):
        print("Re-saving model without external data...")
        onnx_model = onnx.load(output_path, load_external_data=True)
        onnx_model.ir_version = 9
        from onnx.external_data_helper import convert_model_to_external_data
        # Clear external data references and internalize all tensors
        for tensor in onnx_model.graph.initializer:
            tensor.ClearField('data_location')
            tensor.ClearField('external_data')
        onnx.save(onnx_model, output_path, save_as_external_data=False)
        if os.path.exists(data_file):
            os.remove(data_file)
        print("External data file removed.")

    # Verify the model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX model verified. IR version: {onnx_model.ir_version}")
    print(f"Model size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    export_model()
