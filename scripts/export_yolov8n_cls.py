import torch
import torch.nn as nn
import onnx
import os
import struct

# Max IR version supported by ONNX Runtime 1.17.0
TARGET_IR_VERSION = 9


def save_classify_weights(classify_module, output_path):
    """Save the Classify module's Conv+BN+Linear weights as a binary file.

    Format:
        [int32] in_channels (256 for yolov8n-cls)
        [int32] out_channels (1280)
        --- Conv2d weights (1x1, no bias) ---
        [float32 * out_channels * in_channels] conv weight (row-major: [out_ch, in_ch])
        --- BatchNorm parameters ---
        [float32 * out_channels] bn weight (gamma)
        [float32 * out_channels] bn bias (beta)
        [float32 * out_channels] bn running_mean
        [float32 * out_channels] bn running_var
        --- Linear weights ---
        [int32] linear_in_features (1280)
        [int32] linear_out_features (1000)
        [float32 * out * in] linear weight (row-major: [out_features, in_features])
        [float32 * out] linear bias
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Access submodules
    conv_block = classify_module.conv  # Conv module: conv2d + bn + act
    conv2d = conv_block.conv           # nn.Conv2d(256, 1280, 1, 1, bias=False)
    bn = conv_block.bn                 # nn.BatchNorm2d(1280)
    linear = classify_module.linear    # nn.Linear(1280, 1000)

    with open(output_path, 'wb') as f:
        in_ch = conv2d.in_channels
        out_ch = conv2d.out_channels
        f.write(struct.pack('i', in_ch))
        f.write(struct.pack('i', out_ch))

        # Conv2d weight: [out_ch, in_ch, 1, 1] -> flatten to [out_ch * in_ch]
        conv_w = conv2d.weight.data.reshape(-1).cpu().numpy()
        for v in conv_w:
            f.write(struct.pack('f', float(v)))

        # BatchNorm
        for arr in [bn.weight.data, bn.bias.data, bn.running_mean.data, bn.running_var.data]:
            for v in arr.cpu().numpy():
                f.write(struct.pack('f', float(v)))

        # Linear
        lin_in = linear.in_features
        lin_out = linear.out_features
        f.write(struct.pack('i', lin_in))
        f.write(struct.pack('i', lin_out))

        lin_w = linear.weight.data.reshape(-1).cpu().numpy()
        for v in lin_w:
            f.write(struct.pack('f', float(v)))
        for v in linear.bias.data.cpu().numpy():
            f.write(struct.pack('f', float(v)))

    print(f"Classify weights saved to {output_path}")
    print(f"  Conv2d: ({in_ch}, {out_ch}, 1, 1)")
    print(f"  BN: {out_ch} channels")
    print(f"  Linear: ({lin_in}, {lin_out})")
    print(f"  File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")


def export_model():
    from ultralytics import YOLO

    # Load YOLOv8n-cls pretrained model
    print("Loading YOLOv8n-cls model...")
    model = YOLO("yolov8n-cls.pt")

    # Print model structure for reference
    print("\nModel layers:")
    for i, layer in enumerate(model.model.model):
        print(f"  [{i}] {layer.__class__.__name__}")

    # Extract backbone (layers 0-8, before Classify head)
    # Output: [batch, 256, 7, 7] for 224x224 input
    backbone_layers = model.model.model[:9]

    class YOLOv8nClsBackbone(nn.Module):
        def __init__(self, layers):
            super().__init__()
            self.layers = nn.Sequential(*layers)

        def forward(self, x):
            return self.layers(x)

    backbone = YOLOv8nClsBackbone(backbone_layers)
    backbone.eval()

    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224)

    # Verify output shape
    with torch.no_grad():
        output = backbone(dummy_input)
        print(f"\nBackbone output shape: {output.shape}")  # Expected: [1, 256, 7, 7]

    # Define output path for backbone ONNX
    backbone_path = "app/src/main/assets/yolov8n_cls_backbone.onnx"
    os.makedirs(os.path.dirname(backbone_path), exist_ok=True)

    print(f"\nExporting backbone to {backbone_path}...")
    torch.onnx.export(
        backbone,
        dummy_input,
        backbone_path,
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
    print("Backbone exported successfully.")

    # Downgrade IR version for compatibility with ONNX Runtime 1.17.0
    print("Downgrading IR version to %d..." % TARGET_IR_VERSION)
    onnx_model = onnx.load(backbone_path)
    onnx_model.ir_version = TARGET_IR_VERSION
    onnx.save(onnx_model, backbone_path)

    # If an external data file was created, reload and re-save without it
    data_file = backbone_path + ".data"
    if os.path.exists(data_file):
        print("Re-saving model without external data...")
        onnx_model = onnx.load(backbone_path, load_external_data=True)
        onnx_model.ir_version = TARGET_IR_VERSION
        for tensor in onnx_model.graph.initializer:
            tensor.ClearField('data_location')
            tensor.ClearField('external_data')
        onnx.save(onnx_model, backbone_path, save_as_external_data=False)
        if os.path.exists(data_file):
            os.remove(data_file)
        print("External data file removed.")

    # Verify the backbone model
    onnx_model = onnx.load(backbone_path)
    onnx.checker.check_model(onnx_model)
    print(f"Backbone ONNX verified. IR version: {onnx_model.ir_version}")
    print(f"Backbone size: {os.path.getsize(backbone_path) / 1024 / 1024:.2f} MB")

    # Save Classify head weights (Conv+BN+Linear)
    classify_module = model.model.model[9]
    weights_path = "app/src/main/assets/yolov8n_cls_head_weights.bin"
    save_classify_weights(classify_module, weights_path)

    print("\nExport complete!")
    print(f"  Backbone ONNX: {backbone_path}")
    print(f"  Classify weights: {weights_path}")


if __name__ == "__main__":
    export_model()
