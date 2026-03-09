package com.zhuo.traindemo.model

/**
 * Type of backbone used for feature extraction.
 */
enum class BackboneType(
    val displayName: String,
    val onnxAsset: String,
    val backboneChannels: Int,
    val featureHeight: Int,
    val featureWidth: Int,
    val convOutChannels: Int,
    val pretrainedWeightsAsset: String?
) {
    /** ConvNeXtV2-atto backbone: output [batch, 320, 7, 7] */
    CONVNEXT(
        displayName = "ConvNeXt",
        onnxAsset = "feature_extractor.onnx",
        backboneChannels = 320,
        featureHeight = 7,
        featureWidth = 7,
        convOutChannels = 1280,
        pretrainedWeightsAsset = null
    ),

    /** YOLOv8n-cls backbone: output [batch, 256, 7, 7] */
    YOLOV8N_CLS(
        displayName = "YOLOv8n-cls",
        onnxAsset = "yolov8n_cls_backbone.onnx",
        backboneChannels = 256,
        featureHeight = 7,
        featureWidth = 7,
        convOutChannels = 1280,
        pretrainedWeightsAsset = "yolov8n_cls_head_weights.bin"
    )
}
