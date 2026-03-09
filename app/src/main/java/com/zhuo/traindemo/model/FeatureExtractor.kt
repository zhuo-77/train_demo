package com.zhuo.traindemo.model

import android.graphics.Bitmap
import android.content.Context
import java.io.InputStream
import java.nio.FloatBuffer
import java.util.Collections
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.OnnxTensor

class FeatureExtractor(context: Context, modelAsset: String = "feature_extractor.onnx") {

    private val env: OrtEnvironment
    private val session: OrtSession

    init {
        env = OrtEnvironment.getEnvironment()
        // Load model from assets
        val modelBytes = context.assets.open(modelAsset).readBytes()
        session = env.createSession(modelBytes)
    }

    fun extract(bitmap: Bitmap): FloatArray {
        // Resize to 224x224
        val resized = Bitmap.createScaledBitmap(bitmap, 224, 224, true)
        val intValues = IntArray(224 * 224)
        resized.getPixels(intValues, 0, 224, 0, 0, 224, 224)

        // Prepare input tensor (NCHW format, normalized)
        val floatBuffer = FloatBuffer.allocate(3 * 224 * 224)
        for (i in 0 until 224 * 224) {
            val pixel = intValues[i]
            val r = ((pixel shr 16) and 0xFF) / 255.0f
            val g = ((pixel shr 8) and 0xFF) / 255.0f
            val b = (pixel and 0xFF) / 255.0f

            // Normalize (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            floatBuffer.put(i, (r - 0.485f) / 0.229f)
            floatBuffer.put(224 * 224 + i, (g - 0.456f) / 0.224f)
            floatBuffer.put(2 * 224 * 224 + i, (b - 0.406f) / 0.225f)
        }
        floatBuffer.rewind()

        // Create Tensor
        val inputName = session.inputNames.iterator().next()
        val inputTensor = OnnxTensor.createTensor(env, floatBuffer, longArrayOf(1, 3, 224, 224))

        // Run inference
        val results = session.run(Collections.singletonMap(inputName, inputTensor))

        // Get output
        val outputTensor = results[0] as OnnxTensor
        val outputBuffer = outputTensor.floatBuffer
        val outputArray = FloatArray(outputBuffer.remaining())
        outputBuffer.get(outputArray)

        // Cleanup
        inputTensor.close()
        results.close()

        return outputArray
    }
}
