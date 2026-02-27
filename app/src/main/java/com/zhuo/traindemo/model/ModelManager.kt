package com.zhuo.traindemo.model

import android.content.Context
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.File

class ModelManager(private val context: Context) {
    private val modelFile = File(context.filesDir, "trained_head.bin")

    fun saveModel(head: TrainableHead, classLabels: List<String>) {
        DataOutputStream(modelFile.outputStream()).use { dos ->
            // Save Class Labels
            dos.writeInt(classLabels.size)
            for (label in classLabels) {
                dos.writeUTF(label)
            }
            // Save Head Parameters
            dos.writeInt(head.inputDim)
            dos.writeInt(head.numClasses)

            // Weights
            dos.writeInt(head.weights.size)
            for (w in head.weights) {
                dos.writeFloat(w)
            }
            // Bias
            dos.writeInt(head.bias.size)
            for (b in head.bias) {
                dos.writeFloat(b)
            }
            // Optimizer Step
            dos.writeInt(head.t)
        }
    }

    fun loadModel(): Pair<TrainableHead, MutableList<String>>? {
        if (!modelFile.exists()) return null

        try {
            DataInputStream(modelFile.inputStream()).use { dis ->
                // Load Class Labels
                val numLabels = dis.readInt()
                val labels = mutableListOf<String>()
                for (i in 0 until numLabels) {
                    labels.add(dis.readUTF())
                }

                // Load Head Parameters
                val inputDim = dis.readInt()
                val numClasses = dis.readInt()

                if (numClasses != numLabels) {
                    // Mismatch, maybe corrupted or logic error
                    return null
                }

                val head = TrainableHead(inputDim, numClasses)

                // Weights
                val weightSize = dis.readInt()
                if (weightSize != head.weights.size) return null
                for (i in 0 until weightSize) {
                    head.weights[i] = dis.readFloat()
                }

                // Bias
                val biasSize = dis.readInt()
                if (biasSize != head.bias.size) return null
                for (i in 0 until biasSize) {
                    head.bias[i] = dis.readFloat()
                }

                // Optimizer Step (try-catch for backward compatibility if file exists but shorter, though in this new project it shouldn't matter)
                try {
                    head.t = dis.readInt()
                } catch (e: Exception) {
                    // Ignore if missing, assume 0
                }

                return Pair(head, labels)
            }
        } catch (e: Exception) {
            e.printStackTrace()
            return null
        }
    }
}
