package com.zhuo.traindemo.model

import android.content.Context
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.File

class ModelManager(private val context: Context) {
    private val modelFile = File(context.filesDir, "trained_head.bin")
    private val classifierFile = File(context.filesDir, "trained_classifier.bin")

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

                // Optimizer Step
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

    /**
     * Save a TrainableClassifier (ConvBNSiLU + Head) and class labels.
     */
    fun saveClassifier(classifier: TrainableClassifier, classLabels: List<String>) {
        DataOutputStream(classifierFile.outputStream()).use { dos ->
            // Save Class Labels
            dos.writeInt(classLabels.size)
            for (label in classLabels) {
                dos.writeUTF(label)
            }

            // Save ConvBNSiLU parameters
            val conv = classifier.convBlock
            dos.writeInt(conv.inChannels)
            dos.writeInt(conv.outChannels)

            // Conv weights
            dos.writeInt(conv.convWeight.size)
            for (w in conv.convWeight) dos.writeFloat(w)

            // BN gamma, beta, running_mean, running_var
            for (arr in arrayOf(conv.bnGamma, conv.bnBeta, conv.bnRunningMean, conv.bnRunningVar)) {
                dos.writeInt(arr.size)
                for (v in arr) dos.writeFloat(v)
            }
            dos.writeInt(conv.t)

            // Save Head parameters
            val head = classifier.head
            dos.writeInt(head.inputDim)
            dos.writeInt(head.numClasses)

            dos.writeInt(head.weights.size)
            for (w in head.weights) dos.writeFloat(w)

            dos.writeInt(head.bias.size)
            for (b in head.bias) dos.writeFloat(b)

            dos.writeInt(head.t)

            // Save backboneLrRatio
            dos.writeFloat(classifier.backboneLrRatio)
        }
    }

    /**
     * Load a TrainableClassifier (ConvBNSiLU + Head) and class labels.
     */
    fun loadClassifier(): Pair<TrainableClassifier, MutableList<String>>? {
        if (!classifierFile.exists()) return null

        try {
            DataInputStream(classifierFile.inputStream()).use { dis ->
                // Load Class Labels
                val numLabels = dis.readInt()
                val labels = mutableListOf<String>()
                for (i in 0 until numLabels) {
                    labels.add(dis.readUTF())
                }

                // Load ConvBNSiLU parameters
                val inChannels = dis.readInt()
                val outChannels = dis.readInt()

                val convWeightSize = dis.readInt()
                if (convWeightSize != inChannels * outChannels) return null
                val convWeight = FloatArray(convWeightSize) { dis.readFloat() }

                val bnGammaSize = dis.readInt()
                val bnGamma = FloatArray(bnGammaSize) { dis.readFloat() }
                val bnBetaSize = dis.readInt()
                val bnBeta = FloatArray(bnBetaSize) { dis.readFloat() }
                val bnMeanSize = dis.readInt()
                val bnMean = FloatArray(bnMeanSize) { dis.readFloat() }
                val bnVarSize = dis.readInt()
                val bnVar = FloatArray(bnVarSize) { dis.readFloat() }
                val convT = dis.readInt()

                // Load Head parameters
                val inputDim = dis.readInt()
                val numClasses = dis.readInt()
                if (numClasses != numLabels) return null

                val headWeightSize = dis.readInt()
                if (headWeightSize != inputDim * numClasses) return null
                val headWeights = FloatArray(headWeightSize) { dis.readFloat() }

                val headBiasSize = dis.readInt()
                if (headBiasSize != numClasses) return null
                val headBias = FloatArray(headBiasSize) { dis.readFloat() }

                val headT = dis.readInt()

                // Build classifier
                val classifier = TrainableClassifier(inChannels, outChannels, numClasses)
                classifier.convBlock.convWeight = convWeight
                classifier.convBlock.bnGamma = bnGamma
                classifier.convBlock.bnBeta = bnBeta
                classifier.convBlock.bnRunningMean = bnMean
                classifier.convBlock.bnRunningVar = bnVar
                classifier.convBlock.t = convT
                classifier.head.weights = headWeights
                classifier.head.bias = headBias
                classifier.head.t = headT

                try {
                    classifier.backboneLrRatio = dis.readFloat()
                } catch (e: Exception) {
                    // Use default
                }

                return Pair(classifier, labels)
            }
        } catch (e: Exception) {
            e.printStackTrace()
            return null
        }
    }

    /**
     * Load pretrained Conv+BN weights from assets (exported by export_yolov8n_cls.py).
     * Returns a TrainableConvBNSiLU with pretrained weights.
     */
    fun loadPretrainedConvWeights(assetName: String): TrainableConvBNSiLU? {
        try {
            DataInputStream(context.assets.open(assetName)).use { dis ->
                val inChannels = readLittleEndianInt(dis)
                val outChannels = readLittleEndianInt(dis)

                val conv = TrainableConvBNSiLU(inChannels, outChannels)

                // Conv weights
                for (i in conv.convWeight.indices) {
                    conv.convWeight[i] = readLittleEndianFloat(dis)
                }
                // BN gamma
                for (i in conv.bnGamma.indices) {
                    conv.bnGamma[i] = readLittleEndianFloat(dis)
                }
                // BN beta
                for (i in conv.bnBeta.indices) {
                    conv.bnBeta[i] = readLittleEndianFloat(dis)
                }
                // BN running_mean
                for (i in conv.bnRunningMean.indices) {
                    conv.bnRunningMean[i] = readLittleEndianFloat(dis)
                }
                // BN running_var
                for (i in conv.bnRunningVar.indices) {
                    conv.bnRunningVar[i] = readLittleEndianFloat(dis)
                }

                return conv
            }
        } catch (e: Exception) {
            e.printStackTrace()
            return null
        }
    }

    // Python struct.pack uses little-endian by default; Java DataInputStream uses big-endian
    private fun readLittleEndianInt(dis: DataInputStream): Int {
        val b = ByteArray(4)
        dis.readFully(b)
        return (b[0].toInt() and 0xFF) or
                ((b[1].toInt() and 0xFF) shl 8) or
                ((b[2].toInt() and 0xFF) shl 16) or
                ((b[3].toInt() and 0xFF) shl 24)
    }

    private fun readLittleEndianFloat(dis: DataInputStream): Float {
        return Float.fromBits(readLittleEndianInt(dis))
    }
}
