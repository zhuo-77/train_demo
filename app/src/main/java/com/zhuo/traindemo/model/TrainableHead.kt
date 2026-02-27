package com.zhuo.traindemo.model

import com.zhuo.traindemo.math.MathUtils
import java.util.Random
import kotlin.math.sqrt

class TrainableHead(val inputDim: Int, val numClasses: Int) {
    // Weights: [inputDim, numClasses] (flattened)
    var weights = FloatArray(inputDim * numClasses)
    // Bias: [numClasses]
    var bias = FloatArray(numClasses)

    // Adam optimizer state
    private var mWeights = FloatArray(weights.size)
    private var vWeights = FloatArray(weights.size)
    private var mBias = FloatArray(bias.size)
    private var vBias = FloatArray(bias.size)
    var t = 0

    // Hyperparameters
    private val beta1 = 0.9f
    private val beta2 = 0.999f
    private val epsilon = 1e-8f
    private val dropoutRate = 0.5f // 50% dropout
    private val weightDecay = 0.05f // L2 regularization for AdamW
    private val maxGradNorm = 2.0f // Gradient clipping threshold
    private val random = Random()

    init {
        // Xavier initialization
        val limit = sqrt(6.0f / (inputDim + numClasses))
        for (i in weights.indices) {
            weights[i] = (random.nextFloat() * 2 * limit - limit)
        }
        // Bias initialized to 0
    }

    /**
     * Forward pass with Dropout support.
     * Input: [Batch, Channels, H, W] flattened. We perform GAP first -> [Batch, Channels].
     * Then Linear -> [Batch, NumClasses].
     * Returns: Logits [Batch, NumClasses] (flattened)
     */
    fun forward(input: FloatArray, batchSize: Int, channels: Int, height: Int, width: Int, training: Boolean = false): FloatArray {
        var gapOutput = globalAveragePool(input, batchSize, channels, height, width)

        // Apply Dropout if training
        if (training) {
            // We apply dropout to the features after GAP (before Linear)
            // Mask size: [Batch, Channels]
            for (i in gapOutput.indices) {
                if (random.nextFloat() < dropoutRate) {
                    gapOutput[i] = 0f
                } else {
                    // Scale inverted dropout
                    gapOutput[i] = gapOutput[i] / (1.0f - dropoutRate)
                }
            }
        }

        val output = FloatArray(batchSize * numClasses)
        for (b in 0 until batchSize) {
            for (c in 0 until numClasses) {
                var sum = bias[c]
                for (i in 0 until channels) {
                    sum += gapOutput[b * channels + i] * weights[i * numClasses + c]
                }
                output[b * numClasses + c] = sum
            }
        }
        return output
    }

    // Helper to store dropout mask for backward pass?
    // Since we do re-forward inside trainStep, we can implement dropout inside trainStep logic specifically
    // to keep the mask consistent.

    /**
     * Backward pass and Update.
     * Input: Original input features [Batch, C, H, W]
     * Logits: Output of forward pass (before softmax) [Batch, NumClasses]
     * Targets: Class indices [Batch]
     */
    fun trainStep(input: FloatArray, batchSize: Int, channels: Int, height: Int, width: Int, targets: IntArray, learningRate: Float, labelSmoothing: Float = 0.1f): Float {
        // 1. GAP
        var gapOutput = globalAveragePool(input, batchSize, channels, height, width)

        // 2. Apply Dropout and store mask
        val dropoutMask = FloatArray(gapOutput.size)
        for (i in gapOutput.indices) {
            if (random.nextFloat() < dropoutRate) {
                dropoutMask[i] = 0f
                gapOutput[i] = 0f
            } else {
                dropoutMask[i] = 1.0f / (1.0f - dropoutRate)
                gapOutput[i] = gapOutput[i] * dropoutMask[i]
            }
        }

        // 3. Linear Forward
        val logits = FloatArray(batchSize * numClasses)
        for (b in 0 until batchSize) {
            for (c in 0 until numClasses) {
                var sum = bias[c]
                for (i in 0 until channels) {
                    sum += gapOutput[b * channels + i] * weights[i * numClasses + c]
                }
                logits[b * numClasses + c] = sum
            }
        }

        // 4. Softmax & Loss (with Label Smoothing)
        val probs = FloatArray(logits.size)
        var totalLoss = 0f

        for (b in 0 until batchSize) {
            val offset = b * numClasses
            // Softmax for this sample
            var maxVal = Float.NEGATIVE_INFINITY
            for (i in 0 until numClasses) maxVal = maxOf(maxVal, logits[offset + i])

            var sum = 0f
            for (i in 0 until numClasses) {
                probs[offset + i] = kotlin.math.exp(logits[offset + i] - maxVal)
                sum += probs[offset + i]
            }
            for (i in 0 until numClasses) probs[offset + i] /= sum

            // Cross Entropy Loss with Label Smoothing
            // Target distribution: (1 - epsilon) * one_hot + epsilon / numClasses
            val targetIdx = targets[b]
            for (i in 0 until numClasses) {
                var targetProb = labelSmoothing / numClasses
                if (i == targetIdx) {
                    targetProb += (1.0f - labelSmoothing)
                }
                totalLoss -= targetProb * kotlin.math.ln(maxOf(probs[offset + i], 1e-7f))
            }
        }

        // 5. Backward
        // dL/dLogits = probs - target_distribution
        val dLogits = FloatArray(logits.size)
        for (b in 0 until batchSize) {
            val targetIdx = targets[b]
            for (c in 0 until numClasses) {
                var targetProb = labelSmoothing / numClasses
                if (c == targetIdx) {
                    targetProb += (1.0f - labelSmoothing)
                }

                dLogits[b * numClasses + c] = probs[b * numClasses + c] - targetProb
                dLogits[b * numClasses + c] /= batchSize.toFloat() // Average gradients
            }
        }

        // dL/dWeights = GAP(input)^T * dL/dLogits
        // GAP(input) here is the dropped-out version!
        val dWeights = FloatArray(weights.size)
        // gapOutput is already dropped-out

        for (i in 0 until channels) {
            for (j in 0 until numClasses) {
                var sum = 0f
                for (b in 0 until batchSize) {
                    sum += gapOutput[b * channels + i] * dLogits[b * numClasses + j]
                }
                dWeights[i * numClasses + j] = sum
            }
        }

        // dL/dBias = sum(dL/dLogits) across batch
        val dBias = FloatArray(numClasses)
        for (j in 0 until numClasses) {
            var sum = 0f
            for (b in 0 until batchSize) {
                sum += dLogits[b * numClasses + j]
            }
            dBias[j] = sum
        }

        // Gradient Clipping (L2 Norm)
        var globalNormSq = 0f
        for(g in dWeights) globalNormSq += g*g
        for(g in dBias) globalNormSq += g*g
        val globalNorm = kotlin.math.sqrt(globalNormSq)

        // If global norm is greater than threshold, clip gradients
        if (globalNorm > maxGradNorm) {
            val scale = maxGradNorm / (globalNorm + 1e-6f) // Add epsilon to avoid div by zero
            for(i in dWeights.indices) dWeights[i] *= scale
            for(i in dBias.indices) dBias[i] *= scale
        }

        // 6. Update (AdamW)
        t++
        // AdamW: Decay weights before Adam update, but only if applyDecay is true
        adamWUpdate(weights, dWeights, mWeights, vWeights, learningRate, applyDecay = true)
        adamWUpdate(bias, dBias, mBias, vBias, learningRate, applyDecay = false) // Usually no weight decay on bias

        return totalLoss / batchSize
    }

    private fun globalAveragePool(input: FloatArray, batchSize: Int, channels: Int, height: Int, width: Int): FloatArray {
        val output = FloatArray(batchSize * channels)
        val spatialSize = (height * width).toFloat()

        for (b in 0 until batchSize) {
            for (c in 0 until channels) {
                var sum = 0f
                for (h in 0 until height) {
                    for (w in 0 until width) {
                        // Input layout assumed: NCHW flattened -> [b][c][h][w]
                        // Index: b*(C*H*W) + c*(H*W) + h*W + w
                        val idx = b * (channels * height * width) + c * (height * width) + h * width + w
                        sum += input[idx]
                    }
                }
                output[b * channels + c] = sum / spatialSize
            }
        }
        return output
    }

    private fun adamWUpdate(params: FloatArray, grads: FloatArray, m: FloatArray, v: FloatArray, lr: Float, applyDecay: Boolean) {
        val beta1Pow = Math.pow(beta1.toDouble(), t.toDouble()).toFloat()
        val beta2Pow = Math.pow(beta2.toDouble(), t.toDouble()).toFloat()

        // Typical AdamW applies decay like: theta = theta - lr * (grad + decay * theta)
        // Decoupled: theta = theta - lr * decay * theta - lr * adam_step

        for (i in params.indices) {
            // Apply weight decay
            if (applyDecay) {
                params[i] = params[i] * (1.0f - lr * weightDecay)
            }

            // Adam Update
            m[i] = beta1 * m[i] + (1 - beta1) * grads[i]
            v[i] = beta2 * v[i] + (1 - beta2) * grads[i] * grads[i]

            val mHat = m[i] / (1 - beta1Pow)
            val vHat = v[i] / (1 - beta2Pow)

            params[i] -= lr * mHat / (kotlin.math.sqrt(vHat) + epsilon)
        }
    }
}
