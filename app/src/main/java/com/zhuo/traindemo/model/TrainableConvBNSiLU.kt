package com.zhuo.traindemo.model

import java.util.Random
import kotlin.math.exp
import kotlin.math.sqrt

/**
 * Hand-written Conv2d(1x1) + BatchNorm + SiLU with forward and backward pass.
 *
 * Used as the trainable "last few layers" in semi-frozen training mode.
 * In YOLOv8n-cls, this corresponds to the Conv block inside the Classify module
 * that transforms backbone features (256 channels) to 1280 channels.
 *
 * All gradient computation is hand-written without autograd.
 */
class TrainableConvBNSiLU(val inChannels: Int, val outChannels: Int) {

    // Conv2d(1x1) weights: [outChannels, inChannels] (no bias)
    var convWeight = FloatArray(outChannels * inChannels)

    // BatchNorm parameters (gamma, beta are trainable; mean, var are frozen)
    var bnGamma = FloatArray(outChannels) { 1.0f }
    var bnBeta = FloatArray(outChannels)
    var bnRunningMean = FloatArray(outChannels)
    var bnRunningVar = FloatArray(outChannels) { 1.0f }
    private val bnEps = 1e-5f

    // AdamW optimizer state for conv weights
    private var mConvW = FloatArray(convWeight.size)
    private var vConvW = FloatArray(convWeight.size)

    // AdamW optimizer state for BN gamma
    private var mBnGamma = FloatArray(outChannels)
    private var vBnGamma = FloatArray(outChannels)

    // AdamW optimizer state for BN beta
    private var mBnBeta = FloatArray(outChannels)
    private var vBnBeta = FloatArray(outChannels)

    var t = 0

    // Hyperparameters
    private val beta1 = 0.9f
    private val beta2 = 0.999f
    private val epsilon = 1e-8f
    private val weightDecay = 0.01f
    private val maxGradNorm = 2.0f

    init {
        // Kaiming initialization for conv weights
        val std = sqrt(2.0f / inChannels)
        val random = Random()
        for (i in convWeight.indices) {
            convWeight[i] = (random.nextGaussian() * std).toFloat()
        }
    }

    /**
     * Forward pass: Conv2d(1x1) -> BatchNorm(eval) -> SiLU
     *
     * @param input [batchSize, inChannels, height, width] flattened
     * @return output [batchSize, outChannels, height, width] flattened
     */
    fun forward(
        input: FloatArray, batchSize: Int, height: Int, width: Int
    ): FloatArray {
        val convOut = conv1x1Forward(input, batchSize, height, width)
        val bnOut = batchNormForward(convOut, batchSize, height, width)
        return siluForward(bnOut)
    }

    /**
     * Backward pass and parameter update.
     * Computes gradients through SiLU -> BN -> Conv2d(1x1).
     *
     * @param input Original input [batchSize, inChannels, height, width] flattened
     * @param dOutput Gradient from downstream [batchSize, outChannels, height, width] flattened
     * @param learningRate Learning rate for this parameter group
     * @return dInput Gradient w.r.t. input [batchSize, inChannels, height, width] flattened
     */
    fun backward(
        input: FloatArray, dOutput: FloatArray,
        batchSize: Int, height: Int, width: Int,
        learningRate: Float
    ): FloatArray {
        // Re-compute intermediates for backward pass
        val convOut = conv1x1Forward(input, batchSize, height, width)
        val bnOut = batchNormForward(convOut, batchSize, height, width)

        // Backward through SiLU
        val dBnOut = siluBackward(bnOut, dOutput)

        // Backward through BatchNorm (eval mode)
        val dConvOut = batchNormBackward(convOut, dBnOut, batchSize, height, width, learningRate)

        // Backward through Conv2d(1x1)
        val dInput = conv1x1Backward(input, dConvOut, batchSize, height, width, learningRate)

        return dInput
    }

    // ---- Conv2d(1x1) ----

    /**
     * Conv2d with 1x1 kernel: effectively a matrix multiply at each spatial location.
     * output[b, co, h, w] = sum_{ci}(weight[co, ci] * input[b, ci, h, w])
     */
    internal fun conv1x1Forward(
        input: FloatArray, batchSize: Int, height: Int, width: Int
    ): FloatArray {
        val output = FloatArray(batchSize * outChannels * height * width)
        for (b in 0 until batchSize) {
            for (h in 0 until height) {
                for (w in 0 until width) {
                    for (co in 0 until outChannels) {
                        var sum = 0f
                        for (ci in 0 until inChannels) {
                            val inIdx = b * (inChannels * height * width) + ci * (height * width) + h * width + w
                            sum += convWeight[co * inChannels + ci] * input[inIdx]
                        }
                        val outIdx = b * (outChannels * height * width) + co * (height * width) + h * width + w
                        output[outIdx] = sum
                    }
                }
            }
        }
        return output
    }

    /**
     * Backward through Conv2d(1x1).
     * Computes dWeight and dInput, updates conv weights.
     */
    private fun conv1x1Backward(
        input: FloatArray, dOutput: FloatArray,
        batchSize: Int, height: Int, width: Int,
        learningRate: Float
    ): FloatArray {
        val dWeight = FloatArray(convWeight.size)
        val dInput = FloatArray(input.size)

        for (b in 0 until batchSize) {
            for (h in 0 until height) {
                for (w in 0 until width) {
                    for (co in 0 until outChannels) {
                        val outIdx = b * (outChannels * height * width) + co * (height * width) + h * width + w
                        val dOut = dOutput[outIdx]

                        for (ci in 0 until inChannels) {
                            val inIdx = b * (inChannels * height * width) + ci * (height * width) + h * width + w

                            // dWeight[co, ci] += dOutput[b,co,h,w] * input[b,ci,h,w]
                            dWeight[co * inChannels + ci] += dOut * input[inIdx]

                            // dInput[b,ci,h,w] += weight[co,ci] * dOutput[b,co,h,w]
                            dInput[inIdx] += convWeight[co * inChannels + ci] * dOut
                        }
                    }
                }
            }
        }

        // Average gradients over batch
        val scale = 1.0f / batchSize
        for (i in dWeight.indices) dWeight[i] *= scale

        // Gradient clipping
        clipGradients(dWeight)

        // Update conv weights
        t++  // Shared step counter
        adamWUpdate(convWeight, dWeight, mConvW, vConvW, learningRate, applyDecay = true)

        return dInput
    }

    // ---- BatchNorm (eval mode) ----

    /**
     * BatchNorm forward in eval mode: uses frozen running_mean and running_var.
     * x_norm = (x - running_mean) / sqrt(running_var + eps)
     * output = gamma * x_norm + beta
     */
    internal fun batchNormForward(
        input: FloatArray, batchSize: Int, height: Int, width: Int
    ): FloatArray {
        val output = FloatArray(input.size)
        for (b in 0 until batchSize) {
            for (c in 0 until outChannels) {
                val invStd = 1.0f / sqrt(bnRunningVar[c] + bnEps)
                for (h in 0 until height) {
                    for (w in 0 until width) {
                        val idx = b * (outChannels * height * width) + c * (height * width) + h * width + w
                        val xNorm = (input[idx] - bnRunningMean[c]) * invStd
                        output[idx] = bnGamma[c] * xNorm + bnBeta[c]
                    }
                }
            }
        }
        return output
    }

    /**
     * Backward through eval-mode BatchNorm.
     * Only gamma and beta are trainable.
     */
    private fun batchNormBackward(
        input: FloatArray, dOutput: FloatArray,
        batchSize: Int, height: Int, width: Int,
        learningRate: Float
    ): FloatArray {
        val dGamma = FloatArray(outChannels)
        val dBeta = FloatArray(outChannels)
        val dInput = FloatArray(input.size)

        for (b in 0 until batchSize) {
            for (c in 0 until outChannels) {
                val invStd = 1.0f / sqrt(bnRunningVar[c] + bnEps)
                for (h in 0 until height) {
                    for (w in 0 until width) {
                        val idx = b * (outChannels * height * width) + c * (height * width) + h * width + w
                        val xNorm = (input[idx] - bnRunningMean[c]) * invStd

                        // dGamma[c] += dOutput * x_norm
                        dGamma[c] += dOutput[idx] * xNorm
                        // dBeta[c] += dOutput
                        dBeta[c] += dOutput[idx]
                        // dInput = dOutput * gamma * invStd
                        dInput[idx] = dOutput[idx] * bnGamma[c] * invStd
                    }
                }
            }
        }

        // Average gradients over batch
        val scale = 1.0f / batchSize
        for (i in dGamma.indices) dGamma[i] *= scale
        for (i in dBeta.indices) dBeta[i] *= scale

        // Gradient clipping
        clipGradients(dGamma, dBeta)

        // Update BN parameters (gamma, beta)
        adamWUpdate(bnGamma, dGamma, mBnGamma, vBnGamma, learningRate, applyDecay = false)
        adamWUpdate(bnBeta, dBeta, mBnBeta, vBnBeta, learningRate, applyDecay = false)

        return dInput
    }

    // ---- SiLU activation ----

    /**
     * SiLU forward: silu(x) = x * sigmoid(x)
     */
    internal fun siluForward(input: FloatArray): FloatArray {
        val output = FloatArray(input.size)
        for (i in input.indices) {
            val sigmoid = 1.0f / (1.0f + exp(-input[i]))
            output[i] = input[i] * sigmoid
        }
        return output
    }

    /**
     * SiLU backward: d_silu(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
     */
    private fun siluBackward(input: FloatArray, dOutput: FloatArray): FloatArray {
        val dInput = FloatArray(input.size)
        for (i in input.indices) {
            val sigmoid = 1.0f / (1.0f + exp(-input[i]))
            dInput[i] = dOutput[i] * sigmoid * (1.0f + input[i] * (1.0f - sigmoid))
        }
        return dInput
    }

    // ---- Optimizer ----

    private fun clipGradients(vararg gradArrays: FloatArray) {
        var normSq = 0f
        for (grads in gradArrays) {
            for (g in grads) normSq += g * g
        }
        val norm = sqrt(normSq)
        if (norm > maxGradNorm) {
            val scale = maxGradNorm / (norm + 1e-6f)
            for (grads in gradArrays) {
                for (i in grads.indices) grads[i] *= scale
            }
        }
    }

    private fun adamWUpdate(
        params: FloatArray, grads: FloatArray,
        m: FloatArray, v: FloatArray,
        lr: Float, applyDecay: Boolean
    ) {
        val beta1Pow = Math.pow(beta1.toDouble(), t.toDouble()).toFloat()
        val beta2Pow = Math.pow(beta2.toDouble(), t.toDouble()).toFloat()

        for (i in params.indices) {
            if (applyDecay) {
                params[i] *= (1.0f - lr * weightDecay)
            }

            m[i] = beta1 * m[i] + (1 - beta1) * grads[i]
            v[i] = beta2 * v[i] + (1 - beta2) * grads[i] * grads[i]

            val mHat = m[i] / (1 - beta1Pow)
            val vHat = v[i] / (1 - beta2Pow)

            params[i] -= lr * mHat / (sqrt(vHat) + epsilon)
        }
    }

    /**
     * Reset optimizer state (e.g., when changing learning rate schedule).
     */
    fun resetOptimizer() {
        mConvW = FloatArray(convWeight.size)
        vConvW = FloatArray(convWeight.size)
        mBnGamma = FloatArray(outChannels)
        vBnGamma = FloatArray(outChannels)
        mBnBeta = FloatArray(outChannels)
        vBnBeta = FloatArray(outChannels)
        t = 0
    }
}
