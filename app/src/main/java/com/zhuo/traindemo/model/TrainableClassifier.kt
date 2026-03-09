package com.zhuo.traindemo.model

/**
 * Training mode for the classifier.
 */
enum class TrainingMode {
    /** Only the linear head is trained; Conv+BN layers are frozen. */
    HEAD_ONLY,
    /** Semi-frozen: head is trained with higher LR, Conv weights with lower LR. BN params are frozen. */
    SEMI_FROZEN
}

/**
 * Trainable classifier that combines:
 *   1. TrainableConvBNSiLU (Conv2d 1x1 + BN + SiLU) — the "last few layers"
 *   2. TrainableHead (GAP + Dropout + Linear) — the classification head
 *
 * Supports two training modes:
 *   - HEAD_ONLY: Only the linear head (TrainableHead) is trained
 *   - SEMI_FROZEN: Both Conv weights (lower LR) and head (higher LR) are trained; BN params are frozen
 *
 * In SEMI_FROZEN mode, the learning rate for Conv weights is scaled down by [backboneLrRatio].
 * BN parameters (gamma, beta) are always frozen to avoid training instability.
 *
 * All gradient computation is hand-written without autograd.
 */
class TrainableClassifier(
    val backboneOutChannels: Int,
    val convOutChannels: Int,
    numClasses: Int
) {
    val convBlock = TrainableConvBNSiLU(backboneOutChannels, convOutChannels)
    val head = TrainableHead(convOutChannels, numClasses)

    /** LR multiplier for the Conv+BN layers relative to the head LR. */
    var backboneLrRatio = 0.1f

    /**
     * Forward pass (inference): Conv+BN+SiLU -> GAP -> Linear
     *
     * @param backboneFeatures [batchSize, backboneOutChannels, height, width] flattened
     * @return logits [batchSize, numClasses]
     */
    fun forward(
        backboneFeatures: FloatArray,
        batchSize: Int, height: Int, width: Int
    ): FloatArray {
        val convOut = convBlock.forward(backboneFeatures, batchSize, height, width)
        return head.forward(convOut, batchSize, convOutChannels, height, width)
    }

    /**
     * Training step with the specified mode.
     *
     * @param backboneFeatures [batchSize, backboneOutChannels, height, width] flattened
     * @param targets class labels [batchSize]
     * @param headLr learning rate for the classification head
     * @param mode training mode (HEAD_ONLY or SEMI_FROZEN)
     * @return loss value
     */
    fun trainStep(
        backboneFeatures: FloatArray,
        batchSize: Int, height: Int, width: Int,
        targets: IntArray,
        headLr: Float,
        mode: TrainingMode,
        labelSmoothing: Float = 0.1f
    ): Float {
        // Forward through Conv+BN+SiLU
        val convOut = convBlock.forward(backboneFeatures, batchSize, height, width)

        return when (mode) {
            TrainingMode.HEAD_ONLY -> {
                // Only train the head; Conv layers are frozen
                head.trainStep(convOut, batchSize, convOutChannels, height, width, targets, headLr, labelSmoothing)
            }
            TrainingMode.SEMI_FROZEN -> {
                // Train both head (higher LR) and Conv+BN (lower LR)
                // 1. Head forward + backward, also compute input gradient
                val (loss, dConvOut) = head.trainStepWithGrad(
                    convOut, batchSize, convOutChannels, height, width,
                    targets, headLr, labelSmoothing,
                    computeInputGrad = true
                )

                // 2. Backward through Conv+BN+SiLU with lower LR
                val convLr = headLr * backboneLrRatio
                convBlock.backward(
                    backboneFeatures, dConvOut!!,
                    batchSize, height, width,
                    convLr
                )

                loss
            }
        }
    }
}
