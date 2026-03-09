package com.zhuo.traindemo

import com.zhuo.traindemo.model.TrainableClassifier
import com.zhuo.traindemo.model.TrainableConvBNSiLU
import com.zhuo.traindemo.model.TrainableHead
import com.zhuo.traindemo.model.TrainingMode
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.sqrt

class SemiFrozenTrainingTest {

    // ---- Conv2d 1x1 Forward ----

    @Test
    fun testConv1x1Forward() {
        // 2 input channels -> 3 output channels, 1x1 spatial
        val conv = TrainableConvBNSiLU(2, 3)
        // Set identity-like BN (gamma=1, beta=0, mean=0, var=1)
        for (i in conv.bnGamma.indices) conv.bnGamma[i] = 1.0f
        for (i in conv.bnBeta.indices) conv.bnBeta[i] = 0.0f
        for (i in conv.bnRunningMean.indices) conv.bnRunningMean[i] = 0.0f
        for (i in conv.bnRunningVar.indices) conv.bnRunningVar[i] = 1.0f

        // Set known conv weights: [outC=3, inC=2]
        // w[0,0]=1, w[0,1]=0, w[1,0]=0, w[1,1]=1, w[2,0]=1, w[2,1]=1
        conv.convWeight = floatArrayOf(1f, 0f, 0f, 1f, 1f, 1f)

        // Input: batch=1, in_ch=2, h=1, w=1 -> [0.5, 0.3]
        val input = floatArrayOf(0.5f, 0.3f)

        // Conv output before BN/SiLU:
        // out[0] = 1*0.5 + 0*0.3 = 0.5
        // out[1] = 0*0.5 + 1*0.3 = 0.3
        // out[2] = 1*0.5 + 1*0.3 = 0.8
        val convOut = conv.conv1x1Forward(input, 1, 1, 1)
        assertEquals(0.5f, convOut[0], 1e-5f)
        assertEquals(0.3f, convOut[1], 1e-5f)
        assertEquals(0.8f, convOut[2], 1e-5f)
    }

    // ---- BatchNorm Forward (eval mode) ----

    @Test
    fun testBatchNormForward() {
        val conv = TrainableConvBNSiLU(2, 2)
        conv.bnGamma = floatArrayOf(2.0f, 0.5f)
        conv.bnBeta = floatArrayOf(1.0f, -1.0f)
        conv.bnRunningMean = floatArrayOf(0.0f, 0.0f)
        conv.bnRunningVar = floatArrayOf(1.0f, 1.0f)

        // Input: [batch=1, ch=2, h=1, w=1] = [3.0, 4.0]
        val input = floatArrayOf(3.0f, 4.0f)

        // x_norm[0] = (3.0 - 0.0) / sqrt(1.0 + 1e-5) ≈ 3.0
        // output[0] = 2.0 * 3.0 + 1.0 = 7.0
        // x_norm[1] = (4.0 - 0.0) / sqrt(1.0 + 1e-5) ≈ 4.0
        // output[1] = 0.5 * 4.0 + (-1.0) = 1.0
        val output = conv.batchNormForward(input, 1, 1, 1)
        assertEquals(7.0f, output[0], 1e-3f)
        assertEquals(1.0f, output[1], 1e-3f)
    }

    // ---- SiLU Forward ----

    @Test
    fun testSiLUForward() {
        val conv = TrainableConvBNSiLU(1, 1)
        val input = floatArrayOf(0.0f, 1.0f, -1.0f)
        val output = conv.siluForward(input)

        // silu(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
        assertEquals(0.0f, output[0], 1e-6f)
        // silu(1) = 1 * sigmoid(1) = 1 * 0.7310... = 0.7310...
        val sig1 = 1.0f / (1.0f + exp(-1.0f))
        assertEquals(sig1, output[1], 1e-5f)
        // silu(-1) = -1 * sigmoid(-1) = -1 * 0.2689... = -0.2689...
        val sigNeg1 = 1.0f / (1.0f + exp(1.0f))
        assertEquals(-sigNeg1, output[2], 1e-5f)
    }

    // ---- Full forward pass (Conv + BN + SiLU) ----

    @Test
    fun testFullForward() {
        // Simple case: 1 in channel, 1 out channel, 1x1 spatial
        val conv = TrainableConvBNSiLU(1, 1)
        conv.convWeight = floatArrayOf(2.0f) // weight=2
        conv.bnGamma = floatArrayOf(1.0f)
        conv.bnBeta = floatArrayOf(0.0f)
        conv.bnRunningMean = floatArrayOf(0.0f)
        conv.bnRunningVar = floatArrayOf(1.0f)

        val input = floatArrayOf(1.0f)
        // Conv: 2.0 * 1.0 = 2.0
        // BN: (2.0 - 0) / sqrt(1 + 1e-5) * 1 + 0 ≈ 2.0
        // SiLU: 2.0 * sigmoid(2.0)
        val expectedConv = 2.0f
        val expectedSilu = expectedConv * (1.0f / (1.0f + exp(-expectedConv)))

        val output = conv.forward(input, 1, 1, 1)
        assertEquals(expectedSilu, output[0], 1e-4f)
    }

    // ---- Gradient Numerical Check ----

    @Test
    fun testConvGradientNumerical() {
        // Numerical gradient check for Conv weight gradients
        val inC = 2
        val outC = 2
        val h = 2
        val w = 2
        val batchSize = 1

        val conv = TrainableConvBNSiLU(inC, outC)
        // Use identity BN
        conv.bnGamma = FloatArray(outC) { 1.0f }
        conv.bnBeta = FloatArray(outC) { 0.0f }
        conv.bnRunningMean = FloatArray(outC) { 0.0f }
        conv.bnRunningVar = FloatArray(outC) { 1.0f }
        // Fixed conv weights
        conv.convWeight = floatArrayOf(0.1f, 0.2f, 0.3f, 0.4f)

        val input = floatArrayOf(
            // ch0: h=2,w=2
            1.0f, 2.0f, 3.0f, 4.0f,
            // ch1: h=2,w=2
            0.5f, 1.5f, 2.5f, 3.5f
        )

        // Compute a simple scalar loss = sum(output) for gradient checking
        val output = conv.forward(input, batchSize, h, w)
        val dOutput = FloatArray(output.size) { 1.0f } // dL/dOutput = 1 for sum loss

        // Compute analytical gradient w.r.t. input
        val dInput = conv.backward(input, dOutput, batchSize, h, w, 0.0f) // LR=0 to avoid updating

        // Numerical gradient check for dInput
        val eps = 1e-3f
        for (i in input.indices) {
            val inputPlus = input.clone()
            inputPlus[i] += eps
            // Need fresh conv for numerical check (reset weights since backward updated them with LR=0)
            val conv2 = TrainableConvBNSiLU(inC, outC)
            conv2.bnGamma = FloatArray(outC) { 1.0f }
            conv2.bnBeta = FloatArray(outC) { 0.0f }
            conv2.bnRunningMean = FloatArray(outC) { 0.0f }
            conv2.bnRunningVar = FloatArray(outC) { 1.0f }
            conv2.convWeight = floatArrayOf(0.1f, 0.2f, 0.3f, 0.4f)

            val outPlus = conv2.forward(inputPlus, batchSize, h, w)

            val inputMinus = input.clone()
            inputMinus[i] -= eps
            val conv3 = TrainableConvBNSiLU(inC, outC)
            conv3.bnGamma = FloatArray(outC) { 1.0f }
            conv3.bnBeta = FloatArray(outC) { 0.0f }
            conv3.bnRunningMean = FloatArray(outC) { 0.0f }
            conv3.bnRunningVar = FloatArray(outC) { 1.0f }
            conv3.convWeight = floatArrayOf(0.1f, 0.2f, 0.3f, 0.4f)

            val outMinus = conv3.forward(inputMinus, batchSize, h, w)

            // Numerical gradient = (loss+ - loss-) / (2*eps)
            // loss = sum(output)
            val lossPlus = outPlus.sum()
            val lossMinus = outMinus.sum()
            val numGrad = (lossPlus - lossMinus) / (2 * eps)

            // Compare
            assertEquals(
                "dInput[$i] mismatch",
                numGrad, dInput[i], 0.05f // Allow some tolerance due to float precision
            )
        }
    }

    // ---- TrainableHead with input gradient ----

    @Test
    fun testHeadInputGradient() {
        val head = TrainableHead(2, 2)
        head.weights = floatArrayOf(0.1f, 0.2f, 0.3f, 0.4f)
        head.bias = floatArrayOf(0f, 0f)

        // Input: batch=1, channels=2, h=1, w=1
        val input = floatArrayOf(1.0f, 0.5f)
        val targets = intArrayOf(0)

        val (loss, dInput) = head.trainStepWithGrad(
            input, 1, 2, 1, 1, targets, 0.0f, // LR=0
            computeInputGrad = true
        )

        assertTrue("Loss should be positive", loss > 0)
        assertTrue("dInput should not be null", dInput != null)
        assertEquals("dInput should have same size as input", input.size, dInput!!.size)

        // Verify gradient is non-zero
        var hasNonZero = false
        for (g in dInput) {
            if (g != 0f) hasNonZero = true
        }
        assertTrue("dInput should have non-zero values", hasNonZero)
    }

    // ---- TrainableClassifier HEAD_ONLY mode ----

    @Test
    fun testClassifierHeadOnly() {
        val classifier = TrainableClassifier(2, 3, 2)
        // Set known conv weights
        classifier.convBlock.convWeight = floatArrayOf(1f, 0f, 0f, 1f, 1f, 1f) // [3, 2]
        classifier.convBlock.bnGamma = FloatArray(3) { 1.0f }
        classifier.convBlock.bnBeta = FloatArray(3) { 0.0f }
        classifier.convBlock.bnRunningMean = FloatArray(3) { 0.0f }
        classifier.convBlock.bnRunningVar = FloatArray(3) { 1.0f }

        val input = floatArrayOf(1.0f, 0.5f) // batch=1, ch=2, h=1, w=1
        val targets = intArrayOf(0)

        val convWeightsBefore = classifier.convBlock.convWeight.clone()
        val headWeightsBefore = classifier.head.weights.clone()

        val loss = classifier.trainStep(
            input, 1, 1, 1, targets, 0.01f,
            TrainingMode.HEAD_ONLY
        )

        assertTrue("Loss should be positive", loss > 0)

        // In HEAD_ONLY mode, conv weights should NOT change
        assertArrayEquals("Conv weights should not change in HEAD_ONLY mode",
            convWeightsBefore, classifier.convBlock.convWeight, 1e-10f)

        // Head weights should change
        var headChanged = false
        for (i in classifier.head.weights.indices) {
            if (classifier.head.weights[i] != headWeightsBefore[i]) headChanged = true
        }
        assertTrue("Head weights should change in HEAD_ONLY mode", headChanged)
    }

    // ---- TrainableClassifier SEMI_FROZEN mode ----

    @Test
    fun testClassifierSemiFrozen() {
        val classifier = TrainableClassifier(2, 3, 2)
        classifier.convBlock.convWeight = floatArrayOf(1f, 0f, 0f, 1f, 1f, 1f)
        classifier.convBlock.bnGamma = FloatArray(3) { 1.0f }
        classifier.convBlock.bnBeta = FloatArray(3) { 0.0f }
        classifier.convBlock.bnRunningMean = FloatArray(3) { 0.0f }
        classifier.convBlock.bnRunningVar = FloatArray(3) { 1.0f }

        val input = floatArrayOf(1.0f, 0.5f)
        val targets = intArrayOf(0)

        val convWeightsBefore = classifier.convBlock.convWeight.clone()
        val headWeightsBefore = classifier.head.weights.clone()

        val loss = classifier.trainStep(
            input, 1, 1, 1, targets, 0.01f,
            TrainingMode.SEMI_FROZEN
        )

        assertTrue("Loss should be positive", loss > 0)

        // In SEMI_FROZEN mode, both conv and head weights should change
        var convChanged = false
        for (i in classifier.convBlock.convWeight.indices) {
            if (classifier.convBlock.convWeight[i] != convWeightsBefore[i]) convChanged = true
        }
        assertTrue("Conv weights should change in SEMI_FROZEN mode", convChanged)

        var headChanged = false
        for (i in classifier.head.weights.indices) {
            if (classifier.head.weights[i] != headWeightsBefore[i]) headChanged = true
        }
        assertTrue("Head weights should change in SEMI_FROZEN mode", headChanged)
    }

    // ---- Semi-frozen training convergence ----

    @Test
    fun testSemiFrozenConvergence() {
        val classifier = TrainableClassifier(2, 3, 2)
        // Initialize with small random-ish weights
        classifier.convBlock.convWeight = floatArrayOf(0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f)
        classifier.convBlock.bnGamma = FloatArray(3) { 1.0f }
        classifier.convBlock.bnBeta = FloatArray(3) { 0.0f }
        classifier.convBlock.bnRunningMean = FloatArray(3) { 0.0f }
        classifier.convBlock.bnRunningVar = FloatArray(3) { 1.0f }
        classifier.backboneLrRatio = 0.1f

        // Two distinct inputs for two classes
        val input0 = floatArrayOf(1.0f, 0.0f)
        val input1 = floatArrayOf(0.0f, 1.0f)

        var firstLoss = 0f
        var lastLoss = 0f

        for (step in 0 until 100) {
            val loss0 = classifier.trainStep(input0, 1, 1, 1, intArrayOf(0), 0.01f, TrainingMode.SEMI_FROZEN)
            val loss1 = classifier.trainStep(input1, 1, 1, 1, intArrayOf(1), 0.01f, TrainingMode.SEMI_FROZEN)
            val avgLoss = (loss0 + loss1) / 2

            if (step == 0) firstLoss = avgLoss
            if (step == 99) lastLoss = avgLoss
        }

        // Loss should generally decrease (or at least not explode)
        assertTrue("Training should not explode (last loss: $lastLoss)", lastLoss < 10.0f)
    }

    // ---- Differential learning rate ----

    @Test
    fun testDifferentialLearningRate() {
        // Train two identical classifiers: one with higher backbone LR ratio
        val clf1 = TrainableClassifier(2, 3, 2)
        val clf2 = TrainableClassifier(2, 3, 2)

        // Same initial weights
        val initConvW = floatArrayOf(0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f)
        clf1.convBlock.convWeight = initConvW.clone()
        clf2.convBlock.convWeight = initConvW.clone()

        for (clf in listOf(clf1, clf2)) {
            clf.convBlock.bnGamma = FloatArray(3) { 1.0f }
            clf.convBlock.bnBeta = FloatArray(3) { 0.0f }
            clf.convBlock.bnRunningMean = FloatArray(3) { 0.0f }
            clf.convBlock.bnRunningVar = FloatArray(3) { 1.0f }
        }

        // Same head weights
        val initHeadW = clf1.head.weights.clone()
        clf2.head.weights = initHeadW.clone()
        clf2.head.bias = clf1.head.bias.clone()

        clf1.backboneLrRatio = 0.1f  // Conv LR = 0.001
        clf2.backboneLrRatio = 0.5f  // Conv LR = 0.005

        val input = floatArrayOf(1.0f, 0.5f)
        val targets = intArrayOf(0)

        clf1.trainStep(input, 1, 1, 1, targets, 0.01f, TrainingMode.SEMI_FROZEN)
        clf2.trainStep(input, 1, 1, 1, targets, 0.01f, TrainingMode.SEMI_FROZEN)

        // clf2 should have larger conv weight changes (higher backbone LR)
        var totalChange1 = 0f
        var totalChange2 = 0f
        for (i in initConvW.indices) {
            totalChange1 += abs(clf1.convBlock.convWeight[i] - initConvW[i])
            totalChange2 += abs(clf2.convBlock.convWeight[i] - initConvW[i])
        }

        assertTrue("Higher backbone LR should cause larger weight changes ($totalChange2 > $totalChange1)",
            totalChange2 > totalChange1)
    }

    // ---- BN parameters should be frozen in both modes ----

    @Test
    fun testBnParamsFrozenInHeadOnly() {
        val classifier = TrainableClassifier(2, 3, 2)
        classifier.convBlock.convWeight = floatArrayOf(1f, 0f, 0f, 1f, 1f, 1f)
        classifier.convBlock.bnGamma = FloatArray(3) { 1.0f }
        classifier.convBlock.bnBeta = FloatArray(3) { 0.0f }
        classifier.convBlock.bnRunningMean = FloatArray(3) { 0.0f }
        classifier.convBlock.bnRunningVar = FloatArray(3) { 1.0f }

        val input = floatArrayOf(1.0f, 0.5f)
        val targets = intArrayOf(0)

        val bnGammaBefore = classifier.convBlock.bnGamma.clone()
        val bnBetaBefore = classifier.convBlock.bnBeta.clone()
        val bnMeanBefore = classifier.convBlock.bnRunningMean.clone()
        val bnVarBefore = classifier.convBlock.bnRunningVar.clone()

        for (step in 0 until 10) {
            classifier.trainStep(input, 1, 1, 1, targets, 0.01f, TrainingMode.HEAD_ONLY)
        }

        assertArrayEquals("BN gamma should not change in HEAD_ONLY mode",
            bnGammaBefore, classifier.convBlock.bnGamma, 0f)
        assertArrayEquals("BN beta should not change in HEAD_ONLY mode",
            bnBetaBefore, classifier.convBlock.bnBeta, 0f)
        assertArrayEquals("BN running mean should not change in HEAD_ONLY mode",
            bnMeanBefore, classifier.convBlock.bnRunningMean, 0f)
        assertArrayEquals("BN running var should not change in HEAD_ONLY mode",
            bnVarBefore, classifier.convBlock.bnRunningVar, 0f)
    }

    @Test
    fun testBnParamsFrozenInSemiFrozen() {
        val classifier = TrainableClassifier(2, 3, 2)
        classifier.convBlock.convWeight = floatArrayOf(1f, 0f, 0f, 1f, 1f, 1f)
        classifier.convBlock.bnGamma = FloatArray(3) { 1.0f }
        classifier.convBlock.bnBeta = FloatArray(3) { 0.0f }
        classifier.convBlock.bnRunningMean = FloatArray(3) { 0.0f }
        classifier.convBlock.bnRunningVar = FloatArray(3) { 1.0f }

        val input = floatArrayOf(1.0f, 0.5f)
        val targets = intArrayOf(0)

        val bnGammaBefore = classifier.convBlock.bnGamma.clone()
        val bnBetaBefore = classifier.convBlock.bnBeta.clone()
        val bnMeanBefore = classifier.convBlock.bnRunningMean.clone()
        val bnVarBefore = classifier.convBlock.bnRunningVar.clone()

        for (step in 0 until 10) {
            classifier.trainStep(input, 1, 1, 1, targets, 0.01f, TrainingMode.SEMI_FROZEN)
        }

        // Conv weights SHOULD change, but BN params should NOT
        var convChanged = false
        val initConv = floatArrayOf(1f, 0f, 0f, 1f, 1f, 1f)
        for (i in classifier.convBlock.convWeight.indices) {
            if (classifier.convBlock.convWeight[i] != initConv[i]) convChanged = true
        }
        assertTrue("Conv weights should change in SEMI_FROZEN mode", convChanged)

        assertArrayEquals("BN gamma should not change in SEMI_FROZEN mode",
            bnGammaBefore, classifier.convBlock.bnGamma, 0f)
        assertArrayEquals("BN beta should not change in SEMI_FROZEN mode",
            bnBetaBefore, classifier.convBlock.bnBeta, 0f)
        assertArrayEquals("BN running mean should not change in SEMI_FROZEN mode",
            bnMeanBefore, classifier.convBlock.bnRunningMean, 0f)
        assertArrayEquals("BN running var should not change in SEMI_FROZEN mode",
            bnVarBefore, classifier.convBlock.bnRunningVar, 0f)
    }
}
