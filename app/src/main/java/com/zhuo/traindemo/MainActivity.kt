package com.zhuo.traindemo

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.util.Log
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.ImageView
import android.widget.Spinner
import android.util.Size
import android.view.View
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.zhuo.traindemo.data.Dataset
import com.zhuo.traindemo.model.FeatureExtractor
import com.zhuo.traindemo.model.ModelManager
import com.zhuo.traindemo.model.TrainableHead
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.Executors
import kotlin.math.max

class MainActivity : AppCompatActivity() {

    private lateinit var viewFinder: PreviewView
    private lateinit var imgPreview: ImageView
    private lateinit var txtStatus: TextView
    private lateinit var btnCapture: Button
    private lateinit var btnTrain: Button
    private lateinit var btnAddClass: Button
    private lateinit var spinnerLabel: Spinner
    private lateinit var progressBar: ProgressBar

    private val dataset = Dataset()
    private lateinit var featureExtractor: FeatureExtractor
    private var trainableHead: TrainableHead? = null
    private lateinit var modelManager: ModelManager

    private val classLabels = mutableListOf("Class 0", "Class 1")
    private lateinit var labelAdapter: ArrayAdapter<String>

    private val cameraExecutor = Executors.newSingleThreadExecutor()
    private var isTraining = false
    private var isAnalyzing = true

    // Model parameters
    private val featureChannels = 320 // ConvNeXt Atto
    private val featureHeight = 7
    private val featureWidth = 7

    private val trainExecutor = Executors.newFixedThreadPool(4)
    private val trainDispatcher = trainExecutor.asCoroutineDispatcher()
    private val trainScope = CoroutineScope(trainDispatcher)
    private var trainingJob: Job? = null

    // Store camera provider to unbind/bind preview
    private var cameraProvider: ProcessCameraProvider? = null
    private var preview: Preview? = null
    private var imageAnalyzer: ImageAnalysis? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        viewFinder = findViewById(R.id.viewFinder)
        imgPreview = findViewById(R.id.imgPreview)
        txtStatus = findViewById(R.id.txtStatus)
        btnCapture = findViewById(R.id.btnCapture)
        btnTrain = findViewById(R.id.btnTrain)
        btnAddClass = findViewById(R.id.btnAddClass)
        spinnerLabel = findViewById(R.id.spinnerLabel)
        progressBar = findViewById(R.id.progressBar)

        // Initialize Spinner
        labelAdapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, classLabels)
        labelAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        spinnerLabel.adapter = labelAdapter

        // Initialize Feature Extractor
        try {
            featureExtractor = FeatureExtractor(this)
            modelManager = ModelManager(this)

            // Try to load model
            val loaded = modelManager.loadModel()
            if (loaded != null) {
                trainableHead = loaded.first
                classLabels.clear()
                classLabels.addAll(loaded.second)
                labelAdapter.notifyDataSetChanged()
                Toast.makeText(this, "Loaded saved model", Toast.LENGTH_SHORT).show()
            } else {
                // Initialize Head
                trainableHead = TrainableHead(featureChannels, classLabels.size)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error initializing models", e)
            txtStatus.text = "Error: ${e.message}"
        }

        // Request Permissions
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            requestPermissions.launch(arrayOf(Manifest.permission.CAMERA))
        }

        // Button Listeners
        btnCapture.setOnClickListener { captureImage() }

        btnTrain.setOnClickListener {
            if (isTraining) {
                stopTraining()
            } else {
                startTraining()
            }
        }

        btnAddClass.setOnClickListener {
            addNewClass()
        }
    }

    private val requestPermissions = registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { permissions ->
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            Toast.makeText(this, "Permissions not granted by the user.", Toast.LENGTH_SHORT).show()
            finish()
        }
    }

    private fun allPermissionsGranted() = ContextCompat.checkSelfPermission(
        this, Manifest.permission.CAMERA
    ) == PackageManager.PERMISSION_GRANTED

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            cameraProvider = cameraProviderFuture.get()

            val targetResolution = Size(640, 480)

            preview = Preview.Builder()
                .setTargetResolution(targetResolution)
                .build()
                .also {
                    it.setSurfaceProvider(viewFinder.surfaceProvider)
                }

            imageAnalyzer = ImageAnalysis.Builder()
                .setTargetResolution(targetResolution)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor) { image ->
                        processImage(image)
                    }
                }

            bindCameraUseCases()

        }, ContextCompat.getMainExecutor(this))
    }

    private fun bindCameraUseCases() {
        val provider = cameraProvider ?: return
        val selector = CameraSelector.DEFAULT_BACK_CAMERA

        try {
            provider.unbindAll()

            val useCases = mutableListOf<androidx.camera.core.UseCase>()

            // If analyzing (inference mode), bind everything (Preview + Analysis)
            // If training, we stop preview to save resources, but we might want to stop analysis too?
            // The request says "Stop Preview during training".
            // Analysis is used for inference. During training we pause inference (isAnalyzing=false).
            // So we can unbind both or just Preview.
            // However, usually we unbind everything during heavy training if we don't need camera.

            if (!isTraining) {
                preview?.let { useCases.add(it) }
                imageAnalyzer?.let { useCases.add(it) }
            }

            if (useCases.isNotEmpty()) {
                provider.bindToLifecycle(
                    this, selector, *useCases.toTypedArray()
                )
            }

        } catch (exc: Exception) {
            Log.e(TAG, "Use case binding failed", exc)
        }
    }

    // Captured bitmap to be used for adding to dataset
    private var currentBitmap: Bitmap? = null

    private fun processImage(imageProxy: ImageProxy) {
        if (!isAnalyzing) {
            imageProxy.close()
            return
        }

        val bitmap = imageProxy.toBitmap()
        // Rotate if needed (simplified, assumes portrait)
        val matrix = Matrix()
        matrix.postRotate(imageProxy.imageInfo.rotationDegrees.toFloat())
        val rotatedBitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)

        currentBitmap = rotatedBitmap

        // Inference logic (2fps limit roughly controlled by analysis time)
        val startTime = System.currentTimeMillis()

        try {
            if (trainableHead != null && !isTraining) {
                val features = featureExtractor.extract(rotatedBitmap)
                // Head expects: input, batch, channels, h, w
                val logits = trainableHead!!.forward(features, 1, featureChannels, featureHeight, featureWidth)

                // Argmax
                var maxIdx = -1
                var maxVal = Float.NEGATIVE_INFINITY
                for (i in logits.indices) {
                    if (logits[i] > maxVal) {
                        maxVal = logits[i]
                        maxIdx = i
                    }
                }

                runOnUiThread {
                    if (maxIdx >= 0 && maxIdx < classLabels.size) {
                        txtStatus.text = "Prediction: ${classLabels[maxIdx]} (Logit: ${String.format("%.2f", maxVal)})"
                    }
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Inference error", e)
        } finally {
            imageProxy.close()
        }

        // Simple throttle
        val endTime = System.currentTimeMillis()
        val duration = endTime - startTime
        if (duration < 500) { // Max ~2fps
            Thread.sleep(500 - duration)
        }
    }

    private fun captureImage() {
        val bitmap = currentBitmap ?: return
        val labelIndex = spinnerLabel.selectedItemPosition

        // Save to dataset
        dataset.addImage(bitmap, labelIndex)

        runOnUiThread {
            imgPreview.setImageBitmap(bitmap)
            Toast.makeText(this, "Captured for ${classLabels[labelIndex]}. Total: ${dataset.size()}", Toast.LENGTH_SHORT).show()
        }
    }

    private fun startTraining() {
        if (dataset.size() == 0) {
            Toast.makeText(this, "No data collected!", Toast.LENGTH_SHORT).show()
            return
        }

        isTraining = true
        isAnalyzing = false // Pause inference updates
        btnTrain.text = "Stop"
        btnCapture.isEnabled = false
        progressBar.visibility = View.VISIBLE
        progressBar.max = 100
        progressBar.progress = 0

        // Unbind camera
        bindCameraUseCases()

        trainingJob = trainScope.launch {
            val batchSize = 16
            val epochs = 100
            val initialLr = 0.001f
            val minLr = 0.00001f

            var epoch = 0
            while (isTraining && epoch < epochs) {
                epoch++

                // Cosine Learning Rate Schedule
                val progress = epoch.toFloat() / epochs
                val lr = minLr + 0.5f * (initialLr - minLr) * (1 + kotlin.math.cos(Math.PI * progress)).toFloat()

                val batch = dataset.getBatch(batchSize)
                if (batch.isEmpty()) continue

                var totalLoss = 0f

                // Process batch
                // We need to stack inputs. This is inefficient in pure Kotlin loop, but works for demo.
                // Better: Pre-extract features for dataset if possible, but we have augmentation.
                // So: Augment -> Extract -> Stack -> TrainStep.

                // To support batching in TrainableHead, we need to flatten the whole batch into one array.
                // Input size: Batch * Channels * H * W
                val batchInput = FloatArray(batchSize * featureChannels * featureHeight * featureWidth)
                val batchTargets = IntArray(batchSize)

                for (i in 0 until batchSize) {
                    val item = batch[i]
                    val features = featureExtractor.extract(item.bitmap) // Returns [C*H*W]
                    System.arraycopy(features, 0, batchInput, i * features.size, features.size)
                    batchTargets[i] = item.label
                }

                // Train step
                val loss = trainableHead!!.trainStep(
                    batchInput,
                    batchSize,
                    featureChannels,
                    featureHeight,
                    featureWidth,
                    batchTargets,
                    lr,
                    labelSmoothing = 0.1f
                )

                withContext(Dispatchers.Main) {
                    txtStatus.text = "Epoch $epoch/$epochs, Loss: ${String.format("%.4f", loss)}"
                    progressBar.progress = epoch
                }

                // Check if user stopped
                if (!isTraining) break
            }
            // Training finished or stopped
            withContext(Dispatchers.Main) {
                stopTraining()
                txtStatus.text = "Training Complete. Resuming Inference."
            }
        }
    }

    private fun stopTraining() {
        isTraining = false
        isAnalyzing = true
        btnTrain.text = "Train"
        btnCapture.isEnabled = true
        progressBar.visibility = View.GONE
        // trainingJob?.cancel() // Don't cancel immediately if called from within the job

        // Save model
        trainableHead?.let {
            modelManager.saveModel(it, classLabels)
            Toast.makeText(this, "Model Saved", Toast.LENGTH_SHORT).show()
        }

        // Resume camera
        bindCameraUseCases()
    }

    override fun onPause() {
        super.onPause()
        if (isTraining) {
             stopTraining()
        }
    }

    private fun addNewClass() {
        val newClassIndex = classLabels.size
        classLabels.add("Class $newClassIndex")
        labelAdapter.notifyDataSetChanged()

        // Re-initialize head to accommodate new class (naive approach: reset weights)
        // In a real app, we might want to keep existing weights or use a flexible head.
        // For this demo, we reset.
        trainableHead = TrainableHead(featureChannels, classLabels.size)
        Toast.makeText(this, "Added Class $newClassIndex. Model Reset.", Toast.LENGTH_SHORT).show()
    }

    companion object {
        private const val TAG = "TrainDemo"
    }
}
