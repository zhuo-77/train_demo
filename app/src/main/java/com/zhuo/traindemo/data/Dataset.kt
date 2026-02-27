package com.zhuo.traindemo.data

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.ColorMatrix
import android.graphics.ColorMatrixColorFilter
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Rect
import java.util.Random

data class LabeledImage(val bitmap: Bitmap, val label: Int)

class Dataset {
    private val images = mutableListOf<LabeledImage>()
    private val random = Random()

    fun addImage(bitmap: Bitmap, label: Int) {
        images.add(LabeledImage(bitmap, label))
    }

    fun clear() {
        images.clear()
    }

    fun size(): Int = images.size

    fun getBatch(batchSize: Int): List<LabeledImage> {
        if (images.isEmpty()) return emptyList()
        val batch = mutableListOf<LabeledImage>()
        for (i in 0 until batchSize) {
            val idx = random.nextInt(images.size)
            batch.add(augment(images[idx]))
        }
        return batch
    }

    private fun augment(original: LabeledImage): LabeledImage {
        // Mainstream augmentations applied sequentially and independently
        var bitmap = original.bitmap.copy(Bitmap.Config.ARGB_8888, true)

        // 1. Random Horizontal Flip (50%)
        if (random.nextBoolean()) {
            val flipped = horizontalFlip(bitmap)
            // If flip fails or returns same (shouldn't happen with Matrix logic but being safe), use original
            bitmap = flipped
        }

        // 2. Random Crop (50%)
        if (random.nextBoolean()) {
            bitmap = randomCrop(bitmap)
        }

        // 3. Color Jitter (50%)
        if (random.nextBoolean()) {
             bitmap = colorJitter(bitmap)
        }

        // 4. Random Erase (50%)
        if (random.nextBoolean()) {
            bitmap = randomErase(bitmap)
        }

        return LabeledImage(bitmap, original.label)
    }

    private fun horizontalFlip(bitmap: Bitmap): Bitmap {
        val matrix = Matrix().apply { preScale(-1f, 1f) }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    private fun randomCrop(bitmap: Bitmap): Bitmap {
        val width = bitmap.width
        val height = bitmap.height
        val cropFactor = 0.8f + (random.nextFloat() * 0.2f)
        val newWidth = (width * cropFactor).toInt()
        val newHeight = (height * cropFactor).toInt()

        // Ensure new dimensions are valid (>0)
        if (newWidth <= 0 || newHeight <= 0) return bitmap

        val x = random.nextInt(width - newWidth + 1)
        val y = random.nextInt(height - newHeight + 1)

        return Bitmap.createBitmap(bitmap, x, y, newWidth, newHeight)
    }

    private fun colorJitter(bitmap: Bitmap): Bitmap {
        val cm = ColorMatrix()
        // Brightness: -25 to +25
        val brightness = (random.nextFloat() - 0.5f) * 50
        // Contrast: 0.8 to 1.2
        val contrast = 1f + (random.nextFloat() - 0.5f) * 0.4f
        // Saturation: 0.8 to 1.2
        val saturation = 1f + (random.nextFloat() - 0.5f) * 0.4f

        cm.setSaturation(saturation)

        // Contrast scaling matrix
        val scale = contrast
        val translate = (-.5f * scale + .5f) * 255.0f
        val contrastMatrix = floatArrayOf(
            scale, 0f, 0f, 0f, translate,
            0f, scale, 0f, 0f, translate,
            0f, 0f, scale, 0f, translate,
            0f, 0f, 0f, 1f, 0f
        )
        cm.postConcat(ColorMatrix(contrastMatrix))

        // Brightness is simpler but often done via offset in matrix or LightingColorFilter.
        // ColorMatrix supports translation on R, G, B in the 5th column.
        // We can add brightness to the existing matrix.
        // The array is [ a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t ]
        // e, j, o are offsets.
        val array = cm.array
        array[4] += brightness
        array[9] += brightness
        array[14] += brightness

        val paint = Paint().apply {
            colorFilter = ColorMatrixColorFilter(cm)
        }

        val result = Bitmap.createBitmap(bitmap.width, bitmap.height, bitmap.config ?: Bitmap.Config.ARGB_8888)
        val canvas = Canvas(result)
        canvas.drawBitmap(bitmap, 0f, 0f, paint)
        return result
    }

    private fun randomErase(bitmap: Bitmap): Bitmap {
        val width = bitmap.width
        val height = bitmap.height
        val eraseWidth = (width * (0.1f + random.nextFloat() * 0.2f)).toInt()
        val eraseHeight = (height * (0.1f + random.nextFloat() * 0.2f)).toInt()

        val x = random.nextInt(width - eraseWidth + 1)
        val y = random.nextInt(height - eraseHeight + 1)

        val canvas = Canvas(bitmap)
        val paint = Paint().apply {
            color = Color.GRAY // Erase with gray
            style = Paint.Style.FILL
        }
        canvas.drawRect(Rect(x, y, x + eraseWidth, y + eraseHeight), paint)
        return bitmap
    }
}
