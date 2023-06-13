package ai.onnxruntime.sam

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import ai.onnxruntime.sam.npy.Npy
import android.graphics.Bitmap
import android.util.Log
import java.nio.FloatBuffer

/**
 * Created by suli on 2023/6/11
 **/

internal data class ModelResult(
    var maskBitmap: Bitmap,
)

internal data class ModelScale(
    var width: Int,
    var height: Int,
    var samScale: Float,
)

internal data class ModelInput(
    // 缩放后的坐标位置
    var x: Int,
    var y: Int,
    // 正选: 1, 反选: 0
    var type: Int
)

const val TAG = "SegmentAnyThing"

internal class SegmentAnything {

    fun execute(
        embedding: Npy,
        modelScale: ModelScale,
        clicks: Array<ModelInput>,
        ortEnv: OrtEnvironment,
        ortSession: OrtSession
    ): ModelResult {
        // embeddings
        val embeddingShape = LongArray(embedding.shape.size)
        for (i in embeddingShape.indices) {
            embeddingShape[i] = embedding.shape[i].toLong()
        }
        val embeddingData = embedding.floatElements()
        val embeddingTensor = OnnxTensor.createTensor(
            ortEnv,
            FloatBuffer.wrap(embeddingData),
            embeddingShape,
        )
        Log.d(
            TAG,
            "embedding shape:[${embedding.shape.joinToString()}], data size:${embeddingData.size}"
        )

        Log.d(TAG, "model scale:${modelScale}")

        // clicks
        val n = clicks.size
        val pointCoords = FloatBuffer.allocate(2 * (n + 1))
        val pointLabels = FloatBuffer.allocate(n + 1)
        for (i in 0 until n) {
            pointCoords.put(2 * i, clicks[i].x * modelScale.samScale)
            pointCoords.put(2 * i + 1, clicks[i].y * modelScale.samScale)
            pointLabels.put(i, clicks[i].type.toFloat())
        }
        pointCoords.put(2 * n, 0.0F)
        pointCoords.put(2 * n + 1, 0.0F)
        pointLabels.put(n, -1.0F)

        val pointCoordsTensor = OnnxTensor.createTensor(
            ortEnv,
            pointCoords,
            longArrayOf(1, (n + 1).toLong(), 2)
        )

        val pointLabelsTensor = OnnxTensor.createTensor(
            ortEnv,
            pointLabels,
            longArrayOf(1, (n + 1).toLong()),
        )

        val imageSizeTensor = OnnxTensor.createTensor(
            ortEnv,
            floatArrayOf(modelScale.height.toFloat(), modelScale.width.toFloat())
        )

        // There is no previous mask, so default to an empty tensor
        val maskInputTensor = OnnxTensor.createTensor(
            ortEnv,
            FloatBuffer.allocate(256 * 256),
            longArrayOf(1, 1, 256, 256),
        )

        // There is no previous mask, so default to 0
        val hasMaskInputTensor = OnnxTensor.createTensor(ortEnv, floatArrayOf(0F))

        // Step 3: call ort inferenceSession run
        val output = ortSession.run(
            mapOf(
                "image_embeddings" to embeddingTensor,
                // 选点的坐标 ( 2 * (n + 1))
                "point_coords" to pointCoordsTensor,
                // 点是正选还是反选 (n + 1)
                "point_labels" to pointLabelsTensor,
                // 原生图片大小
                "orig_im_size" to imageSizeTensor,
                // 手涂的mask区域(暂未使用)
                "mask_input" to maskInputTensor,
                "has_mask_input" to hasMaskInputTensor,
            )
        )

        val result: ModelResult
        // Step 4: output analysis
        output.use {
            Log.d(TAG, "model output name:${ortSession.outputNames}")
            val tensor = output.get(ortSession.outputNames.first()).get()
            val maskData = (tensor as OnnxTensor).floatBuffer.array()

            val shape = (output.get(0).info as TensorInfo).shape
            val width = shape[3].toInt()
            val height = shape[2].toInt()

            Log.d(TAG, "mask data size:${maskData.size}, width:$width, height:$height")

            val rgbaImageData = arrayToImageData(maskData, width, height)
            val outputMaskBitmap = bitmapFromRgba(width, height, rgbaImageData)
            result = ModelResult(outputMaskBitmap)
        }

        // close
        embeddingTensor.close()
        pointCoordsTensor.close()
        pointLabelsTensor.close()
        imageSizeTensor.close()
        maskInputTensor.close()
        hasMaskInputTensor.close()

        return result
    }

    private fun arrayToImageData(input: FloatArray, width: Int, height: Int): ByteArray {
        val (r, g, b, a) = byteArrayOf(0, 0x72, 0xBD.toByte(), 0xFF.toByte())
        val data = ByteArray(4 * width * height) { 0 }
        for (i in input.indices) {
            if (input[i] > 0) {
                data[4 * i + 0] = r
                data[4 * i + 1] = g
                data[4 * i + 2] = b
                data[4 * i + 3] = a
            }
        }
        return data
    }


    private fun bitmapFromRgba(width: Int, height: Int, bytes: ByteArray): Bitmap {
        val pixels = IntArray(bytes.size / 4)
        var j = 0
        for (i in pixels.indices) {
            val R = bytes[j++].toInt() and 0xff
            val G = bytes[j++].toInt() and 0xff
            val B = bytes[j++].toInt() and 0xff
            val A = bytes[j++].toInt() and 0xff
            val pixel = A shl 24 or (R shl 16) or (G shl 8) or B
            pixels[i] = pixel
        }
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        bitmap.setPixels(pixels, 0, width, 0, 0, width, height)
        return bitmap
    }
}
