package ai.onnxruntime.sam

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.extensions.OrtxPackage
import ai.onnxruntime.sam.npy.Npy
import android.graphics.BitmapFactory
import android.os.Build
import android.os.Bundle
import android.widget.Button
import android.widget.ImageView
import androidx.appcompat.app.AppCompatActivity
import java.io.File
import java.io.FileOutputStream


class MainActivity : AppCompatActivity() {
    private var ortEnv: OrtEnvironment = OrtEnvironment.getEnvironment()
    private lateinit var ortSession: OrtSession

    private lateinit var inputImage: ImageView
    private lateinit var maskImage: ImageView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        inputImage = findViewById(R.id.ivInput)
        inputImage.setImageBitmap(BitmapFactory.decodeStream(resources.openRawResource(R.raw.dogs)))

        maskImage = findViewById(R.id.ivMask)

        // Initialize Ort Session and register the onnxruntime extensions package that contains the custom operators.
        // Note: These are used to decode the input image into the format the original model requires,
        // and to encode the model output into png format
        val sessionOptions: OrtSession.SessionOptions = OrtSession.SessionOptions()
        sessionOptions.registerCustomOpLibrary(OrtxPackage.getLibraryPath())
        ortSession = ortEnv.createSession(readModel(), sessionOptions)

        findViewById<Button>(R.id.btnExecute).setOnClickListener {
            performMaskDecoder(ortSession)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        ortEnv.close()
        ortSession.close()
    }

    private fun readModel(): ByteArray {
        val modelId = R.raw.prompt_encoder_mask_decoder_quantized
        return resources.openRawResource(modelId).readBytes()
    }

    private val LONG_SIDE_LENGTH = 1024

    private var embedding: Npy? = null
    private fun performMaskDecoder(ortSession: OrtSession) {
        val samModel = SegmentAnything()
        if (embedding == null) {
            embedding = Npy(resources.openRawResource(R.raw.dogs_embedding))
        }

        // image: 1072x603
        val w = 1072
        val h = 603
        val samScale: Float = LONG_SIDE_LENGTH / (Math.max(w, h) * 1.0F)
        val modelScale = ModelScale(
            width = w,
            height = h,
            samScale = samScale,
        )

        // mock clicks
        val (x, y) = arrayOf(542, 388)

        val clicks = Array(1) {
            ModelInput(
                (x * samScale).toInt(),
                (y * samScale).toInt(),
                1
            )
        }

        val result = samModel.execute(
            embedding = embedding!!,
            modelScale = modelScale,
            clicks = clicks,
            ortEnv = ortEnv,
            ortSession = ortSession,
        )
        updateUI(result)
    }

    private fun updateUI(result: ModelResult) {
        maskImage.setImageBitmap(result.maskBitmap)
    }
}
