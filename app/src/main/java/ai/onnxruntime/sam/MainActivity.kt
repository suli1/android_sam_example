package ai.onnxruntime.sam

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.extensions.OrtxPackage
import ai.onnxruntime.sam.npy.Npy
import android.annotation.SuppressLint
import android.graphics.BitmapFactory
import android.graphics.Rect
import android.graphics.RectF
import android.os.Bundle
import android.util.Log
import android.view.MotionEvent
import android.widget.Button
import android.widget.ImageView
import android.widget.RadioGroup
import androidx.appcompat.app.AppCompatActivity


class MainActivity : AppCompatActivity() {
    private var ortEnv: OrtEnvironment = OrtEnvironment.getEnvironment()
    private lateinit var ortSession: OrtSession

    private lateinit var inputImage: ImageView
    private lateinit var maskImage: ImageView

    private lateinit var modelScale: ModelScale
    private val rect = Rect()

    private val clicks = mutableListOf<ModelInput>()

    private var clickType = 1

    @SuppressLint("ClickableViewAccessibility")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        inputImage = findViewById(R.id.ivInput)
        inputImage.setImageBitmap(BitmapFactory.decodeStream(resources.openRawResource(R.raw.dogs)))

        maskImage = findViewById(R.id.ivMask)

        inputImage.setOnTouchListener { _, event ->
            when (event.action) {
                MotionEvent.ACTION_DOWN -> {
                    // Do something when the user touches the ImageView
                    inputImage.getDrawingRect(rect)
                    val matrix = inputImage.imageMatrix
                    val rectF = RectF(rect)
                    matrix.mapRect(rectF)
                    rect.set(
                        rectF.left.toInt(),
                        rectF.top.toInt(),
                        rectF.right.toInt(),
                        rectF.bottom.toInt()
                    )
                    val click = ModelInput(event.x.toInt(), (event.y - rect.top).toInt(), clickType)
                    Log.d(
                        TAG,
                        "action down (${event.x}, ${event.y}), $rect, click point:${click}"
                    )
                    if (click.x < modelScale.width && click.y < modelScale.height) {
                        Log.d(TAG, "add click to perform")
                        clicks.add(click)
                        performMaskDecoder()
                    }

                }
            }
            false
        }

        findViewById<RadioGroup>(R.id.radioGroupMask).setOnCheckedChangeListener { _, checkId ->
            when (checkId) {
                R.id.rbtnAddMask -> {
                    clickType = 1
                }

                R.id.rbtnRemoveMask -> {
                    clickType = 0
                }
            }
        }

        findViewById<Button>(R.id.btnRest).setOnClickListener {
            clicks.clear()
            maskImage.setImageBitmap(null)
        }

        prepareSession()
    }

    override fun onDestroy() {
        super.onDestroy()
        ortEnv.close()
        ortSession.close()
    }

    private fun prepareSession() {
        // Initialize Ort Session and register the onnxruntime extensions package that contains the custom operators.
        // Note: These are used to decode the input image into the format the original model requires,
        // and to encode the model output into png format
        val sessionOptions: OrtSession.SessionOptions = OrtSession.SessionOptions()
        sessionOptions.registerCustomOpLibrary(OrtxPackage.getLibraryPath())
        val modelBytes =
            resources.openRawResource(R.raw.prompt_encoder_mask_decoder_quantized).readBytes()
        ortSession = ortEnv.createSession(modelBytes, sessionOptions)

        // image: 1072x603
        val w = 1072
        val h = 603
        val samScale: Float = LONG_SIDE_LENGTH / (Math.max(w, h) * 1.0F)
        modelScale = ModelScale(
            width = w,
            height = h,
            samScale = samScale,
        )
    }

    private val LONG_SIDE_LENGTH = 1024

    private var embedding: Npy? = null

    private fun performMaskDecoder() {
        val samModel = SegmentAnything()
        if (embedding == null) {
            embedding = Npy(resources.openRawResource(R.raw.dogs_embedding))
        }

        // mock clicks
//        val (x, y) = arrayOf(542, 388)
//
//        val clicks = Array(1) {
//            ModelInput(
//                (x * modelScale!!.samScale).toInt(),
//                (y * modelScale!!.samScale).toInt(),
//                1
//            )
//        }

        val result = samModel.execute(
            embedding = embedding!!,
            modelScale = modelScale,
            clicks = clicks.toTypedArray(),
            ortEnv = ortEnv,
            ortSession = ortSession,
        )
        updateUI(result)
    }

    private fun updateUI(result: ModelResult) {
        maskImage.setImageBitmap(result.maskBitmap)
    }
}
