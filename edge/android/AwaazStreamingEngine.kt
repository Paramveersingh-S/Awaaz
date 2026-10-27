package ai.awaaz.edge

import ai.onnxruntime.*
import java.nio.FloatBuffer

/**
 * On-device streaming recognition engine wrapping ONNX Runtime Mobile.
 *
 * Implements rolling window buffer management, dynamic chunk slicing,
 * and state-preserving CTC token collapse.
 */
class AwaazStreamingEngine(
    val modelPath: String,
    val vocab: List<String>,
    val blankId: Int = 0,
    val chunkDurationMs: Int = 640,
    val leftChunks: Int = 8,
    val featureExtractor: FeatureExtractor = FeatureExtractor()
) {
    private var env: OrtEnvironment? = null
    private var session: OrtSession? = null

    // State tracking
    private var lastEmittedTokenId = -1
    private val audioBuffer = ArrayList<Float>()
    private val confirmedTokens = ArrayList<String>()

    init {
        env = OrtEnvironment.getEnvironment()
        val opts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(2)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        session = env?.createSession(modelPath, opts)
    }

    /**
     * Consumes incoming PCM audio chunk, runs inference, and returns new decoded words.
     */
    fun processChunk(pcmChunk: FloatArray): StreamingResult {
        val startTime = System.nanoTime()

        // 1. Add to rolling audio history
        for (sample in pcmChunk) {
            audioBuffer.add(sample)
        }

        // Limit maximum history to (leftChunks + 1) * chunkSize
        val maxBufferSize = 16000 * ((chunkDurationMs / 1000.0) * (leftChunks + 1)).toInt()
        if (audioBuffer.size > maxBufferSize) {
            val trimCount = audioBuffer.size - maxBufferSize
            repeat(trimCount) { audioBuffer.removeAt(0) }
        }

        // 2. Extract log-mel features
        val audioArray = FloatArray(audioBuffer.size) { audioBuffer[it] }
        val melFeatures = featureExtractor.extract(audioArray) // [T, 80]
        val totalFrames = melFeatures.size

        if (totalFrames < 4) {
            return StreamingResult("", "", 0.0f, 0L)
        }

        // 3. Prepare tensor for ONNX Runtime [1, totalFrames, 80]
        val tensorShape = longArrayOf(1L, totalFrames.toLong(), 80L)
        val flatBuffer = FloatBuffer.allocate(totalFrames * 80)
        for (t in 0 until totalFrames) {
            for (m in 0 until 80) {
                flatBuffer.put(melFeatures[t][m])
            }
        }
        flatBuffer.rewind()

        val tensor = OnnxTensor.createTensor(env, flatBuffer, tensorShape)
        val inputs = mapOf("features" to tensor)

        // 4. Run inference
        val outputs = session?.run(inputs)
        val logitsTensor = outputs?.get(0)?.value as? Array<Array<FloatArray>> // [1, subFrames, vocab]

        val newTokens = ArrayList<String>()

        if (logitsTensor != null && logitsTensor.isNotEmpty()) {
            val subFrames = logitsTensor[0].size
            // Only examine the newest chunk of frames corresponding to newest chunk
            val newestFrames = minOf(16, subFrames)
            val startFrame = subFrames - newestFrames

            for (t in startFrame until subFrames) {
                val frameLogits = logitsTensor[0][t]
                var bestId = 0
                var bestScore = -Float.MAX_VALUE
                for (v in frameLogits.indices) {
                    if (frameLogits[v] > bestScore) {
                        bestScore = frameLogits[v]
                        bestId = v
                    }
                }

                // Incremental CTC collapse across chunk boundaries
                if (bestId != lastEmittedTokenId) {
                    if (bestId != blankId && bestId in vocab.indices) {
                        val piece = vocab[bestId]
                        newTokens.add(piece)
                        confirmedTokens.add(piece)
                    }
                    lastEmittedTokenId = bestId
                }
            }
        }

        outputs?.close()
        tensor.close()

        val elapsedNs = System.nanoTime() - startTime
        val latencyMs = elapsedNs / 1_000_000L
        val chunkDurationSec = chunkDurationMs / 1000.0f
        val rtf = (latencyMs / 1000.0f) / chunkDurationSec

        val partialText = newTokens.joinToString("").replace(" ", " ").trim()
        val fullTranscript = confirmedTokens.joinToString("").replace(" ", " ").trim()

        return StreamingResult(
            partial = partialText,
            transcript = fullTranscript,
            realTimeFactor = rtf,
            latencyMs = latencyMs
        )
    }

    fun reset() {
        audioBuffer.clear()
        confirmedTokens.clear()
        lastEmittedTokenId = -1
    }

    fun close() {
        session?.close()
        env?.close()
    }

    data class StreamingResult(
        val partial: String,
        val transcript: String,
        val realTimeFactor: Float,
        val latencyMs: Long
    )
}
