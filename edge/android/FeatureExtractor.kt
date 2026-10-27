package ai.awaaz.edge

import kotlin.math.*

/**
 * Native Android feature extractor implementing docs/FEATURES.md.
 *
 * Computes 80-bin log-mel filterbanks from 16 kHz PCM audio with centered
 * reflection padding, 25ms periodic Hann window, 10ms hop, and CMVN normalization.
 */
class FeatureExtractor(
    val sampleRate: Int = 16000,
    val numMels: Int = 80,
    val frameLengthSamples: Int = 400, // 25ms
    val hopLengthSamples: Int = 160,   // 10ms
    val cmvnMean: FloatArray? = null,
    val cmvnStd: FloatArray? = null
) {
    private val fftSize = 512
    private val halfFft = fftSize / 2 + 1
    private val window: FloatArray = FloatArray(frameLengthSamples)
    private val melFilterbank: Array<FloatArray>

    init {
        // Periodic Hann window: w[n] = 0.5 * (1 - cos(2*pi*n / N))
        for (i in 0 until frameLengthSamples) {
            window[i] = (0.5 * (1.0 - cos(2.0 * PI * i / frameLengthSamples))).toFloat()
        }
        melFilterbank = createMelFilterbank(sampleRate, fftSize, numMels, 20.0f, 7600.0f)
    }

    /**
     * Converts Hz to Mel scale using HTK formula.
     */
    private fun hzToMel(hz: Float): Float = 2595.0f * log10(1.0f + hz / 700.0f)

    /**
     * Converts Mel scale to Hz.
     */
    private fun melToHz(mel: Float): Float = 700.0f * (10.0f.pow(mel / 2595.0f) - 1.0f)

    private fun createMelFilterbank(
        sr: Int,
        nFft: Int,
        nMels: Int,
        fMin: Float,
        fMax: Float
    ): Array<FloatArray> {
        val minMel = hzToMel(fMin)
        val maxMel = hzToMel(fMax)
        val melPoints = FloatArray(nMels + 2)
        val melStep = (maxMel - minMel) / (nMels + 1)

        for (i in 0 until nMels + 2) {
            melPoints[i] = minMel + i * melStep
        }

        val binPoints = IntArray(nMels + 2)
        for (i in 0 until nMels + 2) {
            val hz = melToHz(melPoints[i])
            binPoints[i] = floor((nFft + 1) * hz / sr).toInt()
        }

        val filters = Array(nMels) { FloatArray(halfFft) }
        for (m in 0 until nMels) {
            val left = binPoints[m]
            val center = binPoints[m + 1]
            val right = binPoints[m + 2]

            for (k in left until center) {
                if (k in 0 until halfFft && center > left) {
                    filters[m][k] = (k - left).toFloat() / (center - left)
                }
            }
            for (k in center until right) {
                if (k in 0 until halfFft && right > center) {
                    filters[m][k] = (right - k).toFloat() / (right - center)
                }
            }
        }
        return filters
    }

    /**
     * Extracts log-mel features from raw 1D float PCM audio with centered reflection padding.
     *
     * @return 2D FloatArray of shape [numFrames, numMels]
     */
    fun extract(audio: FloatArray): Array<FloatArray> {
        val pad = frameLengthSamples / 2
        val paddedAudio = FloatArray(audio.size + 2 * pad)

        // Reflection padding
        for (i in 0 until pad) {
            paddedAudio[i] = audio[min(audio.size - 1, pad - i)]
        }
        for (i in audio.indices) {
            paddedAudio[pad + i] = audio[i]
        }
        for (i in 0 until pad) {
            paddedAudio[pad + audio.size + i] = audio[max(0, audio.size - 2 - i)]
        }

        val numFrames = (paddedAudio.size - frameLengthSamples) / hopLengthSamples + 1
        val features = Array(numFrames) { FloatArray(numMels) }

        val fftReal = FloatArray(fftSize)
        val fftImag = FloatArray(fftSize)

        for (t in 0 until numFrames) {
            val offset = t * hopLengthSamples
            // Windowing
            for (i in 0 until fftSize) {
                if (i < frameLengthSamples) {
                    fftReal[i] = paddedAudio[offset + i] * window[i]
                } else {
                    fftReal[i] = 0.0f
                }
                fftImag[i] = 0.0f
            }

            // Power spectrum via DFT
            val powerSpec = FloatArray(halfFft)
            for (k in 0 until halfFft) {
                var realSum = 0.0f
                var imagSum = 0.0f
                val angleFactor = -2.0 * PI * k / fftSize
                for (n in 0 until frameLengthSamples) {
                    val angle = angleFactor * n
                    val sample = fftReal[n]
                    realSum += sample * cos(angle).toFloat()
                    imagSum += sample * sin(angle).toFloat()
                }
                powerSpec[k] = realSum * realSum + imagSum * imagSum
            }

            // Mel filterbank multiplication & natural log floor 1e-5
            for (m in 0 until numMels) {
                var energy = 0.0f
                val filter = melFilterbank[m]
                for (k in 0 until halfFft) {
                    energy += powerSpec[k] * filter[k]
                }
                val logMel = ln(max(energy, 1e-5f))

                // Apply Global CMVN if stats provided
                var normalized = logMel
                if (cmvnMean != null && cmvnStd != null) {
                    normalized = (normalized - cmvnMean[m]) / (cmvnStd[m] + 1e-8f)
                }
                features[t][m] = normalized
            }
        }

        return features
    }

    /**
     * Self-test method validating feature parity against Python reference vectors.
     *
     * @param referenceFeatures Expected 2D float array from Python export test kit.
     * @param tolerance Maximum absolute difference allowable (target <= 1e-3).
     * @return ParityResult with status and measured delta.
     */
    fun verifyFeatureParity(
        testAudio: FloatArray,
        referenceFeatures: Array<FloatArray>,
        tolerance: Float = 1e-3f
    ): ParityResult {
        val extracted = extract(testAudio)
        var maxDiff = 0.0f

        val frames = min(extracted.size, referenceFeatures.size)
        for (t in 0 until frames) {
            for (m in 0 until numMels) {
                val diff = abs(extracted[t][m] - referenceFeatures[t][m])
                if (diff > maxDiff) {
                    maxDiff = diff
                }
            }
        }

        val passed = maxDiff <= tolerance
        return ParityResult(passed = passed, maxDifference = maxDiff, frameCount = frames)
    }

    data class ParityResult(val passed: Boolean, val maxDifference: Float, val frameCount: Int)
}
