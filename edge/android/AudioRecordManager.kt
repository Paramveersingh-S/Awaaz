package ai.awaaz.edge

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn

/**
 * Manages low-latency real-time 16 kHz mono audio capture via Android AudioRecord.
 */
class AudioRecordManager(
    val sampleRate: Int = 16000,
    val chunkDurationMs: Int = 640
) {
    private val channelConfig = AudioFormat.CHANNEL_IN_MONO
    private val audioFormat = AudioFormat.ENCODING_PCM_16BIT
    private val chunkSizeSamples = (sampleRate * (chunkDurationMs / 1000.0)).toInt()

    private var audioRecord: AudioRecord? = null
    private var isRecording = false

    @SuppressLint("MissingPermission")
    fun startRecordingFlow(): Flow<FloatArray> = flow {
        val minBufferSize = AudioRecord.getMinBufferSize(sampleRate, channelConfig, audioFormat)
        val bufferSize = maxOf(minBufferSize, chunkSizeSamples * 2)

        audioRecord = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            sampleRate,
            channelConfig,
            audioFormat,
            bufferSize
        )

        val shortBuffer = ShortArray(chunkSizeSamples)
        val floatBuffer = FloatArray(chunkSizeSamples)

        audioRecord?.startRecording()
        isRecording = true

        try {
            while (isRecording && currentCoroutineContext().isActive) {
                var samplesRead = 0
                while (samplesRead < chunkSizeSamples && isRecording) {
                    val read = audioRecord?.read(
                        shortBuffer,
                        samplesRead,
                        chunkSizeSamples - samplesRead
                    ) ?: -1

                    if (read > 0) {
                        samplesRead += read
                    } else {
                        break
                    }
                }

                if (samplesRead == chunkSizeSamples) {
                    // Convert 16-bit PCM to normalized float [-1.0, 1.0]
                    for (i in 0 until chunkSizeSamples) {
                        floatBuffer[i] = shortBuffer[i] / 32768.0f
                    }
                    emit(floatBuffer.copyOf())
                }
            }
        } finally {
            stopRecording()
        }
    }.flowOn(Dispatchers.IO)

    fun stopRecording() {
        isRecording = false
        try {
            audioRecord?.stop()
            audioRecord?.release()
        } catch (_: Exception) {
        } finally {
            audioRecord = null
        }
    }
}
