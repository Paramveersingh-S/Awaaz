import AVFoundation
import Foundation

/**
 * Manages real-time audio input capture via AVAudioEngine on iOS and watchOS.
 * Taps audio hardware at 16 kHz mono Float32 format.
 */
public class AudioEngineManager: NSObject {
    private let audioEngine = AVAudioEngine()
    private let targetSampleRate: Double = 16000.0
    public var onAudioChunk: (([Float]) -> Void)?

    public override init() {
        super.init()
    }

    public func startRecording(chunkDurationMs: Int = 640) throws {
        let inputNode = audioEngine.inputNode
        let inputFormat = inputNode.outputFormat(forBus: 0)

        // Target 16kHz mono format
        guard let recordingFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: targetSampleRate,
            channels: 1,
            interleaved: false
        ) else {
            throw NSError(domain: "ai.awaaz.edge", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to create AVAudioFormat"])
        }

        guard let formatConverter = AVAudioConverter(from: inputFormat, to: recordingFormat) else {
            throw NSError(domain: "ai.awaaz.edge", code: 2, userInfo: [NSLocalizedDescriptionKey: "Failed to create AVAudioConverter"])
        }

        let chunkSizeSamples = AVAudioFrameCount((targetSampleRate * Double(chunkDurationMs) / 1000.0))

        inputNode.installTap(onBus: 0, bufferSize: chunkSizeSamples, format: inputFormat) { [weak self] (buffer, time) in
            guard let self = self else { return }

            let capacityRatio = self.targetSampleRate / inputFormat.sampleRate
            let convertedCapacity = AVAudioFrameCount(Double(buffer.frameLength) * capacityRatio)
            guard let convertedBuffer = AVAudioPCMBuffer(pcmFormat: recordingFormat, frameCapacity: convertedCapacity) else { return }

            var error: NSError?
            formatConverter.convert(to: convertedBuffer, error: &error) { inNumPackets, outStatus in
                outStatus.pointee = .haveData
                return buffer
            }

            if error == nil, let channelData = convertedBuffer.floatChannelData?[0] {
                let frameLength = Int(convertedBuffer.frameLength)
                let floatArray = Array(UnsafeBufferPointer(start: channelData, count: frameLength))
                self.onAudioChunk?(floatArray)
            }
        }

        audioEngine.prepare()
        try audioEngine.start()
    }

    public func stopRecording() {
        audioEngine.inputNode.removeTap(onBus: 0)
        audioEngine.stop()
    }
}
