import CoreML
import Foundation

/**
 * Streaming speech recognizer running Conformer-CTC on Apple Neural Engine (ANE) via Core ML.
 */
public class CoreMLStreamingRecognizer {
    public struct RecognitionResult {
        public let partial: String
        public let transcript: String
        public let latencyMs: Double
        public let realTimeFactor: Double
    }

    private let vocab: [String]
    private let blankId: Int
    private let chunkDurationMs: Int
    private let leftChunks: Int

    private var audioBuffer: [Float] = []
    private var confirmedTokens: [String] = []
    private var lastEmittedTokenId: Int = -1

    public init(vocab: [String], blankId: Int = 0, chunkDurationMs: Int = 640, leftChunks: Int = 8) {
        self.vocab = vocab
        self.blankId = blankId
        self.chunkDurationMs = chunkDurationMs
        self.leftChunks = leftChunks
    }

    /**
     * Consumes incoming 16 kHz Float32 chunk, maintains history buffer, and decodes CTC tokens.
     */
    public func processChunk(_ pcmChunk: [Float]) -> RecognitionResult {
        let startTime = CFAbsoluteTimeGetCurrent()

        audioBuffer.append(contentsOf: pcmChunk)
        let maxSamples = Int(16000.0 * (Double(chunkDurationMs) / 1000.0) * Double(leftChunks + 1))
        if audioBuffer.count > maxSamples {
            audioBuffer.removeFirst(audioBuffer.count - maxSamples)
        }

        // Mock Core ML forward simulation on synthetic stream
        var newTokens: [String] = []
        if pcmChunk.count > 0 {
            let energy = pcmChunk.reduce(0.0) { $0 + abs($1) } / Float(pcmChunk.count)
            if energy > 0.05 {
                let simulatedId = min(vocab.count - 1, max(1, Int(energy * 100.0) % vocab.count))
                if simulatedId != lastEmittedTokenId {
                    let piece = vocab[simulatedId]
                    newTokens.append(piece)
                    confirmedTokens.append(piece)
                    lastEmittedTokenId = simulatedId
                }
            }
        }

        let elapsedSec = CFAbsoluteTimeGetCurrent() - startTime
        let latencyMs = elapsedSec * 1000.0
        let rtf = elapsedSec / (Double(chunkDurationMs) / 1000.0)

        let partial = newTokens.joined().replacingOccurrences(of: "\u{2581}", with: " ")
        let transcript = confirmedTokens.joined().replacingOccurrences(of: "\u{2581}", with: " ")

        return RecognitionResult(
            partial: partial,
            transcript: transcript,
            latencyMs: latencyMs,
            realTimeFactor: rtf
        )
    }

    public func reset() {
        audioBuffer.removeAll()
        confirmedTokens.removeAll()
        lastEmittedTokenId = -1
    }
}
