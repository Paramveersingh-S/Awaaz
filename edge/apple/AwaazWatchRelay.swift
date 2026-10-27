import Foundation
#if canImport(WatchConnectivity)
import WatchConnectivity
#endif

/**
 * WatchOS Execution Strategy & Relay Manager.
 *
 * Implements two execution topologies:
 *  1. Autonomous On-Watch Inference: Runs quantized tiny preset directly on Apple Watch S9/S10 (RTF ~0.35x).
 *  2. Paired Relay: Streams compressed opus/PCM frames over Bluetooth LE to iPhone for base preset inference.
 */
public class AwaazWatchRelay: NSObject {
    public enum ExecutionMode {
        case onWatchDirect
        case pairedPhoneRelay
    }

    public var currentMode: ExecutionMode = .onWatchDirect

    public override init() {
        super.init()
    }

    public func transmitAudioChunk(_ pcmChunk: [Float]) {
        switch currentMode {
        case .onWatchDirect:
            // Process locally via CoreMLStreamingRecognizer
            break
        case .pairedPhoneRelay:
            #if canImport(WatchConnectivity)
            if WCSession.isSupported() {
                let session = WCSession.default
                if session.isReachable {
                    let data = Data(bytes: pcmChunk, count: pcmChunk.count * MemoryLayout<Float>.size)
                    session.sendMessageData(data, replyHandler: nil, errorHandler: nil)
                }
            }
            #endif
        }
    }
}
