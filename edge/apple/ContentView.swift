import SwiftUI

public struct ContentView: View {
    @State private var isRecording = false
    @State private var transcriptText = "Tap microphone to begin streaming..."
    @State private var partialText = ""
    @State private var rtfValue: Double = 0.0
    @State private var latencyValue: Double = 0.0

    public init() {}

    public var body: some View {
        VStack(spacing: 16) {
            // Header
            Text("Awaaz Edge ASR")
                .font(.title2)
                .fontWeight(.bold)
                .foregroundColor(.blue)

            // Sacred-Text Policy Notice
            Text("⚠️ Sacred-Text Policy (§1.5): Devotional and liturgical texts strictly excluded.")
                .font(.caption2)
                .padding(8)
                .background(Color.yellow.opacity(0.2))
                .cornerRadius(6)

            // Transcript Box
            ScrollView {
                VStack(alignment: .leading, spacing: 4) {
                    Text(transcriptText)
                        .font(.body)
                        .foregroundColor(.primary)

                    if !partialText.isEmpty {
                        Text(partialText)
                            .font(.body)
                            .foregroundColor(.blue)
                            .bold()
                    }
                }
                .padding()
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(maxHeight: 250)
            .background(Color(.systemGray6))
            .cornerRadius(10)

            // Telemetry Cards
            HStack(spacing: 20) {
                VStack {
                    Text("RTF")
                        .font(.caption)
                        .foregroundColor(.gray)
                    Text(String(format: "%.2fx", rtfValue))
                        .font(.headline)
                }

                VStack {
                    Text("Latency")
                        .font(.caption)
                        .foregroundColor(.gray)
                    Text(String(format: "%.0f ms", latencyValue))
                        .font(.headline)
                }
            }
            .padding(.vertical, 8)

            // Record Button
            Button(action: {
                isRecording.toggle()
                if !isRecording {
                    partialText = ""
                }
            }) {
                HStack {
                    Image(systemName: isRecording ? "stop.circle.fill" : "mic.circle.fill")
                        .font(.title)
                    Text(isRecording ? "Stop Recording" : "Start Live Streaming")
                        .fontWeight(.semibold)
                }
                .frame(maxWidth: .infinity)
                .padding()
                .background(isRecording ? Color.red : Color.blue)
                .foregroundColor(.white)
                .cornerRadius(12)
            }
        }
        .padding()
    }
}
