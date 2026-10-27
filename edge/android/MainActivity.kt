package ai.awaaz.edge

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {

    private var audioManager: AudioRecordManager? = null
    private var streamingEngine: AwaazStreamingEngine? = null
    private var telemetry: TelemetryDashboard? = null

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted: Boolean ->
        // Handle microphone permission grant
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        audioManager = AudioRecordManager()
        telemetry = TelemetryDashboard(this)

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED
        ) {
            requestPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
        }

        setContent {
            AwaazEdgeScreen()
        }
    }

    @OptIn(ExperimentalMaterial3Api::class)
    @Composable
    fun AwaazEdgeScreen() {
        val coroutineScope = rememberCoroutineScope()
        var isStreaming by remember { mutableStateOf(false) }
        var transcript by remember { mutableStateOf("Press 'Start Live Streaming' to begin...") }
        var currentPartial by remember { mutableStateOf("") }
        var rtf by remember { mutableFloatStateOf(0.0f) }
        var latencyMs by remember { mutableLongStateOf(0L) }
        var ramMb by remember { mutableFloatStateOf(0.0f) }
        var batteryPct by remember { mutableFloatStateOf(100.0f) }

        Scaffold(
            topBar = {
                TopAppBar(
                    title = { Text("Awaaz Edge ASR", fontWeight = FontWeight.Bold) },
                    colors = TopAppBarDefaults.topAppBarColors(
                        containerColor = Color(0xFF1E3A8A),
                        titleContentColor = Color.White
                    )
                )
            }
        ) { padding ->
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(padding)
                    .padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                // Sacred-Text Policy Callout Card
                Card(
                    colors = CardDefaults.cardColors(containerColor = Color(0xFFFEF3C7)),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text(
                        text = "⚠️ Sacred-Text Policy (§1.5): Awaaz is designed for everyday conversation. It excludes Gurbani and devotional texts.",
                        fontSize = 12.sp,
                        color = Color(0xFF92400E),
                        modifier = Modifier.padding(12.dp)
                    )
                }

                // Live Streaming Transcript Box
                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .weight(1.0f)
                ) {
                    LazyColumn(
                        modifier = Modifier
                            .fillMaxSize()
                            .padding(16.dp)
                    ) {
                        item {
                            Text(
                                text = transcript,
                                fontSize = 18.sp,
                                lineHeight = 26.sp,
                                fontWeight = FontWeight.Medium
                            )
                        }
                        if (currentPartial.isNotEmpty()) {
                            item {
                                Text(
                                    text = " $currentPartial",
                                    fontSize = 18.sp,
                                    lineHeight = 26.sp,
                                    color = Color(0xFF2563EB),
                                    fontWeight = FontWeight.Bold
                                )
                            }
                        }
                    }
                }

                // Telemetry Metrics Grid
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(containerColor = Color(0xFFF1F5F9))
                ) {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(12.dp),
                        horizontalArrangement = Arrangement.SpaceAround
                    ) {
                        MetricItem("RTF", String.format("%.2fx", rtf))
                        MetricItem("Latency", "${latencyMs} ms")
                        MetricItem("RAM", String.format("%.1f MB", ramMb))
                        MetricItem("Battery", String.format("%.0f%%", batteryPct))
                    }
                }

                // Control Button
                Button(
                    onClick = {
                        isStreaming = !isStreaming
                        if (isStreaming) {
                            coroutineScope.launch {
                                audioManager?.startRecordingFlow()?.collect { pcmChunk ->
                                    val result = streamingEngine?.processChunk(pcmChunk)
                                    if (result != null) {
                                        transcript = result.transcript
                                        currentPartial = result.partial
                                        rtf = result.realTimeFactor
                                        latencyMs = result.latencyMs

                                        ramMb = telemetry?.getMemoryUsageMb() ?: 0.0f
                                        batteryPct = telemetry?.getBatteryPercentage() ?: 100.0f
                                    }
                                }
                            }
                        } else {
                            audioManager?.stopRecording()
                            currentPartial = ""
                        }
                    },
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(52.dp),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = if (isStreaming) Color(0xFFDC2626) else Color(0xFF1E3A8A)
                    )
                ) {
                    Text(
                        text = if (isStreaming) "⏹ Stop Recognition" else "🎙️ Start Live Streaming",
                        fontSize = 16.sp,
                        fontWeight = FontWeight.Bold
                    )
                }
            }
        }
    }

    @Composable
    fun MetricItem(label: String, value: String) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Text(text = label, fontSize = 11.sp, color = Color.Gray)
            Text(text = value, fontSize = 14.sp, fontWeight = FontWeight.Bold, color = Color.Black)
        }
    }
}
