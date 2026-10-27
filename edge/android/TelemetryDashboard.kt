package ai.awaaz.edge

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager

/**
 * Edge telemetry tracking real-time latency, compute factor, memory, and battery.
 */
class TelemetryDashboard(private val context: Context) {
    private var lastRecordedBatteryLevel: Float = getBatteryPercentage()
    private var lastBatteryTimestamp: Long = System.currentTimeMillis()

    /**
     * Reads current system battery percentage [0 - 100].
     */
    fun getBatteryPercentage(): Float {
        val batteryStatus: Intent? = IntentFilter(Intent.ACTION_BATTERY_CHANGED).let { filter ->
            context.registerReceiver(null, filter)
        }
        val level = batteryStatus?.getIntExtra(BatteryManager.EXTRA_LEVEL, -1) ?: -1
        val scale = batteryStatus?.getIntExtra(BatteryManager.EXTRA_SCALE, -1) ?: -1
        return if (level >= 0 && scale > 0) (level * 100.0f / scale) else 100.0f
    }

    /**
     * Estimates battery discharge rate in percent per hour (%/hr).
     */
    fun getDischargeRatePerHour(): Float {
        val currentLevel = getBatteryPercentage()
        val currentTime = System.currentTimeMillis()
        val elapsedHours = (currentTime - lastBatteryTimestamp) / (1000.0f * 3600.0f)

        return if (elapsedHours > 0.001f) {
            val delta = maxOf(0.0f, lastRecordedBatteryLevel - currentLevel)
            delta / elapsedHours
        } else {
            0.0f
        }
    }

    /**
     * Reads current JVM heap memory allocated by Awaaz in MB.
     */
    fun getMemoryUsageMb(): Float {
        val runtime = Runtime.getRuntime()
        val usedMem = runtime.totalMemory() - runtime.freeMemory()
        return usedMem / (1024.0f * 1024.0f)
    }

    /**
     * Produces a snapshot report of all on-device edge telemetry metrics.
     */
    fun createSnapshot(rtf: Float, latencyMs: Long): TelemetrySnapshot {
        return TelemetrySnapshot(
            realTimeFactor = rtf,
            latencyMs = latencyMs,
            memoryMb = getMemoryUsageMb(),
            batteryPercent = getBatteryPercentage(),
            dischargeRatePerHour = getDischargeRatePerHour()
        )
    }

    data class TelemetrySnapshot(
        val realTimeFactor: Float,
        val latencyMs: Long,
        val memoryMb: Float,
        val batteryPercent: Float,
        val dischargeRatePerHour: Float
    )
}
