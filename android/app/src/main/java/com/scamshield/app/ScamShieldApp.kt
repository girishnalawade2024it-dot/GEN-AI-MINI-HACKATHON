package com.scamshield.app

import android.app.Application
import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.Context
import android.os.Build
import android.util.Log
import com.scamshield.app.ml.ScamClassifier
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch

class ScamShieldApp : Application() {

    private val applicationScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)

    companion object {
        const val CHANNEL_ALERT_ID = "scamshield_alert_channel"
        const val CHANNEL_SERVICE_ID = "scamshield_service_channel"
        private const val TAG = "ScamShieldApp"

        @Volatile
        private var instance: ScamShieldApp? = null

        fun getInstance(): ScamShieldApp {
            return instance ?: synchronized(this) {
                instance ?: throw IllegalStateException("ScamShieldApp has not been initialized yet")
            }
        }
    }

    override fun onCreate() {
        super.onCreate()
        instance = this
        createNotificationChannels()

        // Eagerly warm up the on-device ONNX runtime engine in background
        applicationScope.launch {
            try {
                Log.d(TAG, "Initializing on-device ScamClassifier engine...")
                ScamClassifier.getInstance(applicationContext).warmUp()
                Log.d(TAG, "ScamClassifier engine warmed up successfully.")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to warm up ScamClassifier", e)
            }
        }
    }

    private fun createNotificationChannels() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val alertChannel = NotificationChannel(
                CHANNEL_ALERT_ID,
                "Scam Warnings & Overlays",
                NotificationManager.IMPORTANCE_HIGH
            ).apply {
                description = "Urgent alerts when a scam message or smishing attempt is intercepted"
                enableVibration(true)
            }

            val serviceChannel = NotificationChannel(
                CHANNEL_SERVICE_ID,
                "ScamShield Active Protection",
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = "Background protection service status"
            }

            val notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            notificationManager.createNotificationChannel(alertChannel)
            notificationManager.createNotificationChannel(serviceChannel)
        }
    }
}
