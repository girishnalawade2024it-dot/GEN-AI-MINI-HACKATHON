package com.scamshield.app.service

import android.app.Notification
import android.service.notification.NotificationListenerService
import android.service.notification.StatusBarNotification
import android.util.Log
import com.scamshield.app.ml.ScamClassifier
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch

/**
 * Intercepts notifications from external messaging apps (WhatsApp, Telegram, Messenger, etc.),
 * evaluates message content with on-device Transformer model, and launches Truecaller alert.
 */
class ScamNotificationListener : NotificationListenerService() {

    companion object {
        private const val TAG = "ScamNotificationListener"

        // Targeted messaging apps to inspect
        private val MONITORED_PACKAGES = setOf(
            "com.whatsapp",
            "com.whatsapp.w4b",
            "org.telegram.messenger",
            "org.telegram.plus",
            "org.thoughtcrime.securesms", // Signal
            "com.facebook.orca",          // Messenger
            "com.google.android.apps.messaging",
            "com.samsung.android.messaging",
            "com.viber.voip"
        )
    }

    private val serviceScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)

    override fun onNotificationPosted(sbn: StatusBarNotification?) {
        super.onNotificationPosted(sbn)
        if (sbn == null) return

        val packageName = sbn.packageName
        if (!MONITORED_PACKAGES.contains(packageName)) {
            return
        }

        val notification = sbn.notification ?: return
        val extras = notification.extras ?: return

        // Ignore group summary notifications
        val isGroupSummary = (notification.flags and Notification.FLAG_GROUP_SUMMARY) != 0
        if (isGroupSummary) {
            return
        }

        val title = extras.getCharSequence(Notification.EXTRA_TITLE)?.toString() ?: "Unknown Contact"
        val text = extras.getCharSequence(Notification.EXTRA_TEXT)?.toString() ?: ""
        val bigText = extras.getCharSequence(Notification.EXTRA_BIG_TEXT)?.toString() ?: ""

        val candidateText = if (bigText.length > text.length) bigText else text
        if (candidateText.isBlank() || candidateText.length < 5) {
            return
        }

        serviceScope.launch {
            try {
                val classifier = ScamClassifier.getInstance(applicationContext)
                val result = classifier.classify(candidateText, threshold = ScamClassifier.DEFAULT_SCAM_THRESHOLD)

                if (result.isScam) {
                    Log.w(
                        TAG,
                        "🚨 SCAM INTERCEPTED via $packageName from '$title'! Confidence: ${result.percentageString}"
                    )

                    // Launch Truecaller-style overlay alert
                    OverlayService.startOverlay(
                        context = applicationContext,
                        sender = title,
                        messageText = candidateText,
                        scamProbability = result.confidence,
                        source = getAppDisplayName(packageName)
                    )
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error evaluating incoming messaging notification", e)
            }
        }
    }

    private fun getAppDisplayName(packageName: String): String {
        return when (packageName) {
            "com.whatsapp", "com.whatsapp.w4b" -> "WhatsApp"
            "org.telegram.messenger", "org.telegram.plus" -> "Telegram"
            "org.thoughtcrime.securesms" -> "Signal"
            "com.facebook.orca" -> "Messenger"
            else -> "Messaging App"
        }
    }
}
