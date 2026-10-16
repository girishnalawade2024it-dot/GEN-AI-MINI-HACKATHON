package com.scamshield.app.receiver

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.provider.Telephony
import android.util.Log
import com.scamshield.app.ml.ScamClassifier
import com.scamshield.app.service.OverlayService
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch

/**
 * High-priority SMS Broadcast Receiver.
 * Intercepts incoming SMS, concatenates multi-part messages, runs on-device
 * quantized Transformer inference, and triggers Truecaller-style overlay if confidence >= 85%.
 */
class SmsBroadcastReceiver : BroadcastReceiver() {

    companion object {
        private const val TAG = "SmsBroadcastReceiver"
    }

    private val receiverScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)

    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action != Telephony.Sms.Intents.SMS_RECEIVED_ACTION) {
            return
        }

        // Keep broadcast alive while performing asynchronous ML inference
        val pendingResult = goAsync()

        receiverScope.launch {
            try {
                val messages = Telephony.Sms.Intents.getMessagesFromIntent(intent)
                if (messages.isNullOrEmpty()) {
                    Log.w(TAG, "No SMS messages extracted from intent")
                    return@launch
                }

                // Group multi-part SMS messages by originating sender
                val senderToBodyMap = HashMap<String, StringBuilder>()
                for (sms in messages) {
                    val sender = sms.displayOriginatingAddress ?: sms.originatingAddress ?: "Unknown Sender"
                    val body = sms.displayMessageBody ?: sms.messageBody ?: ""

                    if (!senderToBodyMap.containsKey(sender)) {
                        senderToBodyMap[sender] = StringBuilder()
                    }
                    senderToBodyMap[sender]?.append(body)
                }

                val classifier = ScamClassifier.getInstance(context)

                for ((sender, bodyBuilder) in senderToBodyMap) {
                    val fullMessage = bodyBuilder.toString().trim()
                    if (fullMessage.isEmpty()) continue

                    Log.d(TAG, "Intercepted SMS from '$sender': ${fullMessage.take(40)}...")

                    val result = classifier.classify(
                        text = fullMessage,
                        threshold = ScamClassifier.DEFAULT_SCAM_THRESHOLD
                    )

                    if (result.isScam) {
                        Log.w(
                            TAG,
                            "⚠️ SCAM DETECTED! Sender: $sender, Prob: ${result.percentageString}, Latency: ${result.latencyMs}ms"
                        )

                        // Trigger Truecaller-style overlay
                        OverlayService.startOverlay(
                            context = context,
                            sender = sender,
                            messageText = fullMessage,
                            scamProbability = result.confidence,
                            source = "SMS"
                        )

                        // If allowed by OS permissions, abort further broadcast dispatch
                        try {
                            abortBroadcast()
                        } catch (e: Exception) {
                            Log.d(TAG, "abortBroadcast not supported or not default SMS app: ${e.message}")
                        }
                    } else {
                        Log.d(TAG, "Message classified as SAFE HAM (Confidence: ${result.percentageString})")
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error processing incoming SMS in broadcast receiver", e)
            } finally {
                pendingResult.finish()
            }
        }
    }
}
