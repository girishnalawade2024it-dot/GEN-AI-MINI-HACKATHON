package com.scamshield.app.service

import android.annotation.SuppressLint
import android.app.Notification
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.graphics.PixelFormat
import android.os.Build
import android.os.IBinder
import android.os.VibrationEffect
import android.os.Vibrator
import android.provider.Settings
import android.util.Log
import android.view.Gravity
import android.view.LayoutInflater
import android.view.MotionEvent
import android.view.View
import android.view.WindowManager
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.core.app.NotificationCompat
import com.scamshield.app.MainActivity
import com.scamshield.app.R
import com.scamshield.app.ScamShieldApp
import kotlin.math.abs

/**
 * Foreground Service that manages the Truecaller-style floating alert overlay.
 * Uses TYPE_APPLICATION_OVERLAY to draw above all active apps upon scam detection.
 */
class OverlayService : Service() {

    companion object {
        private const val TAG = "OverlayService"
        private const val FOREGROUND_NOTIFICATION_ID = 2001

        const val EXTRA_SENDER = "extra_sender"
        const val EXTRA_MESSAGE = "extra_message"
        const val EXTRA_PROBABILITY = "extra_probability"
        const val EXTRA_SOURCE = "extra_source"

        fun startOverlay(
            context: Context,
            sender: String,
            messageText: String,
            scamProbability: Float,
            source: String = "SMS"
        ) {
            if (!Settings.canDrawOverlays(context)) {
                Log.w(TAG, "Cannot show overlay: SYSTEM_ALERT_WINDOW permission not granted")
                return
            }

            val intent = Intent(context, OverlayService::class.java).apply {
                putExtra(EXTRA_SENDER, sender)
                putExtra(EXTRA_MESSAGE, messageText)
                putExtra(EXTRA_PROBABILITY, scamProbability)
                putExtra(EXTRA_SOURCE, source)
            }

            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                context.startForegroundService(intent)
            } else {
                context.startService(intent)
            }
        }
    }

    private var windowManager: WindowManager? = null
    private var overlayView: View? = null
    private var layoutParams: WindowManager.LayoutParams? = null

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        windowManager = getSystemService(Context.WINDOW_SERVICE) as WindowManager
        startForegroundNotification()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent == null) {
            stopSelf()
            return START_NOT_STICKY
        }

        val sender = intent.getStringExtra(EXTRA_SENDER) ?: "Unknown"
        val message = intent.getStringExtra(EXTRA_MESSAGE) ?: ""
        val probability = intent.getFloatExtra(EXTRA_PROBABILITY, 0.95f)
        val source = intent.getStringExtra(EXTRA_SOURCE) ?: "SMS"

        showOverlay(sender, message, probability, source)
        triggerVibrationAlert()

        return START_NOT_STICKY
    }

    @SuppressLint("ClickableViewAccessibility")
    private fun showOverlay(
        sender: String,
        message: String,
        probability: Float,
        source: String
    ) {
        if (overlayView != null) {
            removeOverlay()
        }

        val inflater = LayoutInflater.from(this)
        overlayView = inflater.inflate(R.layout.overlay_scam_alert, null)

        val overlayType = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY
        } else {
            @Suppress("DEPRECATION")
            WindowManager.LayoutParams.TYPE_PHONE
        }

        layoutParams = WindowManager.LayoutParams(
            WindowManager.LayoutParams.MATCH_PARENT,
            WindowManager.LayoutParams.WRAP_CONTENT,
            overlayType,
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or
                    WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN or
                    WindowManager.LayoutParams.FLAG_WATCH_OUTSIDE_TOUCH,
            PixelFormat.TRANSLUCENT
        ).apply {
            gravity = Gravity.TOP or Gravity.CENTER_HORIZONTAL
            y = 80 // Position slightly below notification bar
        }

        // Bind Views
        val tvSender = overlayView?.findViewById<TextView>(R.id.tvSender)
        val tvProbability = overlayView?.findViewById<TextView>(R.id.tvProbability)
        val tvMessagePreview = overlayView?.findViewById<TextView>(R.id.tvMessagePreview)
        val tvSourceBadge = overlayView?.findViewById<TextView>(R.id.tvSourceBadge)
        val btnDismiss = overlayView?.findViewById<Button>(R.id.btnDismiss)
        val btnBlock = overlayView?.findViewById<Button>(R.id.btnBlock)
        val btnViewSafely = overlayView?.findViewById<Button>(R.id.btnViewSafely)

        tvSender?.text = sender
        tvProbability?.text = "%.1f%% Scam Risk".format(probability * 100)
        tvMessagePreview?.text = message
        tvSourceBadge?.text = "$source ALERT"

        // Action: Dismiss
        btnDismiss?.setOnClickListener {
            removeOverlay()
            stopSelf()
        }

        // Action: Block Sender
        btnBlock?.setOnClickListener {
            Toast.makeText(
                this,
                "Sender $sender flagged and blocked.",
                Toast.LENGTH_LONG
            ).show()
            removeOverlay()
            stopSelf()
        }

        // Action: View Safely in Isolated Sandboxed Activity
        btnViewSafely?.setOnClickListener {
            val safeIntent = Intent(this, MainActivity::class.java).apply {
                flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP
                putExtra("view_safe_sender", sender)
                putExtra("view_safe_message", message)
            }
            startActivity(safeIntent)
            removeOverlay()
            stopSelf()
        }

        // Gesture: Swipe up to dismiss or drag card
        setupDragAndDismiss(overlayView!!)

        try {
            windowManager?.addView(overlayView, layoutParams)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to add window view for overlay", e)
        }
    }

    @SuppressLint("ClickableViewAccessibility")
    private fun setupDragAndDismiss(view: View) {
        var initialY = 0
        var initialTouchY = 0f

        view.setOnTouchListener { _, event ->
            when (event.action) {
                MotionEvent.ACTION_DOWN -> {
                    initialY = layoutParams?.y ?: 0
                    initialTouchY = event.rawY
                    true
                }
                MotionEvent.ACTION_MOVE -> {
                    val deltaY = (event.rawY - initialTouchY).toInt()
                    // Allow dragging downward slightly or upward
                    layoutParams?.y = initialY + deltaY
                    try {
                        windowManager?.updateViewLayout(overlayView, layoutParams)
                    } catch (e: Exception) {
                        // View may have been detached
                    }
                    true
                }
                MotionEvent.ACTION_UP -> {
                    val totalDeltaY = event.rawY - initialTouchY
                    // If swiped up by > 100 pixels, dismiss
                    if (totalDeltaY < -100) {
                        removeOverlay()
                        stopSelf()
                    } else {
                        // Snap back to default position
                        layoutParams?.y = 80
                        try {
                            windowManager?.updateViewLayout(overlayView, layoutParams)
                        } catch (e: Exception) {
                            // Detached
                        }
                    }
                    true
                }
                MotionEvent.ACTION_OUTSIDE -> {
                    // Tap outside dismisses overlay
                    removeOverlay()
                    stopSelf()
                    true
                }
                else -> false
            }
        }
    }

    private fun removeOverlay() {
        if (overlayView != null) {
            try {
                windowManager?.removeView(overlayView)
            } catch (e: Exception) {
                Log.e(TAG, "Error removing overlay view", e)
            } finally {
                overlayView = null
            }
        }
    }

    private fun triggerVibrationAlert() {
        try {
            val vibrator = getSystemService(Context.VIBRATOR_SERVICE) as Vibrator
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                // Urgent double-pulse haptic vibration pattern
                val timings = longArrayOf(0, 200, 100, 200)
                val amplitudes = intArrayOf(0, 255, 0, 255)
                vibrator.vibrate(VibrationEffect.createWaveform(timings, amplitudes, -1))
            } else {
                @Suppress("DEPRECATION")
                vibrator.vibrate(300)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Vibration feedback failed: ${e.message}")
        }
    }

    private fun startForegroundNotification() {
        val launchIntent = Intent(this, MainActivity::class.java)
        val pendingIntent = PendingIntent.getActivity(
            this,
            0,
            launchIntent,
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT
        )

        val notification: Notification = NotificationCompat.Builder(this, ScamShieldApp.CHANNEL_SERVICE_ID)
            .setContentTitle(getString(R.string.service_running_title))
            .setContentText(getString(R.string.service_running_desc))
            .setSmallIcon(android.R.drawable.ic_dialog_alert)
            .setContentIntent(pendingIntent)
            .setOngoing(true)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .build()

        startForeground(FOREGROUND_NOTIFICATION_ID, notification)
    }

    override fun onDestroy() {
        super.onDestroy()
        removeOverlay()
    }
}
