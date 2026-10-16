package com.scamshield.app

import android.Manifest
import android.content.Intent
import android.os.Build
import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.cardview.widget.CardView
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.scamshield.app.ml.ScamClassifier
import com.scamshield.app.service.OverlayService
import com.scamshield.app.util.PermissionUtils
import kotlinx.coroutines.launch

class MainActivity : AppCompatActivity() {

    private lateinit var btnGrantOverlay: Button
    private lateinit var btnGrantSms: Button
    private lateinit var btnGrantNotifications: Button
    private lateinit var etTestInput: EditText
    private lateinit var btnRunInference: Button
    private lateinit var btnSimulateOverlay: Button
    private lateinit var btnSampleScam: Button
    private lateinit var btnSampleHam: Button
    private lateinit var tvInferenceResult: TextView
    private lateinit var cardSafeViewer: CardView
    private lateinit var tvSafeSender: TextView
    private lateinit var tvSafeBody: TextView

    private val requestSmsPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted ->
            if (isGranted) {
                Toast.makeText(this, "SMS Interception Permission Granted", Toast.LENGTH_SHORT).show()
            } else {
                Toast.makeText(this, "SMS Permission is required to inspect incoming SMS", Toast.LENGTH_LONG).show()
            }
            updatePermissionStates()
        }

    private val requestNotificationPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) {
            updatePermissionStates()
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        initViews()
        setupListeners()
        handleIncomingSafeViewIntent(intent)
        requestNotificationPermissionIfRequired()
    }

    override fun onResume() {
        super.onResume()
        updatePermissionStates()
    }

    override fun onNewIntent(intent: Intent?) {
        super.onNewIntent(intent)
        setIntent(intent)
        handleIncomingSafeViewIntent(intent)
    }

    private fun initViews() {
        btnGrantOverlay = findViewById(R.id.btnGrantOverlay)
        btnGrantSms = findViewById(R.id.btnGrantSms)
        btnGrantNotifications = findViewById(R.id.btnGrantNotifications)
        etTestInput = findViewById(R.id.etTestInput)
        btnRunInference = findViewById(R.id.btnRunInference)
        btnSimulateOverlay = findViewById(R.id.btnSimulateOverlay)
        btnSampleScam = findViewById(R.id.btnSampleScam)
        btnSampleHam = findViewById(R.id.btnSampleHam)
        tvInferenceResult = findViewById(R.id.tvInferenceResult)
        cardSafeViewer = findViewById(R.id.cardSafeViewer)
        tvSafeSender = findViewById(R.id.tvSafeSender)
        tvSafeBody = findViewById(R.id.tvSafeBody)

        // Set default sample text
        etTestInput.setText("URGENT: Your Chase bank account has been locked due to suspicious activity. Verify immediately at http://chase-security-verify.com/login")
    }

    private fun setupListeners() {
        // Overlay Permission Button
        btnGrantOverlay.setOnClickListener {
            if (!PermissionUtils.hasOverlayPermission(this)) {
                startActivity(PermissionUtils.getOverlayPermissionIntent(this))
            } else {
                Toast.makeText(this, "Overlay permission is already active", Toast.LENGTH_SHORT).show()
            }
        }

        // SMS Permission Button
        btnGrantSms.setOnClickListener {
            if (!PermissionUtils.hasSmsPermission(this)) {
                requestSmsPermissionLauncher.launch(Manifest.permission.RECEIVE_SMS)
            } else {
                Toast.makeText(this, "SMS permission already granted", Toast.LENGTH_SHORT).show()
            }
        }

        // Notification Listener Button
        btnGrantNotifications.setOnClickListener {
            if (!PermissionUtils.isNotificationListenerEnabled(this)) {
                startActivity(PermissionUtils.getNotificationListenerSettingsIntent())
            } else {
                Toast.makeText(this, "Notification listener already enabled", Toast.LENGTH_SHORT).show()
            }
        }

        // Sample Presets
        btnSampleScam.setOnClickListener {
            etTestInput.setText("USPS: Your parcel #98214 is held due to an unpaid $1.50 customs fee. Resolve now: http://usps-redelivery-fee.top")
        }

        btnSampleHam.setOnClickListener {
            etTestInput.setText("Your Chase security code is 849201. Never share this code with anyone.")
        }

        // Run On-Device ML Inference
        btnRunInference.setOnClickListener {
            val text = etTestInput.text.toString().trim()
            if (text.isEmpty()) {
                Toast.makeText(this, "Please enter some text to classify", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }

            tvInferenceResult.text = "Running on-device quantized Transformer inference..."
            lifecycleScope.launch {
                val classifier = ScamClassifier.getInstance(applicationContext)
                val result = classifier.classify(text, threshold = ScamClassifier.DEFAULT_SCAM_THRESHOLD)

                val statusEmoji = if (result.isScam) "🚨 SCAM DETECTED" else "✅ SAFE / LEGITIMATE"
                val textColor = if (result.isScam) {
                    ContextCompat.getColor(this@MainActivity, R.color.danger_red_glow)
                } else {
                    ContextCompat.getColor(this@MainActivity, R.color.safe_green)
                }

                tvInferenceResult.setTextColor(textColor)
                tvInferenceResult.text = """
                    $statusEmoji
                    • Confidence: ${result.percentageString}
                    • On-Device Latency: ${result.latencyMs} ms
                    • Model: INT8 Quantized MobileBERT / DistilBERT
                    • Cloud Roundtrips: 0 (Strictly Local)
                """.trimIndent()
            }
        }

        // Simulate Truecaller Alert Overlay
        btnSimulateOverlay.setOnClickListener {
            if (!PermissionUtils.hasOverlayPermission(this)) {
                Toast.makeText(
                    this,
                    "Please enable Floating Overlay permission first",
                    Toast.LENGTH_LONG
                ).show()
                startActivity(PermissionUtils.getOverlayPermissionIntent(this))
                return@setOnClickListener
            }

            val text = etTestInput.text.toString().trim()
            Toast.makeText(this, "Launching Truecaller Alert Overlay...", Toast.LENGTH_SHORT).show()
            OverlayService.startOverlay(
                context = this,
                sender = "+1 (800) 555-0199",
                messageText = if (text.isNotEmpty()) text else "URGENT: Verify your account immediately at http://scam-link.top",
                scamProbability = 0.984f,
                source = "SMS"
            )
        }
    }

    private fun updatePermissionStates() {
        val hasOverlay = PermissionUtils.hasOverlayPermission(this)
        btnGrantOverlay.text = if (hasOverlay) "Active ✓" else "Enable"
        btnGrantOverlay.isEnabled = !hasOverlay

        val hasSms = PermissionUtils.hasSmsPermission(this)
        btnGrantSms.text = if (hasSms) "Granted ✓" else "Grant"
        btnGrantSms.isEnabled = !hasSms

        val hasListener = PermissionUtils.isNotificationListenerEnabled(this)
        btnGrantNotifications.text = if (hasListener) "Active ✓" else "Enable"
        btnGrantNotifications.isEnabled = !hasListener
    }

    private fun requestNotificationPermissionIfRequired() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            if (!PermissionUtils.hasNotificationPostPermission(this)) {
                requestNotificationPermissionLauncher.launch(Manifest.permission.POST_NOTIFICATIONS)
            }
        }
    }

    private fun handleIncomingSafeViewIntent(intent: Intent?) {
        val sender = intent?.getStringExtra("view_safe_sender")
        val message = intent?.getStringExtra("view_safe_message")

        if (!sender.isNullOrEmpty() && !message.isNullOrEmpty()) {
            cardSafeViewer.visibility = View.VISIBLE
            tvSafeSender.text = "Sender: $sender (Hyperlinks Disarmed)"
            tvSafeBody.text = message
            Toast.makeText(this, "Message rendered safely in isolated viewer", Toast.LENGTH_SHORT).show()
        }
    }
}
