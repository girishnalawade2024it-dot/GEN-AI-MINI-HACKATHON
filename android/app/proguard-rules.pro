# Keep ONNX Runtime JNI bindings and native methods intact
-keep class ai.onnxruntime.** { *; }
-dontwarn ai.onnxruntime.**
