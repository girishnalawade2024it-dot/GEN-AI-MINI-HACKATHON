package com.scamshield.app.ml

import android.content.Context
import android.util.Log
import java.io.BufferedReader
import java.io.InputStreamReader
import java.util.Locale

/**
 * High-performance, zero-dependency WordPiece Tokenizer for on-device Bert/DistilBert inference.
 * Replicates the Hugging Face uncased WordPiece tokenization pipeline directly in native Kotlin.
 */
class WordPieceTokenizer(
    private val vocab: Map<String, Long>,
    private val maxSeqLength: Int = 64
) {

    companion object {
        private const val TAG = "WordPieceTokenizer"
        private const val PAD_TOKEN = "[PAD]"
        private const val UNK_TOKEN = "[UNK]"
        private const val CLS_TOKEN = "[CLS]"
        private const val SEP_TOKEN = "[SEP]"

        private const val PAD_ID = 0L
        private const val UNK_ID = 100L
        private const val CLS_ID = 101L
        private const val SEP_ID = 102L

        /**
         * Loads vocab.txt directly from the Android app assets folder.
         */
        fun fromAssets(context: Context, assetFileName: String = "vocab.txt"): WordPieceTokenizer {
            val vocabMap = HashMap<String, Long>(30522)
            try {
                context.assets.open(assetFileName).use { inputStream ->
                    BufferedReader(InputStreamReader(inputStream, Charsets.UTF_8)).use { reader ->
                        var line: String? = reader.readLine()
                        var index = 0L
                        while (line != null) {
                            val trimmed = line.trim()
                            if (trimmed.isNotEmpty()) {
                                vocabMap[trimmed] = index
                            }
                            index++
                            line = reader.readLine()
                        }
                    }
                }
                Log.d(TAG, "Loaded ${vocabMap.size} tokens into WordPiece vocabulary from $assetFileName")
            } catch (e: Exception) {
                Log.e(TAG, "Error loading vocab from assets: ${e.message}", e)
            }
            return WordPieceTokenizer(vocabMap)
        }
    }

    data class TokenizedInput(
        val inputIds: LongArray,
        val attentionMask: LongArray
    ) {
        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (javaClass != other?.javaClass) return false
            other as TokenizedInput
            if (!inputIds.contentEquals(other.inputIds)) return false
            if (!attentionMask.contentEquals(other.attentionMask)) return false
            return true
        }

        override fun hashCode(): Int {
            var result = inputIds.contentHashCode()
            result = 31 * result + attentionMask.contentHashCode()
            return result
        }
    }

    /**
     * Tokenizes incoming raw message text into fixed-length input_ids and attention_mask.
     */
    fun encode(text: String): TokenizedInput {
        val tokens = ArrayList<Long>()
        tokens.add(vocab[CLS_TOKEN] ?: CLS_ID)

        val words = whitespaceAndPunctuationTokenize(cleanAndNormalize(text))

        for (word in words) {
            val subTokens = wordPieceTokenize(word)
            for (subToken in subTokens) {
                // Reserve 1 slot for [SEP]
                if (tokens.size >= maxSeqLength - 1) {
                    break
                }
                tokens.add(subToken)
            }
            if (tokens.size >= maxSeqLength - 1) {
                break
            }
        }

        tokens.add(vocab[SEP_TOKEN] ?: SEP_ID)

        val inputIds = LongArray(maxSeqLength) { PAD_ID }
        val attentionMask = LongArray(maxSeqLength) { 0L }

        for (i in 0 until tokens.size) {
            inputIds[i] = tokens[i]
            attentionMask[i] = 1L
        }

        return TokenizedInput(inputIds, attentionMask)
    }

    /**
     * Strips control characters, normalizes whitespace, and converts to lowercase.
     */
    private fun cleanAndNormalize(text: String): String {
        val sb = StringBuilder()
        for (char in text) {
            if (char == '\u0000' || char == '\uFFFD' || isControl(char)) {
                continue
            }
            if (isWhitespace(char)) {
                sb.append(' ')
            } else {
                sb.append(char)
            }
        }
        return sb.toString().lowercase(Locale.ROOT)
    }

    /**
     * Splits text on whitespace and punctuation characters, isolating punctuation tokens.
     */
    private fun whitespaceAndPunctuationTokenize(text: String): List<String> {
        val result = ArrayList<String>()
        val currentWord = StringBuilder()

        for (char in text) {
            if (isPunctuation(char)) {
                if (currentWord.isNotEmpty()) {
                    result.add(currentWord.toString())
                    currentWord.clear()
                }
                result.add(char.toString())
            } else if (isWhitespace(char)) {
                if (currentWord.isNotEmpty()) {
                    result.add(currentWord.toString())
                    currentWord.clear()
                }
            } else {
                currentWord.append(char)
            }
        }

        if (currentWord.isNotEmpty()) {
            result.add(currentWord.toString())
        }
        return result
    }

    /**
     * Greedy longest-match WordPiece tokenization algorithm.
     */
    private fun wordPieceTokenize(word: String): List<Long> {
        if (word.length > 100) {
            return listOf(vocab[UNK_TOKEN] ?: UNK_ID)
        }

        val subTokens = ArrayList<Long>()
        var start = 0

        while (start < word.length) {
            var end = word.length
            var curSubstr: String? = null
            var curTokenId: Long? = null

            while (start < end) {
                var substr = word.substring(start, end)
                if (start > 0) {
                    substr = "##$substr"
                }
                if (vocab.containsKey(substr)) {
                    curSubstr = substr
                    curTokenId = vocab[substr]
                    break
                }
                end--
            }

            if (curSubstr == null || curTokenId == null) {
                return listOf(vocab[UNK_TOKEN] ?: UNK_ID)
            }

            subTokens.add(curTokenId)
            start = end
        }

        return subTokens
    }

    private fun isWhitespace(char: Char): Boolean {
        return char.isWhitespace()
    }

    private fun isControl(char: Char): Boolean {
        return Character.isISOControl(char)
    }

    private fun isPunctuation(c: Char): Boolean {
        val type = Character.getType(c).toByte()
        return type == Character.CONNECTOR_PUNCTUATION ||
                type == Character.DASH_PUNCTUATION ||
                type == Character.START_PUNCTUATION ||
                type == Character.END_PUNCTUATION ||
                type == Character.INITIAL_QUOTE_PUNCTUATION ||
                type == Character.FINAL_QUOTE_PUNCTUATION ||
                type == Character.OTHER_PUNCTUATION ||
                (c in '!'..'/' || c in ':'..'@' || c in '['..'`' || c in '{'..'~')
    }
}
