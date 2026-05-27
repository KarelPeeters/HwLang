package com.karelpeeters.hwlang.jetbrains

import com.intellij.lexer.LexerBase
import com.intellij.psi.TokenType
import com.intellij.psi.tree.IElementType

class HwlangLexer : LexerBase() {
    private var buffer: CharSequence = ""
    private var startOffset: Int = 0
    private var endOffset: Int = 0
    private var tokenStart: Int = 0
    private var tokenEnd: Int = 0
    private var tokenType: IElementType? = null

    override fun start(buffer: CharSequence, startOffset: Int, endOffset: Int, initialState: Int) {
        this.buffer = buffer
        this.startOffset = startOffset
        this.endOffset = endOffset
        tokenStart = startOffset
        tokenEnd = startOffset
        tokenType = null
        advance()
    }

    override fun getState(): Int = 0

    override fun getTokenType(): IElementType? = tokenType

    override fun getTokenStart(): Int = tokenStart

    override fun getTokenEnd(): Int = tokenEnd

    override fun advance() {
        if (tokenEnd >= endOffset) {
            tokenStart = endOffset
            tokenType = null
            return
        }

        tokenStart = tokenEnd
        val c = buffer[tokenStart]
        when {
            c.isWhitespace() -> {
                tokenEnd = consumeWhile(tokenStart + 1, Char::isWhitespace)
                tokenType = TokenType.WHITE_SPACE
            }

            c == '/' && peek(tokenStart + 1) == '/' -> {
                tokenEnd = tokenStart + 2
                while (tokenEnd < endOffset && buffer[tokenEnd] != '\n' && buffer[tokenEnd] != '\r') {
                    tokenEnd += 1
                }
                tokenType = HwlangTokenTypes.LINE_COMMENT
            }

            c == '/' && peek(tokenStart + 1) == '*' -> {
                tokenEnd = consumeNestedBlockComment(tokenStart)
                tokenType = HwlangTokenTypes.BLOCK_COMMENT
            }

            c == 'r' && peek(tokenStart + 1) == '"' -> {
                tokenEnd = consumeString(tokenStart + 2)
                tokenType = HwlangTokenTypes.STRING
            }

            c == '"' -> {
                tokenEnd = consumeString(tokenStart + 1)
                tokenType = HwlangTokenTypes.STRING
            }

            isIdentifierStart(c) -> {
                tokenEnd = consumeWhile(tokenStart + 1, ::isIdentifierContinue)
                val text = buffer.subSequence(tokenStart, tokenEnd).toString()
                tokenType = when {
                    text == "_" -> HwlangTokenTypes.OPERATOR
                    KEYWORDS.contains(text) -> HwlangTokenTypes.KEYWORD
                    else -> HwlangTokenTypes.IDENTIFIER
                }
            }

            c.isDigit() -> {
                tokenEnd = consumeWhile(tokenStart + 1, ::isNumberContinue)
                tokenType = HwlangTokenTypes.NUMBER
            }

            else -> {
                val fixed = FIXED_TOKENS.firstOrNull { (literal, _) -> matchesLiteral(tokenStart, literal) }
                if (fixed != null) {
                    tokenEnd = tokenStart + fixed.first.length
                    tokenType = fixed.second
                } else {
                    tokenEnd = tokenStart + 1
                    tokenType = TokenType.BAD_CHARACTER
                }
            }
        }
    }

    override fun getBufferSequence(): CharSequence = buffer

    override fun getBufferEnd(): Int = endOffset

    private fun consumeNestedBlockComment(start: Int): Int {
        var index = start + 2
        var depth = 1
        while (index < endOffset) {
            when {
                index + 1 < endOffset && buffer[index] == '/' && buffer[index + 1] == '*' -> {
                    depth += 1
                    index += 2
                }

                index + 1 < endOffset && buffer[index] == '*' && buffer[index + 1] == '/' -> {
                    depth -= 1
                    index += 2
                    if (depth == 0) {
                        return index
                    }
                }

                else -> index += 1
            }
        }
        return endOffset
    }

    private fun consumeString(start: Int): Int {
        var index = start
        while (index < endOffset) {
            val c = buffer[index]
            when {
                c == '\\' && index + 1 < endOffset -> index += 2
                c == '"' -> return index + 1
                else -> index += 1
            }
        }
        return endOffset
    }

    private fun consumeWhile(start: Int, predicate: (Char) -> Boolean): Int {
        var index = start
        while (index < endOffset && predicate(buffer[index])) {
            index += 1
        }
        return index
    }

    private fun peek(index: Int): Char? = if (index < endOffset) buffer[index] else null

    private fun matchesLiteral(index: Int, literal: String): Boolean {
        if (index + literal.length > endOffset) {
            return false
        }
        for (offset in literal.indices) {
            if (buffer[index + offset] != literal[offset]) {
                return false
            }
        }
        return true
    }

    private companion object {
        val KEYWORDS = setOf(
            "import",
            "type",
            "struct",
            "enum",
            "self",
            "ports",
            "port",
            "module",
            "interface",
            "instance",
            "fn",
            "comb",
            "clock",
            "clocked",
            "const",
            "val",
            "var",
            "wire",
            "reg",
            "ref",
            "deref",
            "in",
            "out",
            "async",
            "sync",
            "return",
            "break",
            "continue",
            "true",
            "false",
            "undef",
            "if",
            "else",
            "loop",
            "match",
            "for",
            "while",
            "pub",
            "as",
            "external",
            "__builtin",
            "unsafe_value_with_domain",
            "id_from_str",
        )

        val FIXED_TOKENS = listOf(
            "..=" to HwlangTokenTypes.OPERATOR,
            "+.." to HwlangTokenTypes.OPERATOR,
            "->" to HwlangTokenTypes.OPERATOR,
            "=>" to HwlangTokenTypes.OPERATOR,
            "::" to HwlangTokenTypes.OPERATOR,
            ".." to HwlangTokenTypes.OPERATOR,
            "&&" to HwlangTokenTypes.OPERATOR,
            "||" to HwlangTokenTypes.OPERATOR,
            "^^" to HwlangTokenTypes.OPERATOR,
            "==" to HwlangTokenTypes.OPERATOR,
            "!=" to HwlangTokenTypes.OPERATOR,
            ">=" to HwlangTokenTypes.OPERATOR,
            ">>" to HwlangTokenTypes.OPERATOR,
            "<=" to HwlangTokenTypes.OPERATOR,
            "<<" to HwlangTokenTypes.OPERATOR,
            "+=" to HwlangTokenTypes.OPERATOR,
            "-=" to HwlangTokenTypes.OPERATOR,
            "*=" to HwlangTokenTypes.OPERATOR,
            "/=" to HwlangTokenTypes.OPERATOR,
            "%=" to HwlangTokenTypes.OPERATOR,
            "&=" to HwlangTokenTypes.OPERATOR,
            "|=" to HwlangTokenTypes.OPERATOR,
            "^=" to HwlangTokenTypes.OPERATOR,
            "**" to HwlangTokenTypes.OPERATOR,
            ";" to HwlangTokenTypes.SEMICOLON,
            "," to HwlangTokenTypes.COMMA,
            "{" to HwlangTokenTypes.BRACES,
            "}" to HwlangTokenTypes.BRACES,
            "(" to HwlangTokenTypes.PARENTHESES,
            ")" to HwlangTokenTypes.PARENTHESES,
            "[" to HwlangTokenTypes.BRACKETS,
            "]" to HwlangTokenTypes.BRACKETS,
            ":" to HwlangTokenTypes.OPERATOR,
            "." to HwlangTokenTypes.OPERATOR,
            "=" to HwlangTokenTypes.OPERATOR,
            ">" to HwlangTokenTypes.OPERATOR,
            "<" to HwlangTokenTypes.OPERATOR,
            "&" to HwlangTokenTypes.OPERATOR,
            "|" to HwlangTokenTypes.OPERATOR,
            "^" to HwlangTokenTypes.OPERATOR,
            "+" to HwlangTokenTypes.OPERATOR,
            "-" to HwlangTokenTypes.OPERATOR,
            "*" to HwlangTokenTypes.OPERATOR,
            "/" to HwlangTokenTypes.OPERATOR,
            "%" to HwlangTokenTypes.OPERATOR,
            "!" to HwlangTokenTypes.OPERATOR,
        )

        fun isIdentifierStart(c: Char): Boolean = c == '_' || c in 'a'..'z' || c in 'A'..'Z'

        fun isIdentifierContinue(c: Char): Boolean = isIdentifierStart(c) || c.isDigit()

        fun isNumberContinue(c: Char): Boolean =
            c.isDigit() || c == '_' || c == 'b' || c == 'x' || c in 'a'..'f' || c in 'A'..'F'
    }
}
