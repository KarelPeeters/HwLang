package com.karelpeeters.hwlang.jetbrains

import com.intellij.psi.TokenType
import junit.framework.TestCase.assertEquals
import org.junit.Test

class HwlangLexerTest {
    @Test
    fun tokenizesKeywordsCommentsStringsAndNumbers() {
        val tokens = lex("pub module top { val x = 0x2a; /* nested /* ok */ comment */ \"hi\" // tail")

        assertEquals(
            listOf(
                "KEYWORD:pub",
                "KEYWORD:module",
                "IDENTIFIER:top",
                "BRACES:{",
                "KEYWORD:val",
                "IDENTIFIER:x",
                "OPERATOR:=",
                "NUMBER:0x2a",
                "SEMICOLON:;",
                "BLOCK_COMMENT:/* nested /* ok */ comment */",
                "STRING:\"hi\"",
                "LINE_COMMENT:// tail",
            ),
            tokens,
        )
    }

    @Test
    fun tokenizesRawStrings() {
        assertEquals(listOf("STRING:r\"raw {still string}\""), lex("r\"raw {still string}\""))
    }

    private fun lex(text: String): List<String> {
        val lexer = HwlangLexer()
        lexer.start(text)

        val tokens = mutableListOf<String>()
        while (true) {
            val tokenType = lexer.tokenType ?: break
            if (tokenType != TokenType.WHITE_SPACE) {
                val tokenText = text.substring(lexer.tokenStart, lexer.tokenEnd)
                tokens += "${tokenType}:$tokenText"
            }
            lexer.advance()
        }
        return tokens
    }
}
