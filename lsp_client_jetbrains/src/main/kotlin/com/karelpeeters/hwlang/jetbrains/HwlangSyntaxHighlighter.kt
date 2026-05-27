package com.karelpeeters.hwlang.jetbrains

import com.intellij.openapi.editor.DefaultLanguageHighlighterColors
import com.intellij.openapi.editor.HighlighterColors
import com.intellij.openapi.editor.colors.TextAttributesKey
import com.intellij.openapi.fileTypes.SyntaxHighlighterBase
import com.intellij.psi.TokenType
import com.intellij.psi.tree.IElementType

class HwlangSyntaxHighlighter : SyntaxHighlighterBase() {
    override fun getHighlightingLexer() = HwlangLexer()

    override fun getTokenHighlights(tokenType: IElementType): Array<TextAttributesKey> =
        when (tokenType) {
            HwlangTokenTypes.LINE_COMMENT -> pack(LINE_COMMENT)
            HwlangTokenTypes.BLOCK_COMMENT -> pack(BLOCK_COMMENT)
            HwlangTokenTypes.STRING -> pack(STRING)
            HwlangTokenTypes.NUMBER -> pack(NUMBER)
            HwlangTokenTypes.KEYWORD -> pack(KEYWORD)
            HwlangTokenTypes.IDENTIFIER -> pack(IDENTIFIER)
            HwlangTokenTypes.OPERATOR -> pack(OPERATOR)
            HwlangTokenTypes.BRACES -> pack(BRACES)
            HwlangTokenTypes.BRACKETS -> pack(BRACKETS)
            HwlangTokenTypes.PARENTHESES -> pack(PARENTHESES)
            HwlangTokenTypes.COMMA -> pack(COMMA)
            HwlangTokenTypes.SEMICOLON -> pack(SEMICOLON)
            TokenType.BAD_CHARACTER -> pack(BAD_CHARACTER)
            else -> emptyArray()
        }

    companion object {
        val LINE_COMMENT: TextAttributesKey =
            TextAttributesKey.createTextAttributesKey("HWLANG.LINE_COMMENT", DefaultLanguageHighlighterColors.LINE_COMMENT)
        val BLOCK_COMMENT: TextAttributesKey =
            TextAttributesKey.createTextAttributesKey("HWLANG.BLOCK_COMMENT", DefaultLanguageHighlighterColors.BLOCK_COMMENT)
        val STRING: TextAttributesKey =
            TextAttributesKey.createTextAttributesKey("HWLANG.STRING", DefaultLanguageHighlighterColors.STRING)
        val NUMBER: TextAttributesKey =
            TextAttributesKey.createTextAttributesKey("HWLANG.NUMBER", DefaultLanguageHighlighterColors.NUMBER)
        val KEYWORD: TextAttributesKey =
            TextAttributesKey.createTextAttributesKey("HWLANG.KEYWORD", DefaultLanguageHighlighterColors.KEYWORD)
        val IDENTIFIER: TextAttributesKey =
            TextAttributesKey.createTextAttributesKey("HWLANG.IDENTIFIER", DefaultLanguageHighlighterColors.IDENTIFIER)
        val OPERATOR: TextAttributesKey =
            TextAttributesKey.createTextAttributesKey("HWLANG.OPERATOR", DefaultLanguageHighlighterColors.OPERATION_SIGN)
        val BRACES: TextAttributesKey =
            TextAttributesKey.createTextAttributesKey("HWLANG.BRACES", DefaultLanguageHighlighterColors.BRACES)
        val BRACKETS: TextAttributesKey =
            TextAttributesKey.createTextAttributesKey("HWLANG.BRACKETS", DefaultLanguageHighlighterColors.BRACKETS)
        val PARENTHESES: TextAttributesKey =
            TextAttributesKey.createTextAttributesKey("HWLANG.PARENTHESES", DefaultLanguageHighlighterColors.PARENTHESES)
        val COMMA: TextAttributesKey =
            TextAttributesKey.createTextAttributesKey("HWLANG.COMMA", DefaultLanguageHighlighterColors.COMMA)
        val SEMICOLON: TextAttributesKey =
            TextAttributesKey.createTextAttributesKey("HWLANG.SEMICOLON", DefaultLanguageHighlighterColors.SEMICOLON)
        val BAD_CHARACTER: TextAttributesKey =
            TextAttributesKey.createTextAttributesKey("HWLANG.BAD_CHARACTER", HighlighterColors.BAD_CHARACTER)
    }
}
