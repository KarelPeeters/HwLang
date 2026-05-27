package com.karelpeeters.hwlang.jetbrains

import com.intellij.psi.tree.IElementType

class HwlangTokenType(debugName: String) : IElementType(debugName, HwlangLanguage)

object HwlangTokenTypes {
    @JvmField
    val LINE_COMMENT = HwlangTokenType("LINE_COMMENT")

    @JvmField
    val BLOCK_COMMENT = HwlangTokenType("BLOCK_COMMENT")

    @JvmField
    val STRING = HwlangTokenType("STRING")

    @JvmField
    val NUMBER = HwlangTokenType("NUMBER")

    @JvmField
    val KEYWORD = HwlangTokenType("KEYWORD")

    @JvmField
    val IDENTIFIER = HwlangTokenType("IDENTIFIER")

    @JvmField
    val OPERATOR = HwlangTokenType("OPERATOR")

    @JvmField
    val BRACES = HwlangTokenType("BRACES")

    @JvmField
    val BRACKETS = HwlangTokenType("BRACKETS")

    @JvmField
    val PARENTHESES = HwlangTokenType("PARENTHESES")

    @JvmField
    val COMMA = HwlangTokenType("COMMA")

    @JvmField
    val SEMICOLON = HwlangTokenType("SEMICOLON")
}
