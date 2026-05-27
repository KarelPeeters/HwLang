package com.karelpeeters.hwlang.jetbrains

import com.intellij.lang.ASTFactory
import com.intellij.lang.ASTNode
import com.intellij.lang.ParserDefinition
import com.intellij.lang.PsiParser
import com.intellij.lexer.Lexer
import com.intellij.openapi.project.Project
import com.intellij.openapi.util.NlsSafe
import com.intellij.psi.FileViewProvider
import com.intellij.psi.PsiElement
import com.intellij.psi.PsiFile
import com.intellij.psi.PlainTextTokenTypes
import com.intellij.psi.TokenType
import com.intellij.psi.tree.IFileElementType
import com.intellij.psi.tree.TokenSet
import com.intellij.psi.util.PsiUtilCore

class HwlangParserDefinition : ParserDefinition {
    override fun createLexer(project: Project): Lexer = HwlangLexer()

    override fun createParser(project: Project): PsiParser {
        throw UnsupportedOperationException("HWLang uses LSP-only parsing in JetBrains")
    }

    override fun getFileNodeType(): IFileElementType = FILE

    override fun getWhitespaceTokens(): TokenSet = TokenSet.create(TokenType.WHITE_SPACE)

    override fun getCommentTokens(): TokenSet = TokenSet.create(HwlangTokenTypes.LINE_COMMENT, HwlangTokenTypes.BLOCK_COMMENT)

    override fun getStringLiteralElements(): TokenSet = TokenSet.create(HwlangTokenTypes.STRING)

    override fun createElement(node: ASTNode): PsiElement = PsiUtilCore.NULL_PSI_ELEMENT

    override fun createFile(viewProvider: FileViewProvider): PsiFile = HwlangPsiFile(viewProvider)

    override fun spaceExistenceTypeBetweenTokens(left: ASTNode, right: ASTNode): ParserDefinition.SpaceRequirements {
        return ParserDefinition.SpaceRequirements.MAY
    }

    companion object {
        private val FILE = object : IFileElementType(HwlangLanguage) {
            override fun parseContents(chameleon: ASTNode): ASTNode {
                val chars = chameleon.chars
                return ASTFactory.leaf(PlainTextTokenTypes.PLAIN_TEXT, chars)
            }

            override fun toString(): @NlsSafe String = "HWLANG_FILE"
        }
    }
}
