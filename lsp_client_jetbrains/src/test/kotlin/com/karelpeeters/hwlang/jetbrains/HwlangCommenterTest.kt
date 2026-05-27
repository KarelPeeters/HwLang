package com.karelpeeters.hwlang.jetbrains

import com.intellij.codeInsight.generation.actions.CommentByBlockCommentAction
import com.intellij.codeInsight.generation.actions.CommentByLineCommentAction
import com.intellij.lang.LanguageCommenters
import com.intellij.testFramework.fixtures.BasePlatformTestCase
import junit.framework.TestCase.assertEquals
import junit.framework.TestCase.assertNotNull
import junit.framework.TestCase.assertTrue

class HwlangCommenterTest : BasePlatformTestCase() {
    fun testLineCommentAction() {
        myFixture.configureByText(
            "aaa.kh",
            """
                pub module top {
                    <caret>foo = 1;
                }
            """.trimIndent(),
        )

        assertEquals(HwlangLanguage.id, myFixture.file.language.id)
        assertNotNull(LanguageCommenters.INSTANCE.forLanguage(HwlangLanguage))

        myFixture.testAction(CommentByLineCommentAction())

        assertEquals(
            """
                pub module top {
                //    foo = 1;
                }<caret>
            """.trimIndent(),
            dumpEditorState(),
        )
    }

    fun testBlockCommentAction() {
        myFixture.configureByText(
            "aaa.kh",
            """
                <selection>foo = 1;</selection>
            """.trimIndent(),
        )

        assertEquals(HwlangLanguage.id, myFixture.file.language.id)
        assertNotNull(LanguageCommenters.INSTANCE.forLanguage(HwlangLanguage))

        myFixture.testAction(CommentByBlockCommentAction())

        assertTrue(myFixture.editor.document.text.contains("foo = 1;"))
        assertTrue(myFixture.editor.document.text.contains("/*"))
        assertTrue(myFixture.editor.document.text.contains("*/"))
    }

    private fun dumpEditorState(): String {
        val text = myFixture.editor.document.text
        val selectionModel = myFixture.editor.selectionModel
        val caretOffset = myFixture.editor.caretModel.offset

        val markers = buildList {
            add(caretOffset to "<caret>")
            if (selectionModel.hasSelection()) {
                add(selectionModel.selectionStart to "<selection>")
                add(selectionModel.selectionEnd to "</selection>")
            }
        }.sortedByDescending { it.first }

        val result = StringBuilder(text)
        for ((offset, marker) in markers) {
            result.insert(offset, marker)
        }
        return result.toString()
    }
}
