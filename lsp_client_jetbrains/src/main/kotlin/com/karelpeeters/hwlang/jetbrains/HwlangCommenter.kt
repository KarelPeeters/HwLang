package com.karelpeeters.hwlang.jetbrains

import com.intellij.lang.Commenter

class HwlangCommenter : Commenter {
    override fun getLineCommentPrefix(): String = "//"

    override fun getBlockCommentPrefix(): String = "/*"

    override fun getBlockCommentSuffix(): String = "*/"

    override fun getCommentedBlockCommentPrefix(): String = "/*"

    override fun getCommentedBlockCommentSuffix(): String = "*/"
}
