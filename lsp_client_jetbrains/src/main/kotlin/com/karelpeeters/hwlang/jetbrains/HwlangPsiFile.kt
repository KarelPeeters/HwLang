package com.karelpeeters.hwlang.jetbrains

import com.intellij.extapi.psi.PsiFileBase
import com.intellij.openapi.fileTypes.FileType
import com.intellij.psi.FileViewProvider

class HwlangPsiFile(viewProvider: FileViewProvider) : PsiFileBase(viewProvider, HwlangLanguage) {
    override fun getFileType(): FileType = HwlangFileType.INSTANCE

    override fun toString(): String = "HWLang File"
}
