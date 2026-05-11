package com.karelpeeters.hwlang.jetbrains

import com.intellij.icons.AllIcons
import com.intellij.openapi.fileTypes.LanguageFileType
import javax.swing.Icon

class HwlangFileType private constructor() : LanguageFileType(HwlangLanguage) {
    override fun getName(): String = "HWLang"

    override fun getDescription(): String = "HWLang source file"

    override fun getDefaultExtension(): String = "kh"

    override fun getIcon(): Icon = AllIcons.FileTypes.Custom

    companion object {
        @JvmField
        val INSTANCE = HwlangFileType()
    }
}

