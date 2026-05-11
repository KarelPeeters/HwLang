package com.karelpeeters.hwlang.jetbrains

import com.intellij.execution.configurations.GeneralCommandLine
import com.intellij.openapi.components.service
import com.intellij.openapi.project.Project
import com.intellij.openapi.vfs.VirtualFile
import com.intellij.platform.lsp.api.LspServer
import com.intellij.platform.lsp.api.LspServerSupportProvider
import com.intellij.platform.lsp.api.ProjectWideLspServerDescriptor
import com.intellij.platform.lsp.api.customization.LspCustomization
import com.intellij.platform.lsp.api.customization.LspFormattingSupport
import com.intellij.platform.lsp.api.lsWidget.LspServerWidgetItem

class HwlangLspServerSupportProvider : LspServerSupportProvider {
    override fun fileOpened(project: Project, file: VirtualFile, serverStarter: LspServerSupportProvider.LspServerStarter) {
        if (!isHwlangFile(file)) {
            return
        }

        val settings = project.service<HwlangLspSettings>()
        project.service<HwlangBinaryWatcher>().updateServerPath(settings.resolvedServerPathOrNull())

        val configurationError = settings.configurationErrorMessage()
        if (configurationError != null) {
            if (project.service<HwlangBinaryWatcher>().rememberProblem(configurationError)) {
                settings.notifyConfigurationProblem(file, configurationError)
            }
            return
        }

        serverStarter.ensureServerStarted(HwlangLspServerDescriptor(project))
    }

    override fun createLspServerWidgetItem(lspServer: LspServer, currentFile: VirtualFile?): LspServerWidgetItem {
        return LspServerWidgetItem(lspServer, currentFile, HwlangFileType.INSTANCE.icon, HwlangLspConfigurable::class.java)
    }
}

private class HwlangLspServerDescriptor(project: Project) : ProjectWideLspServerDescriptor(project, "HWLang") {
    override fun isSupportedFile(file: VirtualFile): Boolean = isHwlangFile(file)

    override val lspCustomization: LspCustomization = object : LspCustomization() {
        override val formattingCustomizer = object : LspFormattingSupport() {
            override fun shouldFormatThisFileExclusivelyByServer(
                file: VirtualFile,
                ideCanFormatThisFileItself: Boolean,
                serverExplicitlyWantsToFormatThisFile: Boolean,
            ): Boolean {
                return isHwlangFile(file)
            }
        }
    }

    override fun createCommandLine(): GeneralCommandLine {
        val settings = project.service<HwlangLspSettings>()
        val configurationError = settings.configurationErrorMessage()
        if (configurationError != null) {
            throw RuntimeException(configurationError)
        }

        val serverPath = settings.resolvedServerPathOrNull()
            ?: throw RuntimeException("Configure the path to hwl_lsp_server in Settings | Tools | HWLang LSP.")

        return GeneralCommandLine(serverPath.toString())
    }
}
