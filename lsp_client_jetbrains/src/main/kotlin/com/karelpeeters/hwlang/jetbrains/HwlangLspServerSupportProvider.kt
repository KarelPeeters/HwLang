package com.karelpeeters.hwlang.jetbrains

import com.intellij.execution.configurations.GeneralCommandLine
import com.intellij.lang.annotation.AnnotationHolder
import com.intellij.lang.annotation.HighlightSeverity
import com.intellij.openapi.components.service
import com.intellij.openapi.editor.colors.CodeInsightColors
import com.intellij.openapi.project.Project
import com.intellij.openapi.util.TextRange
import com.intellij.openapi.util.text.StringUtil
import com.intellij.openapi.vfs.VirtualFile
import com.intellij.platform.lsp.api.LspServer
import com.intellij.platform.lsp.api.LspServerSupportProvider
import com.intellij.platform.lsp.api.ProjectWideLspServerDescriptor
import com.intellij.platform.lsp.api.customization.LspCustomization
import com.intellij.platform.lsp.api.customization.LspDiagnosticsSupport
import com.intellij.platform.lsp.api.customization.LspFormattingSupport
import com.intellij.platform.lsp.api.lsWidget.LspServerWidgetItem
import org.eclipse.lsp4j.Diagnostic
import org.eclipse.lsp4j.DiagnosticRelatedInformation

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
        override val diagnosticsCustomizer = object : LspDiagnosticsSupport() {
            override fun getTooltip(diagnostic: Diagnostic): String = buildDiagnosticTooltip(diagnostic)

            override fun createAnnotation(
                holder: AnnotationHolder,
                diagnostic: Diagnostic,
                textRange: TextRange,
                quickFixes: List<com.intellij.codeInsight.intention.IntentionAction>,
            ) {
                super.createAnnotation(holder, diagnostic, textRange, quickFixes)

                val currentFileUrl = holder.currentAnnotationSession.file.virtualFile.url
                for (related in diagnostic.relatedInformation.orEmpty()) {
                    val location = related.location ?: continue
                    if (location.uri != currentFileUrl) {
                        continue
                    }

                    val relatedRange = lspRangeToTextRange(holder.currentAnnotationSession.file.text, location.range)
                        ?: continue
                    if (relatedRange == textRange) {
                        continue
                    }

                    holder.newAnnotation(HighlightSeverity.INFORMATION, related.message)
                        .range(relatedRange)
                        .tooltip(buildRelatedTooltip(diagnostic, related))
                        .textAttributes(CodeInsightColors.INFORMATION_ATTRIBUTES)
                        .create()
                }
            }
        }

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

private fun buildDiagnosticTooltip(diagnostic: Diagnostic): String {
    val message = escapePreservingLines(diagnostic.message)
    val source = diagnostic.source?.let(::escapePreservingLines)
    val code = diagnostic.code?.toString()?.let(::escapePreservingLines)
    val codeDescriptionHref = diagnostic.codeDescription?.href?.toString()?.let(::escapePreservingLines)
    val related = diagnostic.relatedInformation.orEmpty()

    return buildString {
        append("<html>")
        append(message)

        if (source != null || code != null) {
            append("<br><br><b>Source</b>: ")
            append(source ?: "unknown")
            if (code != null) {
                append(" [")
                append(code)
                append(']')
            }
        }

        if (codeDescriptionHref != null) {
            append("<br><b>Code description</b>: ")
            append(codeDescriptionHref)
        }

        if (related.isNotEmpty()) {
            append("<br><br><b>Related information</b><br>")
            for (info in related) {
                append("&bull; ")
                append(formatRelatedLocation(info))
                append(escapePreservingLines(info.message))
                append("<br>")
            }
        }

        append("</html>")
    }
}

private fun buildRelatedTooltip(diagnostic: Diagnostic, related: DiagnosticRelatedInformation): String {
    return buildString {
        append("<html><b>Related information</b><br>")
        append(escapePreservingLines(related.message))
        append("<br><br><b>Diagnostic</b><br>")
        append(escapePreservingLines(diagnostic.message))

        val otherRelated = diagnostic.relatedInformation.orEmpty().filter { it !== related }
        if (otherRelated.isNotEmpty()) {
            append("<br><br><b>Other related information</b><br>")
            for (info in otherRelated) {
                append("&bull; ")
                append(formatRelatedLocation(info))
                append(escapePreservingLines(info.message))
                append("<br>")
            }
        }

        append("</html>")
    }
}

private fun formatRelatedLocation(info: DiagnosticRelatedInformation): String {
    val location = info.location ?: return ""
    val uri = location.uri
    val fileName = uri.substringAfterLast('/').substringAfterLast('\\')
    val start = location.range?.start ?: return "${escapePreservingLines(fileName)}: "
    return buildString {
        append(escapePreservingLines(fileName))
        append(':')
        append(start.line + 1)
        append(':')
        append(start.character + 1)
        append(": ")
    }
}

private fun escapePreservingLines(text: String): String =
    StringUtil.escapeXmlEntities(text).replace("\n", "<br>")

private fun lspRangeToTextRange(fileText: String, range: org.eclipse.lsp4j.Range?): TextRange? {
    if (range == null) {
        return null
    }

    val start = lspPositionToOffset(fileText, range.start.line, range.start.character) ?: return null
    val end = lspPositionToOffset(fileText, range.end.line, range.end.character) ?: return null
    return TextRange.create(start, end)
}

private fun lspPositionToOffset(text: String, targetLine: Int, targetColumnUtf16: Int): Int? {
    if (targetLine < 0 || targetColumnUtf16 < 0) {
        return null
    }

    var line = 0
    var offset = 0
    while (line < targetLine) {
        if (offset >= text.length) {
            return null
        }
        val nextNewline = text.indexOf('\n', offset)
        offset = if (nextNewline >= 0) nextNewline + 1 else return null
        line += 1
    }

    var utf16 = 0
    var current = offset
    while (current < text.length && utf16 < targetColumnUtf16) {
        val c = text[current]
        if (c == '\n') {
            return null
        }
        utf16 += if (Character.isHighSurrogate(c) && current + 1 < text.length && Character.isLowSurrogate(text[current + 1])) {
            current += 2
            2
        } else {
            current += 1
            1
        }
    }

    return if (utf16 == targetColumnUtf16) current else null
}
