package com.karelpeeters.hwlang.jetbrains

import com.intellij.openapi.components.service
import com.intellij.openapi.fileChooser.FileChooserDescriptor
import com.intellij.openapi.options.SearchableConfigurable
import com.intellij.openapi.project.Project
import com.intellij.openapi.ui.TextFieldWithBrowseButton
import com.intellij.ui.dsl.builder.AlignX
import com.intellij.ui.dsl.builder.panel
import javax.swing.JComponent

class HwlangLspConfigurable(private val project: Project) : SearchableConfigurable {
    private val serverPathField = TextFieldWithBrowseButton()

    init {
        val descriptor = FileChooserDescriptor(true, false, false, false, false, false)
        serverPathField.addBrowseFolderListener(
            "Select HWLang LSP server",
            "Choose the hwl_lsp_server executable to launch for .kh files.",
            project,
            descriptor,
        )
        serverPathField.toolTipText = "/path/to/hwl_lsp_server"
    }

    private val panel = panel {
        row("Server path:") {
            cell(serverPathField)
                .align(AlignX.FILL)
                .resizableColumn()
        }
        row {
            text("Path to the external hwl_lsp_server executable used for .kh files.")
        }
        row {
            text("This plugin requires a supported commercial JetBrains IDE with the public LSP API.")
        }
    }

    override fun getId(): String = HwlangLspSettings.CONFIGURABLE_ID

    override fun getDisplayName(): String = "HWLang LSP"

    override fun createComponent(): JComponent = panel

    override fun isModified(): Boolean = serverPathField.text.trim() != project.service<HwlangLspSettings>().configuredServerPath.orEmpty()

    override fun apply() {
        project.service<HwlangLspSettings>().setServerPath(serverPathField.text)
    }

    override fun reset() {
        serverPathField.text = project.service<HwlangLspSettings>().configuredServerPath.orEmpty()
    }
}
