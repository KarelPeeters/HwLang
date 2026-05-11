package com.karelpeeters.hwlang.jetbrains

import com.intellij.notification.NotificationAction
import com.intellij.notification.NotificationGroupManager
import com.intellij.notification.NotificationType
import com.intellij.openapi.application.ApplicationManager
import com.intellij.openapi.components.PersistentStateComponent
import com.intellij.openapi.components.Service
import com.intellij.openapi.components.State
import com.intellij.openapi.components.Storage
import com.intellij.openapi.components.service
import com.intellij.openapi.options.ShowSettingsUtil
import com.intellij.openapi.project.Project
import com.intellij.openapi.vfs.VirtualFile
import com.intellij.platform.lsp.api.LspServerManager
import java.nio.file.Files
import java.nio.file.InvalidPathException
import java.nio.file.Path
import java.nio.file.Paths

@Service(Service.Level.PROJECT)
@State(name = "HwlangLspSettings", storages = [Storage("hwlang-lsp.xml")])
class HwlangLspSettings(private val project: Project) : PersistentStateComponent<HwlangLspSettings.State> {
    data class State(var serverPath: String = "")

    private var persistedState = State()

    var configuredServerPath: String?
        get() = persistedState.serverPath.trim().ifEmpty { null }
        private set(value) {
            persistedState = State(value.orEmpty())
        }

    override fun getState(): State = persistedState

    override fun loadState(state: State) {
        persistedState = state
    }

    fun setServerPath(value: String?) {
        val normalized = value?.trim()?.ifEmpty { null }
        if (normalized == configuredServerPath) {
            return
        }

        configuredServerPath = normalized
        project.service<HwlangBinaryWatcher>().updateServerPath(resolvedServerPathOrNull())
        restartHwlangServerAsync(project)
    }

    fun resolvedServerPathOrNull(): Path? {
        val configured = configuredServerPath ?: return null
        return try {
            resolveConfiguredPath(project, configured)
        } catch (_: InvalidPathException) {
            null
        }
    }

    fun configurationErrorMessage(): String? {
        val configured = configuredServerPath ?: return "Configure the path to hwl_lsp_server in Settings | Tools | HWLang LSP."
        val path = try {
            resolveConfiguredPath(project, configured)
        } catch (_: InvalidPathException) {
            return "The configured HWLang server path is not a valid filesystem path: $configured"
        }

        if (!Files.exists(path)) {
            return "The configured HWLang server path does not exist: $path"
        }
        if (Files.isDirectory(path)) {
            return "The configured HWLang server path points to a directory instead of an executable: $path"
        }

        return null
    }

    fun notifyConfigurationProblem(file: VirtualFile, message: String) {
        project.service<HwlangBinaryWatcher>().rememberProblem(message)
        NotificationGroupManager.getInstance()
            .getNotificationGroup(NOTIFICATION_GROUP_ID)
            .createNotification(message, NotificationType.WARNING)
            .setTitle("HWLang LSP is not configured for ${file.name}")
            .addAction(NotificationAction.createSimple("Open Settings") {
                openSettings(project)
            })
            .notify(project)
    }

    companion object {
        const val CONFIGURABLE_ID = "hwlang.lsp"
        const val NOTIFICATION_GROUP_ID = "HWLang LSP"

        fun openSettings(project: Project) {
            ShowSettingsUtil.getInstance().showSettingsDialog(project, HwlangLspConfigurable::class.java)
        }
    }
}

internal fun isHwlangFile(file: VirtualFile): Boolean = file.extension.equals(HwlangFileType.INSTANCE.defaultExtension, ignoreCase = true)

internal fun resolveConfiguredPath(project: Project, configuredPath: String): Path {
    val path = Paths.get(configuredPath)
    val resolved = if (path.isAbsolute) {
        path
    } else {
        val basePath = project.basePath?.let(Paths::get)
        (basePath ?: Paths.get("").toAbsolutePath()).resolve(path)
    }
    return resolved.normalize().toAbsolutePath()
}

internal fun restartHwlangServerAsync(project: Project) {
    ApplicationManager.getApplication().invokeLater(
        {
            if (!project.isDisposed) {
                LspServerManager.getInstance(project).stopAndRestartIfNeeded(HwlangLspServerSupportProvider::class.java)
            }
        },
        project.disposed,
    )
}
