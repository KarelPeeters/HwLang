package com.karelpeeters.hwlang.jetbrains

import com.intellij.openapi.Disposable
import com.intellij.openapi.application.ApplicationManager
import com.intellij.openapi.components.Service
import com.intellij.openapi.components.service
import com.intellij.openapi.diagnostic.Logger
import com.intellij.openapi.project.Project
import java.io.IOException
import java.nio.file.ClosedWatchServiceException
import java.nio.file.FileSystems
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.StandardWatchEventKinds.ENTRY_CREATE
import java.nio.file.StandardWatchEventKinds.ENTRY_MODIFY
import java.nio.file.StandardWatchEventKinds.OVERFLOW
import java.nio.file.WatchEvent
import java.nio.file.WatchService
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicLong

@Service(Service.Level.PROJECT)
class HwlangBinaryWatcher(private val project: Project) : Disposable {
    private val log = Logger.getInstance(HwlangBinaryWatcher::class.java)
    private val lastRestartNanos = AtomicLong(0)
    private val lock = Any()

    private var watchedBinary: Path? = null
    private var watchedFileName: Path? = null
    private var watchService: WatchService? = null
    private var lastProblemMessage: String? = null

    init {
        updateServerPath(project.service<HwlangLspSettings>().resolvedServerPathOrNull())
        ApplicationManager.getApplication().executeOnPooledThread { watchLoop() }
    }

    fun updateServerPath(path: Path?) {
        val normalized = path?.normalize()?.toAbsolutePath()

        synchronized(lock) {
            if (normalized == watchedBinary) {
                return
            }

            closeWatchService()
            watchedBinary = normalized
            watchedFileName = normalized?.fileName
            lastProblemMessage = null

            val parent = normalized?.parent ?: return
            if (!Files.isDirectory(parent)) {
                return
            }

            watchService = FileSystems.getDefault().newWatchService().also { service ->
                parent.register(service, ENTRY_CREATE, ENTRY_MODIFY)
            }
        }
    }

    fun rememberProblem(message: String): Boolean {
        synchronized(lock) {
            if (message == lastProblemMessage) {
                return false
            }
            lastProblemMessage = message
            return true
        }
    }

    private fun watchLoop() {
        while (!project.isDisposed) {
            val service = synchronized(lock) { watchService } ?: run {
                TimeUnit.MILLISECONDS.sleep(250)
                continue
            }

            val key = try {
                service.poll(1, TimeUnit.SECONDS)
            } catch (_: ClosedWatchServiceException) {
                continue
            } catch (e: InterruptedException) {
                Thread.currentThread().interrupt()
                return
            }

            if (key == null) {
                continue
            }

            val shouldRestart = key.pollEvents().any(::matchesWatchedBinaryChange)
            key.reset()

            if (shouldRestart) {
                synchronized(lock) {
                    lastProblemMessage = null
                }
                requestRestart("HWLang server binary changed on disk")
            }
        }
    }

    private fun matchesWatchedBinaryChange(event: WatchEvent<*>): Boolean {
        if (event.kind() == OVERFLOW) {
            return true
        }

        val changedName = event.context() as? Path ?: return false
        val targetName = synchronized(lock) { watchedFileName } ?: return false
        return changedName == targetName
    }

    private fun requestRestart(reason: String) {
        val now = System.nanoTime()
        val previous = lastRestartNanos.get()
        if (now - previous < RESTART_DEBOUNCE_NANOS) {
            return
        }
        if (!lastRestartNanos.compareAndSet(previous, now)) {
            return
        }

        log.info(reason)
        restartHwlangServerAsync(project)
    }

    override fun dispose() {
        synchronized(lock) {
            closeWatchService()
            watchedBinary = null
            watchedFileName = null
            lastProblemMessage = null
        }
    }

    private fun closeWatchService() {
        try {
            watchService?.close()
        } catch (e: IOException) {
            log.warn("Failed to close HWLang binary watch service", e)
        }
        watchService = null
    }

    companion object {
        private const val RESTART_DEBOUNCE_NANOS = 500_000_000L
    }
}

