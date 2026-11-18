import type { Plugin } from "@opencode/plugin"
import { tool, type ToolDefinition } from "@opencode-ai/plugin"
import { getConfig } from "../config"
import { Logger } from "../logger"
import { PromptStore } from "../prompts/store"
import { createCompressMessageTool, createCompressRangeTool } from "../compress"
import { attachCompressionDuration } from "../compress/state"
import { createCommandExecuteHandler, createSystemPromptHandler } from "../hooks"
import {
    createSessionState,
    ensureSessionInitialized,
    checkSession,
    saveSessionState,
    syncToolCache,
    type SessionState,
    type WithParts,
} from "../state"
import { assignMessageRefs } from "../message-ids"
import { applyPendingManualTrigger } from "../commands/manual"
import {
    buildPriorityMap,
    buildToolIdList,
    injectCompressNudges,
    injectMessageIds,
    injectExtendedSubAgentResults,
    prune,
    stripHallucinations,
    syncCompressionBlocks,
} from "../messages"
import { countTokens, getCurrentParams } from "../token-utils"
import { isMessageCompacted } from "../state/utils"
import { recentContextMessages } from "../compress/context-window"
import { matchesGlob } from "../protected-patterns"
import { history, project } from "./messages"
import { analyzeContextTokens } from "../commands/context"
import { buildStatsReport } from "../commands/stats"
import {
    createOutOfBandRuntime,
    createOutOfBandTools,
    DCP_NOTICE_PREFIX,
    isolateOutOfBandTools,
    notifyOverLimit as notifyOutOfBandOverLimit,
    removeDcpNotices,
    runOutOfBandAgent,
    type OutOfBandToolHost,
} from "../compress/oob"
import { rpc } from "./rpc"
import { getPluginIdentity } from "../plugin-identity"

// Extension point for future model-invisible V2 reports. Never use synthetic()
// here: its text would enter the model's context, unlike V1 ignored messages.
export async function report(logger: Logger, text: string, sessionID?: string) {
    logger.debug("V2 report (display pending)", { sessionID, text })
}

export async function setup(ctx: Plugin.Context) {
    const warnings = {
        tui: {
            showToast: async (input: { body: { message: string } }) => {
                console.warn(`DCP: ${input.body.message}`)
            },
        },
    }
    const config = getConfig({ directory: ctx.location.directory, client: warnings })
    if (!config.enabled) {
        await ctx.rpc.register(
            { ...rpc, methods: { status: rpc.methods.status } },
            { status: async () => ({ enabled: false, ...getPluginIdentity() }) },
        )
        return
    }
    const logger = new Logger(config.debug)
    const prompts = new PromptStore(
        logger,
        ctx.location.directory,
        config.experimental.customPrompts,
        "compact",
    )
    const sessions = new Map<string, SessionState>()
    const queues = new Map<string, Promise<unknown>>()
    const limits = new Map<string, number>()
    const oobSessions = new Set<string>()
    const activeContextIds = new Map<string, Set<string>>()
    const fullContext = new Map<string, WithParts[]>()
    const partialContextWarned = new Set<string>()

    function mergeContextMessages(
        cached: WithParts[] | undefined,
        historyMessages: WithParts[],
    ): WithParts[] {
        if (!cached?.length) {
            return historyMessages
        }
        const historyIds = new Set(historyMessages.map((message) => message.info.id))
        return [...cached.filter((message) => !historyIds.has(message.info.id)), ...historyMessages]
    }
    const oobRuntime = createOutOfBandRuntime()

    async function isOutOfBandSession(sessionID: string): Promise<boolean> {
        if (oobSessions.has(sessionID)) {
            return true
        }
        try {
            const session = await ctx.session.get({ sessionID })
            if ((session?.metadata as Record<string, unknown> | undefined)?.dcpOob === true) {
                oobSessions.add(sessionID)
                return true
            }
        } catch {
            // If the lookup fails, fall through to normal handling.
        }
        return false
    }
    const aliases: Record<string, string> = {
        task: "subagent",
        bash: "shell",
        apply_patch: "patch",
    }
    for (const list of [
        config.compress.protectedTools,
        config.commands.protectedTools,
        config.strategies.deduplication.protectedTools,
        config.strategies.purgeErrors.protectedTools,
    ]) {
        for (const name of [...list])
            if (aliases[name] && !list.includes(aliases[name]!)) list.push(aliases[name]!)
    }

    function serial<T>(sessionID: string, operation: () => Promise<T>): Promise<T> {
        const pending = (queues.get(sessionID) ?? Promise.resolve()).catch(() => {}).then(operation)
        queues.set(sessionID, pending)
        void pending
            .finally(() => {
                if (queues.get(sessionID) === pending) queues.delete(sessionID)
            })
            .catch(() => {})
        return pending
    }

    const client = {
        session: {
            get: async ({ path }: { path: { id: string } }) => ({
                data: await ctx.session.get({ sessionID: path.id }),
            }),
            messages: async ({ path }: { path: { id: string } }) => {
                const [entries, session] = await Promise.all([
                    ctx.session.context({ sessionID: path.id }),
                    ctx.session.get({ sessionID: path.id }),
                ])
                return {
                    data: mergeContextMessages(fullContext.get(path.id), history(entries, session)),
                }
            },
            prompt: async (input: {
                path: { id: string }
                body: { parts: Array<{ text: string }> }
            }) =>
                report(logger, input.body.parts.map((part) => part.text).join("\n"), input.path.id),
        },
        tui: {
            showToast: async (input: { body: { message: string } }) =>
                report(logger, input.body.message),
        },
    }

    async function load(sessionID: string, agentID?: string) {
        const [session, entries] = await Promise.all([
            ctx.session.get({ sessionID }),
            ctx.session.context({ sessionID }),
        ])
        const selected =
            agentID ??
            session.agent ??
            entries.findLast((entry) => entry.type === "assistant")?.agent
        if (!selected) throw new Error("DCP commands require a session with a selected agent")
        const { data: agent } = await ctx.agent.get({ agentID: selected })
        let state = sessions.get(sessionID)
        if (!state) {
            state = createSessionState("compact")
            sessions.set(sessionID, state)
        }
        const messages = mergeContextMessages(fullContext.get(sessionID), history(entries, session))
        await ensureSessionInitialized(
            client,
            state,
            sessionID,
            logger,
            messages,
            config.manualMode.enabled,
        )
        await checkSession(client, state, logger, messages, config.manualMode.enabled)
        const rule = [...agent.permissions, ...(session.permissions ?? [])].findLast(
            (rule) => matchesGlob("compress", rule.action) && matchesGlob("*", rule.resource),
        )
        state.compressPermission =
            config.compress.permission === "deny"
                ? "deny"
                : rule?.effect === "deny"
                  ? "deny"
                  : config.compress.permission === "ask" || rule?.effect === "ask"
                    ? "ask"
                    : "allow"
        return { state, entries, session, messages }
    }

    function allowed(state: SessionState) {
        if (state.isSubAgent && !config.experimental.allowSubAgents)
            throw new Error("DCP compression is disabled in subagents")
        if (state.compressPermission === "deny") throw new Error("DCP compression is denied")
    }

    // permission.assert exists from OpenCode 2.0.29; the plugin types used here predate it.
    const permission = ctx.permission as typeof ctx.permission & {
        assert?: (input: {
            sessionID: string
            agent?: string
            action: string
            resources: string[]
            save?: string[]
            source?: { type: "tool"; messageID: string; id: string }
        }) => Promise<void>
    }

    await ctx.model.transform((editor) => {
        limits.clear()
        for (const model of editor.list())
            limits.set(`${model.providerID}/${model.id}`, model.limit.context)
    })
    for (const kind of ["context", "compaction"] as const)
        await ctx.session.hook(kind, async (event) => {
            const outOfBand = await isOutOfBandSession(event.sessionID)
            isolateOutOfBandTools(event.tools ?? {}, outOfBand)
            if (outOfBand) {
                return
            }
            return serial(event.sessionID, async () => {
                const { state, entries, session, messages } = await load(
                    event.sessionID,
                    event.agent,
                )
                if (state.isSubAgent && !config.experimental.allowSubAgents) {
                    delete event.tools?.compress
                    return
                }
                if (state.compressPermission === "deny") delete event.tools?.compress
                state.modelContextLimit = limits.get(`${event.model.providerID}/${event.model.id}`)
                state.systemPromptTokens = countTokens(
                    event.system.map((part) => part.text).join("\n"),
                )
                const view = project(event.messages, entries, {
                    ...session,
                    agent: event.agent,
                    model: event.model,
                })
                const entryIds = new Set(entries.map((entry) => entry.id))
                const missingFromContext = event.messages.filter(
                    (message) => message.id && !entryIds.has(message.id),
                ).length
                if (missingFromContext > 0 && !partialContextWarned.has(event.sessionID)) {
                    partialContextWarned.add(event.sessionID)
                    logger.warn("DCP is projecting native messages missing from session.context", {
                        sessionID: event.sessionID,
                        missing: missingFromContext,
                        native: event.messages.length,
                        context: entries.length,
                    })
                }
                if (kind === "context") {
                    const tail = view.messages
                        .filter((message) => !entryIds.has(message.info.id))
                        .map((message) => structuredClone(message))
                    if (tail.length > 0) {
                        fullContext.set(event.sessionID, tail)
                    } else {
                        fullContext.delete(event.sessionID)
                    }
                }
                removeDcpNotices(view.messages)
                stripHallucinations(view.messages, state.idFormat)
                assignMessageRefs(state, view.messages)
                // Compaction may select only a prefix; block origins can be in the retained tail.
                const syncMessages = new Map<string, WithParts>()
                for (const message of messages) syncMessages.set(message.info.id, message)
                for (const message of view.messages) syncMessages.set(message.info.id, message)
                syncCompressionBlocks(state, logger, [...syncMessages.values()])
                syncToolCache(state, config, logger, view.messages)
                buildToolIdList(state, view.messages)
                prune(state, logger, config, view.messages, view.summaryBase)
                activeContextIds.set(
                    event.sessionID,
                    new Set(view.messages.map((message) => message.info.id)),
                )
                await injectExtendedSubAgentResults(
                    client,
                    state,
                    logger,
                    view.messages,
                    config.experimental.allowSubAgents,
                )
                const priorities = buildPriorityMap(config, state, view.messages)
                prompts.reload()
                injectCompressNudges(
                    state,
                    config,
                    logger,
                    view.messages,
                    prompts.getRuntimePrompts(),
                    priorities,
                    kind === "context",
                )
                injectMessageIds(state, config, view.messages, priorities)
                applyPendingManualTrigger(state, view.messages, logger)
                event.messages = view.restore()
                const system = { system: event.system.map((part) => part.text) }
                await createSystemPromptHandler(
                    state,
                    logger,
                    config,
                    prompts,
                )(
                    {
                        sessionID: event.sessionID,
                        model: { limit: { context: state.modelContextLimit ?? 0 } },
                    },
                    system,
                )
                event.system = system.system.map((text, index) => ({
                    ...event.system[index],
                    type: "text",
                    text,
                }))
                await logger.saveContext(event.sessionID, view.messages)
            })
        })

    const define = (state: SessionState): ToolDefinition =>
        (config.compress.mode === "message" ? createCompressMessageTool : createCompressRangeTool)({
            client,
            state,
            logger,
            config,
            prompts,
            activeMessageIds: state.sessionId ? activeContextIds.get(state.sessionId) : undefined,
        })
    const oobHost: OutOfBandToolHost = {
        config,
        logger,
        runtime: oobRuntime,
        loadTarget: async (sessionID) => {
            const loaded = await load(sessionID)
            const params = getCurrentParams(loaded.state, loaded.messages, logger)
            if (params.providerId && params.modelId) {
                const limit = limits.get(`${params.providerId}/${params.modelId}`)
                if (limit) {
                    loaded.state.modelContextLimit = limit
                }
            }
            const activeIds = activeContextIds.get(sessionID)
            const messages = activeIds
                ? loaded.messages.filter(
                      (message) =>
                          activeIds.has(message.info.id) &&
                          !isMessageCompacted(loaded.state, message),
                  )
                : recentContextMessages(loaded.state, loaded.messages)
            return { state: loaded.state, messages, allMessages: loaded.messages }
        },
        legacyFor: (state: SessionState) =>
            createCompressRangeTool({
                client,
                state,
                logger,
                config,
                prompts,
                activeMessageIds: state.sessionId
                    ? activeContextIds.get(state.sessionId)
                    : undefined,
            }) as any,
    }
    const oobToolDefinitions = createOutOfBandTools(oobHost)

    await ctx.tool.transform((editor) => {
        if (config.compress.permission !== "deny") {
            const definition = define(createSessionState("compact"))
            editor.add({
                name: "compress",
                description: definition.description,
                input: tool.schema.object(definition.args),
                options: { codemode: false, permission: "compress" },
                execute: (input, context) =>
                    serial(context.sessionID, async () => {
                        const { state } = await load(context.sessionID, context.agent)
                        allowed(state)
                        const started = Date.now()
                        const legacy = define(state)
                        const content = await legacy.execute(input, {
                            sessionID: context.sessionID,
                            messageID: context.messageID,
                            callID: context.id,
                            agent: context.agent,
                            directory: ctx.location.directory,
                            worktree: ctx.location.directory,
                            abort: new AbortController().signal,
                            ask: async () => {
                                allowed(state)
                                if (state.compressPermission !== "ask") return
                                if (!permission.assert)
                                    throw new Error(
                                        "DCP: compress permission 'ask' requires OpenCode 2.0.29 or newer. Compression was not performed.",
                                    )
                                await permission.assert({
                                    sessionID: context.sessionID,
                                    agent: context.agent,
                                    action: "compress",
                                    resources: ["*"],
                                    save: ["*"],
                                    source: {
                                        type: "tool",
                                        messageID: context.messageID,
                                        id: context.id,
                                    },
                                })
                            },
                            metadata: ({ title }: { title?: string }) => {
                                void context.progress({ title })
                            },
                        } as Parameters<typeof legacy.execute>[1])
                        attachCompressionDuration(
                            state.prune.messages,
                            context.messageID,
                            context.id,
                            Date.now() - started,
                        )
                        await saveSessionState(state, logger)
                        // Both shared compression executors return text; the V1 helper's
                        // public return type also permits unrelated attachment results.
                        return { content: content as string }
                    }),
            })
        }
        for (const entry of oobToolDefinitions) {
            editor.add({
                name: entry.name,
                description: entry.definition.description,
                input: tool.schema.object(entry.definition.args),
                output: tool.schema.string(),
                options: { codemode: false, permission: "dcp_oob" },
                execute: (input, context) =>
                    entry.definition.execute(input as any, context as any) as any,
            })
        }
    })
    if (config.commands.enabled)
        await ctx.command.transform((editor) => {
            for (const name of ["dcp", "dcp-compress"])
                editor.add({
                    name,
                    description: name === "dcp" ? "DCP commands" : "Trigger DCP manual compression",
                    execute: async (invocation) => {
                        try {
                            const prompt = await serial(invocation.sessionID, async () => {
                                const { state, messages } = await load(invocation.sessionID)
                                const params = getCurrentParams(state, messages, logger)
                                if (params.providerId && params.modelId) {
                                    const limit = limits.get(
                                        `${params.providerId}/${params.modelId}`,
                                    )
                                    if (limit) {
                                        state.modelContextLimit = limit
                                    }
                                }
                                const permission =
                                    state.compressPermission ?? config.compress.permission
                                if (
                                    name === "dcp-compress" ||
                                    invocation.prompt.text.trim().split(/\s+/)[0] === "compress"
                                )
                                    allowed(state)
                                const output = { parts: [] }
                                await createCommandExecuteHandler(
                                    client,
                                    state,
                                    logger,
                                    config,
                                    ctx.location.directory,
                                    { global: { compress: permission }, agents: {} },
                                    {
                                        modelLimit: state.modelContextLimit,
                                        notifyOverLimit: async ({
                                            sessionID,
                                            live,
                                            limit,
                                            mode,
                                        }) => {
                                            await notifyOutOfBandOverLimit({
                                                session: ctx.session as any,
                                                sessionID,
                                                logger,
                                                live,
                                                limit,
                                                mode,
                                            })
                                        },
                                        sendMessage: async ({ sessionID, text }) => {
                                            await ctx.session.synthetic({
                                                sessionID,
                                                text: `${DCP_NOTICE_PREFIX}\n${text}`,
                                                description: "DCP command output",
                                            })
                                        },
                                        runOutOfBand: async ({
                                            sessionID,
                                            instruction,
                                            messages,
                                        }) =>
                                            runOutOfBandAgent({
                                                session: ctx.session as any,
                                                host: oobHost,
                                                state,
                                                messages,
                                                sessionID,
                                                directory: ctx.location.directory,
                                                instruction,
                                                onChildSessionCreated: (childSessionID) =>
                                                    oobSessions.add(childSessionID),
                                                onChildSessionRemoved: (childSessionID) =>
                                                    oobSessions.delete(childSessionID),
                                            }),
                                    },
                                )(
                                    {
                                        command: name,
                                        sessionID: invocation.sessionID,
                                        arguments: invocation.prompt.text,
                                    },
                                    output,
                                )
                                state.compressPermission = permission
                                const pending = state.pendingManualTrigger
                                if (!pending) return
                                allowed(state)
                                state.pendingManualTrigger = null
                                return pending.prompt
                            })
                            if (prompt)
                                await ctx.session.prompt({
                                    sessionID: invocation.sessionID,
                                    text: prompt,
                                    delivery: invocation.delivery,
                                })
                        } catch (error: any) {
                            const message = error?.message ?? String(error)
                            logger.error("DCP command failed", {
                                command: name,
                                sessionID: invocation.sessionID,
                                error: message,
                            })
                            try {
                                await ctx.session.synthetic({
                                    sessionID: invocation.sessionID,
                                    text: `DCP internal notice: DCP command failed: ${message}`,
                                    description: "DCP command error",
                                })
                            } catch (sendError: any) {
                                logger.error("Failed to report DCP command error", {
                                    error: sendError?.message ?? String(sendError),
                                })
                            }
                        }
                    },
                })
        })
    await ctx.rpc.register(rpc, {
        status: async () => ({ enabled: config.commands.enabled, ...getPluginIdentity() }),
        snapshot: ({ sessionID }) =>
            serial(sessionID, async () => {
                const { state, messages } = await load(sessionID)
                syncCompressionBlocks(state, logger, messages)
                return {
                    manualMode: !!state.manualMode,
                    canCompress:
                        state.compressPermission !== "deny" &&
                        (!state.isSubAgent || config.experimental.allowSubAgents),
                    context: analyzeContextTokens(state, messages),
                    stats: await buildStatsReport(state, logger),
                }
            }),
        manual: ({ sessionID, enabled }) =>
            serial(sessionID, async () => {
                const { state } = await load(sessionID)
                state.manualMode = enabled ? "active" : false
                await saveSessionState(state, logger)
                return {}
            }),
    })
    logger.info("DCP V2 initialized")
    return () => {
        sessions.clear()
        limits.clear()
        fullContext.clear()
        partialContextWarned.clear()
        oobRuntime.runs.clear()
        oobSessions.clear()
    }
}
