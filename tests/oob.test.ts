import assert from "node:assert/strict"
import test from "node:test"
import { existsSync, unlinkSync } from "node:fs"
import { homedir } from "node:os"
import { join } from "node:path"
import {
    buildOutOfBandPrompt,
    buildTranscript,
    extractAssistantText,
    isDcpNoticeMessage,
    liveContextTokens,
    parseOutOfBandFlag,
    removeDcpNotices,
    resolveContextTarget,
    resolveOutOfBandDecision,
    runOutOfBandCompression,
    selectOutOfBandBlock,
    selectOutOfBandRange,
} from "../lib/compress/oob"
import { createCommandExecuteHandler } from "../lib/hooks"
import { assignMessageRefs } from "../lib/message-ids"
import { createSessionState } from "../lib/state"

function message(id: string, role: "user" | "assistant", created: number, text: string): any {
    return {
        info: {
            id,
            role,
            sessionID: "ses_test",
            time: { created },
            agent: "build",
            model: { providerID: "provider", modelID: "model" },
            tokens:
                role === "assistant"
                    ? { input: 5000, output: 100, reasoning: 0, cache: { read: 0, write: 0 } }
                    : undefined,
        },
        parts: [{ id: `${id}:text`, type: "text", text }],
    }
}

function logger(): any {
    return {
        info() {},
        debug() {},
        warn() {},
        error() {},
        saveContext: async () => {},
    }
}

function config(overrides: Record<string, any> = {}): any {
    return {
        compress: {
            mode: "range",
            permission: "allow",
            maxContextLimit: 1,
            outOfBand: "allow",
            outOfBandChunkTokens: 100000,
            outOfBandMaxChunks: 1,
            ...overrides,
        },
    }
}

function commandConfig(mode: string): any {
    return {
        ...config({ outOfBand: mode }),
        enabled: true,
        commands: { enabled: true, protectedTools: [] },
        manualMode: { enabled: false, automaticStrategies: true },
    }
}

function commandClient(messages: any[]): any {
    return {
        session: {
            messages: async () => ({ data: messages }),
            get: async () => ({ data: { id: "session-1", agent: "build" } }),
        },
    }
}

test("parses out-of-band flags and preserves the focus", () => {
    assert.deepEqual(parseOutOfBandFlag("find auth --oob"), {
        requested: true,
        cleaned: "find auth",
    })
    assert.deepEqual(parseOutOfBandFlag("--out-of-band"), { requested: true, cleaned: "" })
    assert.deepEqual(parseOutOfBandFlag("find --out-of-band auth"), {
        requested: true,
        cleaned: "find auth",
    })
    assert.deepEqual(parseOutOfBandFlag("find auth"), {
        requested: false,
        cleaned: "find auth",
    })
})

test("resolves the out-of-band decision for each mode", () => {
    assert.equal(
        resolveOutOfBandDecision({ requested: true, mode: "off", overLimit: false }),
        "run",
    )
    assert.equal(
        resolveOutOfBandDecision({ requested: false, mode: "always", overLimit: false }),
        "run",
    )
    assert.equal(
        resolveOutOfBandDecision({ requested: false, mode: "allow", overLimit: true }),
        "run",
    )
    assert.equal(
        resolveOutOfBandDecision({ requested: false, mode: "ask", overLimit: true }),
        "notify",
    )
    assert.equal(
        resolveOutOfBandDecision({ requested: false, mode: "off", overLimit: true }),
        "notify",
    )
    assert.equal(
        resolveOutOfBandDecision({ requested: false, mode: "allow", overLimit: false }),
        "normal",
    )
})

test("resolves numeric and percentage targets against the model limit", () => {
    assert.equal(resolveContextTarget(config({ maxContextLimit: 160000 }), 1000000), 160000)
    assert.equal(
        resolveContextTarget(config({ maxContextLimit: 160000, minContextLimit: 100000 }), 1000000),
        100000,
    )
    assert.equal(resolveContextTarget(config({ maxContextLimit: "10%" }), 1000000), 100000)
    assert.equal(resolveContextTarget(config({ maxContextLimit: "200%" }), 1000000), 1000000)
})

test("recognizes DCP internal notices but not ordinary messages", () => {
    assert.equal(isDcpNoticeMessage(message("m1", "user", 1, "DCP internal notice: hello")), true)
    assert.equal(isDcpNoticeMessage(message("m2", "user", 2, "hello")), false)
})

test("removes DCP notices before messages reach the model", () => {
    const messages = [
        message("m1", "user", 1, "DCP internal notice: started"),
        message("m2", "user", 2, "real user text"),
        message("m3", "user", 3, "DCP internal notice: finished"),
    ]
    assert.equal(removeDcpNotices(messages), 2)
    assert.deepEqual(
        messages.map((item) => item.info.id),
        ["m2"],
    )
})

test("selects the largest active block for recompression", () => {
    const state = createSessionState()
    const base = {
        runId: 1,
        active: true,
        deactivatedByUser: false,
        compressedTokens: 1000,
        durationMs: 1,
        topic: "t",
        startId: "@1@",
        endId: "@1@",
        anchorMessageId: "m1",
        compressMessageId: "m1",
        includedBlockIds: [],
        consumedBlockIds: [],
        parentBlockIds: [],
        directMessageIds: [],
        directToolIds: [],
        effectiveMessageIds: ["m1"],
        effectiveToolIds: [],
        createdAt: 1,
        summary: "summary",
    }
    state.prune.messages.blocksById.set(1, { ...base, blockId: 1, summaryTokens: 20000 } as any)
    state.prune.messages.blocksById.set(2, { ...base, blockId: 2, summaryTokens: 40000 } as any)
    state.prune.messages.blocksById.set(3, {
        ...base,
        blockId: 3,
        summaryTokens: 50000,
        active: false,
    } as any)

    assert.equal(selectOutOfBandBlock(state)?.blockId, 2)
    assert.equal(selectOutOfBandBlock(state, new Set([2]))?.blockId, 1)
})

test("selects the oldest uncompressed range and skips compacted messages", () => {
    const state = createSessionState("compact")
    const messages = [
        message("m1", "user", 1, "first"),
        message("m2", "assistant", 2, "second"),
        message("m3", "user", 3, "current"),
    ]
    assignMessageRefs(state, messages)

    const initial = selectOutOfBandRange(state, messages, 0, 2, 1000000)
    assert.ok(initial)
    assert.equal(initial.startRef, "@1@")
    assert.equal(initial.endRef, "@2@")

    state.prune.messages.blocksById.set(1, {
        blockId: 1,
        runId: 1,
        active: true,
        deactivatedByUser: false,
        compressedTokens: 100,
        summaryTokens: 10,
        durationMs: 1,
        topic: "t",
        startId: "@2@",
        endId: "@2@",
        anchorMessageId: "m2",
        compressMessageId: "m2",
        includedBlockIds: [],
        consumedBlockIds: [],
        parentBlockIds: [],
        directMessageIds: ["m2"],
        directToolIds: [],
        effectiveMessageIds: ["m2"],
        effectiveToolIds: [],
        createdAt: 1,
        summary: "summary",
    })
    state.prune.messages.byMessageId.set("m2", {
        tokenCount: 10,
        allBlockIds: [1],
        activeBlockIds: [1],
    })

    const afterCompaction = selectOutOfBandRange(state, messages, 0, 2, 1000000)
    assert.ok(afterCompaction)
    assert.equal(afterCompaction.startRef, "@1@")
    assert.equal(afterCompaction.endRef, "@1@")
})

test("builds a transcript and prompt that mark the transcript as untrusted", () => {
    const messages = [
        message("m1", "user", 1, "hello"),
        {
            info: { id: "m2", role: "assistant", time: { created: 2 }, sessionID: "s" },
            parts: [
                {
                    type: "tool",
                    callID: "call_1",
                    tool: "read",
                    state: {
                        status: "completed",
                        input: { path: "/tmp/file" },
                        output: "file contents",
                    },
                },
            ],
        } as any,
    ]

    const transcript = buildTranscript(messages, 0, 1)
    assert.match(transcript, /hello/)
    assert.match(transcript, /\[tool read\]/)
    assert.match(transcript, /file contents/)

    const prompt = buildOutOfBandPrompt(transcript, "auth work")
    assert.match(prompt, /untrusted conversation data/)
    assert.match(prompt, /Focus: auth work/)
    assert.match(prompt, /<transcript>/)
})

test("extracts the last assistant text and ignores trailing non-text messages", () => {
    const messages = [
        message("m1", "assistant", 1, "first"),
        message("m2", "assistant", 2, "last"),
        { info: { id: "m3", role: "user", time: { created: 3 } }, parts: [] } as any,
    ]
    assert.equal(extractAssistantText(messages), "last")
})

test("runs one out-of-band chunk through a fresh child session and applies its summary", async () => {
    const state = createSessionState("compact")
    state.sessionId = "ses_target"
    state.modelContextLimit = 1_000_000
    const messages = [
        message("m1", "user", 1, "first ".repeat(800)),
        message("m2", "assistant", 2, "second ".repeat(800)),
        message("m3", "user", 3, "current"),
    ]
    const notices: string[] = []
    const prompts: any[] = []
    const removals: any[] = []
    const creates: any[] = []
    const applied: any[] = []
    const childEntries = [
        { type: "user", id: "c1", text: "compress this", time: { created: 1 } },
        {
            type: "assistant",
            id: "c2",
            agent: "build",
            model: { providerID: "provider", id: "model" },
            time: { created: 2 },
            tokens: { input: 10, output: 2, reasoning: 0, cache: { read: 0, write: 0 } },
            content: [{ type: "text", text: "the compact summary" }],
        },
    ]

    const session = {
        async create(input: any) {
            creates.push(input)
            return { id: "child" }
        },
        async prompt(input: any) {
            prompts.push(input)
        },
        async wait() {},
        async context() {
            return childEntries
        },
        async get() {
            return { id: "child", agent: "build", model: { providerID: "provider", id: "model" } }
        },
        async synthetic(input: any) {
            notices.push(input.text)
        },
        async remove(input: any) {
            removals.push(input)
        },
    }

    const legacy = {
        async execute(args: any, toolCtx: any) {
            applied.push({ args, toolCtx })
            state.prune.messages.blocksById.set(1, {
                blockId: 1,
                runId: 1,
                active: true,
                deactivatedByUser: false,
                compressedTokens: 500,
                summaryTokens: 100,
                durationMs: 1,
                topic: args.topic,
                startId: args.content[0].startId,
                endId: args.content[0].endId,
                anchorMessageId: "m2",
                compressMessageId: toolCtx.messageID,
                includedBlockIds: [],
                consumedBlockIds: [],
                parentBlockIds: [],
                directMessageIds: ["m1", "m2"],
                directToolIds: [],
                effectiveMessageIds: ["m1", "m2"],
                effectiveToolIds: [],
                createdAt: 1,
                summary: args.content[0].summary,
            })
            return "ok"
        },
    }

    const before = liveContextTokens(state, messages)
    assert.ok(before > 1)

    const result = await runOutOfBandCompression({
        session: session as any,
        client: {} as any,
        state,
        config: config(),
        logger: logger(),
        sessionID: "ses_target",
        messages,
        directory: "/tmp/project",
        legacy,
    })

    assert.equal(result.chunks, 1)
    assert.equal(result.stopped, "max-chunks")
    assert.ok(result.after < result.before)
    assert.equal(creates.length, 1)
    assert.equal(creates[0].metadata.dcpOob, true)
    assert.equal(creates[0].agent, "build")
    assert.match(prompts[0].text, /the compact summary is requested|untrusted conversation data/)
    assert.equal(applied.length, 1)
    assert.equal(applied[0].args.content[0].startId, "@1@")
    assert.equal(applied[0].args.content[0].endId, "@2@")
    assert.equal(applied[0].args.content[0].summary, "the compact summary")
    assert.equal(applied[0].toolCtx.messageID, "m3")
    assert.equal(removals.length, 1)
    assert.equal(state.manualMode, false)
    assert.equal(notices.length, 2)
    assert.match(notices[0], /started/)
    assert.match(notices[1], /finished/)
})

test("restores manual mode when an out-of-band chunk fails", async () => {
    const state = createSessionState()
    state.manualMode = "active"
    state.modelContextLimit = 1_000_000
    const messages = [
        message("m1", "user", 1, "first ".repeat(800)),
        message("m2", "assistant", 2, "second ".repeat(800)),
        message("m3", "user", 3, "third"),
    ]
    const session = {
        async create() {
            return { id: "child" }
        },
        async prompt() {},
        async wait() {},
        async context() {
            return [
                {
                    type: "assistant",
                    id: "c1",
                    agent: "build",
                    model: { providerID: "provider", id: "model" },
                    time: { created: 1 },
                    tokens: { input: 10, output: 2, reasoning: 0, cache: { read: 0, write: 0 } },
                    content: [{ type: "text", text: "summary" }],
                },
            ]
        },
        async get() {
            return { id: "child", agent: "build", model: { providerID: "provider", id: "model" } }
        },
        async synthetic() {},
        async remove() {},
    }
    const legacy = {
        async execute() {
            throw new Error("chunk failed")
        },
    }

    const result = await runOutOfBandCompression({
        session: session as any,
        client: {} as any,
        state,
        config: config({ maxContextLimit: 1 }),
        logger: logger(),
        sessionID: "ses_target",
        messages,
        directory: "/tmp/project",
        legacy: legacy as any,
    })

    assert.equal(result.stopped, "failed")
    assert.equal(result.chunks, 0)
    assert.equal(state.manualMode, "active")
})

test("command host refuses out-of-band when compress permission is ask", async () => {
    const calls = { ran: 0 }
    const cfg = commandConfig("allow")
    cfg.compress.permission = "ask"
    const handler = createCommandExecuteHandler(
        commandClient([message("m1", "user", 1, "first"), message("m2", "assistant", 2, "second")]),
        createSessionState(),
        logger(),
        cfg,
        "/tmp",
        { global: undefined, agents: {} },
        {
            modelLimit: 1,
            runOutOfBand: async () => {
                calls.ran += 1
                return { chunks: 0, before: 0, after: 0, stopped: "no-range" }
            },
        },
    )

    await assert.rejects(
        handler(
            {
                command: "dcp-compress",
                sessionID: "session-oob-ask-permission",
                arguments: "--oob",
            },
            { parts: [] },
        ),
        /requires compress permission/,
    )
    assert.equal(calls.ran, 0)
})

test("command host notifies instead of running out-of-band in ask mode", async () => {
    const calls = { notified: 0, ran: 0, focus: "" }
    const state = createSessionState()
    const handler = createCommandExecuteHandler(
        commandClient([message("m1", "user", 1, "first"), message("m2", "assistant", 2, "second")]),
        state,
        logger(),
        commandConfig("ask"),
        "/tmp",
        { global: undefined, agents: {} },
        {
            modelLimit: 1,
            notifyOverLimit: async () => {
                calls.notified += 1
            },
            runOutOfBand: async ({ focus }) => {
                calls.ran += 1
                calls.focus = focus
                return { chunks: 0, before: 0, after: 0, stopped: "no-range" }
            },
        },
    )

    await handler(
        { command: "dcp-compress", sessionID: "session-oob-ask", arguments: "focus" },
        { parts: [] },
    )

    assert.equal(calls.notified, 1)
    assert.equal(calls.ran, 0)
    assert.equal(state.pendingManualTrigger, null)
})

test("command host runs out-of-band when over the limit and allowed", async () => {
    const calls = { notified: 0, ran: 0, focus: "" }
    const state = createSessionState()
    const handler = createCommandExecuteHandler(
        commandClient([message("m1", "user", 1, "first"), message("m2", "assistant", 2, "second")]),
        state,
        logger(),
        commandConfig("allow"),
        "/tmp",
        { global: undefined, agents: {} },
        {
            modelLimit: 1,
            notifyOverLimit: async () => {
                calls.notified += 1
            },
            runOutOfBand: async ({ focus }) => {
                calls.ran += 1
                calls.focus = focus
                return { chunks: 0, before: 0, after: 0, stopped: "no-range" }
            },
        },
    )

    const output = { parts: [] as any[] }
    await handler(
        { command: "dcp-compress", sessionID: "session-oob-allow", arguments: "auth work" },
        output,
    )

    assert.equal(calls.notified, 0)
    assert.equal(calls.ran, 1)
    assert.equal(calls.focus, "auth work")
    assert.equal(state.pendingManualTrigger, null)
    assert.equal(output.parts.length, 1)
    assert.equal(output.parts[0].ignored, true)
    assert.equal(output.parts[0].synthetic, true)
})

test("command host keeps the normal prompt when under the limit", async () => {
    const calls = { notified: 0, ran: 0 }
    const state = createSessionState()
    const handler = createCommandExecuteHandler(
        commandClient([message("m1", "user", 1, "first"), message("m2", "assistant", 2, "second")]),
        state,
        logger(),
        commandConfig("ask"),
        "/tmp",
        { global: undefined, agents: {} },
        {
            modelLimit: 1_000_000,
            notifyOverLimit: async () => {
                calls.notified += 1
            },
            runOutOfBand: async () => {
                calls.ran += 1
                return { chunks: 0, before: 0, after: 0, stopped: "no-range" }
            },
        },
    )

    const output = { parts: [] as any[] }
    await handler(
        { command: "dcp-compress", sessionID: "session-oob-normal", arguments: "auth work" },
        output,
    )

    assert.equal(calls.notified, 0)
    assert.equal(calls.ran, 0)
    assert.ok(state.pendingManualTrigger)
    assert.equal(output.parts.length, 1)
})

test("recompresses an active summary block when it dominates the context", async () => {
    const sessionID = `ses_oob_block_${Date.now()}`
    const state = createSessionState()
    state.sessionId = sessionID
    state.modelContextLimit = 1_000_000
    const messages = [
        message("m1", "user", 1, "first"),
        message("m2", "assistant", 2, "second"),
        message("m3", "user", 3, "current"),
    ]
    state.prune.messages.blocksById.set(1, {
        blockId: 1,
        runId: 1,
        active: true,
        deactivatedByUser: false,
        compressedTokens: 1000,
        summaryTokens: 20000,
        durationMs: 1,
        topic: "old block",
        startId: "@1@",
        endId: "@2@",
        anchorMessageId: "m1",
        compressMessageId: "m2",
        includedBlockIds: [],
        consumedBlockIds: [],
        parentBlockIds: [],
        directMessageIds: ["m1", "m2"],
        directToolIds: [],
        effectiveMessageIds: ["m1", "m2"],
        effectiveToolIds: [],
        createdAt: 1,
        summary: "old summary",
    } as any)
    state.prune.messages.activeBlockIds.add(1)
    state.prune.messages.activeByAnchorMessageId.set("m1", 1)
    state.prune.messages.nextBlockId = 2
    state.prune.messages.byMessageId.set("m1", {
        tokenCount: 1000,
        allBlockIds: [1],
        activeBlockIds: [1],
    })
    state.prune.messages.byMessageId.set("m2", {
        tokenCount: 1000,
        allBlockIds: [1],
        activeBlockIds: [1],
    })

    const notices: string[] = []
    const session = {
        async create() {
            return { id: "child" }
        },
        async prompt() {},
        async wait() {},
        async context() {
            return [
                {
                    type: "assistant",
                    id: "c1",
                    agent: "build",
                    model: { providerID: "provider", id: "model" },
                    time: { created: 1 },
                    tokens: { input: 10, output: 2, reasoning: 0, cache: { read: 0, write: 0 } },
                    content: [{ type: "text", text: "short summary" }],
                },
            ]
        },
        async get() {
            return { id: "child", agent: "build", model: { providerID: "provider", id: "model" } }
        },
        async synthetic(input: any) {
            notices.push(input.text)
        },
        async remove() {},
    }

    const result = await runOutOfBandCompression({
        session: session as any,
        client: {} as any,
        state,
        config: config({ maxContextLimit: 1 }),
        logger: logger(),
        sessionID,
        messages,
        directory: "/tmp/project",
        legacy: { execute: async () => "" },
    })

    assert.equal(result.chunks, 1)
    assert.equal(result.stopped, "target")
    assert.equal(state.prune.messages.blocksById.get(1)?.active, false)
    const newer = [...state.prune.messages.blocksById.values()].find((block) => block.blockId !== 1)
    assert.ok(newer)
    assert.equal(newer?.active, true)
    assert.match(newer?.summary ?? "", /short summary/)
    assert.ok(state.prune.messages.byMessageId.get("m1")?.activeBlockIds.includes(newer!.blockId))
    assert.ok(notices.every((text) => text.startsWith("DCP internal notice: ")))

    const dataHome = process.env.XDG_DATA_HOME || join(homedir(), ".local", "share")
    const statePath = join(dataHome, "opencode", "storage", "plugin", "dcp", `${sessionID}.json`)
    if (existsSync(statePath)) {
        unlinkSync(statePath)
    }
})
