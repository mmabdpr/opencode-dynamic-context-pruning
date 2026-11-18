import assert from "node:assert/strict"
import test from "node:test"
import {
    applyRecompressedBlock,
    buildOutOfBandAgentPrompt,
    buildTranscript,
    extractAssistantText,
    extractProtectedTail,
    isolateOutOfBandTools,
    isDcpNoticeMessage,
    liveContextTokens,
    OOB_DEFAULT_INSTRUCTION,
    parseOutOfBandFlag,
    projectContextTokens,
    PROTECTED_CONTENT_HEADINGS,
    removeDcpNotices,
    resolveOutOfBandDecision,
    selectOutOfBandBlock,
    selectOutOfBandRange,
} from "../lib/compress/oob"
import { createCommandExecuteHandler } from "../lib/hooks"
import { assignMessageRefs } from "../lib/message-ids"
import { createSessionState } from "../lib/state"
import { countTokens } from "../lib/token-utils"
import { wrapCompressedSummary } from "../lib/compress/state"

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

function buildActiveBlockState(summaryBody = "old summary") {
    const state = createSessionState("compact")
    state.sessionId = null
    const messages = [
        message("m1", "user", 1, "first"),
        message("m2", "assistant", 2, "second"),
        message("m3", "user", 3, "current"),
    ]
    assignMessageRefs(state, messages)
    const summary = wrapCompressedSummary(1, summaryBody, "compact")
    const block = {
        blockId: 1,
        runId: 1,
        active: true,
        deactivatedByUser: false,
        compressedTokens: 1000,
        summaryTokens: countTokens(summary),
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
        summary,
    } as any
    state.prune.messages.blocksById.set(1, block)
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
    return { state, messages, block }
}

test("parses out-of-band flags and preserves the free-form instruction", () => {
    assert.deepEqual(parseOutOfBandFlag("find auth --oob"), {
        requested: true,
        cleaned: "find auth",
    })
    assert.deepEqual(parseOutOfBandFlag("--out-of-band"), { requested: true, cleaned: "" })
    assert.deepEqual(parseOutOfBandFlag("keep only auth decisions until 20% remains --oob"), {
        requested: true,
        cleaned: "keep only auth decisions until 20% remains",
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
        "normal",
    )
    assert.equal(
        resolveOutOfBandDecision({ requested: false, mode: "allow", overLimit: false }),
        "normal",
    )
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

test("selects the largest active block and skips oversized inactive blocks", () => {
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

test("builds a transcript marked as untrusted", () => {
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
})

test("extracts the last assistant text and ignores trailing non-text messages", () => {
    const messages = [
        message("m1", "assistant", 1, "first"),
        message("m2", "assistant", 2, "last"),
        { info: { id: "m3", role: "user", time: { created: 3 } }, parts: [] } as any,
    ]
    assert.equal(extractAssistantText(messages), "last")
})

test("builds an autonomous prompt that carries the user's instruction verbatim", () => {
    const instruction = "Compress until about half the context remains, then leave auth alone."
    const prompt = buildOutOfBandAgentPrompt({
        targetSessionID: "ses_target",
        projectedTokens: 500000,
        modelLimit: 1000000,
        instruction,
        round: 2,
        roundWriteBudget: 4,
        roundReadBudget: 400000,
        globalWritesRemaining: 30,
        globalReadsRemaining: 7000000,
    })

    assert.match(prompt, /autonomous out-of-band context compression agent/)
    assert.match(prompt, /ses_target/)
    assert.match(prompt, /500,000/)
    assert.match(prompt, /1,000,000/)
    assert.ok(prompt.includes(instruction))
    assert.match(prompt, /does not parse it/)
    assert.match(prompt, /untrusted/)
    assert.match(prompt, /dcp_oob_overview/)
    assert.match(prompt, /Do not claim the target was reached/)
    assert.match(prompt, /worker round 2/)
    assert.match(prompt, /fresh context/)
    assert.match(prompt, /at most 4 successful write/)
    assert.match(prompt, /400,000 read tokens/)
    assert.match(prompt, /DCP-OOB: DONE/)
    assert.match(prompt, /DCP-OOB: CONTINUE/)
    assert.match(prompt, /omit the control line.*treats it as DONE/i)
})

test("uses the default instruction only when the user instruction is empty", () => {
    const prompt = buildOutOfBandAgentPrompt({
        targetSessionID: "ses_target",
        projectedTokens: 10,
        modelLimit: 0,
        instruction: "   ",
    })
    assert.ok(prompt.includes(OOB_DEFAULT_INSTRUCTION))
    assert.match(prompt, /unknown/)
})

test("isolates dcp_oob tools by session kind", () => {
    const ordinary = {
        compress: {},
        shell: {},
        dcp_oob_overview: {},
        dcp_oob_read: {},
        dcp_oob_compress: {},
    }
    isolateOutOfBandTools(ordinary, false)
    assert.deepEqual(Object.keys(ordinary).sort(), ["compress", "shell"])

    const oob = {
        compress: {},
        shell: {},
        dcp_oob_overview: {},
        dcp_oob_read: {},
        dcp_oob_compress: {},
    }
    isolateOutOfBandTools(oob, true)
    assert.deepEqual(Object.keys(oob).sort(), [
        "dcp_oob_compress",
        "dcp_oob_overview",
        "dcp_oob_read",
    ])
})

test("extracts the protected tail starting at the earliest heading", () => {
    const summary = [
        "[Compressed conversation section]",
        "normal summary",
        "",
        PROTECTED_CONTENT_HEADINGS[1],
        "prompt payload",
        "",
        PROTECTED_CONTENT_HEADINGS[2],
        "tool payload",
        "",
        "@b1@",
    ].join("\n")
    const tail = extractProtectedTail(summary)
    assert.ok(tail.startsWith(PROTECTED_CONTENT_HEADINGS[1]))
    assert.match(tail, /tool payload/)
    assert.doesNotMatch(tail, /normal summary/)
    assert.doesNotMatch(tail, /@b1@/)
})

test("replaces an active block with a smaller summary and preserves protected content", async () => {
    const protectedBody = [
        "old summary ".repeat(50),
        PROTECTED_CONTENT_HEADINGS[0],
        "verbatim user prompt",
        "",
        PROTECTED_CONTENT_HEADINGS[2],
        "protected tool output",
    ].join("\n")
    const { state, messages, block } = buildActiveBlockState(protectedBody)
    state.stats.pruneTokenCounter = 12
    state.stats.totalPruneTokens = 34

    const result = await applyRecompressedBlock({
        state,
        messages,
        logger: logger(),
        block,
        summary: "[Compressed conversation section]\nnew short summary\n\n@b1@",
        topic: "New topic",
    })

    assert.ok(result.summaryTokens < block.summaryTokens)
    assert.equal(state.prune.messages.blocksById.get(1)?.active, false)
    const created = state.prune.messages.blocksById.get(result.blockId)
    assert.ok(created)
    assert.equal(created?.active, true)
    assert.match(created?.summary ?? "", /new short summary/)
    assert.match(created?.summary ?? "", /verbatim user prompt/)
    assert.match(created?.summary ?? "", /protected tool output/)
    assert.equal((created?.summary.match(/\[Compressed conversation section\]/g) ?? []).length, 1)
    assert.match(created?.summary ?? "", /@b2@$/)
    assert.doesNotMatch(created?.summary ?? "", /@b1@/)
    assert.deepEqual(created?.consumedBlockIds, [1])
    assert.deepEqual(created?.includedBlockIds, [1])
    assert.deepEqual(created?.effectiveMessageIds.sort(), ["m1", "m2"])
    assert.equal(state.stats.pruneTokenCounter, 12)
    assert.equal(state.stats.totalPruneTokens, 34)
    assert.ok(state.prune.messages.byMessageId.get("m1")?.activeBlockIds.includes(result.blockId))
})

test("rejects a block replacement that is not smaller after protected content", async () => {
    const { state, messages, block } = buildActiveBlockState("tiny")
    await assert.rejects(
        applyRecompressedBlock({
            state,
            messages,
            logger: logger(),
            block,
            summary: "word ".repeat(5000),
        }),
        /not smaller/,
    )
    assert.equal(state.prune.messages.blocksById.get(1)?.active, true)
})

test("rejects replacing a block whose summary changed after it was read", async () => {
    const { state, messages, block } = buildActiveBlockState("old summary")
    await assert.rejects(
        applyRecompressedBlock({
            state,
            messages,
            logger: logger(),
            block,
            summary: "short",
            expectedSummary: "a different summary",
        }),
        /changed since it was read/,
    )
})

test("command host sends context output through the V2 synthetic channel", async () => {
    const sent: Array<{ sessionID: string; text: string }> = []
    const state = createSessionState()
    state.sessionId = "session-context"
    state.systemPromptTokens = 14_000
    const handler = createCommandExecuteHandler(
        commandClient([message("m1", "user", 1, "first"), message("m2", "assistant", 2, "second")]),
        state,
        logger(),
        commandConfig("off"),
        "/tmp",
        { global: undefined, agents: {} },
        {
            sendMessage: async ({ sessionID, text }) => {
                sent.push({ sessionID, text })
            },
        },
    )

    const output = { parts: [] as any[] }
    await handler({ command: "dcp", sessionID: "session-context", arguments: "context" }, output)

    assert.equal(sent.length, 1)
    assert.equal(sent[0]?.sessionID, "session-context")
    assert.match(sent[0]?.text ?? "", /DCP Context Analysis/)
    assert.match(sent[0]?.text ?? "", /Live Context Breakdown/)
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
                return {
                    chunks: 0,
                    rounds: 0,
                    before: 0,
                    after: 0,
                    stopped: "completed",
                    blocks: 0,
                }
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
            modelLimit: 1,
            notifyOverLimit: async () => {
                calls.notified += 1
            },
            runOutOfBand: async () => {
                calls.ran += 1
                return {
                    chunks: 0,
                    rounds: 0,
                    before: 0,
                    after: 0,
                    stopped: "completed",
                    blocks: 0,
                }
            },
        },
    )

    const output = { parts: [] as any[] }
    await handler(
        { command: "dcp-compress", sessionID: "session-oob-ask", arguments: "focus" },
        output,
    )

    assert.equal(calls.notified, 1)
    assert.equal(calls.ran, 0)
    assert.equal(state.pendingManualTrigger, null)
    assert.equal(output.parts.length, 1)
    assert.equal(output.parts[0].ignored, true)
    assert.equal(output.parts[0].synthetic, true)
})

test("command host passes the free-form instruction to the out-of-band agent", async () => {
    const calls = { notified: 0, ran: 0, instruction: "" }
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
            runOutOfBand: async ({ instruction }) => {
                calls.ran += 1
                calls.instruction = instruction
                return {
                    chunks: 0,
                    rounds: 0,
                    before: 0,
                    after: 0,
                    stopped: "completed",
                    blocks: 0,
                }
            },
        },
    )

    const output = { parts: [] as any[] }
    await handler(
        {
            command: "dcp-compress",
            sessionID: "session-oob-allow",
            arguments: "--oob keep auth decisions until about 20% remains",
        },
        output,
    )

    assert.equal(calls.notified, 0)
    assert.equal(calls.ran, 1)
    assert.equal(calls.instruction, "keep auth decisions until about 20% remains")
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
                return {
                    chunks: 0,
                    rounds: 0,
                    before: 0,
                    after: 0,
                    stopped: "completed",
                    blocks: 0,
                }
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

test("live context tokens stays consistent with the context breakdown", () => {
    const state = createSessionState("compact")
    state.sessionId = null
    const messages = [message("m1", "user", 1, "hello"), message("m2", "assistant", 2, "world")]
    assert.ok(liveContextTokens(state, messages) >= 5000)
})

test("OOB projection trusts a stale reported total only as a fresh bound", () => {
    const build = (blockCreatedAt: number) => {
        const state = createSessionState()
        state.systemPromptTokens = 14_000
        const assistant = message("assistant", "assistant", 2, "done")
        assistant.info.tokens = {
            input: 407_000,
            output: 100,
            reasoning: 0,
            cache: { read: 0, write: 0 },
        }
        const messages = [message("user", "user", 1, "next request"), assistant]
        state.prune.messages.blocksById.set(1, {
            blockId: 1,
            runId: 1,
            active: true,
            deactivatedByUser: false,
            compressedTokens: 10_000,
            summaryTokens: 1_000,
            durationMs: 1,
            topic: "summary",
            startId: "@1@",
            endId: "@1@",
            anchorMessageId: "user",
            compressMessageId: "assistant",
            includedBlockIds: [],
            consumedBlockIds: [],
            parentBlockIds: [],
            directMessageIds: ["user"],
            directToolIds: [],
            effectiveMessageIds: ["user"],
            effectiveToolIds: [],
            createdAt: blockCreatedAt,
            summary: "summary",
        } as any)
        state.prune.messages.activeBlockIds.add(1)
        return { state, messages }
    }

    const stale = build(3)
    const staleProjection = projectContextTokens(stale.state, stale.messages, 0)
    assert.equal(staleProjection.reportedContextTokens, 407_100)
    assert.ok(staleProjection.projectedContextTokens < 20_000)

    const fresh = build(1)
    const freshProjection = projectContextTokens(fresh.state, fresh.messages, 10_000)
    assert.equal(freshProjection.projectedContextTokens, 407_100 - 10_000)
})
