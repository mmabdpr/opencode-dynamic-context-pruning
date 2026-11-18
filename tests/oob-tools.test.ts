import assert from "node:assert/strict"
import test, { after } from "node:test"
import { existsSync, unlinkSync } from "node:fs"
import { homedir } from "node:os"
import { join } from "node:path"
import {
    buildTranscript,
    createOutOfBandRuntime,
    createOutOfBandTools,
    OOB_DEFAULT_INSTRUCTION,
    OOB_DEFAULT_READ_TOKENS,
    OOB_READ_TOKEN_BUDGET,
    OOB_ROUND_READ_TOKEN_BUDGET,
    OOB_WALL_CLOCK_MS,
    parseOutOfBandWorkerDecision,
    PROTECTED_CONTENT_HEADINGS,
    runOutOfBandAgent,
    targetRevision,
    type OobRun,
    type OutOfBandToolHost,
} from "../lib/compress/oob"
import { assignMessageRefs } from "../lib/message-ids"
import { createSessionState, type SessionState, type WithParts } from "../lib/state"
import { countAllMessageTokens, countTokens } from "../lib/token-utils"
import { allocateBlockId, wrapCompressedSummary } from "../lib/compress/state"
import { syncCompressionBlocks } from "../lib/messages"

const sessionsToClean = new Set<string>()
const dataHome = process.env.XDG_DATA_HOME || join(homedir(), ".local", "share")

after(() => {
    for (const sessionID of sessionsToClean) {
        const statePath = join(
            dataHome,
            "opencode",
            "storage",
            "plugin",
            "dcp",
            `${sessionID}.json`,
        )
        if (existsSync(statePath)) {
            unlinkSync(statePath)
        }
    }
})

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
            outOfBandMaxChunks: 3,
            ...overrides,
        },
    }
}

let sessionCounter = 0

function createTarget(options: { withBlock?: boolean } = {}) {
    const sessionID = `ses_oob_tools_${Date.now()}_${sessionCounter++}`
    sessionsToClean.add(sessionID)
    const state = createSessionState("compact")
    state.sessionId = sessionID
    state.compressPermission = "allow"
    state.modelContextLimit = 100000
    const messages: WithParts[] = [
        message("m1", "user", 1, "alpha ".repeat(100)),
        message("m2", "assistant", 2, "beta ".repeat(100)),
        message("m3", "user", 3, "gamma ".repeat(100)),
        message("m4", "user", 4, "current request"),
    ]
    assignMessageRefs(state, messages)

    let block: any
    if (options.withBlock) {
        const body = [
            "old block summary ".repeat(200),
            PROTECTED_CONTENT_HEADINGS[0],
            "verbatim user prompt",
        ].join("\n")
        const summary = wrapCompressedSummary(1, body, "compact")
        block = {
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
            anchorMessageId: "m2",
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
        }
        state.prune.messages.blocksById.set(1, block)
        state.prune.messages.activeBlockIds.add(1)
        state.prune.messages.activeByAnchorMessageId.set("m2", 1)
        state.prune.messages.nextBlockId = 2
        for (const id of ["m1", "m2"]) {
            state.prune.messages.byMessageId.set(id, {
                tokenCount: 500,
                allBlockIds: [1],
                activeBlockIds: [1],
            })
        }
    }

    syncCompressionBlocks(state, logger(), messages)
    return { sessionID, state, messages, block }
}

function createHarness(
    options: { withBlock?: boolean; maxWrites?: number; maxToolCalls?: number } = {},
) {
    const target = createTarget(options)
    const runtime = createOutOfBandRuntime()
    const legacyCalls: Array<{ args: any; toolCtx: any }> = []

    const fakeLegacyFor = (state: SessionState) => ({
        async execute(args: any, toolCtx: any) {
            legacyCalls.push({ args, toolCtx })
            const entry = args.content[0]
            const startId = state.messageIds.byRef.get(entry.startId)
            const endId = state.messageIds.byRef.get(entry.endId)
            if (!startId || !endId) {
                throw new Error("fake legacy could not resolve refs")
            }
            const startIndex = target.messages.findIndex((item) => item.info.id === startId)
            const endIndex = target.messages.findIndex((item) => item.info.id === endId)
            if (startIndex === -1 || endIndex === -1 || endIndex < startIndex) {
                throw new Error("fake legacy range is invalid")
            }
            const included = target.messages.slice(startIndex, endIndex + 1)
            const effectiveMessageIds = included.map((item) => item.info.id)
            const compressedTokens = included.reduce(
                (sum, item) => sum + countAllMessageTokens(item),
                0,
            )
            const blockId = allocateBlockId(state)
            const summary = wrapCompressedSummary(blockId, entry.summary, state.idFormat)
            const summaryTokens = countTokens(summary)
            const anchorMessageId = included[0]!.info.id
            state.prune.messages.blocksById.set(blockId, {
                blockId,
                runId: 1,
                active: true,
                deactivatedByUser: false,
                compressedTokens,
                summaryTokens,
                durationMs: 1,
                topic: args.topic,
                startId: entry.startId,
                endId: entry.endId,
                anchorMessageId,
                compressMessageId: toolCtx.messageID,
                compressCallId: toolCtx.callID,
                includedBlockIds: [],
                consumedBlockIds: [],
                parentBlockIds: [],
                directMessageIds: effectiveMessageIds,
                directToolIds: [],
                effectiveMessageIds,
                effectiveToolIds: [],
                createdAt: Date.now(),
                summary,
            })
            state.prune.messages.activeBlockIds.add(blockId)
            state.prune.messages.activeByAnchorMessageId.set(anchorMessageId, blockId)
            for (const id of effectiveMessageIds) {
                const existing = state.prune.messages.byMessageId.get(id)
                if (existing) {
                    existing.allBlockIds.push(blockId)
                    existing.activeBlockIds.push(blockId)
                } else {
                    state.prune.messages.byMessageId.set(id, {
                        tokenCount: 0,
                        allBlockIds: [blockId],
                        activeBlockIds: [blockId],
                    })
                }
            }
            return "Compressed messages."
        },
    })

    const hostLogger = logger()
    const host: OutOfBandToolHost = {
        config: config({ outOfBandMaxChunks: options.maxWrites ?? 3 }),
        logger: hostLogger,
        runtime,
        loadTarget: async (sessionID) => {
            if (sessionID !== target.sessionID) {
                throw new Error("unexpected target")
            }
            assignMessageRefs(target.state, target.messages)
            syncCompressionBlocks(target.state, hostLogger, target.messages)
            return { state: target.state, messages: target.messages }
        },
        legacyFor: fakeLegacyFor,
    }
    const definitions = createOutOfBandTools(host)
    const tools = new Map(definitions.map((entry) => [entry.name, entry.definition]))

    function registerRun(overrides: Partial<OobRun> = {}): OobRun {
        const run: OobRun = {
            targetSessionID: target.sessionID,
            childSessionID: "child-1",
            runId: "run-1",
            startedAt: Date.now(),
            expiresAt: Date.now() + 60_000,
            agent: "build",
            directory: "/tmp/project",
            instruction: "test instruction",
            toolCalls: 0,
            writes: 0,
            readTokens: 0,
            maxToolCalls: options.maxToolCalls ?? 20,
            maxWrites: options.maxWrites ?? 3,
            maxReadTokens: 2_000_000,
            lastProjectedTokens: 0,
            reclaimedTokens: 0,
            consecutiveNoProgress: 0,
            status: "running",
            initialBlockIds: [...target.state.prune.messages.activeBlockIds],
            revision: targetRevision(target.state, target.messages),
            blockReads: new Map(),
            rangeReads: new Map(),
            ...overrides,
        }
        runtime.register(run)
        return run
    }

    async function call(name: string, args: Record<string, unknown>, childID = "child-1") {
        const definition = tools.get(name)
        assert.ok(definition, `missing tool ${name}`)
        const result = await definition.execute(args as any, { sessionID: childID } as any)
        return typeof result === "string" ? result : (result as { output: string }).output
    }

    return { target, host, runtime, legacyCalls, registerRun, call, tools }
}

function childEntries(report: string) {
    return [
        {
            type: "assistant",
            id: "child-message-1",
            agent: "build",
            model: { providerID: "provider", id: "model" },
            time: { created: 1 },
            tokens: { input: 10, output: 2, reasoning: 0, cache: { read: 0, write: 0 } },
            content: [{ type: "text", text: report }],
        },
    ]
}

test("overview exposes projected tokens, block refs, and paginated ranges", async () => {
    const harness = createHarness({ withBlock: true })
    harness.registerRun()

    const output = await harness.call("dcp_oob_overview", {})
    assert.match(output, /projectedContextTokens: \d+/)
    assert.match(output, /modelContextLimit: 100000/)
    assert.match(output, /activeCompressionBlocks: 1/)
    assert.match(output, /ref: @b1@/)
    assert.match(output, /candidateRanges:/)
    assert.match(output, /startRef: @3@/)
    assert.match(output, /endRef: @3@/)
    assert.match(output, /nextCursor: none/)
})

test("read is cursor-resumable and accounts for cumulative read tokens", async () => {
    const harness = createHarness()
    const run = harness.registerRun()

    const first = await harness.call("dcp_oob_read", {
        startRef: "@1@",
        endRef: "@3@",
        maxTokens: 50,
    })
    assert.match(first, /source: messages @1@\.\.@3@/)
    assert.match(first, /nextCursor: \d+/)
    assert.ok(run.readTokens > 0)
    const next = Number(/nextCursor: (\d+)/.exec(first)![1])

    let output = first
    let cursor = next
    for (let attempt = 0; attempt < 50; attempt++) {
        output = await harness.call("dcp_oob_read", {
            startRef: "@1@",
            endRef: "@3@",
            cursor,
            maxTokens: 50,
        })
        const match = /nextCursor: (\d+|none)/.exec(output)!
        if (match[1] === "none") {
            break
        }
        cursor = Number(match[1])
    }
    assert.match(output, /nextCursor: none/)
    assert.ok(run.readTokens > 50)
    assert.ok(run.readTokens <= run.maxReadTokens)
})

test("read of a block only marks it fully read after the cursor completes", async () => {
    const harness = createHarness({ withBlock: true })
    const run = harness.registerRun()

    const partial = await harness.call("dcp_oob_read", { startRef: "@b1@", maxTokens: 5 })
    assert.match(partial, /nextCursor: \d+/)
    assert.equal(run.blockReads.size, 0)

    const next = Number(/nextCursor: (\d+)/.exec(partial)![1])
    const complete = await harness.call("dcp_oob_read", {
        startRef: "@b1@",
        cursor: next,
        maxTokens: 200000,
    })
    assert.match(complete, /nextCursor: none/)
    assert.equal(run.blockReads.size, 1)
})

test("block replacement bypasses the range tool and preserves protected content", async () => {
    const harness = createHarness({ withBlock: true })
    const run = harness.registerRun()

    await harness.call("dcp_oob_read", { startRef: "@b1@" })
    const output = await harness.call("dcp_oob_compress", {
        topic: "New block",
        startId: "@b1@",
        endId: "@b1@",
        summary: "short replacement summary",
    })

    assert.equal(harness.legacyCalls.length, 0)
    assert.match(output, /block: @b2@/)
    assert.equal(run.writes, 1)
    assert.equal(harness.target.state.prune.messages.blocksById.get(1)?.active, false)
    const created = harness.target.state.prune.messages.blocksById.get(2)
    assert.equal(created?.active, true)
    assert.match(created?.summary ?? "", /short replacement summary/)
    assert.match(created?.summary ?? "", /verbatim user prompt/)
    assert.deepEqual(created?.consumedBlockIds, [1])
})

test("block replacement requires a completed read and rejects changed summaries", async () => {
    const harness = createHarness({ withBlock: true })
    harness.registerRun()

    await assert.rejects(
        harness.call("dcp_oob_compress", {
            topic: "No read",
            startId: "@b1@",
            endId: "@b1@",
            summary: "short",
        }),
        /Read compressed block/,
    )

    await harness.call("dcp_oob_read", { startRef: "@b1@" })
    harness.target.block.summary = harness.target.block.summary.replace(
        "old block summary",
        "OLD BLOCK SUMMARY",
    )
    await assert.rejects(
        harness.call("dcp_oob_compress", {
            topic: "Stale",
            startId: "@b1@",
            endId: "@b1@",
            summary: "short",
        }),
        /changed since it was read/,
    )
})

test("mixed block and message boundaries are rejected", async () => {
    const harness = createHarness({ withBlock: true })
    harness.registerRun()
    await harness.call("dcp_oob_read", { startRef: "@b1@" })

    await assert.rejects(
        harness.call("dcp_oob_compress", {
            topic: "Mixed",
            startId: "@b1@",
            endId: "@3@",
            summary: "short",
        }),
        /Mixed or multi-block ranges/,
    )
    assert.equal(harness.legacyCalls.length, 0)
})

test("raw range compression uses the range tool and enforces the write budget", async () => {
    const harness = createHarness({ maxWrites: 1 })
    const run = harness.registerRun()
    const beforeOverview = await harness.call("dcp_oob_overview", {})
    const beforeProjected = Number(/projectedContextTokens: (\d+)/.exec(beforeOverview)![1])
    await harness.call("dcp_oob_read", {
        startRef: "@1@",
        endRef: "@3@",
        maxTokens: 200_000,
    })

    const output = await harness.call("dcp_oob_compress", {
        topic: "Auth",
        startId: "@1@",
        endId: "@3@",
        summary: "short summary",
    })

    assert.equal(harness.legacyCalls.length, 1)
    assert.match(output, /write: 1\/1/)
    assert.match(output, /status: round write budget exhausted/)
    assert.equal(run.status, "budget-exhausted")
    assert.equal(run.writes, 1)
    assert.ok(run.reclaimedTokens > 0)

    const afterOverview = await harness.call("dcp_oob_overview", {})
    const afterProjected = Number(/projectedContextTokens: (\d+)/.exec(afterOverview)![1])
    assert.ok(afterProjected < beforeProjected)

    await assert.rejects(
        harness.call("dcp_oob_compress", {
            topic: "Again",
            startId: "@1@",
            endId: "@3@",
            summary: "another summary",
        }),
        /write budget is exhausted/,
    )
})

test("a mid-run permission denial blocks writes", async () => {
    const harness = createHarness({ maxWrites: 3 })
    harness.registerRun()
    harness.target.state.compressPermission = "deny"

    await assert.rejects(
        harness.call("dcp_oob_compress", {
            topic: "Denied",
            startId: "@1@",
            endId: "@3@",
            summary: "short",
        }),
        /requires compress permission "allow"/,
    )
    assert.equal(harness.legacyCalls.length, 0)
})

test("a raw range that changed after it was read is rejected", async () => {
    const harness = createHarness({ maxWrites: 3 })
    harness.registerRun()
    await harness.call("dcp_oob_read", { startRef: "@1@", endRef: "@3@" })

    const first = harness.target.messages[0]!.parts[0] as any
    first.text = first.text.replace("alpha", "ALPHA")
    await assert.rejects(
        harness.call("dcp_oob_compress", {
            topic: "Stale read",
            startId: "@1@",
            endId: "@3@",
            summary: "short",
        }),
        /changed since it was read/,
    )
    assert.equal(harness.legacyCalls.length, 0)
})

test("concurrent target changes are rejected before a write", async () => {
    const harness = createHarness({ maxWrites: 3 })
    harness.registerRun()

    harness.target.messages.push(message("m5", "user", 5, "concurrent message"))
    assignMessageRefs(harness.target.state, harness.target.messages)

    await assert.rejects(
        harness.call("dcp_oob_compress", {
            topic: "Stale",
            startId: "@1@",
            endId: "@3@",
            summary: "short",
        }),
        /target context changed/,
    )
    assert.equal(harness.legacyCalls.length, 0)

    await harness.call("dcp_oob_overview", {})
    await harness.call("dcp_oob_read", {
        startRef: "@1@",
        endRef: "@3@",
        maxTokens: 200_000,
    })
    const recovered = await harness.call("dcp_oob_compress", {
        topic: "Recovered",
        startId: "@1@",
        endId: "@3@",
        summary: "short",
    })
    assert.match(recovered, /write applied/)
    assert.equal(harness.legacyCalls.length, 1)
})

test("only one active out-of-band run is allowed per target", () => {
    const harness = createHarness()
    harness.registerRun()
    assert.throws(
        () => harness.registerRun({ childSessionID: "child-2" }),
        /already active for this target/,
    )
})

test("unknown, expired, and over-budget runs fail closed", async () => {
    const unknown = createHarness()
    await assert.rejects(
        unknown.call("dcp_oob_overview", {}, "not-a-run"),
        /no active out-of-band run/,
    )

    const expired = createHarness()
    expired.registerRun({ expiresAt: Date.now() - 1 })
    await assert.rejects(expired.call("dcp_oob_overview", {}), /expired/)
    assert.equal(expired.runtime.runs.size, 0)

    const budget = createHarness({ maxToolCalls: 1 })
    budget.registerRun()
    await budget.call("dcp_oob_overview", {})
    await assert.rejects(budget.call("dcp_oob_overview", {}), /tool-call budget exhausted/)
})

test("runs an autonomous child that reads, decides, writes, and reports", async () => {
    const harness = createHarness({ maxWrites: 3 })
    const state = harness.target.state
    state.manualMode = "active"
    const createdInputs: any[] = []
    const prompts: any[] = []
    const notices: string[] = []
    const removals: any[] = []
    const createdMarks: string[] = []
    const childID = "child-orchestrator"
    const instruction = "Keep only authentication decisions; stop once those are preserved."

    const session = {
        async create(input: any) {
            createdInputs.push(input)
            return { id: childID }
        },
        async prompt(input: any) {
            prompts.push(input)
            const overview = await harness.call("dcp_oob_overview", {}, childID)
            assert.match(overview, /candidateRanges/)
            const read = await harness.call(
                "dcp_oob_read",
                { startRef: "@1@", endRef: "@3@" },
                childID,
            )
            assert.match(read, /alpha/)
            const write = await harness.call(
                "dcp_oob_compress",
                {
                    topic: "Authentication",
                    startId: "@1@",
                    endId: "@3@",
                    summary: "Authentication decisions preserved.",
                },
                childID,
            )
            assert.match(write, /write: 1\/3/)
        },
        async wait() {},
        async context() {
            return childEntries("Compressed one auth range; nothing else left.")
        },
        async get() {
            return { id: childID, agent: "build", model: { providerID: "provider", id: "model" } }
        },
        async synthetic(input: any) {
            notices.push(input.text)
        },
        async interrupt() {},
        async remove(input: any) {
            removals.push(input)
        },
    }

    const result = await runOutOfBandAgent({
        session: session as any,
        host: harness.host,
        state,
        messages: harness.target.messages,
        sessionID: harness.target.sessionID,
        directory: "/tmp/project",
        instruction,
        onChildSessionCreated: (id) => createdMarks.push(id),
    })

    assert.equal(createdMarks[0], childID)
    assert.equal(createdInputs.length, 1)
    assert.equal(createdInputs[0].metadata.dcpOob, true)
    assert.equal(createdInputs[0].metadata.dcpOobTarget, harness.target.sessionID)
    assert.equal(createdInputs[0].metadata.dcpOobRunId, result.runId)
    assert.equal(createdInputs[0].agent, "build")
    assert.equal(createdInputs[0].model.id, "model")
    assert.deepEqual(createdInputs[0].location, { directory: "/tmp/project" })
    assert.equal(createdInputs[0].permissions[0].effect, "deny")
    assert.equal(createdInputs[0].permissions[1].action, "dcp_oob")
    assert.equal(createdInputs[0].permissions[1].effect, "allow")

    assert.equal(prompts.length, 1)
    assert.ok(prompts[0].text.includes(instruction))
    assert.match(prompts[0].text, /does not parse it/)
    assert.match(prompts[0].text, /dcp_oob_read/)

    assert.equal(result.chunks, 1)
    assert.equal(result.stopped, "completed")
    assert.equal(result.blocks, 1)
    assert.ok(result.before > 0)
    assert.ok(result.after < result.before)
    assert.equal(state.manualMode, "active")
    assert.equal(harness.runtime.runs.size, 0)
    assert.equal(removals.length, 1)
    assert.equal(removals[0].sessionID, childID)
    assert.equal(notices.length, 1)
    assert.ok(notices[0].startsWith("DCP internal notice: "))
    assert.match(notices[0], /projected context ~[\d,]+ -> ~[\d,]+ tokens/)
    assert.match(notices[0], /Final worker report:\nCompressed one auth range/)
    assert.match(notices[0], /1 round/)
    assert.doesNotMatch(notices[0], /target reached/i)
})

test("starts a fresh worker round when the previous worker reports CONTINUE", async () => {
    const harness = createHarness({ maxWrites: 8 })
    const state = harness.target.state
    const instruction = "Keep every authentication decision; continue until the context is lean."
    const createdInputs: any[] = []
    const prompts: any[] = []
    const marks: string[] = []
    const runsAtCreate: number[] = []
    const runsAtRemove: number[] = []
    const reports = new Map<string, string>()
    let nextChild = 0

    const session = {
        async create(input: any) {
            runsAtCreate.push(harness.runtime.runs.size)
            createdInputs.push(input)
            return { id: `child-round-${++nextChild}` }
        },
        async prompt(input: any) {
            prompts.push(input)
            const childID = input.sessionID as string
            const overview = await harness.call("dcp_oob_overview", {}, childID)
            assert.match(overview, /candidateRanges:/)
            if (childID === "child-round-1") {
                assert.match(overview, /activeCompressionBlocks: 0/)
                await harness.call("dcp_oob_read", { startRef: "@1@", endRef: "@2@" }, childID)
                await harness.call(
                    "dcp_oob_compress",
                    {
                        topic: "First range",
                        startId: "@1@",
                        endId: "@2@",
                        summary: "first range summary",
                    },
                    childID,
                )
            } else {
                assert.match(overview, /activeCompressionBlocks: 1/)
                await harness.call("dcp_oob_read", { startRef: "@3@" }, childID)
                await harness.call(
                    "dcp_oob_compress",
                    {
                        topic: "Second range",
                        startId: "@3@",
                        endId: "@3@",
                        summary: "second range summary",
                    },
                    childID,
                )
            }
        },
        async wait() {},
        async context({ sessionID }: any) {
            return childEntries(reports.get(sessionID)!)
        },
        async get({ sessionID }: any) {
            return { id: sessionID, agent: "build", model: { providerID: "provider", id: "model" } }
        },
        async synthetic() {},
        async interrupt() {},
        async remove({ sessionID }: any) {
            runsAtRemove.push(harness.runtime.runs.size)
            marks.push(sessionID)
        },
    }
    reports.set("child-round-1", "First range compressed.\nDCP-OOB: CONTINUE")
    reports.set("child-round-2", "Second range compressed.\nDCP-OOB: DONE")

    const result = await runOutOfBandAgent({
        session: session as any,
        host: harness.host,
        state,
        messages: harness.target.messages,
        sessionID: harness.target.sessionID,
        directory: "/tmp/project",
        instruction,
    })

    assert.equal(result.rounds, 2)
    assert.equal(result.chunks, 2)
    assert.equal(result.stopped, "completed")
    assert.equal(result.report, "Second range compressed.\nDCP-OOB: DONE")
    assert.equal(result.budget?.writes, 2)
    assert.equal(result.budget?.maxWrites, 8)
    assert.equal(createdInputs.length, 2)
    assert.deepEqual(runsAtCreate, [0, 0])
    assert.deepEqual(runsAtRemove, [0, 0])
    assert.deepEqual(marks, ["child-round-1", "child-round-2"])
    assert.equal(createdInputs[0].metadata.dcpOobTarget, harness.target.sessionID)
    assert.equal(createdInputs[1].metadata.dcpOobTarget, harness.target.sessionID)
    assert.notEqual(createdInputs[0].metadata.dcpOobRunId, createdInputs[1].metadata.dcpOobRunId)
    assert.match(createdInputs[0].title, /round 1/)
    assert.match(createdInputs[1].title, /round 2/)
    assert.equal(prompts.length, 2)
    assert.ok(prompts[0].text.includes(instruction))
    assert.ok(prompts[1].text.includes(instruction))
    assert.match(prompts[0].text, /worker round 1/)
    assert.match(prompts[1].text, /worker round 2/)
    assert.match(prompts[0].text, /at most 4 successful write\(s\)/)
    assert.match(prompts[1].text, /at most 4 successful write\(s\)/)
    assert.equal(harness.runtime.runs.size, 0)
})

test("continues after a worker exhausts its round write budget even if it reports DONE", async () => {
    const harness = createHarness({ maxWrites: 8 })
    const created: string[] = []
    let nextChild = 0

    const session = {
        async create() {
            const id = `child-budget-done-${++nextChild}`
            created.push(id)
            return { id }
        },
        async prompt(input: any) {
            if (input.sessionID !== "child-budget-done-1") {
                return
            }
            const childID = input.sessionID
            await harness.call("dcp_oob_read", { startRef: "@1@", endRef: "@2@" }, childID)
            await harness.call(
                "dcp_oob_compress",
                {
                    topic: "First budgeted range",
                    startId: "@1@",
                    endId: "@2@",
                    summary: "first budgeted summary",
                },
                childID,
            )
            await harness.call("dcp_oob_read", { startRef: "@3@" }, childID)
            await harness.call(
                "dcp_oob_compress",
                {
                    topic: "Second budgeted range",
                    startId: "@3@",
                    endId: "@3@",
                    summary: "second budgeted summary",
                },
                childID,
            )
        },
        async wait() {},
        async context({ sessionID }: any) {
            return childEntries(
                sessionID === "child-budget-done-1"
                    ? "Hit the round budget.\nDCP-OOB: DONE"
                    : "No more ranges.\nDCP-OOB: DONE",
            )
        },
        async get({ sessionID }: any) {
            return { id: sessionID, agent: "build", model: { providerID: "provider", id: "model" } }
        },
        async synthetic() {},
        async interrupt() {},
        async remove() {},
    }

    const result = await runOutOfBandAgent({
        session: session as any,
        host: harness.host,
        state: harness.target.state,
        messages: harness.target.messages,
        sessionID: harness.target.sessionID,
        directory: "/tmp/project",
        instruction: "compress as much as possible",
        roundWriteBudget: 2,
    })

    assert.equal(result.rounds, 2)
    assert.equal(result.chunks, 2)
    assert.equal(result.stopped, "completed")
    assert.equal(result.budget?.writes, 2)
    assert.equal(result.budget?.maxWrites, 8)
    assert.deepEqual(created, ["child-budget-done-1", "child-budget-done-2"])
    assert.equal(result.report, "No more ranges.\nDCP-OOB: DONE")
})

test("a worker that requests CONTINUE without net progress stops the loop", async () => {
    const harness = createHarness({ maxWrites: 8 })
    const createdInputs: any[] = []
    let nextChild = 0

    const session = {
        async create(input: any) {
            createdInputs.push(input)
            return { id: `child-stalled-${++nextChild}` }
        },
        async prompt(input: any) {
            const childID = input.sessionID as string
            await harness.call("dcp_oob_read", { startRef: "@1@", endRef: "@3@" }, childID)
            await harness.call(
                "dcp_oob_compress",
                {
                    topic: "Oversized",
                    startId: "@1@",
                    endId: "@3@",
                    summary: "oversized ".repeat(5000),
                },
                childID,
            )
        },
        async wait() {},
        async context() {
            return childEntries("Compression did not reduce tokens.\nDCP-OOB: CONTINUE")
        },
        async get() {
            return {
                id: "child-stalled-1",
                agent: "build",
                model: { providerID: "provider", id: "model" },
            }
        },
        async synthetic() {},
        async interrupt() {},
        async remove() {},
    }

    const result = await runOutOfBandAgent({
        session: session as any,
        host: harness.host,
        state: harness.target.state,
        messages: harness.target.messages,
        sessionID: harness.target.sessionID,
        directory: "/tmp/project",
        instruction: "compress everything",
    })

    assert.equal(result.rounds, 1)
    assert.equal(result.chunks, 1)
    assert.equal(result.stopped, "no-progress")
    assert.equal(createdInputs.length, 1)
    assert.equal(harness.runtime.runs.size, 0)
})

test("global write budget prevents another worker round", async () => {
    const harness = createHarness({ maxWrites: 1 })
    let creates = 0

    const session = {
        async create() {
            creates += 1
            return { id: "child-global-write-budget" }
        },
        async prompt(input: any) {
            const childID = input.sessionID as string
            await harness.call("dcp_oob_read", { startRef: "@1@" }, childID)
            await harness.call(
                "dcp_oob_compress",
                { topic: "One", startId: "@1@", endId: "@1@", summary: "one" },
                childID,
            )
        },
        async wait() {},
        async context() {
            return childEntries("Need more.\nDCP-OOB: CONTINUE")
        },
        async get() {
            return {
                id: "child-global-write-budget",
                agent: "build",
                model: { providerID: "provider", id: "model" },
            }
        },
        async synthetic() {},
        async interrupt() {},
        async remove() {},
    }

    const result = await runOutOfBandAgent({
        session: session as any,
        host: harness.host,
        state: harness.target.state,
        messages: harness.target.messages,
        sessionID: harness.target.sessionID,
        directory: "/tmp/project",
        instruction: "compress more",
    })

    assert.equal(creates, 1)
    assert.equal(result.rounds, 1)
    assert.equal(result.chunks, 1)
    assert.equal(result.stopped, "budget")
})

test("worker-round defaults are sized for multi-million-token targets", () => {
    assert.equal(OOB_WALL_CLOCK_MS, 30 * 60 * 1000)
    assert.equal(OOB_READ_TOKEN_BUDGET, 8_000_000)
    assert.equal(OOB_ROUND_READ_TOKEN_BUDGET, 600_000)
    assert.equal(OOB_DEFAULT_READ_TOKENS, 200_000)
})

test("worker control lines default to DONE when missing or ambiguous", () => {
    assert.deepEqual(parseOutOfBandWorkerDecision(undefined), {
        signal: "done",
        explicit: false,
    })
    assert.deepEqual(parseOutOfBandWorkerDecision("ordinary report"), {
        signal: "done",
        explicit: false,
    })
    assert.deepEqual(parseOutOfBandWorkerDecision("report\nDCP-OOB: CONTINUE"), {
        signal: "continue",
        explicit: true,
    })
    assert.deepEqual(parseOutOfBandWorkerDecision("DCP-OOB: DONE"), {
        signal: "done",
        explicit: true,
    })
    assert.deepEqual(parseOutOfBandWorkerDecision("DCP-OOB: DONE\nDCP-OOB: CONTINUE"), {
        signal: "done",
        explicit: true,
    })
})

test("per-round read budgets refresh while global reads accumulate", async () => {
    const harness = createHarness({ maxWrites: 8 })
    const firstReadCost = countTokens(buildTranscript(harness.target.messages, 0, 0))
    const secondReadCost = countTokens(buildTranscript(harness.target.messages, 2, 2))
    const roundReadBudget = Math.max(firstReadCost, secondReadCost) + 5
    const globalReadBudget = firstReadCost + secondReadCost + 10
    const runRefs: OobRun[] = []
    const prompts: any[] = []
    let nextChild = 0

    const session = {
        async create() {
            return { id: `child-read-${++nextChild}` }
        },
        async prompt(input: any) {
            prompts.push(input)
            const childID = input.sessionID as string
            const run = harness.runtime.runs.get(childID)
            assert.ok(run)
            runRefs.push(run)
            const ref = childID.endsWith("1") ? "@1@" : "@3@"
            await harness.call("dcp_oob_read", { startRef: ref }, childID)
            await harness.call(
                "dcp_oob_compress",
                { topic: "Read", startId: ref, endId: ref, summary: "short" },
                childID,
            )
        },
        async wait() {},
        async context({ sessionID }: any) {
            return childEntries(
                sessionID.endsWith("1")
                    ? "Need another round.\nDCP-OOB: CONTINUE"
                    : "Finished.\nDCP-OOB: DONE",
            )
        },
        async get({ sessionID }: any) {
            return { id: sessionID, agent: "build", model: { providerID: "provider", id: "model" } }
        },
        async synthetic() {},
        async interrupt() {},
        async remove() {},
    }

    const result = await runOutOfBandAgent({
        session: session as any,
        host: harness.host,
        state: harness.target.state,
        messages: harness.target.messages,
        sessionID: harness.target.sessionID,
        directory: "/tmp/project",
        instruction: "compress bounded sections",
        roundReadTokenBudget: roundReadBudget,
        readTokenBudget: globalReadBudget,
    })

    assert.equal(result.rounds, 2)
    assert.equal(result.chunks, 2)
    assert.equal(runRefs.length, 2)
    assert.equal(runRefs[0]!.maxReadTokens, roundReadBudget)
    assert.equal(
        runRefs[1]!.maxReadTokens,
        Math.min(roundReadBudget, globalReadBudget - runRefs[0]!.readTokens),
    )
    assert.ok(runRefs[0]!.readTokens > 0)
    assert.ok(runRefs[1]!.readTokens > 0)
    assert.equal(result.budget?.readTokens, runRefs[0]!.readTokens + runRefs[1]!.readTokens)
    assert.equal(result.budget?.maxReadTokens, globalReadBudget)
    assert.equal(prompts.length, 2)
})

test("global read budget stops before another worker round", async () => {
    const harness = createHarness({ maxWrites: 8 })
    const readCost = countTokens(buildTranscript(harness.target.messages, 0, 0))
    const createdInputs: any[] = []

    const session = {
        async create(input: any) {
            createdInputs.push(input)
            return { id: "child-read-budget" }
        },
        async prompt(input: any) {
            const childID = input.sessionID as string
            await harness.call("dcp_oob_read", { startRef: "@1@" }, childID)
            await harness.call(
                "dcp_oob_compress",
                { topic: "One", startId: "@1@", endId: "@1@", summary: "one" },
                childID,
            )
        },
        async wait() {},
        async context() {
            return childEntries("Need more.\nDCP-OOB: CONTINUE")
        },
        async get() {
            return {
                id: "child-read-budget",
                agent: "build",
                model: { providerID: "provider", id: "model" },
            }
        },
        async synthetic() {},
        async interrupt() {},
        async remove() {},
    }

    const result = await runOutOfBandAgent({
        session: session as any,
        host: harness.host,
        state: harness.target.state,
        messages: harness.target.messages,
        sessionID: harness.target.sessionID,
        directory: "/tmp/project",
        instruction: "compress more",
        roundReadTokenBudget: readCost,
        readTokenBudget: readCost,
    })

    assert.equal(createdInputs.length, 1)
    assert.equal(result.rounds, 1)
    assert.equal(result.chunks, 1)
    assert.equal(result.stopped, "budget")
    assert.equal(result.budget?.readTokens, readCost)
    assert.equal(result.budget?.maxReadTokens, readCost)
})

test("default round read budget is capped by the child model limit", async () => {
    const harness = createHarness({ maxWrites: 1 })
    const state = harness.target.state
    state.modelContextLimit = 100000
    let maxReadTokens: number | undefined

    const session = {
        async create() {
            return { id: "child-read-cap" }
        },
        async prompt(input: any) {
            maxReadTokens = harness.runtime.runs.get(input.sessionID)?.maxReadTokens
        },
        async wait() {},
        async context() {
            return childEntries("No writes.\nDCP-OOB: DONE")
        },
        async get() {
            return {
                id: "child-read-cap",
                agent: "build",
                model: { providerID: "provider", id: "model" },
            }
        },
        async synthetic() {},
        async interrupt() {},
        async remove() {},
    }

    const result = await runOutOfBandAgent({
        session: session as any,
        host: harness.host,
        state,
        messages: harness.target.messages,
        sessionID: harness.target.sessionID,
        directory: "/tmp/project",
        instruction: "compress",
    })

    assert.equal(maxReadTokens, 50000)
    assert.equal(result.budget?.roundReadTokens, 50000)
})

test("timeout revokes the run before interrupting and reports the wall-clock stop", async () => {
    const harness = createHarness({ maxWrites: 3 })
    const state = harness.target.state
    const notices: string[] = []
    const childID = "child-timeout"
    let runsAtInterrupt = -1

    const session = {
        async create() {
            return { id: childID }
        },
        async prompt() {},
        async wait() {
            return new Promise(() => {})
        },
        async context() {
            return []
        },
        async get() {
            return { id: childID, agent: "build", model: { providerID: "provider", id: "model" } }
        },
        async synthetic(input: any) {
            notices.push(input.text)
        },
        async interrupt() {
            runsAtInterrupt = harness.runtime.runs.size
        },
        async remove() {},
    }

    const result = await runOutOfBandAgent({
        session: session as any,
        host: harness.host,
        state,
        messages: harness.target.messages,
        sessionID: harness.target.sessionID,
        directory: "/tmp/project",
        instruction: "compress as much as possible",
        timeoutMs: 10,
    })

    assert.equal(result.stopped, "timeout")
    assert.equal(result.chunks, 0)
    assert.equal(runsAtInterrupt, 0)
    assert.equal(harness.runtime.runs.size, 0)
    assert.match(notices[0], /wall-clock limit reached/)
    assert.match(notices[0], /revoked and interrupted/)
})

test("orchestrator reports budget exhaustion without claiming the target was reached", async () => {
    const harness = createHarness({ maxWrites: 1 })
    const state = harness.target.state
    const notices: string[] = []
    const childID = "child-budget"

    const session = {
        async create() {
            return { id: childID }
        },
        async prompt() {
            await harness.call(
                "dcp_oob_read",
                { startRef: "@1@", endRef: "@3@", maxTokens: 200_000 },
                childID,
            )
            await harness.call(
                "dcp_oob_compress",
                {
                    topic: "Budget",
                    startId: "@1@",
                    endId: "@3@",
                    summary: "short",
                },
                childID,
            )
        },
        async wait() {},
        async context() {
            return childEntries("Used the available write budget.\nDCP-OOB: CONTINUE")
        },
        async get() {
            return { id: childID, agent: "build", model: { providerID: "provider", id: "model" } }
        },
        async synthetic(input: any) {
            notices.push(input.text)
        },
        async interrupt() {},
        async remove() {},
    }

    const result = await runOutOfBandAgent({
        session: session as any,
        host: harness.host,
        state,
        messages: harness.target.messages,
        sessionID: harness.target.sessionID,
        directory: "/tmp/project",
        instruction: "",
    })

    assert.equal(result.stopped, "budget")
    assert.equal(result.chunks, 1)
    assert.match(notices[0], /budget exhausted/)
    assert.match(notices[0], /may still need more compression/)
    assert.doesNotMatch(notices[0], /target reached/i)
})

test("an empty user instruction falls back to the default agent instruction", async () => {
    const harness = createHarness()
    const prompts: any[] = []
    const session = {
        async create() {
            return { id: "child-default" }
        },
        async prompt(input: any) {
            prompts.push(input)
        },
        async wait() {},
        async context() {
            return []
        },
        async get() {
            return {
                id: "child-default",
                agent: "build",
                model: { providerID: "provider", id: "model" },
            }
        },
        async synthetic() {},
        async interrupt() {},
        async remove() {},
    }

    await runOutOfBandAgent({
        session: session as any,
        host: harness.host,
        state: harness.target.state,
        messages: harness.target.messages,
        sessionID: harness.target.sessionID,
        directory: "/tmp/project",
        instruction: "   ",
    })

    assert.ok(prompts[0].text.includes(OOB_DEFAULT_INSTRUCTION))
})
