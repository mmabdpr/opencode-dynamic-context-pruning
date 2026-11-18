import assert from "node:assert/strict"
import test from "node:test"
import type { PluginConfig } from "../lib/config"
import { analyzeContextTokens } from "../lib/commands/context"
import { cacheSystemPromptTokens } from "../lib/ui/utils"
import { isContextOverLimits } from "../lib/messages/inject/utils"
import { wrapCompressedSummary } from "../lib/compress/state"
import { createSessionState, type WithParts } from "../lib/state"
import type { CompressionBlock } from "../lib/state"
import { countTokens, getCurrentParams, getCurrentTokenUsage } from "../lib/token-utils"

function buildConfig(maxContextLimit: number, minContextLimit = 1): PluginConfig {
    return {
        enabled: true,
        debug: false,
        pruneNotification: "off",
        pruneNotificationType: "chat",
        commands: {
            enabled: true,
            protectedTools: [],
        },
        manualMode: {
            enabled: false,
            automaticStrategies: true,
        },
        turnProtection: {
            enabled: false,
            turns: 4,
        },
        experimental: {
            allowSubAgents: false,
            customPrompts: false,
        },
        protectedFilePatterns: [],
        compress: {
            mode: "message",
            permission: "allow",
            showCompression: false,
            summaryBuffer: true,
            maxContextLimit,
            minContextLimit,
            nudgeFrequency: 5,
            iterationNudgeThreshold: 15,
            nudgeForce: "soft",
            protectedTools: ["task"],
            protectTags: false,
            protectUserMessages: false,
        },
        strategies: {
            deduplication: {
                enabled: true,
                protectedTools: [],
            },
            purgeErrors: {
                enabled: true,
                turns: 4,
                protectedTools: [],
            },
        },
    }
}

function textPart(messageID: string, sessionID: string, id: string, text: string) {
    return {
        id,
        messageID,
        sessionID,
        type: "text" as const,
        text,
    }
}

function repeatedWord(word: string, count: number): string {
    return Array.from({ length: count }, () => word).join(" ")
}

function buildCompactedMessages(): WithParts[] {
    const sessionID = "ses_compaction_token_usage"

    return [
        {
            info: {
                id: "msg-user-summary",
                role: "user",
                sessionID,
                agent: "assistant",
                time: { created: 1 },
            } as WithParts["info"],
            parts: [
                textPart(
                    "msg-user-summary",
                    sessionID,
                    "msg-user-summary-part",
                    `[Compressed conversation section]\n${repeatedWord("summary", 120)}`,
                ),
            ],
        },
        {
            info: {
                id: "msg-assistant-summary",
                role: "assistant",
                sessionID,
                agent: "assistant",
                summary: true,
                time: { created: 2 },
                tokens: {
                    input: 86000,
                    output: 1200,
                    reasoning: 300,
                    cache: {
                        read: 5000,
                        write: 0,
                    },
                },
            } as WithParts["info"],
            parts: [
                textPart(
                    "msg-assistant-summary",
                    sessionID,
                    "msg-assistant-summary-part",
                    `Compaction summary. ${repeatedWord("carry", 180)}`,
                ),
            ],
        },
        {
            info: {
                id: "msg-user-follow-up",
                role: "user",
                sessionID,
                agent: "assistant",
                time: { created: 3 },
            } as WithParts["info"],
            parts: [
                textPart(
                    "msg-user-follow-up",
                    sessionID,
                    "msg-user-follow-up-part",
                    `Continue from here. ${repeatedWord("next", 40)}`,
                ),
            ],
        },
    ]
}

function buildPostCompactionAssistantMessage(): WithParts {
    const sessionID = "ses_compaction_token_usage"

    return {
        info: {
            id: "msg-assistant-post-compaction",
            role: "assistant",
            sessionID,
            agent: "assistant",
            time: { created: 4 },
            tokens: {
                input: 2400,
                output: 600,
                reasoning: 150,
                cache: {
                    read: 300,
                    write: 0,
                },
            },
        } as WithParts["info"],
        parts: [
            textPart(
                "msg-assistant-post-compaction",
                sessionID,
                "msg-assistant-post-compaction-part",
                `Fresh post-compaction reply. ${repeatedWord("done", 60)}`,
            ),
        ],
    }
}

function createActiveBlock(
    blockId: number,
    summary: string,
    summaryTokens: number,
): CompressionBlock {
    return {
        blockId,
        runId: blockId,
        active: true,
        deactivatedByUser: false,
        compressedTokens: 0,
        summaryTokens,
        mode: "message",
        topic: `Summary ${blockId}`,
        batchTopic: `Summary ${blockId}`,
        startId: "m0001",
        endId: "m0001",
        anchorMessageId: `msg-${blockId}`,
        compressMessageId: `compress-${blockId}`,
        includedBlockIds: [],
        consumedBlockIds: [],
        parentBlockIds: [],
        directMessageIds: [],
        directToolIds: [],
        effectiveMessageIds: [],
        effectiveToolIds: [],
        createdAt: blockId,
        summary,
    }
}

test("getCurrentTokenUsage returns 0 until a fresh assistant follows compaction", () => {
    const messages = buildCompactedMessages()
    const state = createSessionState()
    state.lastCompaction = 2

    assert.equal(getCurrentTokenUsage(state, messages), 0)
})

test("isContextOverLimits ignores stale summary totals and resumes with fresh reported totals", () => {
    const messages = buildCompactedMessages()
    const state = createSessionState()
    state.lastCompaction = 2

    const staleAssistantTotal = 86000 + 1200 + 300 + 5000
    assert.equal(getCurrentTokenUsage(state, messages), 0)

    const underLimit = isContextOverLimits(
        buildConfig(staleAssistantTotal - 1, 1),
        state,
        undefined,
        undefined,
        messages,
    )

    assert.equal(underLimit.overMaxLimit, false)
    assert.equal(underLimit.overMinLimit, false)

    messages.push(buildPostCompactionAssistantMessage())
    const freshReportedTotal = 2400 + 600 + 150 + 300

    assert.equal(getCurrentTokenUsage(state, messages), freshReportedTotal)

    const overLimit = isContextOverLimits(
        buildConfig(freshReportedTotal - 1, 1),
        state,
        undefined,
        undefined,
        messages,
    )

    assert.equal(overLimit.overMaxLimit, true)
})

test("isContextOverLimits extends the max threshold by active summary tokens", () => {
    const messages = buildCompactedMessages()
    messages.push(buildPostCompactionAssistantMessage())

    const state = createSessionState()
    state.lastCompaction = 2

    const storedSummary = wrapCompressedSummary(7, repeatedWord("summary", 120))
    state.prune.messages.blocksById.set(7, createActiveBlock(7, storedSummary, 1000))
    state.prune.messages.activeBlockIds.add(7)

    const freshReportedTotal = 2400 + 600 + 150 + 300

    const underExtendedLimit = isContextOverLimits(
        buildConfig(freshReportedTotal - 1, 1),
        state,
        undefined,
        undefined,
        messages,
    )

    assert.equal(underExtendedLimit.overMaxLimit, false)

    const overExtendedLimit = isContextOverLimits(
        buildConfig(freshReportedTotal - 1001, 1),
        state,
        undefined,
        undefined,
        messages,
    )

    assert.equal(overExtendedLimit.overMaxLimit, true)
})

test("isContextOverLimits does not extend the max threshold when summaryBuffer is disabled", () => {
    const messages = buildCompactedMessages()
    messages.push(buildPostCompactionAssistantMessage())

    const state = createSessionState()
    state.lastCompaction = 2

    const storedSummary = wrapCompressedSummary(7, repeatedWord("summary", 120))
    state.prune.messages.blocksById.set(7, createActiveBlock(7, storedSummary, 1000))
    state.prune.messages.activeBlockIds.add(7)

    const freshReportedTotal = 2400 + 600 + 150 + 300
    const config = buildConfig(freshReportedTotal - 1, 1)
    config.compress.summaryBuffer = false

    const overLimit = isContextOverLimits(config, state, undefined, undefined, messages)

    assert.equal(overLimit.overMaxLimit, true)
})

test("analyzeContextTokens subtracts a compacted first user from the system estimate", () => {
    const firstUserText = repeatedWord("first", 100)
    const messages: WithParts[] = [
        {
            info: {
                id: "msg-first",
                role: "user",
                sessionID: "ses_context_tokens",
                agent: "assistant",
                time: { created: 1 },
            } as WithParts["info"],
            parts: [textPart("msg-first", "ses_context_tokens", "msg-first-part", firstUserText)],
        },
        {
            info: {
                id: "msg-first-assistant",
                role: "assistant",
                sessionID: "ses_context_tokens",
                agent: "assistant",
                time: { created: 2 },
                tokens: {
                    input: 5000,
                    output: 100,
                    reasoning: 0,
                    cache: { read: 0, write: 0 },
                },
            } as WithParts["info"],
            parts: [
                textPart(
                    "msg-first-assistant",
                    "ses_context_tokens",
                    "msg-first-assistant-part",
                    "First reply",
                ),
            ],
        },
        {
            info: {
                id: "msg-second",
                role: "user",
                sessionID: "ses_context_tokens",
                agent: "assistant",
                time: { created: 3 },
            } as WithParts["info"],
            parts: [textPart("msg-second", "ses_context_tokens", "msg-second-part", "next")],
        },
        {
            info: {
                id: "msg-second-assistant",
                role: "assistant",
                sessionID: "ses_context_tokens",
                agent: "assistant",
                time: { created: 4 },
                tokens: {
                    input: 5100,
                    output: 200,
                    reasoning: 0,
                    cache: { read: 0, write: 0 },
                },
            } as WithParts["info"],
            parts: [
                textPart(
                    "msg-second-assistant",
                    "ses_context_tokens",
                    "msg-second-assistant-part",
                    "Second reply",
                ),
            ],
        },
    ]

    const state = createSessionState()
    state.prune.messages.byMessageId.set("msg-first", {
        tokenCount: countTokens(firstUserText),
        allBlockIds: [1],
        activeBlockIds: [1],
    })
    state.prune.messages.blocksById.set(
        1,
        createActiveBlock(1, wrapCompressedSummary(1, "compressed"), 10),
    )
    state.prune.messages.activeBlockIds.add(1)

    const breakdown = analyzeContextTokens(state, messages)

    assert.equal(breakdown.system, 5000 - countTokens(firstUserText))
    assert.equal(breakdown.user, countTokens("next"))
})

test("uses measured system tokens instead of a historical first-active assistant", () => {
    const state = createSessionState()
    state.systemPromptTokens = 14_000
    state.lastCompaction = 100

    const messages: WithParts[] = [
        {
            info: {
                id: "msg-summary",
                role: "assistant",
                sessionID: "ses_system_metric",
                agent: "assistant",
                summary: true,
                finish: "stop",
                time: { created: 100 },
                tokens: {
                    input: 400_000,
                    output: 200,
                    reasoning: 0,
                    cache: { read: 0, write: 0 },
                },
            } as WithParts["info"],
            parts: [],
        },
        {
            info: {
                id: "msg-next-user",
                role: "user",
                sessionID: "ses_system_metric",
                agent: "assistant",
                time: { created: 150 },
            } as WithParts["info"],
            parts: [textPart("msg-next-user", "ses_system_metric", "next-user", "next request")],
        },
        {
            info: {
                id: "msg-current-assistant",
                role: "assistant",
                sessionID: "ses_system_metric",
                agent: "assistant",
                time: { created: 200 },
                tokens: {
                    input: 407_000,
                    output: 100,
                    reasoning: 0,
                    cache: { read: 0, write: 0 },
                },
            } as WithParts["info"],
            parts: [textPart("msg-current-assistant", "ses_system_metric", "current", "done")],
        },
    ]

    const breakdown = analyzeContextTokens(state, messages)

    assert.equal(breakdown.system, 14_000)
    assert.equal(breakdown.systemKnown, true)
    assert.equal(breakdown.reportedTotal, 407_100)
    assert.equal(breakdown.total, breakdown.reportedTotal)
    assert.ok(breakdown.liveTotal < 20_000)
})

test("reports an unknown system baseline after compaction without a measurement", () => {
    const state = createSessionState()
    state.lastCompaction = 100

    const messages: WithParts[] = [
        {
            info: {
                id: "msg-summary",
                role: "assistant",
                sessionID: "ses_system_unknown",
                agent: "assistant",
                summary: true,
                finish: "stop",
                time: { created: 100 },
                tokens: {
                    input: 400_000,
                    output: 200,
                    reasoning: 0,
                    cache: { read: 0, write: 0 },
                },
            } as WithParts["info"],
            parts: [],
        },
        {
            info: {
                id: "msg-user",
                role: "user",
                sessionID: "ses_system_unknown",
                agent: "assistant",
                time: { created: 150 },
            } as WithParts["info"],
            parts: [textPart("msg-user", "ses_system_unknown", "user", "next")],
        },
        {
            info: {
                id: "msg-assistant",
                role: "assistant",
                sessionID: "ses_system_unknown",
                agent: "assistant",
                time: { created: 200 },
                tokens: {
                    input: 407_000,
                    output: 100,
                    reasoning: 0,
                    cache: { read: 0, write: 0 },
                },
            } as WithParts["info"],
            parts: [textPart("msg-assistant", "ses_system_unknown", "assistant", "done")],
        },
    ]

    const breakdown = analyzeContextTokens(state, messages)

    assert.equal(breakdown.system, 0)
    assert.equal(breakdown.systemKnown, false)
    assert.ok(breakdown.liveTotal < 20_000)
})

test("counts active compression summaries exactly once", () => {
    const state = createSessionState()
    state.systemPromptTokens = 14_000

    const block = createActiveBlock(7, wrapCompressedSummary(7, "compressed history"), 1_000)
    state.prune.messages.blocksById.set(7, block)
    state.prune.messages.activeBlockIds.add(7)
    state.prune.messages.byMessageId.set("old-user", {
        tokenCount: 0,
        allBlockIds: [7],
        activeBlockIds: [7],
    })

    const messages: WithParts[] = [
        {
            info: {
                id: "old-user",
                role: "user",
                sessionID: "ses_summary_metric",
                agent: "assistant",
                time: { created: 1 },
            } as WithParts["info"],
            parts: [textPart("old-user", "ses_summary_metric", "old", "old text")],
        },
        {
            info: {
                id: "visible-assistant",
                role: "assistant",
                sessionID: "ses_summary_metric",
                agent: "assistant",
                time: { created: 2 },
                tokens: {
                    input: 5_000,
                    output: 100,
                    reasoning: 0,
                    cache: { read: 0, write: 0 },
                },
            } as WithParts["info"],
            parts: [textPart("visible-assistant", "ses_summary_metric", "reply", "reply")],
        },
    ]

    const breakdown = analyzeContextTokens(state, messages)

    assert.equal(breakdown.summaries, 1_000)
    assert.equal(breakdown.user, 0)
    assert.equal(
        breakdown.liveTotal,
        breakdown.system +
            breakdown.user +
            breakdown.assistant +
            breakdown.tools +
            breakdown.summaries,
    )
})

test("does not double count an injected DCP summary", () => {
    const state = createSessionState()
    state.systemPromptTokens = 14_000

    const block = createActiveBlock(7, wrapCompressedSummary(7, "stored summary"), 9_999)
    state.prune.messages.blocksById.set(7, block)
    state.prune.messages.activeBlockIds.add(7)

    const messages: WithParts[] = [
        {
            info: {
                id: "msg_dcp_summary_injected",
                role: "user",
                sessionID: "ses_summary_injected",
                agent: "assistant",
                time: { created: 1 },
            } as WithParts["info"],
            parts: [
                textPart(
                    "msg_dcp_summary_injected",
                    "ses_summary_injected",
                    "injected",
                    "injected summary",
                ),
            ],
        },
        {
            info: {
                id: "msg-visible-user",
                role: "user",
                sessionID: "ses_summary_injected",
                agent: "assistant",
                time: { created: 2 },
            } as WithParts["info"],
            parts: [textPart("msg-visible-user", "ses_summary_injected", "hello", "hello")],
        },
        {
            info: {
                id: "msg-visible-assistant",
                role: "assistant",
                sessionID: "ses_summary_injected",
                agent: "assistant",
                time: { created: 3 },
                tokens: {
                    input: 5_000,
                    output: 100,
                    reasoning: 0,
                    cache: { read: 0, write: 0 },
                },
            } as WithParts["info"],
            parts: [textPart("msg-visible-assistant", "ses_summary_injected", "world", "world")],
        },
    ]

    const breakdown = analyzeContextTokens(state, messages)

    assert.equal(breakdown.summaries, countTokens("injected summary"))
    assert.equal(breakdown.user, countTokens("hello"))
})

test("cacheSystemPromptTokens preserves an authoritative value and ignores invalid history", () => {
    const messages: WithParts[] = [
        {
            info: {
                id: "msg-user",
                role: "user",
                sessionID: "ses_cache_system",
                agent: "assistant",
                time: { created: 1 },
            } as WithParts["info"],
            parts: [textPart("msg-user", "ses_cache_system", "first", "hello")],
        },
        {
            info: {
                id: "msg-assistant",
                role: "assistant",
                sessionID: "ses_cache_system",
                agent: "assistant",
                time: { created: 2 },
                tokens: {
                    input: 5_000,
                    output: 100,
                    reasoning: 0,
                    cache: { read: 0, write: 0 },
                },
            } as WithParts["info"],
            parts: [textPart("msg-assistant", "ses_cache_system", "reply", "world")],
        },
    ]

    const measured = createSessionState()
    measured.systemPromptTokens = 14_000
    cacheSystemPromptTokens(measured, messages)
    assert.equal(measured.systemPromptTokens, 14_000)

    const compacted = createSessionState()
    compacted.lastCompaction = 1
    cacheSystemPromptTokens(compacted, messages)
    assert.equal(compacted.systemPromptTokens, undefined)
})

test("getCurrentParams tolerates user messages without model metadata", () => {
    const messages = [
        {
            info: {
                id: "user-without-model",
                role: "user",
                sessionID: "ses_missing_model",
                time: { created: 1 },
            },
            parts: [textPart("user-without-model", "ses_missing_model", "text", "hello")],
        },
    ] as WithParts[]

    const params = getCurrentParams(createSessionState(), messages, {
        debug() {},
    } as any)

    assert.deepEqual(params, {
        providerId: undefined,
        modelId: undefined,
        agent: undefined,
        variant: undefined,
    })
})
