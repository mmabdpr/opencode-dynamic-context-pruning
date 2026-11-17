import assert from "node:assert/strict"
import test from "node:test"
import { remapForkState } from "../lib/state/fork-migration"

function message(id: string, role: "user" | "assistant", created: number, parts: any[] = []): any {
    return {
        info: { id, role, time: { created }, sessionID: "ses_test" },
        parts,
    }
}

function block(id: number, anchor: string, compress: string, extra: Record<string, any> = {}): any {
    return {
        blockId: id,
        runId: id,
        active: true,
        deactivatedByUser: false,
        compressedTokens: 100,
        summaryTokens: 10,
        durationMs: 1,
        topic: "topic",
        startId: "@1@",
        endId: "@2@",
        anchorMessageId: anchor,
        compressMessageId: compress,
        includedBlockIds: [],
        consumedBlockIds: [],
        parentBlockIds: [],
        directMessageIds: [anchor],
        directToolIds: [],
        effectiveMessageIds: [anchor],
        effectiveToolIds: [],
        createdAt: 1,
        summary: "summary",
        ...extra,
    }
}

function parentState(): any {
    return {
        manualMode: true,
        prune: {
            tools: { tool1: 100, tool2: 200, tool3: 300 },
            messages: {
                byMessageId: {
                    p1: { tokenCount: 10, allBlockIds: [1], activeBlockIds: [1] },
                    p2: { tokenCount: 20, allBlockIds: [1], activeBlockIds: [1] },
                    p3: { tokenCount: 30, allBlockIds: [2], activeBlockIds: [2] },
                    p4: { tokenCount: 40, allBlockIds: [2], activeBlockIds: [2] },
                    p5: { tokenCount: 50, allBlockIds: [3], activeBlockIds: [3] },
                },
                blocksById: {
                    "1": block(1, "p2", "p2", {
                        directMessageIds: ["p1", "p2"],
                        effectiveMessageIds: ["p1", "p2"],
                        directToolIds: ["tool1"],
                    }),
                    "2": block(2, "p4", "p4", {
                        directMessageIds: ["p3", "p4"],
                        effectiveMessageIds: ["p3", "p4"],
                        directToolIds: ["tool2"],
                        includedBlockIds: [1, 3],
                    }),
                    "3": block(3, "p6", "p6", {
                        directMessageIds: ["p5", "p6"],
                        directToolIds: ["tool3"],
                    }),
                },
                activeBlockIds: [1, 2, 3],
                activeByAnchorMessageId: { p2: 1, p4: 2, p6: 3 },
                nextBlockId: 4,
                nextRunId: 5,
            },
        },
        nudges: {
            contextLimitAnchors: ["p2", "p6"],
            turnNudgeAnchors: ["p4"],
            iterationNudgeAnchors: ["p6"],
        },
        stats: { pruneTokenCounter: 300, totalPruneTokens: 1000 },
        lastUpdated: "2026-01-01T00:00:00.000Z",
    }
}

function parentMessages(): any[] {
    return [
        message("p1", "user", 1),
        message("p2", "assistant", 2),
        message("p3", "user", 3),
        message("p4", "assistant", 4),
        message("p5", "user", 5),
        message("p6", "assistant", 6),
    ]
}

function forkMessages(): any[] {
    return [
        message("f1", "user", 1, [
            { type: "tool", callID: "tool1", state: { status: "completed", output: "one" } },
        ]),
        message("f2", "assistant", 2),
        message("f3", "user", 3, [
            { type: "tool", callID: "tool2", state: { status: "completed", output: "two" } },
        ]),
        message("f4", "assistant", 4),
        message("f5", "user", 10),
        message("f6", "assistant", 11),
    ]
}

test("remaps parent state onto the fork prefix and drops post-fork blocks", () => {
    const result = remapForkState(parentState(), parentMessages(), forkMessages(), {
        type: "before",
        messageID: "p5",
    })

    assert.ok(result)
    assert.equal(result.prefixLength, 4)
    assert.equal(result.mappedMessageKeys, 4)
    assert.equal(result.totalMessageKeys, 4)
    assert.equal(result.mappedBlocks, 2)
    assert.equal(result.totalBlocks, 3)

    const messages = result.state.prune.messages
    assert.deepEqual(Object.keys(messages.byMessageId).sort(), ["f1", "f2", "f3", "f4"])
    assert.deepEqual(messages.activeBlockIds, [1, 2])
    assert.deepEqual(messages.activeByAnchorMessageId, { f2: 1, f4: 2 })
    assert.equal(messages.blocksById["3"], undefined)
    assert.equal(messages.blocksById["1"].anchorMessageId, "f2")
    assert.equal(messages.blocksById["2"].anchorMessageId, "f4")
    assert.deepEqual(messages.blocksById["1"].directMessageIds, ["f1", "f2"])
    assert.deepEqual(messages.blocksById["2"].includedBlockIds, [1])
    assert.deepEqual(result.state.prune.tools, { tool1: 100, tool2: 200 })
    assert.deepEqual(result.state.nudges.contextLimitAnchors, ["f2"])
    assert.deepEqual(result.state.nudges.turnNudgeAnchors, ["f4"])
    assert.deepEqual(result.state.nudges.iterationNudgeAnchors, [])
    assert.equal(result.state.manualMode, true)
    assert.deepEqual(result.state.stats, { pruneTokenCounter: 300, totalPruneTokens: 1000 })
    assert.equal(messages.nextBlockId, 4)
    assert.equal(messages.nextRunId, 5)

    const serialized = JSON.stringify(result.state)
    assert.equal(serialized.includes('"p1"'), false)
    assert.equal(serialized.includes('"p6"'), false)
})

test("uses the longest matching prefix when the boundary message is unavailable", () => {
    const result = remapForkState(parentState(), parentMessages(), forkMessages())

    assert.ok(result)
    assert.equal(result.prefixLength, 4)
    assert.deepEqual(result.state.prune.messages.activeBlockIds, [1, 2])
})

test("honours an after boundary", () => {
    const fork = [
        message("f1", "user", 1),
        message("f2", "assistant", 2),
        message("f3", "user", 3),
        message("f4", "assistant", 4),
        message("f5", "user", 5),
        message("f6", "assistant", 11),
    ]

    const result = remapForkState(parentState(), parentMessages(), fork, {
        type: "after",
        messageID: "p5",
    })

    assert.ok(result)
    assert.equal(result.prefixLength, 5)
    assert.equal(result.mappedMessageKeys, 5)
    assert.deepEqual(result.state.prune.messages.activeBlockIds, [1, 2])
})

test("ignores post-fork parent keys when checking coverage", () => {
    const state = parentState()
    state.prune.messages.byMessageId = {
        p1: state.prune.messages.byMessageId.p1,
        p2: state.prune.messages.byMessageId.p2,
        p5: state.prune.messages.byMessageId.p5,
    }
    state.prune.messages.blocksById = {
        "1": state.prune.messages.blocksById["1"],
        "3": state.prune.messages.blocksById["3"],
    }
    state.prune.messages.activeBlockIds = [1, 3]
    state.prune.messages.activeByAnchorMessageId = { p2: 1, p6: 3 }

    const result = remapForkState(state, parentMessages(), forkMessages(), {
        type: "before",
        messageID: "p5",
    })

    assert.ok(result)
    assert.equal(result.mappedMessageKeys, 2)
    assert.equal(result.totalMessageKeys, 2)
    assert.equal(result.mappedBlocks, 1)
    assert.deepEqual(Object.keys(result.state.prune.messages.byMessageId).sort(), ["f1", "f2"])
    assert.deepEqual(result.state.prune.messages.activeBlockIds, [1])
})

test("refuses migration when too few parent message keys map", () => {
    const state = parentState()
    state.prune.messages.byMessageId = {
        p5: state.prune.messages.byMessageId.p5,
    }

    const result = remapForkState(state, parentMessages(), forkMessages(), {
        type: "before",
        messageID: "p5",
    })

    assert.equal(result, null)
})

test("stops at the first role or timestamp mismatch", () => {
    const state = parentState()
    state.prune.messages.byMessageId = {
        p1: state.prune.messages.byMessageId.p1,
        p2: state.prune.messages.byMessageId.p2,
        p3: state.prune.messages.byMessageId.p3,
    }
    const fork = forkMessages()
    fork[2] = message("f3", "assistant", 3)

    const result = remapForkState(state, parentMessages(), fork, {
        type: "before",
        messageID: "p5",
    })

    assert.ok(result)
    assert.equal(result.prefixLength, 2)
    assert.deepEqual(result.state.prune.messages.activeBlockIds, [1])
    assert.equal(result.state.prune.messages.blocksById["2"], undefined)
})
