import assert from "node:assert/strict"
import test from "node:test"
import { project } from "../lib/v2/messages"

const session = {
    id: "ses_projection",
    agent: "build",
    model: { providerID: "provider", id: "model" },
} as any

function nativeMessage(id: string, role: "user" | "assistant", text: string) {
    return {
        id,
        role,
        content: [{ type: "text", text }],
    } as any
}

test("projects native messages that are missing from session.context", () => {
    const native = [
        nativeMessage("a", "user", "retained tail A"),
        nativeMessage("b", "assistant", "retained tail B"),
        nativeMessage("c", "user", "post-compaction C"),
    ]
    const entries = [
        {
            id: "c",
            type: "user",
            text: "post-compaction C",
            time: { created: 3 },
        } as any,
    ]

    const view = project(native, entries, session)

    assert.deepEqual(
        view.messages.map((message) => message.info.id),
        ["a", "b", "c"],
    )
    assert.equal(view.messages[0]?.parts[0]?.type, "text")
    assert.equal(view.messages[0]?.parts[0]?.text, "retained tail A")

    assert.deepEqual(
        view.restore().map((message) => message.id),
        ["a", "b", "c"],
    )
})

test("projects the retained tail when only a compaction entry anchors the native set", () => {
    const native = [
        nativeMessage("a", "user", "retained A"),
        nativeMessage("b", "assistant", "retained B"),
        nativeMessage("c", "user", "post-compaction C"),
    ]
    const entries = [
        {
            id: "c",
            type: "compaction",
            status: "completed",
            time: { created: 3 },
        } as any,
    ]

    const view = project(native, entries, session)

    assert.deepEqual(
        view.messages.map((message) => message.info.id),
        ["a", "b"],
    )
    assert.deepEqual(
        view.restore().map((message) => message.id),
        ["a", "b", "c"],
    )
})

test("uses a compaction keepID as the retained-tail anchor", () => {
    const native = [
        nativeMessage("a", "user", "retained tail A"),
        nativeMessage("b", "assistant", "retained tail B"),
        nativeMessage("c", "user", "post-compaction C"),
    ]
    const entries = [
        {
            id: "compaction-x",
            type: "compaction",
            metadata: { keepID: "a" },
            time: { created: 1 },
        } as any,
    ]

    const view = project(native, entries, session)

    assert.deepEqual(
        view.messages.map((message) => message.info.id),
        ["a", "b", "c"],
    )
    assert.deepEqual(
        view.restore().map((message) => message.id),
        ["a", "b", "c"],
    )
})

test("does not project a retained tail when the compaction keepID is absent", () => {
    const native = [
        nativeMessage("a", "user", "retained tail A"),
        nativeMessage("b", "assistant", "retained tail B"),
    ]
    const entries = [
        {
            id: "compaction-x",
            type: "compaction",
            metadata: { keepID: "missing" },
            time: { created: 1 },
        } as any,
    ]

    const view = project(native, entries, session)

    assert.equal(view.messages.length, 0)
    assert.deepEqual(
        view.restore().map((message) => message.id),
        ["a", "b"],
    )
})
