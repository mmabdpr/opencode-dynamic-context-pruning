import assert from "node:assert/strict"
import test from "node:test"
import { recentContextMessages, scopeCompressionMessages } from "../lib/compress/context-window"
import { createSessionState, type WithParts } from "../lib/state"

function message(
    id: string,
    role: "user" | "assistant",
    created: number,
    text: string,
    input = 0,
): WithParts {
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
                    ? {
                          input,
                          output: 0,
                          reasoning: 0,
                          cache: { read: 0, write: 0 },
                      }
                    : undefined,
        },
        parts: [{ id: `${id}:text`, type: "text", text }],
    } as WithParts
}

test("recentContextMessages excludes messages outside the last reported context window", () => {
    const state = createSessionState("compact")
    state.modelContextLimit = 1000
    const messages = [
        message("old-user", "user", 1, "old ".repeat(1000)),
        message("old-assistant", "assistant", 2, "old reply"),
        message("recent-user", "user", 3, "recent ".repeat(10)),
        message("latest-assistant", "assistant", 4, "latest", 100),
    ]

    const selected = recentContextMessages(state, messages)
    const ids = selected.map((item) => item.info.id)

    assert.ok(ids.includes("latest-assistant"))
    assert.ok(ids.includes("recent-user"))
    assert.equal(ids.includes("old-user"), false)
})

test("scopeCompressionMessages keeps only the model-visible message IDs", () => {
    const state = createSessionState("compact")
    const messages = [
        message("archived-user", "user", 1, "archived"),
        message("archived-assistant", "assistant", 2, "archived reply"),
        message("active-user", "user", 3, "active"),
        message("active-assistant", "assistant", 4, "active reply"),
    ]

    const scoped = scopeCompressionMessages(
        state,
        messages,
        new Set(["active-user", "active-assistant"]),
    )

    assert.deepEqual(
        scoped.map((item) => item.info.id),
        ["active-user", "active-assistant"],
    )
})
