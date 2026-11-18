import assert from "node:assert/strict"
import test from "node:test"
import { createOobGate, isCompressionCommand } from "../lib/compress/oob-guard"

function clock(start = 0) {
    let value = start
    return {
        now: () => value,
        advance: (ms: number) => {
            value += ms
        },
    }
}

test("rejects a second OOB entry while the first is active", () => {
    const gate = createOobGate({ now: clock().now })
    const first = gate.enterOob("ses_a", "compress this")
    assert.equal(first.ok, true)
    assert.equal(gate.isOob("ses_a"), true)
    assert.deepEqual(gate.enterOob("ses_a", "another instruction"), {
        ok: false,
        reason: "active",
    })
    if (first.ok) first.release(false)
    assert.equal(gate.isOob("ses_a"), false)
})

test("applies a cooldown only to the same target and instruction after a completed run", () => {
    const time = clock(1000)
    const gate = createOobGate({ now: time.now, cooldownMs: 5000 })
    const first = gate.enterOob("ses_a", "keep auth decisions")
    assert.equal(first.ok, true)
    if (!first.ok) return
    first.release(true)

    assert.deepEqual(gate.enterOob("ses_a", "keep auth decisions"), {
        ok: false,
        reason: "cooldown",
        retryAfterMs: 5000,
    })
    const other = gate.enterOob("ses_a", "keep database decisions")
    assert.equal(other.ok, true)
    if (other.ok) other.release(false)

    time.advance(5000)
    const expired = gate.enterOob("ses_a", "keep auth decisions")
    assert.equal(expired.ok, true)
    if (expired.ok) expired.release(false)
})

test("failed or no-op runs remain immediately retryable", () => {
    const gate = createOobGate({ cooldownMs: 5000 })
    const first = gate.enterOob("ses_a", "same")
    assert.equal(first.ok, true)
    if (first.ok) first.release(false)

    const retry = gate.enterOob("ses_a", "same")
    assert.equal(retry.ok, true)
    if (retry.ok) retry.release(false)
})

test("targets are isolated and recent entries are capped", () => {
    const gate = createOobGate({ cooldownMs: 5000, maxRecent: 1 })
    const a = gate.enterOob("ses_a", "same")
    assert.equal(a.ok, true)
    if (a.ok) a.release(true)
    const b = gate.enterOob("ses_b", "same")
    assert.equal(b.ok, true)
    if (b.ok) b.release(true)

    const aRetry = gate.enterOob("ses_a", "same")
    assert.equal(aRetry.ok, true)
    if (aRetry.ok) aRetry.release(false)
    const bRetry = gate.enterOob("ses_b", "same")
    assert.equal(bRetry.ok, false)
})

test("classifies both compression command spellings", () => {
    assert.equal(isCompressionCommand("dcp-compress", ""), true)
    assert.equal(isCompressionCommand("dcp-compress", "--oob keep decisions"), true)
    assert.equal(isCompressionCommand("dcp", "compress --oob keep decisions"), true)
    assert.equal(isCompressionCommand("dcp", "context"), false)
    assert.equal(isCompressionCommand("dcp", "compress"), true)
})
