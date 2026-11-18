const DEFAULT_OOB_COOLDOWN_MS = 30_000
const DEFAULT_MAX_RECENT = 200

export type OobGateEntry =
    | { ok: true; release: (completed: boolean) => void }
    | { ok: false; reason: "active" }
    | { ok: false; reason: "cooldown"; retryAfterMs: number }

export interface OobGate {
    /** True while an out-of-band run is active for the target. */
    isOob(sessionID: string): boolean
    /**
     * Claim the OOB phase for a target. Call this only after the command has
     * decided to run out-of-band, inside the target's serial queue.
     */
    enterOob(sessionID: string, instruction: string): OobGateEntry
    clear(): void
}

export interface OobGateOptions {
    cooldownMs?: number
    maxRecent?: number
    now?: () => number
}

/**
 * A per-target guard against duplicate out-of-band commands. It does not
 * order work: the session's serial queue still protects target state. The
 * guard only makes retries fail closed before they can queue a second run.
 */
export function createOobGate(options: OobGateOptions = {}): OobGate {
    const cooldownMs = Math.max(0, options.cooldownMs ?? DEFAULT_OOB_COOLDOWN_MS)
    const maxRecent = Math.max(1, options.maxRecent ?? DEFAULT_MAX_RECENT)
    const now = options.now ?? (() => Date.now())
    const active = new Set<string>()
    const recent = new Map<string, number>()

    function key(sessionID: string, instruction: string): string {
        return `${sessionID}\0${instruction.trim()}`
    }

    function pruneRecent(): void {
        while (recent.size > maxRecent) {
            const oldest = recent.keys().next().value
            if (oldest === undefined) return
            recent.delete(oldest)
        }
    }

    return {
        isOob(sessionID) {
            return active.has(sessionID)
        },
        enterOob(sessionID, instruction) {
            if (active.has(sessionID)) {
                return { ok: false, reason: "active" }
            }

            const recentKey = key(sessionID, instruction)
            const until = recent.get(recentKey)
            if (until !== undefined) {
                const current = now()
                if (current < until) {
                    return {
                        ok: false,
                        reason: "cooldown",
                        retryAfterMs: Math.max(1, until - current),
                    }
                }
                recent.delete(recentKey)
            }

            active.add(sessionID)
            let released = false
            return {
                ok: true,
                release(completed) {
                    if (released) return
                    released = true
                    active.delete(sessionID)
                    if (!completed || cooldownMs === 0) return
                    recent.set(recentKey, now() + cooldownMs)
                    pruneRecent()
                },
            }
        },
        clear() {
            active.clear()
            recent.clear()
        },
    }
}

/** The two command spellings DCP registers for compression. */
export function isCompressionCommand(name: string, text: string): boolean {
    if (name === "dcp-compress") return true
    if (name !== "dcp") return false
    return text.trim().split(/\s+/)[0]?.toLowerCase() === "compress"
}
