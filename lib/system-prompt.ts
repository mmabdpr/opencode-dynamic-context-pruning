import type { SessionState, WithParts } from "./state"
import { isIgnoredUserMessage } from "./messages/query"
import { countTokens } from "./token-utils"

export type SystemPromptSource = "state" | "first-request" | "unknown"

export interface SystemPromptResolution {
    tokens: number
    known: boolean
    source: SystemPromptSource
}

function hasKnownValue(value: number | undefined): value is number {
    return typeof value === "number" && Number.isFinite(value) && value >= 0
}

function messageText(message: WithParts): string {
    const parts = Array.isArray(message.parts) ? message.parts : []
    return parts
        .filter((part) => part.type === "text")
        .map((part) => (part as { text?: string }).text ?? "")
        .join("\n")
}

/**
 * Resolve the system-prompt component of the current context.
 *
 * `state.systemPromptTokens` is measured from the system text OpenCode sends
 * on a normal turn, so it is preferred. The first-request residual is only a
 * fallback for sessions that have not been compacted or forked and still
 * contain their genuine first user message and first assistant response.
 * Without one of those baselines the system size is unknown; returning zero
 * is explicitly a partial estimate, not a claim that the prompt is empty.
 */
export function resolveSystemPromptTokens(
    state: SessionState,
    messages: WithParts[],
): SystemPromptResolution {
    if (hasKnownValue(state.systemPromptTokens)) {
        return { tokens: state.systemPromptTokens, known: true, source: "state" }
    }

    const firstAssistantIndex = messages.findIndex((message) => {
        if (message.info.role !== "assistant") {
            return false
        }
        const tokens = (message.info as any).tokens
        return (
            (tokens?.input ?? 0) > 0 ||
            (tokens?.cache?.read ?? 0) > 0 ||
            (tokens?.cache?.write ?? 0) > 0
        )
    })
    if (firstAssistantIndex <= 0) {
        return { tokens: 0, known: false, source: "unknown" }
    }

    const prefix = messages.slice(0, firstAssistantIndex)
    if (
        prefix.some(
            (message) =>
                message.info.summary === true ||
                message.info.id.startsWith("msg_dcp_summary_") ||
                messageText(message).startsWith("[Compressed conversation section]") ||
                messageText(message).startsWith("DCP internal notice: "),
        )
    ) {
        return { tokens: 0, known: false, source: "unknown" }
    }

    // A native compaction means the first assistant still in the list is not
    // the first request, even if the summary message was filtered out.
    if (state.lastCompaction > 0) {
        return { tokens: 0, known: false, source: "unknown" }
    }

    const userMessages = prefix.filter(
        (message) => message.info.role === "user" && !isIgnoredUserMessage(message),
    )
    if (userMessages.length !== 1) {
        return { tokens: 0, known: false, source: "unknown" }
    }

    const firstUserText = messageText(userMessages[0]!)
    if (!firstUserText || firstUserText.startsWith("[Compressed conversation section]")) {
        return { tokens: 0, known: false, source: "unknown" }
    }

    const firstAssistant = messages[firstAssistantIndex]!
    const tokens = (firstAssistant.info as any).tokens
    const firstInput =
        (tokens?.input ?? 0) + (tokens?.cache?.read ?? 0) + (tokens?.cache?.write ?? 0)
    return {
        tokens: Math.max(0, firstInput - countTokens(firstUserText)),
        known: true,
        source: "first-request",
    }
}
