/**
 * DCP Context Command
 * Shows a visual breakdown of token usage in the current session.
 *
 * TOKEN CALCULATION STRATEGY
 * ==========================
 * `reportedTotal` (`total`) is the last provider-reported usage. It is useful
 * when fresh, but after compaction it can still describe messages that are no
 * longer sent to the model.
 *
 * `liveTotal` estimates the content in the current effective prompt:
 *
 *   SYSTEM    = measured state.systemPromptTokens, or a validated first-request
 *               residual. The residual is never used after compaction. When no
 *               baseline exists, systemKnown is false and zero is reported.
 *   USER      = retained user text.
 *   ASSISTANT = retained assistant text, not a residual of the provider total.
 *   TOOLS     = retained tool inputs, outputs, errors, and pruning replacements.
 *   SUMMARIES = active compression block summaries, counted once.
 *
 * The UI uses liveTotal for "current context" and shows reportedTotal
 * separately. OOB projection uses liveTotal as its floor and only applies
 * reclaimed tokens to a fresh reported total.
 */

import type { Logger } from "../logger"
import type { SessionState, WithParts } from "../state"
import { sendIgnoredMessage } from "../ui/notification"
import { formatTokenCount } from "../ui/utils"
import { isIgnoredUserMessage } from "../messages/query"
import {
    PRUNED_QUESTION_INPUT_REPLACEMENT,
    PRUNED_TOOL_ERROR_INPUT_REPLACEMENT,
    PRUNED_TOOL_OUTPUT_REPLACEMENT,
} from "../messages/prune"
import { isMessageCompacted } from "../state/utils"
import { resolveSystemPromptTokens } from "../system-prompt"
import { countTokens, extractCompletedToolOutput, getCurrentParams } from "../token-utils"
import type { AssistantMessage, TextPart, ToolPart } from "@opencode-ai/sdk/v2"

export interface ContextCommandContext {
    client: any
    state: SessionState
    logger: Logger
    sessionId: string
    messages: WithParts[]
}

export interface TokenBreakdown {
    system: number
    user: number
    assistant: number
    tools: number
    summaries: number
    toolCount: number
    toolsInContextCount: number
    prunedTokens: number
    prunedToolCount: number
    prunedMessageCount: number
    /** Last provider-reported usage, kept for compatibility. */
    total: number
    /** Explicit alias for `total`. */
    reportedTotal: number
    /** Estimate of the content in the current effective prompt. */
    liveTotal: number
    /** True when `system` is measured or derived from a valid first request. */
    systemKnown: boolean
    /** True when the reported total postdates the newest active block. */
    reportedFresh: boolean
}

const DCP_SUMMARY_PREFIX = "msg_dcp_summary_"

function stringifyToolValue(value: unknown): string {
    return typeof value === "string" ? value : JSON.stringify(value)
}

function effectiveToolInput(state: SessionState, part: ToolPart): string | undefined {
    let input = part.state?.input
    if (input === undefined) {
        return undefined
    }

    if (state.prune.tools.has(part.callID) && part.state?.status === "error") {
        if (input && typeof input === "object") {
            const copy: Record<string, unknown> = { ...(input as Record<string, unknown>) }
            for (const key of Object.keys(copy)) {
                if (typeof copy[key] === "string") {
                    copy[key] = PRUNED_TOOL_ERROR_INPUT_REPLACEMENT
                }
            }
            input = copy
        }
    }

    if (state.prune.tools.has(part.callID) && part.tool === "question") {
        const questions = (input as Record<string, unknown> | undefined)?.questions
        if (questions !== undefined) {
            input = {
                ...(input as Record<string, unknown>),
                questions: PRUNED_QUESTION_INPUT_REPLACEMENT,
            }
        }
    }

    return stringifyToolValue(input)
}

function effectiveToolOutput(state: SessionState, part: ToolPart): string | undefined {
    if (part.state?.status === "completed") {
        if (
            state.prune.tools.has(part.callID) &&
            part.tool !== "question" &&
            part.tool !== "edit" &&
            part.tool !== "write"
        ) {
            return PRUNED_TOOL_OUTPUT_REPLACEMENT
        }
        return extractCompletedToolOutput(part)
    }
    if (part.state?.status === "error" && part.state.error !== undefined) {
        return stringifyToolValue(part.state.error)
    }
    return undefined
}

export function analyzeContextTokens(state: SessionState, messages: WithParts[]): TokenBreakdown {
    let lastAssistant: AssistantMessage | undefined
    let lastAssistantMessage: WithParts | undefined
    for (let i = messages.length - 1; i >= 0; i--) {
        const msg = messages[i]
        if (msg.info.role === "assistant") {
            const assistantInfo = msg.info as AssistantMessage
            const tokens = assistantInfo.tokens
            if (
                (tokens?.input ?? 0) > 0 ||
                (tokens?.output ?? 0) > 0 ||
                (tokens?.cache?.read ?? 0) > 0 ||
                (tokens?.cache?.write ?? 0) > 0
            ) {
                lastAssistant = assistantInfo
                lastAssistantMessage = msg
                break
            }
        }
    }

    const apiInput = lastAssistant?.tokens?.input || 0
    const apiOutput = lastAssistant?.tokens?.output || 0
    const apiReasoning = lastAssistant?.tokens?.reasoning || 0
    const apiCacheRead = lastAssistant?.tokens?.cache?.read || 0
    const apiCacheWrite = lastAssistant?.tokens?.cache?.write || 0
    const reportedTotal = apiInput + apiOutput + apiReasoning + apiCacheRead + apiCacheWrite

    const userTextParts: string[] = []
    const assistantTextParts: string[] = []
    const toolInputParts: string[] = []
    const toolOutputParts: string[] = []
    const allToolIds = new Set<string>()
    const activeToolIds = new Set<string>()
    const prunedByMessageToolIds = new Set<string>()
    const allMessageIds = new Set<string>()
    const injectedSummaryIds: string[] = []

    for (const msg of messages) {
        allMessageIds.add(msg.info.id)
        const parts = Array.isArray(msg.parts) ? msg.parts : []
        const isCompacted = isMessageCompacted(state, msg)
        const pruneEntry = state.prune.messages.byMessageId.get(msg.info.id)
        const isMessagePruned = !!pruneEntry && pruneEntry.activeBlockIds.length > 0
        const isIgnoredUser = isIgnoredUserMessage(msg)
        const isInjectedSummary = msg.info.id.startsWith(DCP_SUMMARY_PREFIX)

        if (isInjectedSummary && !isCompacted) {
            injectedSummaryIds.push(msg.info.id)
        }

        for (const part of parts) {
            if (part.type === "tool") {
                const toolPart = part as ToolPart
                if (toolPart.callID) {
                    allToolIds.add(toolPart.callID)
                    if (!isCompacted) {
                        activeToolIds.add(toolPart.callID)
                    }
                    if (isMessagePruned) {
                        prunedByMessageToolIds.add(toolPart.callID)
                    }
                }

                if (!isCompacted) {
                    const input = effectiveToolInput(state, toolPart)
                    if (input !== undefined) {
                        toolInputParts.push(input)
                    }
                    const output = effectiveToolOutput(state, toolPart)
                    if (output !== undefined) {
                        toolOutputParts.push(output)
                    }
                }
            } else if (
                part.type === "text" &&
                msg.info.role === "user" &&
                !isIgnoredUser &&
                !isInjectedSummary
            ) {
                const textPart = part as TextPart
                const text = textPart.text || ""
                if (!isCompacted) {
                    userTextParts.push(text)
                }
            } else if (
                part.type === "text" &&
                msg.info.role === "assistant" &&
                msg.info.summary !== true &&
                !isInjectedSummary &&
                !isCompacted
            ) {
                assistantTextParts.push((part as TextPart).text || "")
            }
        }
    }

    const prunedByToolIds = new Set<string>()
    for (const id of allToolIds) {
        if (state.prune.tools.has(id)) {
            prunedByToolIds.add(id)
        }
    }

    const prunedToolIds = new Set<string>([...prunedByToolIds, ...prunedByMessageToolIds])
    const toolsInContextCount = [...activeToolIds].filter((id) => !prunedByToolIds.has(id)).length

    let prunedMessageCount = 0
    for (const [id, entry] of state.prune.messages.byMessageId) {
        if (allMessageIds.has(id) && entry.activeBlockIds.length > 0) {
            prunedMessageCount++
        }
    }

    let summaries = 0
    if (injectedSummaryIds.length > 0) {
        const injectedText = messages
            .filter((msg) => injectedSummaryIds.includes(msg.info.id))
            .flatMap((msg) => (Array.isArray(msg.parts) ? msg.parts : []))
            .filter((part) => part.type === "text")
            .map((part) => (part as TextPart).text || "")
            .join("\n")
        summaries = countTokens(injectedText)
    } else {
        for (const blockId of state.prune.messages.activeBlockIds) {
            const block = state.prune.messages.blocksById.get(blockId)
            if (block?.active) {
                summaries += block.summaryTokens
            }
        }
    }

    const activeBlockCreatedAt = [...state.prune.messages.activeBlockIds]
        .map((blockId) => state.prune.messages.blocksById.get(blockId)?.createdAt ?? 0)
        .reduce((max, value) => Math.max(max, value), 0)
    const reportedFresh =
        !!lastAssistantMessage &&
        (activeBlockCreatedAt === 0 ||
            lastAssistantMessage.info.time.created > activeBlockCreatedAt)

    const system = resolveSystemPromptTokens(state, messages)

    const breakdown: TokenBreakdown = {
        system: system.tokens,
        user: countTokens(userTextParts.join("\n")),
        assistant: countTokens(assistantTextParts.join("\n")),
        tools: countTokens(toolInputParts.join("\n")) + countTokens(toolOutputParts.join("\n")),
        summaries,
        toolCount: allToolIds.size,
        toolsInContextCount,
        prunedTokens: state.stats.totalPruneTokens,
        prunedToolCount: prunedToolIds.size,
        prunedMessageCount,
        total: reportedTotal,
        reportedTotal,
        liveTotal: 0,
        systemKnown: system.known,
        reportedFresh,
    }
    breakdown.liveTotal =
        breakdown.system +
        breakdown.user +
        breakdown.assistant +
        breakdown.tools +
        breakdown.summaries

    return breakdown
}

function createBar(value: number, maxValue: number, width: number, char: string = "█"): string {
    if (maxValue === 0) return ""
    const filled = Math.round((value / maxValue) * width)
    const bar = char.repeat(Math.max(0, filled))
    return bar
}

export function formatContextMessage(breakdown: TokenBreakdown): string {
    const lines: string[] = []
    const barWidth = 30

    const toolsLabel = `Tools (${breakdown.toolsInContextCount})`
    const systemLabel = breakdown.systemKnown ? "System" : "System (unknown)"

    const categories = [
        { label: systemLabel, value: breakdown.system, char: "█" },
        { label: "User", value: breakdown.user, char: "▓" },
        { label: "Assistant", value: breakdown.assistant, char: "▒" },
        { label: toolsLabel, value: breakdown.tools, char: "░" },
        ...(breakdown.summaries > 0
            ? [{ label: "Summaries", value: breakdown.summaries, char: "▚" }]
            : []),
    ] as const

    const maxLabelLen = Math.max(...categories.map((c) => c.label.length))
    const liveTotal = breakdown.liveTotal

    lines.push("╭───────────────────────────────────────────────────────────╮")
    lines.push("│                  DCP Context Analysis                     │")
    lines.push("╰───────────────────────────────────────────────────────────╯")
    lines.push("")
    lines.push("Live Context Breakdown (estimate):")
    lines.push("─".repeat(60))
    lines.push("")

    for (const cat of categories) {
        const bar = createBar(cat.value, liveTotal, barWidth, cat.char)
        const percentage = liveTotal > 0 ? ((cat.value / liveTotal) * 100).toFixed(1) : "0.0"
        const labelWithPct = `${cat.label.padEnd(maxLabelLen)} ${percentage.padStart(5)}% `
        const valueStr = formatTokenCount(cat.value).padStart(13)
        lines.push(`${labelWithPct}│${bar.padEnd(barWidth)}│${valueStr}`)
    }

    lines.push("")
    lines.push("─".repeat(60))
    lines.push("")

    lines.push("Summary:")
    lines.push(`  Current context: ~${formatTokenCount(liveTotal)} (live estimate)`)
    lines.push(
        `  Last API report: ~${formatTokenCount(breakdown.reportedTotal)}${breakdown.reportedFresh ? "" : " (stale after compression)"}`,
    )
    if (!breakdown.systemKnown) {
        lines.push("  System baseline: unknown (live estimate is partial)")
    }

    if (breakdown.prunedTokens > 0) {
        const withoutPruning = liveTotal + breakdown.prunedTokens
        const pruned = []
        if (breakdown.prunedToolCount > 0) pruned.push(`${breakdown.prunedToolCount} tools`)
        if (breakdown.prunedMessageCount > 0)
            pruned.push(`${breakdown.prunedMessageCount} messages`)
        lines.push(
            `  Pruned:          ${pruned.join(", ")} (~${formatTokenCount(breakdown.prunedTokens)})`,
        )
        lines.push(`  Without DCP:     ~${formatTokenCount(withoutPruning)}`)
    }

    lines.push("")

    return lines.join("\n")
}

export async function handleContextCommand(ctx: ContextCommandContext): Promise<void> {
    const { client, state, logger, sessionId, messages } = ctx

    const breakdown = analyzeContextTokens(state, messages)

    const message = formatContextMessage(breakdown)

    const params = getCurrentParams(state, messages, logger)
    await sendIgnoredMessage(client, sessionId, message, params, logger)
}
