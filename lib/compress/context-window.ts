import { analyzeContextTokens } from "../commands/context"
import type { SessionState, WithParts } from "../state"
import { isMessageCompacted } from "../state/utils"
import { countAllMessageTokens } from "../token-utils"

/**
 * Select the part of a session that still fits the last reported model
 * context. OOB compression must not spend writes on archived messages that
 * are no longer sent to the model.
 */
export function recentContextMessages(state: SessionState, messages: WithParts[]): WithParts[] {
    const breakdown = analyzeContextTokens(state, messages)
    const budget = breakdown.total > 0 ? breakdown.total : (state.modelContextLimit ?? 0)
    if (budget <= 0) {
        return messages
    }
    const selected: WithParts[] = []
    let tokens = 0
    for (let index = messages.length - 1; index >= 0; index--) {
        const message = messages[index]
        if (!message?.info?.id || isMessageCompacted(state, message)) {
            continue
        }
        const messageTokens = countAllMessageTokens(message)
        if (selected.length > 0 && tokens + messageTokens > budget) {
            break
        }
        selected.unshift(message)
        tokens += messageTokens
    }
    return selected
}
