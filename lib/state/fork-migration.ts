import type { Logger } from "../logger"
import { loadSessionState, type PersistedSessionState } from "./persistence"
import type { CompressionBlock, PrunedMessageEntry, WithParts } from "./types"

export interface ForkBoundary {
    type: "before" | "after"
    messageID: string
}

export interface ForkRemapResult {
    state: PersistedSessionState
    prefixLength: number
    totalMessageKeys: number
    mappedMessageKeys: number
    totalBlocks: number
    mappedBlocks: number
}

const MIN_MESSAGE_KEY_COVERAGE = 0.5

function messageId(message: WithParts | undefined): string | undefined {
    const id = message?.info?.id
    return typeof id === "string" && id.length > 0 ? id : undefined
}

function sameMessagePrefix(parent: WithParts, fork: WithParts): boolean {
    if (!parent?.info || !fork?.info) {
        return false
    }
    if (parent.info.role !== fork.info.role) {
        return false
    }
    return parent.info.time?.created === fork.info.time?.created
}

function mapMessageIds(ids: string[] | undefined, mapping: Map<string, string>): string[] {
    const mapped: string[] = []
    for (const id of ids ?? []) {
        const forkId = mapping.get(id)
        if (forkId) {
            mapped.push(forkId)
        }
    }
    return [...new Set(mapped)]
}

function mapAnchorIds(ids: string[] | undefined, mapping: Map<string, string>): string[] {
    return mapMessageIds(ids, mapping)
}

function collectToolIds(messages: WithParts[]): Set<string> {
    const toolIds = new Set<string>()
    for (const message of messages) {
        for (const part of message.parts ?? []) {
            if (part.type === "tool" && typeof part.callID === "string" && part.callID.length > 0) {
                toolIds.add(part.callID)
            }
        }
    }
    return toolIds
}

/**
 * Rebuild DCP's persisted compression state for a forked session.
 *
 * OpenCode copies the parent's messages into a fork but assigns new message IDs. DCP keys its
 * compression state by message ID, so the fork otherwise starts with an empty state and reloads
 * the full uncompressed history. This maps the parent's prefix 1:1 onto the fork and drops any
 * blocks that refer to parent messages created after the fork point.
 */
export function remapForkState(
    parentState: PersistedSessionState,
    parentMessages: WithParts[],
    forkMessages: WithParts[],
    boundary?: ForkBoundary,
): ForkRemapResult | null {
    const parentById = new Map<string, number>()
    for (let index = 0; index < parentMessages.length; index++) {
        const id = messageId(parentMessages[index])
        if (id) {
            parentById.set(id, index)
        }
    }

    let prefixLength = Math.min(parentMessages.length, forkMessages.length)
    if (boundary?.messageID) {
        const boundaryIndex = parentById.get(boundary.messageID)
        if (boundaryIndex !== undefined) {
            prefixLength =
                boundary.type === "after"
                    ? Math.min(boundaryIndex + 1, prefixLength)
                    : Math.min(boundaryIndex, prefixLength)
        }
    }

    let matchedPrefix = 0
    while (matchedPrefix < prefixLength) {
        if (!sameMessagePrefix(parentMessages[matchedPrefix], forkMessages[matchedPrefix])) {
            break
        }
        matchedPrefix++
    }

    if (matchedPrefix === 0) {
        return null
    }

    const mapping = new Map<string, string>()
    for (let index = 0; index < matchedPrefix; index++) {
        const parentId = messageId(parentMessages[index])
        const forkId = messageId(forkMessages[index])
        if (parentId && forkId) {
            mapping.set(parentId, forkId)
        }
    }

    const persistedMessages = parentState.prune?.messages
    if (!persistedMessages) {
        return null
    }

    const prefixMessageIds = new Set<string>()
    for (let index = 0; index < matchedPrefix; index++) {
        const id = messageId(parentMessages[index])
        if (id) {
            prefixMessageIds.add(id)
        }
    }

    const totalMessageKeys = Object.keys(persistedMessages.byMessageId ?? {}).filter((id) =>
        prefixMessageIds.has(id),
    ).length
    const mappedMessageKeys = Object.keys(persistedMessages.byMessageId ?? {}).filter((id) =>
        mapping.has(id),
    ).length

    if (totalMessageKeys === 0) {
        return null
    }
    if (mappedMessageKeys / totalMessageKeys < MIN_MESSAGE_KEY_COVERAGE) {
        return null
    }

    const forkToolIds = collectToolIds(forkMessages)
    const blocks = new Map<number, CompressionBlock>()

    for (const [key, block] of Object.entries(persistedMessages.blocksById ?? {})) {
        const blockId = Number.isInteger(block?.blockId) ? block.blockId : Number(key)
        if (!Number.isInteger(blockId) || blockId <= 0) {
            continue
        }

        const anchorMessageId = mapping.get(block.anchorMessageId)
        const compressMessageId = mapping.get(block.compressMessageId)
        if (!anchorMessageId || !compressMessageId) {
            continue
        }

        blocks.set(blockId, {
            ...block,
            blockId,
            anchorMessageId,
            compressMessageId,
            directMessageIds: mapMessageIds(block.directMessageIds, mapping),
            effectiveMessageIds: mapMessageIds(block.effectiveMessageIds, mapping),
            directToolIds: (block.directToolIds ?? []).filter((id) => forkToolIds.has(id)),
            effectiveToolIds: (block.effectiveToolIds ?? []).filter((id) => forkToolIds.has(id)),
        })
    }

    const survivingBlockIds = new Set(blocks.keys())
    for (const block of blocks.values()) {
        block.includedBlockIds = (block.includedBlockIds ?? []).filter((id) =>
            survivingBlockIds.has(id),
        )
        block.consumedBlockIds = (block.consumedBlockIds ?? []).filter((id) =>
            survivingBlockIds.has(id),
        )
        block.parentBlockIds = (block.parentBlockIds ?? []).filter((id) =>
            survivingBlockIds.has(id),
        )
        if (
            block.deactivatedByBlockId !== undefined &&
            !survivingBlockIds.has(block.deactivatedByBlockId)
        ) {
            block.deactivatedByBlockId = undefined
        }
        if (!block.active || block.deactivatedByUser) {
            block.active = false
        }
    }

    const activeBlockIdSet = new Set(
        [...blocks.values()]
            .filter((block) => block.active && !block.deactivatedByUser)
            .map((block) => block.blockId),
    )

    const byMessageId: Record<string, PrunedMessageEntry> = {}
    for (const [parentId, entry] of Object.entries(persistedMessages.byMessageId ?? {})) {
        const forkId = mapping.get(parentId)
        if (!forkId) {
            continue
        }
        const allBlockIds = (entry.allBlockIds ?? []).filter((id) => survivingBlockIds.has(id))
        const activeBlockIds = (entry.activeBlockIds ?? []).filter((id) => activeBlockIdSet.has(id))
        if (allBlockIds.length === 0 && activeBlockIds.length === 0) {
            continue
        }
        byMessageId[forkId] = {
            tokenCount: entry.tokenCount ?? 0,
            allBlockIds,
            activeBlockIds,
        }
    }

    const activeBlockIds = [...activeBlockIdSet].sort((a, b) => a - b)

    const activeByAnchorMessageId: Record<string, number> = {}
    for (const blockId of activeBlockIds) {
        const block = blocks.get(blockId)
        if (block) {
            activeByAnchorMessageId[block.anchorMessageId] = blockId
        }
    }

    const pruneTools: Record<string, number> = {}
    for (const [toolId, tokenCount] of Object.entries(parentState.prune?.tools ?? {})) {
        if (forkToolIds.has(toolId)) {
            pruneTools[toolId] = tokenCount
        }
    }

    const maxBlockId = Math.max(0, ...blocks.keys())
    const maxRunId = Math.max(0, ...[...blocks.values()].map((block) => block.runId ?? 0))

    return {
        state: {
            sessionName: parentState.sessionName,
            manualMode: parentState.manualMode,
            prune: {
                tools: pruneTools,
                messages: {
                    byMessageId,
                    blocksById: Object.fromEntries(
                        [...blocks.entries()].map(([blockId, block]) => [String(blockId), block]),
                    ),
                    activeBlockIds,
                    activeByAnchorMessageId,
                    nextBlockId: Math.max(persistedMessages.nextBlockId ?? 1, maxBlockId + 1),
                    nextRunId: Math.max(persistedMessages.nextRunId ?? 1, maxRunId + 1),
                },
            },
            nudges: {
                contextLimitAnchors: mapAnchorIds(parentState.nudges?.contextLimitAnchors, mapping),
                turnNudgeAnchors: mapAnchorIds(parentState.nudges?.turnNudgeAnchors, mapping),
                iterationNudgeAnchors: mapAnchorIds(
                    parentState.nudges?.iterationNudgeAnchors,
                    mapping,
                ),
            },
            stats: {
                pruneTokenCounter: parentState.stats?.pruneTokenCounter ?? 0,
                totalPruneTokens: parentState.stats?.totalPruneTokens ?? 0,
            },
            lastUpdated: new Date().toISOString(),
        },
        prefixLength: matchedPrefix,
        totalMessageKeys,
        mappedMessageKeys,
        totalBlocks: Object.keys(persistedMessages.blocksById ?? {}).length,
        mappedBlocks: blocks.size,
    }
}

const migrationAttempted = new Set<string>()

export async function migrateForkStateForSession(
    client: any,
    sessionId: string,
    messages: WithParts[],
    logger: Logger,
): Promise<PersistedSessionState | null> {
    if (migrationAttempted.has(sessionId)) {
        return null
    }
    migrationAttempted.add(sessionId)

    try {
        const sessionResult = await client.session.get({ path: { id: sessionId } })
        const session = sessionResult?.data ?? sessionResult
        const fork = (session as any)?.fork as
            | { sessionID?: string; boundary?: ForkBoundary }
            | undefined
        if (!fork?.sessionID) {
            return null
        }

        const parentState = await loadSessionState(fork.sessionID, logger)
        if (!parentState) {
            return null
        }

        const parentResult = await client.session.messages({ path: { id: fork.sessionID } })
        const parentMessages = (parentResult?.data ?? parentResult) as WithParts[]
        if (!Array.isArray(parentMessages) || parentMessages.length === 0) {
            return null
        }

        const remapped = remapForkState(parentState, parentMessages, messages, fork.boundary)
        if (!remapped) {
            logger.warn("Fork state migration skipped: no reliable parent prefix", {
                sessionId,
                parentId: fork.sessionID,
            })
            return null
        }

        logger.info("Fork state migrated", {
            sessionId,
            parentId: fork.sessionID,
            prefixLength: remapped.prefixLength,
            mappedMessageKeys: remapped.mappedMessageKeys,
            totalMessageKeys: remapped.totalMessageKeys,
            mappedBlocks: remapped.mappedBlocks,
            totalBlocks: remapped.totalBlocks,
        })

        return remapped.state
    } catch (error: any) {
        logger.warn("Fork state migration failed", {
            sessionId,
            error: error?.message ?? String(error),
        })
        return null
    }
}
