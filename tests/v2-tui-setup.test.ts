import assert from "node:assert/strict"
import test from "node:test"
import { setup } from "../lib/v2/tui"

type DialogCall = { kind: "set"; value: unknown } | { kind: "show"; render: () => unknown }

interface HarnessOptions {
    route?: { type: string; sessionID?: string }
    status?: () => Promise<{ enabled: boolean }>
    snapshot?: () => Promise<typeof DEFAULT_SNAPSHOT>
    command?: (input: Record<string, unknown>) => Promise<void>
}

const DEFAULT_SNAPSHOT = {
    manualMode: false,
    canCompress: true,
    blockedReason: undefined,
    context: {},
    stats: {},
}

function createHarness(options: HarnessOptions = {}) {
    const calls = {
        status: 0,
        snapshot: 0,
        manual: 0,
        statusCalls: [] as Array<{ input: unknown; options: unknown }>,
        snapshotCalls: [] as Array<{ input: unknown; options: unknown }>,
        command: [] as Array<Record<string, unknown>>,
        dialogs: [] as DialogCall[],
    }
    let slot: any
    let commands: any[] = []

    const client = {
        status: async (input: unknown, callOptions: unknown) => {
            calls.status++
            calls.statusCalls.push({ input, options: callOptions })
            return (options.status ?? (async () => ({ enabled: true })))()
        },
        snapshot: async (input: unknown, callOptions: unknown) => {
            calls.snapshot++
            calls.snapshotCalls.push({ input, options: callOptions })
            return (options.snapshot ?? (async () => DEFAULT_SNAPSHOT))()
        },
        manual: async () => {
            calls.manual++
            return {}
        },
    }

    const ctx: any = {
        client: {
            rpc: (definition: { id: string }) => {
                assert.equal(definition.id, "dcp")
                return client
            },
            session: {
                command: async (input: Record<string, unknown>) => {
                    calls.command.push(input)
                    if (options.command) await options.command(input)
                },
            },
        },
        data: { location: { default: () => ({ directory: "/tmp/dcp-tui-test" }) } },
        renderer: {},
        theme: {},
        ui: {
            dialog: {
                clear: () => {},
                set: (value: unknown) => {
                    calls.dialogs.push({ kind: "set", value })
                },
                show: (render: () => unknown) => {
                    calls.dialogs.push({ kind: "show", render })
                },
            },
            router: {
                current: () => options.route ?? { type: "session", sessionID: "ses_tui_test" },
            },
            slot: (claim: any) => {
                slot = claim
            },
        },
        keymap: {
            layer: (factory: () => { commands: any[] }) => {
                commands = factory().commands
                return () => {}
            },
        },
    }

    return {
        ctx,
        calls,
        slot: () => slot,
        commands: () => commands,
    }
}

async function boot(options: HarnessOptions = {}) {
    const harness = createHarness(options)
    await setup(harness.ctx)
    assert.ok(harness.slot(), "expected a registered UI slot")
    assert.equal(harness.slot().append, "app")
    assert.equal(harness.slot().render(), null)
    return harness
}

test("setup registers the panel command without making an RPC request", async () => {
    const harness = await boot()

    assert.equal(harness.calls.status, 0)
    assert.equal(harness.calls.snapshot, 0)
    assert.equal(harness.commands().length, 1)
    assert.equal(harness.commands()[0].id, "dcp.panel")
    assert.equal(harness.commands()[0].palette, true)
    assert.deepEqual(harness.commands()[0].slash, { name: "dcp", arguments: true })
    assert.equal(harness.commands()[0].group, "DCP")
})

test("opening the panel checks status lazily and shows the disabled dialog", async () => {
    const harness = await boot({ status: async () => ({ enabled: false }) })

    await harness.commands()[0].run()

    assert.equal(harness.calls.status, 1)
    assert.equal(harness.calls.snapshot, 0)
    assert.equal(harness.calls.dialogs.filter((call) => call.kind === "set").length, 1)
    assert.equal(harness.calls.dialogs.filter((call) => call.kind === "show").length, 1)
})

test("opening the panel loads the snapshot when DCP is enabled", async () => {
    const harness = await boot()

    await harness.commands()[0].run()

    assert.equal(harness.calls.status, 1)
    assert.equal(harness.calls.snapshot, 1)
    assert.deepEqual(harness.calls.statusCalls[0], {
        input: {},
        options: { location: { directory: "/tmp/dcp-tui-test" } },
    })
    assert.deepEqual(harness.calls.snapshotCalls[0], {
        input: { sessionID: "ses_tui_test" },
        options: { location: { directory: "/tmp/dcp-tui-test" } },
    })
    assert.equal(harness.calls.dialogs.filter((call) => call.kind === "show").length, 1)
})

test("an unavailable first open is recoverable on the next open", async () => {
    let attempts = 0
    const harness = await boot({
        status: async () => {
            attempts++
            if (attempts === 1) {
                const cause: any = new Error("RPC is unavailable: dcp")
                cause.type = "rpc.unavailable"
                throw cause
            }
            return { enabled: true }
        },
    })

    await harness.commands()[0].run()
    assert.equal(harness.calls.status, 1)
    assert.equal(harness.calls.snapshot, 0)
    assert.equal(harness.calls.dialogs.filter((call) => call.kind === "show").length, 1)

    await harness.commands()[0].run()
    assert.equal(harness.calls.status, 2)
    assert.equal(harness.calls.snapshot, 1)
    assert.equal(harness.calls.dialogs.filter((call) => call.kind === "show").length, 2)
})

test("a snapshot failure is shown as an error and a later open recovers", async () => {
    let attempts = 0
    const harness = await boot({
        snapshot: async () => {
            attempts++
            if (attempts === 1) throw new Error("snapshot unavailable")
            return DEFAULT_SNAPSHOT
        },
    })

    await harness.commands()[0].run()
    assert.equal(harness.calls.status, 1)
    assert.equal(harness.calls.snapshot, 1)
    assert.equal(harness.calls.dialogs.filter((call) => call.kind === "show").length, 1)

    await harness.commands()[0].run()
    assert.equal(harness.calls.status, 2)
    assert.equal(harness.calls.snapshot, 2)
    assert.equal(harness.calls.dialogs.filter((call) => call.kind === "show").length, 2)
})

test("arguments on a non-session route fall back to open guidance", async () => {
    const harness = await boot({ route: { type: "home" } })

    await harness.commands()[0].run(" stats ")

    assert.equal(harness.calls.command.length, 0)
    assert.equal(harness.calls.status, 0)
    assert.equal(harness.calls.snapshot, 0)
    assert.equal(harness.calls.dialogs.filter((call) => call.kind === "show").length, 1)
})

test("a rejected argument command is shown as an error", async () => {
    const harness = await boot({
        command: async () => {
            throw new Error("command unavailable")
        },
    })

    await harness.commands()[0].run(" stats ")

    assert.deepEqual(harness.calls.command, [
        { sessionID: "ses_tui_test", name: "dcp", text: " stats " },
    ])
    assert.equal(harness.calls.status, 0)
    assert.equal(harness.calls.snapshot, 0)
    assert.equal(harness.calls.dialogs.filter((call) => call.kind === "show").length, 1)
})

test("a non-session route shows guidance without making an RPC request", async () => {
    const harness = await boot({ route: { type: "home" } })

    await harness.commands()[0].run()

    assert.equal(harness.calls.status, 0)
    assert.equal(harness.calls.snapshot, 0)
    assert.equal(harness.calls.dialogs.filter((call) => call.kind === "show").length, 1)
})

test("arguments are forwarded to the server command in a session", async () => {
    const harness = await boot()

    await harness.commands()[0].run(" stats ")

    assert.deepEqual(harness.calls.command, [
        { sessionID: "ses_tui_test", name: "dcp", text: " stats " },
    ])
})
