import { execFileSync } from "node:child_process"
import { existsSync, readFileSync } from "node:fs"
import { dirname, join } from "node:path"
import { fileURLToPath } from "node:url"

export interface PluginIdentity {
    version: string
    commit: string
}

const UNKNOWN = "unknown"
let cached: PluginIdentity | undefined

function findPackageRoot(): string | undefined {
    let directory = dirname(fileURLToPath(import.meta.url))
    while (true) {
        const packagePath = join(directory, "package.json")
        if (existsSync(packagePath)) {
            try {
                const packageJson = JSON.parse(readFileSync(packagePath, "utf8"))
                if (packageJson?.name === "@tarquinen/opencode-dcp") {
                    return directory
                }
            } catch {
                // Keep walking; an unrelated or unreadable package.json is not authoritative.
            }
        }
        const parent = dirname(directory)
        if (parent === directory) {
            return undefined
        }
        directory = parent
    }
}

export function getPluginIdentity(): PluginIdentity {
    if (cached) {
        return cached
    }

    const root = findPackageRoot()
    if (!root) {
        cached = { version: UNKNOWN, commit: UNKNOWN }
        return cached
    }

    let version = UNKNOWN
    let commit = UNKNOWN
    try {
        const packageJson = JSON.parse(readFileSync(join(root, "package.json"), "utf8"))
        if (typeof packageJson.version === "string" && packageJson.version.length > 0) {
            version = packageJson.version
        }
        if (typeof packageJson.gitHead === "string" && packageJson.gitHead.length > 0) {
            commit = packageJson.gitHead.slice(0, 7)
        }
    } catch {
        // Fall through to unknown/defaults.
    }

    if (commit === UNKNOWN) {
        try {
            const result = execFileSync("git", ["rev-parse", "--short", "HEAD"], {
                cwd: root,
                encoding: "utf8",
                timeout: 1000,
            }).trim()
            if (result.length > 0) {
                commit = result
            }
        } catch {
            // Published packages have no checkout; version alone is still useful.
        }
    }

    cached = { version, commit }
    return cached
}
