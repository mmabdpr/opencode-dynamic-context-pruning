import assert from "node:assert/strict"
import test from "node:test"
import { readFileSync } from "node:fs"
import { getPluginIdentity } from "../lib/plugin-identity"

test("reports the package version and a commit identity", () => {
    const pkg = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8"))
    const identity = getPluginIdentity()

    assert.equal(identity.version, pkg.version)
    assert.match(identity.commit, /^[0-9a-f]{7,}$|^unknown$/)
})
