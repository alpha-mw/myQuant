"use strict";

const assert = require("assert");
const fs = require("fs");
const path = require("path");

const root = path.resolve(__dirname, "..");
const sample = JSON.parse(
  fs.readFileSync(path.join(root, "sample", "dashboard_snapshot.v3.json"), "utf8")
);

assert.strictEqual(sample.schema_version, "dashboard_contract.v3");
assert.ok(Array.isArray(sample.industries));
assert.strictEqual(Object.prototype.hasOwnProperty.call(sample, "themes"), false);
assert.strictEqual(Object.prototype.hasOwnProperty.call(sample, "theme_protocol"), false);
console.log("dashboard_contract_v3.test.js: PASS");
