#!/usr/bin/env node
// MirrorOPD adaptation of agent-maintenance's Markdown formatting workflow.
// Source and MIT attribution: LICENSE.agent-maintenance and SOURCES.md.
const fs = require("node:fs/promises");
const prettier = require("prettier");
const autocorrect = require("autocorrect-node");

function spacing(source, filename) {
  const saved = [];
  function keep(text) {
    const marker = `MIRROROPDPRESERVED${saved.length}ENDTOKEN`;
    saved.push([marker, text]);
    return marker;
  }
  // AutoCorrect's Markdown parser may also touch fenced strings/HTML attributes.
  // Preserve those contracts and TeX explicitly, then apply layout separately.
  let text = source.replace(
    /(^[ \t]*(`{3,}|~{3,})[^\n]*\n)([\s\S]*?)(^[ \t]*\2[ \t]*$)/gm,
    (_, start, fence, body, end) =>
      start + keep(body.replace(/\n$/, "")) + "\n" + end,
  );
  text = text.replace(/\$\$[\s\S]*?\$\$/g, keep);
  text = text.replace(/(?<!\\)\$(?!\$)(?:\\.|[^$\n])*?(?<!\\)\$/g, keep);
  text = text.replace(/<\/?[A-Za-z][^>]*>/g, keep);
  text = autocorrect.formatFor(text, filename);
  for (const [marker, value] of saved.reverse())
    text = text.replaceAll(marker, () => value);
  return text;
}
async function format(source, filename) {
  return prettier.format(spacing(source, filename), {
    filepath: filename,
    parser: "markdown",
    proseWrap: "preserve",
    tabWidth: 4,
    printWidth: 100,
    embeddedLanguageFormatting: "off",
  });
}
async function main() {
  const mode = process.argv[2];
  if (!["--write", "--check"].includes(mode))
    throw new Error("Use --write or --check");
  let raw = "";
  for await (const chunk of process.stdin) raw += chunk;
  const paths = JSON.parse(raw);
  if (!Array.isArray(paths) || !paths.length)
    throw new Error("No Markdown inputs");
  let changed = 0;
  for (const filename of paths) {
    const before = await fs.readFile(filename, "utf8");
    const after = await format(before, filename);
    if (before === after) continue;
    changed++;
    if (mode === "--write") {
      if ((await fs.readFile(filename, "utf8")) !== before)
        throw new Error(`Concurrent edit: ${filename}`);
      await fs.writeFile(filename, after);
    } else console.error(`Needs formatting: ${filename}`);
  }
  console.log(
    `Markdown ${paths.length}; ${mode === "--write" ? "formatted" : "noncanonical"} ${changed}`,
  );
  if (mode === "--check" && changed) process.exitCode = 1;
}
module.exports = { spacing, format };
if (require.main === module)
  main().catch((error) => {
    console.error(error.message);
    process.exitCode = 2;
  });
