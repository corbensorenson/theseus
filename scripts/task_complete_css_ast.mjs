#!/usr/bin/env node

import fs from 'node:fs'
import path from 'node:path'
import { createRequire } from 'node:module'

function fail(message) {
  process.stderr.write(`${message}\n`)
  process.exit(2)
}

const inputPath = process.argv[2]
const outputPath = process.argv[3]
const input = JSON.parse(fs.readFileSync(inputPath || 0, 'utf8'))
if (!input || typeof input.toolchainRoot !== 'string' || !Array.isArray(input.paths)) {
  fail('expected toolchainRoot and paths')
}

const requireFromToolchain = createRequire(
  path.join(path.resolve(input.toolchainRoot), 'package.json'),
)
const postcss = requireFromToolchain('postcss')

function lineOffsets(source) {
  const offsets = [0]
  for (let index = 0; index < source.length; index += 1) {
    if (source[index] === '\n') offsets.push(index + 1)
  }
  return offsets
}

function characterOffset(position, offsets, sourceLength, includeEnd = false) {
  if (!position || !Number.isInteger(position.line) || !Number.isInteger(position.column)) {
    return null
  }
  const lineOffset = offsets[position.line - 1]
  if (lineOffset === undefined) return null
  const offset = lineOffset + position.column - 1 + (includeEnd ? 1 : 0)
  return Math.min(sourceLength, Math.max(0, offset))
}

function ancestorKinds(node) {
  const kinds = []
  for (let current = node.parent; current && current.type !== 'root'; current = current.parent) {
    if (current.type === 'atrule') kinds.push(`@${current.name}`)
    else kinds.push(current.type)
  }
  return kinds.reverse()
}

function recordsForFile(filePath) {
  const source = fs.readFileSync(filePath, 'utf8')
  const offsets = lineOffsets(source)
  let root
  try {
    root = postcss.parse(source, { from: filePath })
  } catch (error) {
    return { records: [], parseError: String(error).slice(0, 500) }
  }
  const records = []
  root.walkRules((rule) => {
    const start = characterOffset(rule.source?.start, offsets, source.length, false)
    const end = characterOffset(rule.source?.end, offsets, source.length, true)
    if (start === null || end === null || end <= start) return
    const targetBody = source.slice(start, end)
    let reparsed
    try {
      reparsed = postcss.parse(targetBody, { from: undefined })
    } catch {
      return
    }
    const topRules = reparsed.nodes.filter((node) => node.type === 'rule')
    if (topRules.length !== 1 || topRules[0].selector !== rule.selector) return
    const declarations = (rule.nodes || []).filter((node) => node.type === 'decl')
    if (!rule.selector || declarations.length === 0) return
    records.push({
      path: filePath,
      start_byte: Buffer.byteLength(source.slice(0, start), 'utf8'),
      end_byte: Buffer.byteLength(source.slice(0, end), 'utf8'),
      start_char: start,
      end_char: end,
      selector: rule.selector,
      declaration_count: declarations.length,
      ancestor_kinds: ancestorKinds(rule),
      target_body: targetBody,
    })
  })
  return { records, parseError: null }
}

const records = []
const parseErrors = {}
for (const filePath of input.paths) {
  const parsed = recordsForFile(filePath)
  records.push(...parsed.records)
  if (parsed.parseError) parseErrors[filePath] = parsed.parseError
}
const serialized = JSON.stringify({
  postcssVersion: requireFromToolchain('postcss/package.json').version,
  records,
  parseErrors,
})
if (outputPath) fs.writeFileSync(outputPath, serialized)
else process.stdout.write(serialized)
