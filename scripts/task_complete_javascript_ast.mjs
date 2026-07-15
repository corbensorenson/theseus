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
const ts = requireFromToolchain('typescript')

function scriptKind(filePath) {
  const lower = filePath.toLowerCase()
  if (lower.endsWith('.tsx')) return ts.ScriptKind.TSX
  if (lower.endsWith('.jsx')) return ts.ScriptKind.JSX
  if (lower.endsWith('.js') || lower.endsWith('.mjs') || lower.endsWith('.cjs')) {
    return ts.ScriptKind.JS
  }
  return ts.ScriptKind.TS
}

function displayName(node, fallback) {
  if (node.name && typeof node.name.getText === 'function') {
    return node.name.getText()
  }
  return fallback
}

function bodyRecord(sourceFile, body, qualifiedName, sourceText) {
  if (!body || !ts.isBlock(body)) return null
  const open = body.getStart(sourceFile)
  const close = body.end - 1
  if (sourceText[open] !== '{' || sourceText[close] !== '}') return null
  const start = open + 1
  const end = close
  const targetBody = sourceText.slice(start, end)
  const lineStart = sourceText.lastIndexOf('\n', start - 1) + 1
  const indentation = sourceText.slice(lineStart, start).match(/^[ \t]*/)?.[0] ?? ''
  const memberIndent = `${indentation}  `
  const newline = sourceText.includes('\r\n') ? '\r\n' : '\n'
  const starterBody = `${newline}${memberIndent}throw new Error("Theseus task-complete implementation hole")${newline}${indentation}`
  const mutated = sourceText.slice(0, start) + starterBody + sourceText.slice(end)
  const parsedMutation = ts.createSourceFile(
    sourceFile.fileName,
    mutated,
    ts.ScriptTarget.Latest,
    true,
    scriptKind(sourceFile.fileName),
  )
  if (parsedMutation.parseDiagnostics.length > 0) return null
  return {
    qualified_name: qualifiedName,
    start_byte: Buffer.byteLength(sourceText.slice(0, start), 'utf8'),
    end_byte: Buffer.byteLength(sourceText.slice(0, end), 'utf8'),
    target_body: targetBody,
    starter_body: starterBody,
  }
}

function objectLiteralMethods(sourceFile, objectLiteral, prefix, sourceText, records) {
  for (const property of objectLiteral.properties) {
    if (ts.isMethodDeclaration(property) || ts.isGetAccessor(property) || ts.isSetAccessor(property)) {
      const name = displayName(property, 'anonymous')
      const row = bodyRecord(sourceFile, property.body, `${prefix}.${name}`, sourceText)
      if (row) records.push(row)
    } else if (ts.isPropertyAssignment(property)) {
      const name = displayName(property, 'anonymous')
      const initializer = property.initializer
      if (ts.isArrowFunction(initializer) || ts.isFunctionExpression(initializer)) {
        const row = bodyRecord(sourceFile, initializer.body, `${prefix}.${name}`, sourceText)
        if (row) records.push(row)
      }
    }
  }
}

function isFunctionLike(node) {
  return (
    ts.isFunctionDeclaration(node) ||
    ts.isMethodDeclaration(node) ||
    ts.isGetAccessor(node) ||
    ts.isSetAccessor(node) ||
    ts.isConstructorDeclaration(node) ||
    ts.isArrowFunction(node) ||
    ts.isFunctionExpression(node)
  )
}

function functionName(sourceFile, node) {
  if (ts.isConstructorDeclaration(node)) {
    const owner = node.parent && ts.isClassDeclaration(node.parent)
      ? displayName(node.parent, 'default')
      : 'class'
    return `${owner}.constructor`
  }
  if (node.name && typeof node.name.getText === 'function') {
    const ownName = node.name.getText(sourceFile)
    if (node.parent && ts.isClassDeclaration(node.parent)) {
      return `${displayName(node.parent, 'default')}.${ownName}`
    }
    return ownName
  }
  if (node.parent && ts.isVariableDeclaration(node.parent)) {
    return node.parent.name.getText(sourceFile)
  }
  if (node.parent && ts.isPropertyAssignment(node.parent)) {
    return node.parent.name.getText(sourceFile)
  }
  const position = sourceFile.getLineAndCharacterOfPosition(node.getStart(sourceFile))
  return `callback@${position.line + 1}:${position.character + 1}`
}

function recordsForFile(filePath) {
  const sourceText = fs.readFileSync(filePath, 'utf8')
  const sourceFile = ts.createSourceFile(
    filePath,
    sourceText,
    ts.ScriptTarget.Latest,
    true,
    scriptKind(filePath),
  )
  if (sourceFile.parseDiagnostics.length > 0) return { records: [], imports: [] }
  const records = []
  for (const statement of sourceFile.statements) {
    if (ts.isFunctionDeclaration(statement)) {
      const row = bodyRecord(
        sourceFile,
        statement.body,
        displayName(statement, 'default'),
        sourceText,
      )
      if (row) records.push(row)
    } else if (ts.isClassDeclaration(statement)) {
      const className = displayName(statement, 'default')
      for (const member of statement.members) {
        if (
          ts.isMethodDeclaration(member) ||
          ts.isGetAccessor(member) ||
          ts.isSetAccessor(member) ||
          ts.isConstructorDeclaration(member)
        ) {
          const name = ts.isConstructorDeclaration(member)
            ? 'constructor'
            : displayName(member, 'anonymous')
          const row = bodyRecord(sourceFile, member.body, `${className}.${name}`, sourceText)
          if (row) records.push(row)
        }
      }
    } else if (ts.isVariableStatement(statement)) {
      for (const declaration of statement.declarationList.declarations) {
        const name = declaration.name.getText(sourceFile)
        const initializer = declaration.initializer
        if (!initializer) continue
        if (ts.isArrowFunction(initializer) || ts.isFunctionExpression(initializer)) {
          const row = bodyRecord(sourceFile, initializer.body, name, sourceText)
          if (row) records.push(row)
        } else if (ts.isObjectLiteralExpression(initializer)) {
          objectLiteralMethods(sourceFile, initializer, name, sourceText, records)
        }
      }
    } else if (ts.isExportAssignment(statement)) {
      const expression = statement.expression
      if (ts.isArrowFunction(expression) || ts.isFunctionExpression(expression)) {
        const row = bodyRecord(sourceFile, expression.body, 'default', sourceText)
        if (row) records.push(row)
      } else if (ts.isObjectLiteralExpression(expression)) {
        objectLiteralMethods(sourceFile, expression, 'default', sourceText, records)
      }
    }
  }
  const recordedSpans = new Set(
    records.map((row) => `${row.start_byte}:${row.end_byte}`),
  )

  function visitNested(node, scope) {
    if (isFunctionLike(node)) {
      const name = functionName(sourceFile, node)
      const qualifiedName = [...scope, name].join('.')
      const row = bodyRecord(sourceFile, node.body, qualifiedName, sourceText)
      if (row) {
        const key = `${row.start_byte}:${row.end_byte}`
        if (!recordedSpans.has(key)) {
          recordedSpans.add(key)
          records.push(row)
        }
      }
      if (node.body) {
        ts.forEachChild(node.body, (child) => visitNested(child, [...scope, name]))
      }
      return
    }
    ts.forEachChild(node, (child) => visitNested(child, scope))
  }

  visitNested(sourceFile, [])
  const imports = ts.preProcessFile(sourceText, true, true).importedFiles.map(
    (row) => row.fileName,
  )
  return { records, imports }
}

const output = []
const imports = {}
for (const filePath of input.paths) {
  const parsed = recordsForFile(filePath)
  imports[filePath] = parsed.imports
  for (const record of parsed.records) {
    output.push({ path: filePath, ...record })
  }
}
const serialized = JSON.stringify({
  typescriptVersion: ts.version,
  records: output,
  imports,
})
if (outputPath) {
  fs.writeFileSync(outputPath, serialized)
} else {
  process.stdout.write(serialized)
}
