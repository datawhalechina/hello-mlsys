import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { existsSync, readdirSync } from 'node:fs'
import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { chapterHeading, chapters, parts } from './outline.mjs'

const __dirname = dirname(fileURLToPath(import.meta.url))
const repoRoot = join(__dirname, '..', '..')
const writeStubs = process.argv.includes('--write-stubs')
const checkOnly = process.argv.includes('--check')
const changedFiles = []

const alphaNotice = '::: danger ⚠️ Alpha内测版本警告\n此为早期内部构建版本，尚不完整且可能存在错误，欢迎大家提Issue反馈问题或建议。\n:::'

async function writeIfChanged(path, content) {
  const current = existsSync(path) ? await readFile(path, 'utf8') : null
  if (current?.replace(/\r\n/g, '\n') === content.replace(/\r\n/g, '\n')) return

  changedFiles.push(path)
  if (!checkOnly) {
    await mkdir(dirname(path), { recursive: true })
    await writeFile(path, content)
  }
}

function frontmatter(chapter) {
  const heading = chapterHeading(chapter)
  return `---\ntitle: "${heading}"\ndescription: "Hello MLSys · ${heading} · ${chapter.summary}"\n---\n`
}

function chapterMarkdown(chapter) {
  const sections = chapter.sections.map(([title, body], index) => {
    const number = chapter.number ? `${chapter.number}.${index + 1} ` : ''
    return `## ${number}${title}\n\n${body}`
  })

  const blocks = [
    frontmatter(chapter).trimEnd(),
    `# ${chapterHeading(chapter)}`,
    alphaNotice,
    `## 本章导读\n\n> ${chapter.lead}`,
    ...sections,
  ]

  if (chapter.code) {
    blocks.push(`## 本章小结\n\n- 本章目前是 Alpha 阶段的大纲骨架，正式正文会在对应实验跑通后补齐。\n- 与本章相关的代码、实验配置与测量记录会放在 \`${chapter.code}/\`。`)
  }

  blocks.push('## 延伸阅读\n\n- 待补充。')
  return `${blocks.join('\n\n')}\n`
}

// 已有正文只同步 frontmatter 与一级标题，不覆盖作者写下的内容。
function syncChapterMetadata(existing, chapter) {
  const body = existing.replace(/\r\n/g, '\n').replace(/^---\n[\s\S]*?\n---\n/, '').replace(/^\n+/, '')
  return `${frontmatter(chapter)}\n${body}`.replace(/^# .+$/m, `# ${chapterHeading(chapter)}`)
}

async function syncChapters() {
  for (const chapter of chapters) {
    const path = join(repoRoot, chapter.source)
    if (!existsSync(path) || writeStubs) {
      await writeIfChanged(path, chapterMarkdown(chapter))
    } else {
      await writeIfChanged(path, syncChapterMetadata(await readFile(path, 'utf8'), chapter))
    }
  }
}

function hasChapterCode(chapter) {
  if (!chapter.code) return false
  const dir = join(repoRoot, chapter.code)
  return existsSync(dir) && readdirSync(dir).some((name) => name !== '.gitkeep')
}

function readmeDirectory() {
  const lines = [
    '## 目录',
    '',
    '> 章节顺序就是推荐的学习顺序。全书内容覆盖「单机部署、服务评测、单机优化、多机服务、进阶协同」五个模块，其中「单机部署」与「多机服务」正在编写，完成后插入相应位置。vLLM 走通主线，llama.cpp 做对照与端侧；涉及 Radeon 多卡张量并行的结论（第 5、6 章）以本书实测为准。',
    '',
    '| 章节名 | 简介 | 所属模块 | 状态 |',
    '| ---- | ---- | ---- | ---- |',
  ]

  for (const part of parts) {
    lines.push(`| **${part.readmeTitle}** | | | |`)
    for (const chapter of chapters.filter((item) => item.part === part)) {
      const code = hasChapterCode(chapter) ? ` · [代码](./${chapter.code})` : ''
      lines.push(`| [${chapterHeading(chapter)}](./${chapter.source})${code} | ${chapter.summary} | ${chapter.module} | ${chapter.status} |`)
    }
  }

  return `${lines.join('\n')}\n`
}

async function syncReadme() {
  const path = join(repoRoot, 'README.md')
  const current = (await readFile(path, 'utf8')).replace(/\r\n/g, '\n')
  const next = current.replace(/## 目录\n[\s\S]*?\n## 贡献者名单/, `${readmeDirectory()}\n## 贡献者名单`)
  await writeIfChanged(path, next)
}

async function syncCurriculum() {
  const lines = [
    '---',
    'title: 全书目录',
    'description: Hello MLSys 的学习路线、全部章节与附录。',
    'outline: [2, 2]',
    'editLink: false',
    'prev: false',
    'next: false',
    '---',
    '',
    '<!-- 此页由 docs/.vitepress/sync-outline.mjs 生成，请在 outline.mjs 中维护章节。 -->',
    '',
    '# 全书目录',
    '',
    '章节顺序就是推荐的学习顺序。全书内容覆盖「单机部署、服务评测、单机优化、多机服务、进阶协同」五个模块，每章后标注了所属模块；「单机部署」与「多机服务」正在编写，完成后插入相应位置。选修案例和附录可按需查阅。',
    '',
    '::: info 阅读说明',
    '教程处于 Alpha 阶段，部分章节仍为大纲或待补充内容，请以各章现有正文与验证记录为准。',
    ':::',
  ]

  for (const part of parts) {
    lines.push('', `## ${part.readmeTitle}`, '')
    for (const chapter of chapters.filter((item) => item.part === part)) {
      const module = chapter.module && chapter.module !== '——' ? `（${chapter.module}）` : ''
      lines.push(`- [${chapterHeading(chapter)}](${chapter.path}) — ${chapter.summary}${module}`)
    }
  }

  await writeIfChanged(join(repoRoot, 'docs/curriculum/index.md'), `${lines.join('\n')}\n`)
}

async function main() {
  await syncChapters()
  await syncReadme()
  await syncCurriculum()

  if (checkOnly && changedFiles.length > 0) {
    console.error('Docs outline is out of sync:')
    for (const path of changedFiles) console.error(`- ${path}`)
    process.exit(1)
  }

  console.log(changedFiles.length > 0
    ? `Synced docs outline (${changedFiles.length} file${changedFiles.length === 1 ? '' : 's'}).`
    : 'Docs outline already in sync.')
}

main().catch((error) => {
  console.error(error)
  process.exit(1)
})
