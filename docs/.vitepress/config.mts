import { defineConfig } from 'vitepress'
import { navItems, repoUrl, sidebar } from './outline.mjs'

export default defineConfig({
  lang: 'zh-CN',
  title: 'Hello MLSys',
  description: '连接设备，搭建平台，让每一份 AI 算力更高效：基于 AMD 硬件，以 vLLM 等开源推理框架为实践载体，学习从单机部署到多机 LLM 服务',
  base: '/hello-mlsys/',

  cleanUrls: true,

  markdown: {
    math: true,
  },

  themeConfig: {
    nav: navItems,

    sidebar,
    sidebarMenuLabel: '全书目录',
    returnToTopLabel: '返回顶部',
    darkModeSwitchLabel: '外观',
    lightModeSwitchTitle: '切换至浅色',
    darkModeSwitchTitle: '切换至深色',
    skipToContentLabel: '跳转到正文',

    search: {
      provider: 'local',
      options: {
        translations: {
          button: {
            buttonText: '搜索文档',
            buttonAriaLabel: '搜索文档',
          },
          modal: {
            noResultsText: '无法找到相关结果',
            resetButtonTitle: '清除查询条件',
            footer: {
              selectText: '选择',
              navigateText: '切换',
            },
          },
        },
      },
    },

    socialLinks: [
      { icon: 'github', link: repoUrl },
    ],

    editLink: {
      pattern: `${repoUrl}/blob/main/docs/:path`,
      text: '在 GitHub 上编辑此页',
    },

    outline: {
      level: [2, 3],
      label: '本章目录',
    },

    docFooter: {
      prev: '上一章',
      next: '下一章',
    },

    footer: {
      copyright: '本作品采用 <a href="http://creativecommons.org/licenses/by-nc-sa/4.0/" target="_blank">知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议（CC BY-NC-SA 4.0）</a> 进行许可',
    },
  },
})
