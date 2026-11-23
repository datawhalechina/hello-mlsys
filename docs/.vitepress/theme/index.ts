import type { Theme } from 'vitepress'
import DefaultTheme from 'vitepress/theme'
import FormulaDemo from './components/FormulaDemo.vue'
export default {
  extends: DefaultTheme,
  enhanceApp({ app }) { app.component('FormulaDemo', FormulaDemo) },
} satisfies Theme
