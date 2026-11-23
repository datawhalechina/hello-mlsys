<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { withBase } from 'vitepress'
const props = withDefaults(defineProps<{ scene: string; title?: string; chapter?: '1' | '2' }>(), { chapter: '1' })
const frame = ref<HTMLIFrameElement | null>(null)
const height = ref(820)
const src = computed(() => withBase(`/demos/chapter${props.chapter}/embed.html?scene=${encodeURIComponent(props.scene)}`))
function resize(event: MessageEvent) {
  if (event.origin !== window.location.origin || event.source !== frame.value?.contentWindow) return
  const data = event.data
  if (data?.type !== 'hello-mlsys:formula-height' || data.scene !== props.scene) return
  if (typeof data.height === 'number' && Number.isFinite(data.height)) height.value = Math.min(2200, Math.max(320, Math.ceil(data.height)))
}
onMounted(() => window.addEventListener('message', resize))
onBeforeUnmount(() => window.removeEventListener('message', resize))
</script>
<template>
  <div class="formula-demo">
    <iframe ref="frame" :src="src" :title="title || `公式交互动画：${scene}`" :style="{ height: `${height}px` }" loading="lazy" @load="height = Math.max(height, 320)" />
  </div>
</template>
<style scoped>
.formula-demo { margin: 20px 0 28px; }
.formula-demo iframe { display: block; width: 100%; border: 0; color-scheme: light; }
</style>
