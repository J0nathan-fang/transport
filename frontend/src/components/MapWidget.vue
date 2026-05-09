<!-- src/components/MapWidget.vue -->
<template>
  <div class="w-full h-full rounded-lg relative">
    <!-- 1. 地图容器 -->
    <div id="amap-container" class="w-full h-full rounded-lg z-0"></div>

    <!-- 2. 自定义路况开关面板 -->
    <div class="absolute top-4 right-4 z-10 bg-white rounded-lg shadow-lg p-3 flex items-center space-x-3">
      <label for="traffic-switch" class="text-sm font-medium text-slate-700 select-none">显示实时车流量</label>
      <button
        id="traffic-switch"
        @click="toggleTraffic"
        :class="showTraffic ? 'bg-blue-600' : 'bg-slate-300'"
        class="relative inline-flex flex-shrink-0 h-6 w-11 border-2 border-transparent rounded-full cursor-pointer transition-colors ease-in-out duration-200 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
        role="switch"
        :aria-checked="showTraffic"
      >
        <span
          aria-hidden="true"
          :class="showTraffic ? 'translate-x-5' : 'translate-x-0'"
          class="pointer-events-none inline-block h-5 w-5 rounded-full bg-white shadow transform ring-0 transition ease-in-out duration-200"
        ></span>
      </button>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, shallowRef } from 'vue';
import AMapLoader from '@amap/amap-jsapi-loader';

const emit = defineEmits(['map-click']);

// 使用 shallowRef 存储地图实例，避免深度代理带来的性能问题
const map = shallowRef(null);
const trafficLayer = shallowRef(null);
const showTraffic = ref(false);

// 切换路况图层显示状态
const toggleTraffic = () => {
  showTraffic.value = !showTraffic.value;
  if (trafficLayer.value) {
    if (showTraffic.value) {
      trafficLayer.value.show();
    } else {
      trafficLayer.value.hide();
    }
  }
};

onMounted(() => {
  const amapKey = import.meta.env.VITE_AMAP_JS_KEY;
  if (!amapKey) {
    console.error('VITE_AMAP_JS_KEY is not configured in .env file.');
    return;
  }

  AMapLoader.load({
    key: amapKey,
    version: "2.0",
    plugins: ['AMap.TileLayer.Traffic'], // 加载实时路况插件
  })
  .then((AMap) => {
    // 1. 实例化地图
    map.value = new AMap.Map('amap-container', {
      zoom: 14,
      center: [103.9870, 30.7613], // Default to SWJTU
      viewMode: '2D',
      pitch: 0,
    });

    // 2. 实例化实时交通图层
    trafficLayer.value = new AMap.TileLayer.Traffic({
      autoRefresh: true,
      interval: 180000, // 3分钟刷新一次
      opacity: 0.85,
      visible: false, // 初始不可见
      zIndex: 10, // 确保在底图之上
    });

    // 3. 将路况图层添加到地图实例
    map.value.add(trafficLayer.value);

    // 4. 监听地图点击事件
    map.value.on('click', (e) => {
      const coords = {
        lat: e.lnglat.lat,
        lon: e.lnglat.lng,
      };
      emit('map-click', coords);
    });
  })
  .catch((e) => {
    console.error('高德地图 JS API 加载失败', e);
  });
});

// 5. 组件销毁时清理地图实例
onUnmounted(() => {
  if (map.value) {
    map.value.destroy();
  }
});
</script>

<style scoped>
/* 确保高德地图的 logo 和版权信息能正常显示 */
:deep(.amap-logo) {
  display: block !important;
  z-index: 1 !important;
}
:deep(.amap-copyright) {
  display: block !important;
  z-index: 1 !important;
}
</style>
