<!-- src/components/AnalyticsWidget.vue -->
<template>
  <!-- 1. 外层容器 -->
  <div class="h-full w-full bg-slate-50 p-4 rounded-lg flex flex-col gap-4 overflow-y-auto">

    <!-- 空状态提示 -->
    <div v-if="!coords" class="flex-1 flex flex-col justify-center items-center text-center text-slate-500">
      <svg xmlns="http://www.w3.org/2000/svg" class="w-12 h-12 mb-4 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5">
        <path stroke-linecap="round" stroke-linejoin="round" d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
        <path stroke-linecap="round" stroke-linejoin="round" d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
      </svg>
      <p class="font-medium">请在地图上点击一个位置</p>
      <p class="text-sm text-slate-400">以开始交通事故风险分析</p>
    </div>

    <!-- 有坐标时显示内容 -->
    <template v-else>
      <!-- 加载状态 -->
      <div v-if="isLoading" class="flex-1 flex flex-col justify-center items-center text-center text-slate-500">
        <svg class="animate-spin h-8 w-8 text-blue-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
          <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
          <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
        </svg>
        <p class="mt-4 font-medium">正在进行多维风险分析...</p>
      </div>

      <!-- 分析结果 -->
      <template v-else>
        <!-- 2. 卡片 1：标题与定位 -->
        <div class="bg-white rounded-xl shadow-sm p-4 flex-shrink-0 transition-all duration-300">
          <h2 class="font-bold text-lg text-slate-800 mb-2">交通事故风控分析</h2>
          <div class="flex items-center gap-2">
            <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
              <path stroke-linecap="round" stroke-linejoin="round" d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
              <path stroke-linecap="round" stroke-linejoin="round" d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
            </svg>
            <span class="text-sm text-slate-500">分析位置</span>
            <span class="bg-slate-100 text-slate-600 text-xs font-mono px-2 py-1 rounded-full">
              {{ coords.lat.toFixed(4) }}, {{ coords.lon.toFixed(4) }}
            </span>
          </div>
        </div>

        <!-- 3. 卡片 2：综合风险评估 (动态颜色) -->
        <div class="bg-white rounded-xl shadow-sm p-5 flex flex-col items-center justify-center flex-shrink-0">
          <div class="relative flex items-center justify-center w-24 h-24">
            <!-- 动态背景光晕 -->
            <div class="absolute inset-0 rounded-full blur-lg opacity-60 transition-colors duration-500" :class="currentTheme.bgLight"></div>
            <!-- 动态边框环 -->
            <div class="absolute inset-2 rounded-full ring-4 transition-colors duration-500" :class="[currentTheme.bgLightest, currentTheme.ring]"></div>
            <!-- 动态图标与分数 -->
            <div class="relative flex flex-col items-center transition-colors duration-500" :class="currentTheme.text">
              <svg xmlns="http://www.w3.org/2000/svg" class="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
                <path stroke-linecap="round" stroke-linejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
              </svg>
              <span class="text-2xl font-bold mt-1">{{ riskScore }}分</span>
            </div>
          </div>
          <p class="mt-3 font-semibold text-lg transition-colors duration-500" :class="currentTheme.textMedium">
            {{ currentTheme.name }}
          </p>
        </div>

        <!-- 4. 卡片 3：风险归因分析 (动态数据条) -->
        <div class="bg-white rounded-xl shadow-sm p-4 flex-shrink-0">
          <h3 class="text-md font-semibold mb-4 text-slate-700">风险归因分析</h3>
          <div class="space-y-4">
            <!-- 天气状况 -->
            <div>
              <div class="flex justify-between items-center mb-1">
                <span class="text-sm font-medium text-slate-600">天气状况</span>
                <span class="text-xs font-mono text-slate-500">{{ factors.weather.toFixed(0) }}%</span>
              </div>
              <div class="w-full bg-slate-100 rounded-full h-2 overflow-hidden">
                <div class="h-2 rounded-full transition-all duration-1000 ease-out" 
                     :class="getBarColor(factors.weather)" 
                     :style="{ width: factors.weather + '%' }"></div>
              </div>
            </div>
            <!-- 路面湿滑度 -->
            <div>
              <div class="flex justify-between items-center mb-1">
                <span class="text-sm font-medium text-slate-600">路面湿滑度</span>
                <span class="text-xs font-mono text-slate-500">{{ factors.road.toFixed(0) }}%</span>
              </div>
              <div class="w-full bg-slate-100 rounded-full h-2 overflow-hidden">
                <div class="h-2 rounded-full transition-all duration-1000 ease-out delay-100" 
                     :class="getBarColor(factors.road)" 
                     :style="{ width: factors.road + '%' }"></div>
              </div>
            </div>
            <!-- 实时车流量 -->
            <div>
              <div class="flex justify-between items-center mb-1">
                <span class="text-sm font-medium text-slate-600">实时拥堵度</span>
                <span class="text-xs font-mono text-slate-500">{{ factors.traffic.toFixed(0) }}%</span>
              </div>
              <div class="w-full bg-slate-100 rounded-full h-2 overflow-hidden">
                <div class="h-2 rounded-full transition-all duration-1000 ease-out delay-200" 
                     :class="getBarColor(factors.traffic)" 
                     :style="{ width: factors.traffic + '%' }"></div>
              </div>
            </div>
          </div>
        </div>

        <!-- 5. 卡片 4：智能出行建议 (动态主题) -->
        <div class="border rounded-xl p-4 flex-shrink-0 transition-colors duration-500" 
             :class="[currentTheme.bgLightest, currentTheme.border]">
          <div class="flex items-start gap-3">
            <div class="flex-shrink-0">
              <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" :class="currentTheme.text" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
                <path stroke-linecap="round" stroke-linejoin="round" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
              </svg>
            </div>
            <div class="flex-1">
              <h4 class="font-medium mb-1" :class="currentTheme.textDark">智能出行建议</h4>
              <p class="text-sm leading-relaxed" :class="currentTheme.textMedium">
                {{ advice }}
              </p>
            </div>
          </div>
        </div>
      </template>
    </template>

  </div>
</template>

<script setup>
import { ref, watch, defineProps } from 'vue';

const props = defineProps({
  coords: {
    type: Object,
    default: null
  },
});

// 响应式数据
const isLoading = ref(false);
const riskScore = ref(0);
const factors = ref({ weather: 0, road: 0, traffic: 0 });
const advice = ref('');

// 主题颜色配置字典 (Tailwind 类名)
const themes = {
  low: {
    name: '低风险',
    text: 'text-emerald-500',
    textMedium: 'text-emerald-600',
    textDark: 'text-emerald-800',
    bgLightest: 'bg-emerald-50',
    bgLight: 'bg-emerald-100',
    border: 'border-emerald-200',
    ring: 'ring-emerald-100'
  },
  medium: {
    name: '中等风险',
    text: 'text-amber-500',
    textMedium: 'text-amber-600',
    textDark: 'text-amber-800',
    bgLightest: 'bg-amber-50',
    bgLight: 'bg-amber-100',
    border: 'border-amber-200',
    ring: 'ring-amber-100'
  },
  high: {
    name: '高危风险',
    text: 'text-rose-500',
    textMedium: 'text-rose-600',
    textDark: 'text-rose-800',
    bgLightest: 'bg-rose-50',
    bgLight: 'bg-rose-100',
    border: 'border-rose-200',
    ring: 'ring-rose-100'
  }
};

const currentTheme = ref(themes.low);

// 进度条颜色计算器
const getBarColor = (val) => {
  if (val >= 70) return 'bg-rose-500';
  if (val >= 40) return 'bg-amber-500';
  return 'bg-emerald-500';
};

// 监听坐标变化，调用后端机器学习预测接口
watch(() => props.coords, async (newCoords) => {
  if (!newCoords) return;

  isLoading.value = true;
  const { lat, lon } = newCoords;

  try {
    // 调用后端预测接口
    const response = await fetch('/api/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ lng: lon, lat: lat }),
    });

    if (!response.ok) {
      throw new Error(`后端预测服务响应失败 (Status: ${response.status})`);
    }

    const data = await response.json();
    
    // 从后端响应中提取数据
    riskScore.value = Math.round(data.risk_score * 100); // 转换为百分制
    const riskLevel = data.risk_level;
    const features = data.features;
    const weatherInfo = data.weather_info;

    // 更新因子数据（基于特征构造权重）
    factors.value.weather = features.weather_sunny ? 20 : 
                          features.weather_rain ? 90 : 
                          features.weather_fog ? 95 : 
                          features.weather_snow ? 85 : 50;
    
    factors.value.road = weatherInfo && parseFloat(weatherInfo.precip) > 0 ? 
                        Math.min(95, 40 + parseFloat(weatherInfo.precip) * 10) : 15;
    
    // 使用真实的交通拥堵度数据
    factors.value.traffic = data.traffic_info ? data.traffic_info.congestion_score : 
                          (features.rushhour ? 75 : features.night ? 30 : 45);

    // 根据风险等级和交通状况匹配主题和建议
    const trafficDesc = data.traffic_info ? data.traffic_info.description : '';
    const trafficCongestion = data.traffic_info ? `路况${data.traffic_info.congestion_name}` : '';
    
    if (riskLevel === '低' || riskLevel === 'low') {
      currentTheme.value = themes.low;
      advice.value = `当前区域机器学习模型评估为低风险。${trafficCongestion}，气象条件良好，视野清晰。请保持正常速度行驶，注意常规交通安全。`;
    } else if (riskLevel === '中' || riskLevel === 'medium') {
      currentTheme.value = themes.medium;
      advice.value = `当前处于中等风险区域。${trafficDesc || '识别出潜在风险因素'}。请适当降低车速，保持安全车距，注意观察周边路况。`;
    } else {
      currentTheme.value = themes.high;
      advice.value = `⚠️ 警告：机器学习模型评估当前区域为高风险！${trafficDesc || '综合多项因素判断'}事故概率较高。请大幅减速，开启警示灯，非必要建议暂缓通过该路段！`;
    }

  } catch (error) {
    console.error('预测API调用失败:', error);
    
    // 降级处理：使用简单的基于时间的备用方案
    const fallbackSeed = (Math.abs(lat) * 100 + Math.abs(lon) * 100);
    riskScore.value = 25 + (fallbackSeed % 60); // 25-84
    factors.value.weather = 20 + (fallbackSeed % 50);
    factors.value.road = 15 + ((fallbackSeed + 1) % 40);
    factors.value.traffic = 30 + ((fallbackSeed + 2) % 50);
    
    currentTheme.value = riskScore.value < 50 ? themes.low : 
                         riskScore.value < 75 ? themes.medium : themes.high;
    
    advice.value = '预测服务暂时不可用，当前显示为备用评估结果。建议谨慎驾驶，注意观察路况。';
  }

  isLoading.value = false;

}, { immediate: true, deep: true });
</script>

<style scoped>
/* 自定义滚动条样式，保持界面的轻量感 */
.overflow-y-auto::-webkit-scrollbar {
  width: 6px;
}
.overflow-y-auto::-webkit-scrollbar-track {
  background: transparent;
}
.overflow-y-auto::-webkit-scrollbar-thumb {
  background-color: #d1d5db;
  border-radius: 20px;
  border: 3px solid transparent;
}
</style>
