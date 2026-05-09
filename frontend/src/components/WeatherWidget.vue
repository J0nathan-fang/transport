<!-- src/components/WeatherWidget.vue -->
<template>
  <div class="weather-widget-container">
    <div v-if="!hasBeenClicked" class="loading-state">
      <p>点击地图以获取天气信息...</p>
    </div>
    <div v-else-if="loading" class="loading-state">
      <p>正在加载天气数据...</p>
    </div>
    <div v-else-if="error" class="error-state">
      <p>{{ error }}</p>
    </div>
    <div v-else-if="weatherData" class="weather-card">
      <!-- Header Section -->
      <header class="weather-header">
        <div class="location-info">
          <span class="city-name">{{ weatherData.locationName }}</span>
          <h1 class="temperature">{{ weatherData.temp }}°C</h1>
        </div>
        <div class="weather-icon-main">
          <span>{{ getWeatherEmoji(weatherData.weatherText) }}</span>
        </div>
      </header>

      <!-- Middle Section: Overview -->
      <div class="overview-card">
        <div class="overview-item">
          <span class="overview-label">当前天气</span>
          <span class="overview-value">{{ weatherData.weatherText }}</span>
        </div>
        <div class="overview-item">
          <span class="overview-label">体感温度</span>
          <span class="overview-value">{{ weatherData.feelsLike }}°C</span>
        </div>
      </div>

      <!-- Bottom Section: Details Grid -->
      <div class="details-grid">
        <div class="detail-card">
          <div class="detail-info">
            <span class="detail-label">湿度</span>
            <span class="detail-value">{{ weatherData.humidity }}%</span>
          </div>
          <div class="detail-icon">💧</div>
        </div>
        <div class="detail-card">
          <div class="detail-info">
            <span class="detail-label">能见度</span>
            <span class="detail-value">{{ weatherData.vis }} km</span>
          </div>
          <div class="detail-icon">👀</div>
        </div>
        <div class="detail-card">
          <div class="detail-info">
            <span class="detail-label">风速</span>
            <span class="detail-value">{{ weatherData.windSpeed }} km/h</span>
          </div>
          <div class="detail-icon">💨</div>
        </div>
        <div class="detail-card">
          <div class="detail-info">
            <span class="detail-label">风向</span>
            <span class="detail-value">{{ weatherData.windDir }}</span>
          </div>
          <div class="detail-icon">🧭</div>
        </div>
      </div>
    </div>
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

const weatherData = ref(null);
const loading = ref(false);
const error = ref('');
const hasBeenClicked = ref(false);

const getWeatherEmoji = (weatherText) => {
  if (!weatherText) return '🌍';
  if (weatherText.includes('晴')) return '☀️';
  if (weatherText.includes('云') || weatherText.includes('阴')) return '☁️';
  if (weatherText.includes('雨')) return '🌧️';
  if (weatherText.includes('雪')) return '❄️';
  if (weatherText.includes('雷')) return '⚡';
  if (weatherText.includes('风')) return '🌬️';
  if (weatherText.includes('雾')) return '🌫️';
  return '🌍';
};

watch(() => props.coords, async (newCoords) => {
  console.log('WeatherWidget: Coords prop changed!', newCoords);
  // The key fix is here: check for `lon`, not `lng`
  if (newCoords && newCoords.lat && newCoords.lon) {
    hasBeenClicked.value = true;
    loading.value = true;
    error.value = '';
    weatherData.value = null;

    try {
      const response = await fetch(`/api/weather?lat=${newCoords.lat}&lon=${newCoords.lon}`);
      if (!response.ok) {
        throw new Error(`天气服务响应失败 (Status: ${response.status})`);
      }
      const data = await response.json();
      if (!data) {
        throw new Error('返回的天气数据为空');
      }
      weatherData.value = data;
    } catch (e) {
      error.value = '无法获取天气数据，请稍后重试。';
      console.error('[WeatherWidget Fetch Error]', e);
    } finally {
      loading.value = false;
    }
  }
}, { deep: true });
</script>

<style scoped>
.weather-widget-container {
  background-color: #F4F6F9;
  padding: 20px;
  border-radius: 16px;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
  min-height: 380px; /* Ensure a minimum height */
  display: flex;
  flex-direction: column;
  justify-content: center;
}

.loading-state,
.error-state {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100%;
  color: #666;
  font-size: 16px;
  text-align: center;
}

.weather-card {
  display: flex;
  flex-direction: column;
  width: 100%;
}

/* Header */
.weather-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
}

.city-name {
  color: #666;
  font-size: 16px;
  margin-bottom: 4px;
}

.temperature {
  color: #2196f3;
  font-size: 48px;
  font-weight: 600;
  margin: 0;
  line-height: 1;
}

.weather-icon-main span {
  font-size: 80px;
}

/* Overview Card */
.overview-card {
  background-color: #fff;
  border-radius: 12px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
  display: flex;
  padding: 20px;
  margin-bottom: 12px;
}

.overview-item {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  padding: 0 16px;
}

.overview-item:first-child {
  border-right: 1px solid #EFEFEF;
  padding-left: 0;
}
.overview-item:last-child {
  padding-right: 0;
}


.overview-label {
  color: #999;
  font-size: 13px;
}

.overview-value {
  color: #333;
  font-size: 18px;
  margin-top: 8px;
  font-weight: 500;
}

/* Details Grid */
.details-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 12px;
}

.detail-card {
  background-color: #fff;
  border-radius: 12px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
  padding: 16px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.detail-info {
  display: flex;
  flex-direction: column;
}

.detail-label {
  color: #999;
  font-size: 14px;
}

.detail-value {
  color: #2196f3;
  font-size: 18px;
  font-weight: 600;
  margin-top: 4px;
}

.detail-icon {
  font-size: 32px;
  opacity: 0.8;
}
</style>
