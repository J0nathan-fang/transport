 <template>
  <div class="flex h-screen bg-gray-100 font-sans">
    <!-- Sidebar -->
    <aside class="w-64 bg-gray-800 text-white flex flex-col flex-shrink-0">
      <div class="p-4 border-b border-gray-700">
        <h1 class="text-2xl font-bold">Dashboard</h1>
      </div>
      <nav class="flex-1 p-4 space-y-2">
        <a 
          href="#" 
          @click.prevent="setActiveMenu('weather')"
          class="block px-4 py-2 rounded"
          :class="activeMenu === 'weather' ? 'bg-gray-700' : 'hover:bg-gray-700'"
        >
          Weather Map
        </a>
        <a 
          href="#" 
          @click.prevent="setActiveMenu('analytics')"
          class="block px-4 py-2 rounded"
          :class="activeMenu === 'analytics' ? 'bg-gray-700' : 'hover:bg-gray-700'"
        >
          Analytics
        </a>
        <a href="#" class="block px-4 py-2 rounded hover:bg-gray-700">Settings</a>
      </nav>
    </aside>

    <!-- Main Content -->
    <main class="flex-1 p-8 flex flex-col overflow-hidden">
      <h2 class="text-3xl font-bold mb-8 flex-shrink-0">
        {{ activeMenu === 'weather' ? 'Weather Map' : 'Risk Analysis' }}
      </h2>
      
      <div class="flex-1 flex gap-8 min-h-0">
        
        <!-- Map Container -->
        <div class="flex-1 bg-white p-6 rounded-lg shadow flex flex-col min-h-0">
          <h3 class="text-xl font-semibold mb-4 flex-shrink-0">Live Map</h3>
          <div class="flex-1 min-h-0 flex flex-col">
            <MapWidget @map-click="handleMapClick" class="w-full flex-1" />
          </div>
        </div>

        <!-- Right Panel: Weather or Analytics -->
        <div class="w-96 bg-white p-6 rounded-lg shadow flex-shrink-0">
          <WeatherWidget v-if="activeMenu === 'weather'" :coords="clickedCoords" />
          <AnalyticsWidget v-else :coords="clickedCoords" />
        </div>

      </div>
    </main>
  </div>
</template>

<script setup>
import { ref } from 'vue';
import MapWidget from './components/MapWidget.vue';
import WeatherWidget from './components/WeatherWidget.vue';
import AnalyticsWidget from './components/AnalyticsWidget.vue';

const clickedCoords = ref(null);
const activeMenu = ref('weather');

const handleMapClick = (coords) => {
  console.log('App.vue: Map clicked, new coords:', coords);
  clickedCoords.value = coords;
};

const setActiveMenu = (menu) => {
  console.log('Menu switched to:', menu);
  activeMenu.value = menu;
};
</script>

<style>
.leaflet-container {
  width: 100%;
  height: 100%;
  border-radius: 0.5rem;
}
</style>