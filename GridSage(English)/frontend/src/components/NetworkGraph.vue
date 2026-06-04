<template>
  <div class="network-container">
    <!-- Top panel: scenario parameters, global controls, and node modifications -->
    <div class="top-panel">
      <div class="state-header">
        <h3>{{ $t('current_scenario') }}</h3>
        <p class="desc" v-if="displayInfo.model">
          {{ displayInfo.model }} · {{ displayInfo.timeRange }} · {{ displayInfo.step }} min
        </p>
        <p class="desc" v-if="currentState">
          {{ $t('algo_label') }}：{{ currentState.algo_name }}
          · {{ $t('mode_label') }}：{{ modeText }}
        </p>
      </div>

      <!-- Device Summary -->
      <div v-if="snapshot" class="summary-bar">
        <span class="sum-chip">🔌 {{ snapshot.device_summary.bus_count }} {{ $t('chip_nodes') }}</span>
        <span class="sum-chip">🔗 {{ snapshot.device_summary.line_count }} {{ $t('chip_lines') }}</span>
        <span class="sum-chip" v-if="snapshot.device_summary.pv_count">☀️ {{ snapshot.device_summary.pv_count }} PV</span>
        <span class="sum-chip" v-if="snapshot.device_summary.wind_count">🌬️ {{ snapshot.device_summary.wind_count }} Wind</span>
        <span class="sum-chip" v-if="snapshot.device_summary.ess_count">🔋 {{ snapshot.device_summary.ess_count }} ESS</span>
        <span class="sum-chip" v-if="snapshot.device_summary.ev_station_count">🚗 {{ snapshot.device_summary.ev_station_count }} EV</span>
        <span class="sum-chip" v-if="snapshot.device_summary.sop_count">⚡ {{ snapshot.device_summary.sop_count }} SOP</span>
        <span class="sum-chip" v-if="snapshot.device_summary.nop_count">🔀 {{ snapshot.device_summary.nop_count }} NOP</span>
      </div>

      <div v-if="evStationRows.length" class="ev-section">
        <h4>EV Charging Stations</h4>
        <div class="ev-chip-row">
          <span v-for="station in evStationRows" :key="`${station.node}-${station.name}`" class="ev-chip">
            <strong>{{ station.name }}</strong>
            <span>{{ station.node }}</span>
            <small>{{ station.description }}</small>
          </span>
        </div>
      </div>

      <!-- Global Control -->
      <div v-if="displayInfo.model" class="ctrl-section">
        <h4>{{ $t('global_control') }}</h4>
        <div class="ctrl-grid">
          <span>{{ $t('pv_multiplier') }}</span><strong>×{{ fmt(displayInfo.pvMul) }}</strong>
          <span>{{ $t('load_multiplier') }}</span><strong>×{{ fmt(displayInfo.loadMul) }}</strong>
          <span>{{ $t('ev_multiplier') }}</span><strong>×{{ fmt(displayInfo.evMul) }}</strong>
          <span>PV</span><strong :class="displayInfo.usePv?'on':'off'">{{ displayInfo.usePv ? $t('switch_on') : $t('switch_off') }}</strong>
          <span>Wind</span><strong :class="displayInfo.useWind?'on':'off'">{{ displayInfo.useWind ? $t('switch_on') : $t('switch_off') }}</strong>
          <span>ESS</span><strong :class="displayInfo.useEss?'on':'off'">{{ displayInfo.useEss ? $t('switch_on') : $t('switch_off') }}</strong>
          <span>SOP</span><strong :class="displayInfo.useSop?'on':'off'">{{ displayInfo.useSop ? $t('switch_on') : $t('switch_off') }}</strong>
          <span>NOP</span><strong :class="displayInfo.useNop?'on':'off'">{{ displayInfo.useNop ? 'Reconfig' : $t('switch_off') }}</strong>
        </div>
        <div v-if="selectedPlan" class="reconfig-box">
          <strong>{{ selectedPlan.plan_id }}</strong>
          <span>{{ selectedPlan.description_cn || selectedPlan.plan_name }}</span>
        </div>
        <div v-if="hasTimeProfiles" class="tp-hint">{{ $t('time_profile_hint') }}</div>
      </div>

      <!-- Node modification table -->
      <div class="mod-section">
        <h4>{{ $t('mod_table_title') }}</h4>
        <el-table v-if="modRows.length" :data="modRows" size="small" max-height="180" border stripe>
          <el-table-column prop="node" :label="$t('mod_col_node')" width="80" align="center" />
          <el-table-column prop="device_name" :label="$t('mod_col_device')" width="90" align="center" />
          <el-table-column prop="change_type" :label="$t('mod_col_type')" width="100" align="center" />
          <el-table-column prop="change_value" :label="$t('mod_col_value')" align="center" />
        </el-table>
        <p v-else class="empty-hint">{{ $t('mod_empty') }}</p>
      </div>
    </div>

    <!-- Bottom panel: topology graph -->
    <div ref="echartsContainer" class="echarts-graph"></div>
  </div>
</template>

<script setup>
import { ref, watch, onMounted, computed } from 'vue'
import * as echarts from 'echarts'
import axios from 'axios'
import { useI18n } from 'vue-i18n'

const { t } = useI18n()
const props = defineProps(['currentState', 'sessionId'])
const echartsContainer = ref(null)
let chartInstance = null
const snapshot = ref(null)
const snapshotFailed = ref(false)
const api = axios.create({ baseURL: '/api' })

const fmt = (v) => Number(v || 0).toFixed(2)

// ---- Algorithm / Mode display ----
const modeText = computed(() => {
  const st = props.currentState
  if (!st) return ''
  const algo = (st.algo_name || '').toLowerCase()
  const exec = (st.execution_mode || 'evaluate').toLowerCase()
  if (algo === 'baseline') return t('mode_baseline')
  return exec === 'train' ? t('mode_train') : t('mode_evaluate')
})

// ---- Unified display info: prefer snapshot, fall back to currentState ----
const displayInfo = computed(() => {
  const s = snapshot.value
  const st = props.currentState
  if (s) {
    return {
      model: s.grid_model?.toUpperCase() || '',
      timeRange: `${s.global_controls.start_hour}:00 – ${s.global_controls.end_hour}:00`,
      step: s.global_controls.step_minutes,
      pvMul: s.global_controls.pv_multiplier,
      loadMul: s.global_controls.load_multiplier,
      evMul: s.global_controls.ev_multiplier,
      usePv: s.global_controls.use_pv,
      useWind: s.global_controls.use_wind,
      useEss: s.global_controls.use_ess,
      useSop: s.global_controls.use_sop,
      useNop: s.global_controls.use_nop,
    }
  }
  if (st) {
    return {
      model: (st.grid_model || '').toUpperCase(),
      timeRange: `${st.start_hour}:00 – ${st.end_hour}:00`,
      step: st.step_minutes,
      pvMul: st.global_pv_multiplier,
      loadMul: st.global_load_multiplier,
      evMul: st.global_ev_multiplier,
      usePv: st.use_pv,
      useWind: st.use_wind,
      useEss: st.use_ess,
      useSop: st.use_sop,
      useNop: st.use_nop,
    }
  }
  return {}
})

const hasTimeProfiles = computed(() => {
  const tp = snapshot.value?.global_controls?.time_profiles
    || props.currentState?.time_profiles || {}
  return !!(tp.load_multiplier_by_hour || tp.pv_multiplier_by_hour || tp.ev_multiplier_by_hour)
})

const selectedPlan = computed(() => {
  const plan = snapshot.value?.selected_reconfiguration_plan
  return plan && plan.plan_id ? plan : null
})

// ---- Modification rows: prefer snapshot, fall back to currentState ----
const modRows = computed(() => {
  if (snapshot.value?.node_modification_rows?.length) {
    return snapshot.value.node_modification_rows
  }
  const overrides = props.currentState?.node_overrides || {}
  const rows = []
  for (const [node, vals] of Object.entries(overrides)) {
    if (vals.add_load_kw) rows.push({ node, device_name: 'Load', change_type: '+Load', change_value: `+${vals.add_load_kw} kW` })
    if (vals.add_pv_kw) rows.push({ node, device_name: 'PV', change_type: '+PV', change_value: `+${vals.add_pv_kw} kW` })
    if (vals.add_ess_kwh) {
      const essPower = vals.add_ess_power_kw
        ? `, ${vals.add_ess_power_kw} kW`
        : (vals.add_ess_c_rate ? `, ${vals.add_ess_c_rate}C` : '')
      rows.push({ node, device_name: 'ESS', change_type: '+ESS', change_value: `+${vals.add_ess_kwh} kWh${essPower}` })
    }
    if (vals.add_ev_spots) rows.push({ node, device_name: 'EV', change_type: '+EV', change_value: `+${Math.round(vals.add_ev_spots)}` })
    if (vals.add_wind_kw) rows.push({ node, device_name: 'Wind', change_type: '+Wind', change_value: `+${vals.add_wind_kw} kW` })
  }
  return rows
})

const evStationRows = computed(() => {
  const rows = []
  for (const node of snapshot.value?.nodes || []) {
    for (const device of node.base_devices || []) {
      if (device.device_type === 'ev_station') {
        rows.push({
          node: node.id,
          name: device.device_name,
          description: device.description || ''
        })
      }
    }
  }
  return rows
})

const fetchSnapshot = async () => {
  const sid = props.sessionId
  if (!sid) return
  try {
    const res = await api.get(`/grid_snapshot/${sid}`)
    snapshot.value = res.data
    snapshotFailed.value = false
    renderChartFromSnapshot()
  } catch (e) {
    console.warn('grid_snapshot fetch failed, using fallback', e)
    snapshotFailed.value = true
    renderChartFromState()
  }
}

// ---- IEEE-33 fixed coordinates ----
const COORDS_33 = {
  1:[0,0],2:[60,0],3:[120,0],4:[180,0],5:[240,0],6:[300,0],7:[360,0],8:[420,0],
  9:[480,0],10:[540,0],11:[600,0],12:[660,0],13:[720,0],14:[780,0],15:[840,0],
  16:[900,0],17:[960,0],18:[1020,0],
  19:[60,60],20:[60,120],21:[60,180],22:[60,240],
  23:[120,60],24:[120,120],25:[120,180],
  26:[300,60],27:[360,60],28:[420,60],29:[480,60],30:[540,60],31:[600,60],32:[660,60],33:[720,60]
}

const COORDS_69 = {
  1:[0,0],2:[48,0],3:[96,0],4:[144,0],5:[192,0],6:[240,0],7:[288,0],8:[336,0],
  9:[384,0],10:[432,0],11:[480,0],12:[528,0],13:[576,0],14:[624,0],15:[672,0],
  16:[720,0],17:[768,0],18:[816,0],19:[864,0],20:[912,0],21:[960,0],22:[1008,0],
  23:[1056,0],24:[1104,0],25:[1152,0],26:[1200,0],27:[1248,0],
  28:[96,50],29:[96,100],30:[96,150],31:[96,200],32:[96,250],33:[96,300],34:[96,350],35:[96,400],
  36:[144,50],37:[144,100],38:[144,150],39:[144,200],40:[144,250],
  41:[336,50],42:[336,100],43:[336,150],44:[336,200],45:[336,250],46:[336,300],47:[336,350],48:[336,400],
  49:[384,50],50:[384,100],
  51:[480,50],52:[480,100],53:[480,150],
  54:[528,50],55:[528,100],56:[528,150],
  57:[1248,50],58:[1248,100],59:[1248,150],60:[1248,200],61:[1248,250],62:[1248,300],63:[1248,350],
  64:[1248,400],65:[1248,450],66:[1248,500],67:[1248,550],68:[1248,600],69:[1248,650]
}

const EDGES_33 = [
  [1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[8,9],[9,10],[10,11],[11,12],[12,13],
  [13,14],[14,15],[15,16],[16,17],[17,18],
  [2,19],[19,20],[20,21],[21,22],
  [3,23],[23,24],[24,25],
  [6,26],[26,27],[27,28],[28,29],[29,30],[30,31],[31,32],[32,33]
]

const EDGES_69 = [
  [1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[8,9],
  [9,10],[10,11],[11,12],[12,13],[13,14],[14,15],[15,16],
  [16,17],[17,18],[18,19],[19,20],[20,21],[21,22],[22,23],
  [23,24],[24,25],[25,26],[26,27],
  [3,28],[28,29],[29,30],[30,31],[31,32],[32,33],[33,34],[34,35],
  [4,36],[36,37],[37,38],[38,39],[39,40],
  [8,41],[41,42],[42,43],[43,44],[44,45],[45,46],[46,47],[47,48],
  [9,49],[49,50],
  [11,51],[51,52],[52,53],
  [12,54],[54,55],[55,56],
  [27,57],[57,58],[58,59],[59,60],[60,61],[61,62],[62,63],
  [63,64],[64,65],[65,66],[66,67],[67,68],[68,69]
]

function getStaticEdges(model) {
  if (model === 'ieee33') return { edges: EDGES_33, n: 33 }
  if (model === 'ieee69') return { edges: EDGES_69, n: 69 }
  const n = 123
  const e = []; for (let i=1;i<n;i++) e.push([i,i+1])
  return { edges: e, n }
}

function getNodeCoords(model, i) {
  const m = (model || '').toLowerCase()
  if (m === 'ieee33' && COORDS_33[i]) return COORDS_33[i]
  if (m === 'ieee69' && COORDS_69[i]) return COORDS_69[i]
  return [(i % 15) * 60, Math.floor(i / 15) * 60]
}

function buildNodeDeviceMap() {
  const map = {}
  if (!snapshot.value) return map
  for (const node of snapshot.value.nodes) map[node.id] = node
  return map
}

function getDeviceTags(nodeData) {
  if (!nodeData) return []
  const tags = []
  const types = new Set()
  for (const d of nodeData.base_devices || []) types.add(d.device_type)
  if (types.has('pv')) tags.push('PV')
  if (types.has('wind')) tags.push('Wind')
  if (types.has('ess')) tags.push('ESS')
  if (types.has('ev_station')) tags.push('EV')
  if (types.has('sop')) tags.push('SOP')
  if (types.has('nop')) tags.push('NOP')
  if (types.has('generator')) tags.push('Gen')
  return tags
}

function buildTooltipFromSnapshot(nodeData, globalControls) {
  if (!nodeData) return ''
  let html = `<b style="font-size:13px">${t('tooltip_node')}：${nodeData.id}</b><br/><br/>`
  html += `<b>${t('tooltip_base_devices')}：</b><br/>`
  const base = (nodeData.base_devices || []).filter(d => d.device_type !== 'bus')
  if (!base.length) { html += `- ${t('tooltip_none')}<br/>` }
  else { for (const d of base) { html += `- ${d.device_name}${d.description ? '：'+d.description : ''}<br/>` } }
  html += `<br/><b>${t('tooltip_modifications')}：</b><br/>`
  let hasGlobal = false
  if (globalControls) {
    if (globalControls.load_multiplier !== 1.0) { html += `- ${t('tooltip_global_load', { value: globalControls.load_multiplier.toFixed(2) })}<br/>`; hasGlobal = true }
    if (globalControls.pv_multiplier !== 1.0) { html += `- ${t('tooltip_global_pv', { value: globalControls.pv_multiplier.toFixed(2) })}<br/>`; hasGlobal = true }
    if (globalControls.ev_multiplier !== 1.0) { html += `- ${t('tooltip_global_ev', { value: globalControls.ev_multiplier.toFixed(2) })}<br/>`; hasGlobal = true }
  }
  const mods = nodeData.modifications || []
  if (!mods.length && !hasGlobal) { html += `- ${t('tooltip_none')}<br/>` }
  else { for (const m of mods) { html += `- ${m.change_type}：${m.change_value}<br/>` } }
  return html
}

function buildTooltipFallback(bid, overrides) {
  const ov = overrides[bid]
  let html = `<b style="font-size:13px">${t('tooltip_node')}：${bid}</b><br/><br/>`
  html += `<b>${t('tooltip_base_devices')}：</b><br/>- ${t('tooltip_none')}<br/>`
  html += `<br/><b>${t('tooltip_modifications')}：</b><br/>`
  if (!ov || !Object.keys(ov).length) { html += `- ${t('tooltip_none')}<br/>` }
  else { html += `<pre style="margin:0;font-size:11px">${JSON.stringify(ov, null, 2)}</pre>` }
  return html
}

// ---- Render with full snapshot data ----
function renderChartFromSnapshot() {
  if (!chartInstance || !snapshot.value) return
  const data = snapshot.value
  const nodeMap = buildNodeDeviceMap()
  const numNodes = data.device_summary.bus_count
  const overrides = props.currentState?.node_overrides || {}

  const chartNodes = []
  for (let i = 1; i <= numNodes; i++) {
    const bid = `b${i}`
    const nd = nodeMap[bid]
    const isModified = !!(overrides[bid] && Object.keys(overrides[bid]).length)
    const tags = getDeviceTags(nd)
    const hasDevice = tags.length > 0
    const [xPos, yPos] = getNodeCoords(data.grid_model, i)
    let label = String(i)
    if (tags.length) label += '\n' + tags.join(' ')
    if (isModified) label += '\n★'
    chartNodes.push({
      id: bid, name: label, x: xPos, y: yPos,
      symbolSize: isModified ? 38 : (hasDevice ? 30 : 22),
      itemStyle: {
        color: isModified ? '#ef4444' : (hasDevice ? '#8b5cf6' : '#3b82f6'),
        shadowBlur: isModified ? 12 : 0, shadowColor: 'rgba(239,68,68,0.45)',
        borderColor: isModified ? '#fca5a5' : '#fff', borderWidth: isModified ? 3 : 1.5,
      },
      _tooltip: buildTooltipFromSnapshot(nd, data.global_controls),
    })
  }
  const links = (data.edges || []).map(e => {
    const link = { source: e.source, target: e.target }
    if (e.edge_type === 'nop_line') {
      link.lineStyle = { type: 'solid', color: '#16a34a', width: 3 }
    } else if (e.is_opened || e.is_active === false) {
      link.lineStyle = { type: 'dashed', color: '#dc2626', width: 2, opacity: 0.75 }
    }
    return link
  })
  for (const node of data.nodes) {
    for (const d of node.base_devices || []) {
      if (d.device_type === 'sop') {
        const match = (d.description || '').match(/connects (b\d+)\s*-\s*(b\d+)/i)
        if (match && node.id === match[1]) {
          links.push({ source: match[1], target: match[2],
            lineStyle: { type: 'dashed', color: '#f59e0b', width: 2 } })
        }
      }
    }
  }
  applyChartOption(chartNodes, links, data.grid_model, true)
}

// ---- Render fallback from currentState ----
function renderChartFromState() {
  if (!chartInstance || !props.currentState) return
  const st = props.currentState
  const { edges, n: numNodes } = getStaticEdges(st.grid_model)
  const overrides = st.node_overrides || {}

  const chartNodes = []
  for (let i = 1; i <= numNodes; i++) {
    const bid = `b${i}`
    const isModified = !!(overrides[bid] && Object.keys(overrides[bid]).length)
    const [xPos, yPos] = getNodeCoords(st.grid_model, i)
    chartNodes.push({
      id: bid, name: isModified ? `${i}\n★` : String(i), x: xPos, y: yPos,
      symbolSize: isModified ? 35 : 20,
      itemStyle: {
        color: isModified ? '#ef4444' : '#3b82f6',
        shadowBlur: isModified ? 10 : 0, shadowColor: 'rgba(239,68,68,0.45)',
        borderColor: isModified ? '#fca5a5' : '#fff', borderWidth: isModified ? 3 : 1,
      },
      _tooltip: buildTooltipFallback(bid, overrides),
    })
  }
  const links = edges.map(e => ({ source: `b${e[0]}`, target: `b${e[1]}` }))
  applyChartOption(chartNodes, links, st.grid_model, false)
}

function applyChartOption(chartNodes, links, gridModel, hasSnapshot) {
  const modelUpper = (gridModel || 'ieee33').toUpperCase()
  const option = {
    title: {
      text: t('topo_title', { model: modelUpper }),
      subtext: hasSnapshot ? t('topo_sub_full') : t('topo_sub_fallback'),
      left: 'center',
      textStyle: { fontSize: 14, color: '#1f2937' },
      subtextStyle: { fontSize: 11, color: '#9ca3af' },
    },
    tooltip: {
      trigger: 'item', enterable: true, confine: true,
      backgroundColor: 'rgba(255,255,255,0.96)', borderColor: '#e5e7eb',
      textStyle: { color: '#374151', fontSize: 12 },
      formatter: (params) => params.data?._tooltip || '',
    },
    series: [{
      type: 'graph', layout: 'none',
      data: chartNodes, links: links, roam: true,
      edgeSymbol: ['none', 'none'],
      label: { show: true, position: 'bottom', color: '#374151', fontSize: 9, lineHeight: 12, formatter: (p) => p.name },
      lineStyle: { color: '#94a3b8', width: 2, curveness: 0 },
    }],
  }
  chartInstance.setOption(option, true)
}

onMounted(() => {
  if (echartsContainer.value) {
    chartInstance = echarts.init(echartsContainer.value)
    window.addEventListener('resize', () => chartInstance?.resize())
    if (snapshot.value) renderChartFromSnapshot()
    else if (props.currentState) renderChartFromState()
  }
})

watch(() => props.sessionId, (sid) => {
  if (sid) fetchSnapshot()
}, { immediate: true })

watch(() => props.currentState, () => {
  if (props.sessionId) {
    fetchSnapshot()
  } else if (props.currentState) {
    renderChartFromState()
  }
}, { deep: true })

watch(snapshot, (val) => {
  if (val) renderChartFromSnapshot()
}, { deep: true })
</script>

<style scoped>
.network-container {
  padding: 16px; height: 100%;
  display: flex; flex-direction: column; gap: 12px; overflow-y: auto;
}
.top-panel { flex-shrink: 0; }
.state-header h3 { margin: 0 0 4px; color: #1f2937; font-size: 15px; }
.desc { font-size: 12px; color: #6b7280; margin: 2px 0; }

.summary-bar { display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0; }
.sum-chip {
  background: #f0f4ff; color: #3b5bdb; border-radius: 12px;
  padding: 2px 10px; font-size: 11px; white-space: nowrap;
}
.ev-section {
  background: #fff; border: 1px solid #dbeafe; border-radius: 8px;
  padding: 8px 10px; margin-top: 8px;
}
.ev-section h4 { margin: 0 0 6px; font-size: 13px; color: #1d4ed8; }
.ev-chip-row { display: flex; flex-wrap: wrap; gap: 6px; }
.ev-chip {
  display: inline-flex; align-items: center; gap: 6px; max-width: 100%;
  background: #eff6ff; color: #1e40af; border: 1px solid #bfdbfe;
  border-radius: 6px; padding: 4px 8px; font-size: 11px;
}
.ev-chip strong { color: #1e3a8a; }
.ev-chip small { color: #475569; overflow-wrap: anywhere; }

.ctrl-section, .mod-section {
  background: #fff; border: 1px solid #e5e7eb; border-radius: 8px;
  padding: 10px 12px; margin-top: 10px;
}
.ctrl-section h4, .mod-section h4 { margin: 0 0 8px; font-size: 13px; color: #374151; }
.ctrl-grid {
  display: grid; grid-template-columns: 110px 1fr; gap: 4px 10px; font-size: 12px;
}
.ctrl-grid span { color: #6b7280; }
.ctrl-grid strong { color: #1f2937; }
.ctrl-grid .on { color: #16a34a; }
.ctrl-grid .off { color: #dc2626; }
.reconfig-box {
  margin-top: 8px; display: grid; grid-template-columns: 42px 1fr; gap: 8px;
  align-items: start; font-size: 12px; color: #374151;
  border-top: 1px solid #eef2f7; padding-top: 8px;
}
.reconfig-box strong { color: #166534; }
.reconfig-box span { overflow-wrap: anywhere; line-height: 1.4; }
.tp-hint { margin-top: 6px; font-size: 11px; color: #d97706; }
.empty-hint { font-size: 12px; color: #9ca3af; margin: 4px 0; }

.echarts-graph {
  flex: 1; min-height: 320px;
  background: #fff; border-radius: 8px; border: 1px solid #e5e7eb;
}
</style>
