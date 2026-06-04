<template>
  <div class="chat-view">
    <div class="main-split">
      <section class="chat-section">
        <div ref="messagesContainer" class="messages-container">
          <div v-for="(msg, index) in messages" :key="index" :class="['message', msg.role]">
            <div class="avatar">{{ msg.role === 'user' ? 'U' : 'AI' }}</div>
            <div class="content">
              <div class="text" v-html="formatMessage(msg.content)"></div>

              <div v-if="msg.matchedSkills?.length" class="info-panel">
                <div class="panel-title">Scenario Skill Matches</div>
                <div v-for="skill in orderedSkills(msg.matchedSkills)" :key="skill.skill_id" class="skill-card">
                  <div class="skill-card-head">
                    <strong>{{ skill.name_cn || skill.name_en || skill.skill_id }}</strong>
                    <span :class="['role-chip', skill.role]">{{ skill.role }}</span>
                  </div>
                  <div class="skill-meta">
                    <span>{{ skill.skill_id }}</span>
                    <span>{{ skill.category || 'uncategorized' }}</span>
                    <span>score {{ formatNumber(skill.score, 1) }}</span>
                  </div>
                  <p v-if="skill.description_cn">{{ skill.description_cn }}</p>
                </div>
              </div>

              <div v-if="msg.skillWarnings?.length" class="notice-panel warning">
                <div class="panel-title">Skill Warnings</div>
                <div v-for="(warning, idx) in msg.skillWarnings" :key="idx" class="notice-line">
                  {{ warning }}
                </div>
              </div>

              <div v-if="msg.validationMessages?.length" class="info-panel">
                <div class="panel-title">Validation Results</div>
                <div
                  v-for="(item, idx) in msg.validationMessages"
                  :key="idx"
                  :class="['validation-item', item.level]"
                >
                  <strong>{{ item.level }}</strong>
                  <span>{{ item.message }}</span>
                  <small v-if="item.skill_id || item.rule_id">
                    {{ [item.skill_id, item.rule_id].filter(Boolean).join(' / ') }}
                  </small>
                </div>
                <div v-if="hasReject(msg.validationMessages)" class="reject-note">
                  This configuration was rejected. The current state remains unchanged.
                </div>
              </div>

              <div v-if="msg.proposedStateSummary && Object.keys(msg.proposedStateSummary).length" class="info-panel">
                <div class="panel-title">Change Summary</div>
                <div class="summary-grid">
                  <span>Scenario</span>
                  <strong>{{ msg.proposedStateSummary.scenario_name || '-' }}</strong>
                  <span>Global Fields</span>
                  <strong>{{ (msg.proposedStateSummary.changed_fields || []).join(', ') || '-' }}</strong>
                  <span>Node Changes</span>
                  <strong>{{ Object.keys(msg.proposedStateSummary.node_changes || {}).join(', ') || '-' }}</strong>
                  <span>Disabled Devices</span>
                  <strong>{{ Object.keys(msg.proposedStateSummary.disabled_device_changes || {}).join(', ') || '-' }}</strong>
                </div>
              </div>

              <el-collapse v-if="msg.commands?.length" class="mt-2">
                <el-collapse-item title="Delta Commands">
                  <pre class="json-pre">{{ JSON.stringify(msg.commands, null, 2) }}</pre>
                </el-collapse-item>
              </el-collapse>
            </div>
          </div>

          <div v-if="isWaiting" class="message assistant">
            <div class="avatar">AI</div>
            <div class="content"><div class="text">{{ $t('waiting') }}</div></div>
          </div>

          <div v-if="isTaskRunning && liveLogs.length" class="message assistant">
            <div class="avatar sys">SYS</div>
            <div class="content">
              <div class="log-panel">
                <div v-for="(logLine, idx) in liveLogs" :key="idx">{{ logLine }}</div>
              </div>
            </div>
          </div>
        </div>

        <div class="input-area">
          <el-input
            v-model="inputMessage"
            type="textarea"
            :rows="3"
            :placeholder="isTaskRunning ? $t('task_running') : $t('placeholder_input')"
            :disabled="isTaskRunning"
            @keyup.enter.exact="sendMessage"
          />
          <div class="action-buttons">
            <el-button @click="openSkillsLibrary">Scenario Skill Library</el-button>
            <el-button type="success" :disabled="isTaskRunning || isWaiting" @click="runSimulation">
              {{ $t('run_simulation') }}
            </el-button>
            <el-button type="primary" :loading="isWaiting" :disabled="isTaskRunning || !inputMessage.trim()" @click="sendMessage">
              {{ $t('send') }}
            </el-button>
          </div>
        </div>
      </section>

      <aside class="state-section">
        <NetworkGraph :current-state="currentState" :session-id="props.sessionId" />
        <div v-if="currentState" class="state-panel">
          <div class="panel-title">Scenario / RL Configuration</div>
          <div class="state-grid">
            <span>Scenario Name</span><strong>{{ currentState.scenario_name || 'Unnamed Scenario' }}</strong>
            <span>Algorithm</span><strong>{{ currentState.algo_name }}</strong>
            <span>Execution Mode</span><strong>{{ currentState.execution_mode }}</strong>
          </div>
          <p v-if="currentState.scenario_description" class="scenario-desc">{{ currentState.scenario_description }}</p>
          <div v-if="currentState.active_skill_ids?.length" class="chip-list">
            <span v-for="skillId in currentState.active_skill_ids" :key="skillId" class="state-skill-chip">{{ skillId }}</span>
          </div>
          <div v-if="currentState.validation_warnings?.length" class="notice-panel warning compact">
            <div v-for="(warning, idx) in currentState.validation_warnings" :key="idx" class="notice-line">{{ warning }}</div>
          </div>
          <div v-if="timeProfileRows.length" class="time-profile">
            <div class="panel-title">Time-Varying Multiplier Curves</div>
            <el-table :data="timeProfileRows" size="small" max-height="200" border>
              <el-table-column prop="hour" label="Hour" width="70" />
              <el-table-column prop="load" label="Load Multiplier" />
              <el-table-column prop="pv" label="PV Multiplier" />
              <el-table-column prop="ev" label="EV Multiplier" />
            </el-table>
          </div>
        </div>
      </aside>
    </div>

    <el-dialog v-model="skillsDialogVisible" title="Scenario Skill Library" width="760px">
      <el-table :data="skills" height="360" highlight-current-row @row-click="selectSkill">
        <el-table-column prop="skill_id" label="Skill ID" width="190" />
        <el-table-column prop="name_cn" label="Name" width="170" />
        <el-table-column prop="category" label="Category" width="150" />
        <el-table-column prop="default_algorithm" label="Default Algorithm" width="130" />
        <el-table-column prop="description_cn" label="Description" />
      </el-table>
      <div v-if="selectedSkill" class="skill-detail">
        <h3>{{ selectedSkill.skill_id }} / {{ selectedSkill.name_cn }}</h3>
        <p>{{ selectedSkill.description_cn }}</p>
        <div class="detail-columns">
          <div>
            <h4>Recommended Parameters</h4>
            <pre>{{ JSON.stringify(selectedSkill.recommended_parameters || {}, null, 2) }}</pre>
          </div>
          <div>
            <h4>Recommended Metrics</h4>
            <div class="chip-list">
              <span v-for="metric in selectedSkill.recommended_metrics || []" :key="metric" class="state-skill-chip">{{ metric }}</span>
            </div>
            <h4>Validation Rules</h4>
            <div v-for="rule in selectedSkill.validation_rules || []" :key="rule.rule_id" class="validation-rule">
              <strong>{{ rule.rule_id }}</strong> {{ rule.level }}: {{ rule.message_cn }}
            </div>
          </div>
        </div>
      </div>
    </el-dialog>
  </div>
</template>

<script setup>
import { computed, nextTick, onMounted, ref, watch } from 'vue'
import axios from 'axios'
import { marked } from 'marked'
import NetworkGraph from '../components/NetworkGraph.vue'
import { useI18n } from 'vue-i18n'

const { t } = useI18n()
const props = defineProps(['sessionId'])

const messages = ref([])
const inputMessage = ref('')
const isWaiting = ref(false)
const isTaskRunning = ref(false)
const currentState = ref(null)
const messagesContainer = ref(null)
const liveLogs = ref([])
const skillsDialogVisible = ref(false)
const skills = ref([])
const selectedSkill = ref(null)

const api = axios.create({ baseURL: '/api' })

const scrollToBottom = () => {
  nextTick(() => {
    if (messagesContainer.value) {
      messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight
    }
  })
}

const formatMessage = (text) => marked(text || '')
const formatNumber = (value, digits = 2) => Number(value || 0).toFixed(digits)
const orderedSkills = (items) => [...items].sort((a, b) => (a.role === 'primary' ? -1 : 1) - (b.role === 'primary' ? -1 : 1))
const hasReject = (items) => (items || []).some((item) => item.level === 'reject')

const getHourValue = (source, hour, fallback = '-') => {
  if (!source) return fallback
  if (Array.isArray(source)) return source[hour] ?? fallback
  return source[String(hour)] ?? source[hour] ?? source[String(hour).padStart(2, '0')] ?? fallback
}

const timeProfileRows = computed(() => {
  const profile = currentState.value?.time_profiles || {}
  if (!profile.load_multiplier_by_hour && !profile.pv_multiplier_by_hour && !profile.ev_multiplier_by_hour) return []
  return Array.from({ length: 24 }, (_, hour) => ({
    hour,
    load: getHourValue(profile.load_multiplier_by_hour, hour, currentState.value?.global_load_multiplier ?? '-'),
    pv: getHourValue(profile.pv_multiplier_by_hour, hour, currentState.value?.global_pv_multiplier ?? '-'),
    ev: getHourValue(profile.ev_multiplier_by_hour, hour, currentState.value?.global_ev_multiplier ?? '-')
  }))
})

const fetchState = async () => {
  if (!props.sessionId) return
  const res = await api.get(`/state/${props.sessionId}`)
  currentState.value = res.data
}

watch(() => props.sessionId, (newVal) => {
  messages.value = []
  liveLogs.value = []
  if (newVal) {
    messages.value.push({ role: 'assistant', content: t('system_hello') })
    fetchState()
  }
}, { immediate: true })

const sendMessage = async () => {
  if (!inputMessage.value.trim() || isTaskRunning.value || isWaiting.value) return

  const userText = inputMessage.value
  messages.value.push({ role: 'user', content: userText })
  inputMessage.value = ''
  isWaiting.value = true
  scrollToBottom()

  // Compatibility note: keep the legacy lvgs_model_config key so saved settings remain available.
  const savedConfig = localStorage.getItem('lvgs_model_config')
  const modelConfig = savedConfig ? JSON.parse(savedConfig) : {}

  try {
    const res = await api.post('/chat', {
      session_id: props.sessionId,
      user_input: userText,
      api_config: modelConfig
    })
    await fetchState()
    messages.value.push({
      role: 'assistant',
      content: res.data.thoughts,
      commands: res.data.delta_commands,
      matchedSkills: res.data.matched_skills,
      skillWarnings: res.data.skill_warnings,
      validationMessages: res.data.validation_messages,
      proposedStateSummary: res.data.proposed_state_summary
    })
  } catch (error) {
    messages.value.push({
      role: 'assistant',
      content: `${t('req_failed')}: ${error.message}\n${t('check_api')}`
    })
  } finally {
    isWaiting.value = false
    scrollToBottom()
  }
}

const runSimulation = async () => {
  if (isTaskRunning.value) return
  isTaskRunning.value = true
  liveLogs.value = []
  try {
    await api.post(`/run_task?session_id=${props.sessionId}`)
    messages.value.push({ role: 'assistant', content: t('sim_recv') })
    scrollToBottom()
    pollTaskStatus()
  } catch (error) {
    isTaskRunning.value = false
    messages.value.push({ role: 'assistant', content: `${t('req_failed')}: ${error.message}` })
  }
}

const pollTaskStatus = () => {
  const pollInterval = setInterval(async () => {
    try {
      const res = await api.get(`/task_status/${props.sessionId}`)
      const statusData = res.data
      if (statusData.logs?.length) {
        liveLogs.value = statusData.logs.slice(-15)
        scrollToBottom()
      }
      if (['completed', 'error', 'train_completed'].includes(statusData.status)) {
        clearInterval(pollInterval)
        isTaskRunning.value = false
        const finalMsg = statusData.status === 'error'
          ? t('sim_error', { log: (statusData.logs || []).join('\n') })
          : statusData.result?.message || t('train_success')
        messages.value.push({ role: 'assistant', content: finalMsg })
        scrollToBottom()
      }
    } catch (error) {
      clearInterval(pollInterval)
      isTaskRunning.value = false
    }
  }, 2000)
}

const openSkillsLibrary = async () => {
  skillsDialogVisible.value = true
  if (!skills.value.length) {
    const res = await api.get('/skills')
    skills.value = res.data.skills || []
  }
}

const selectSkill = async (row) => {
  const res = await api.get(`/skills/${row.skill_id}`)
  selectedSkill.value = res.data
}

onMounted(() => {
  if (props.sessionId) fetchState()
})
</script>

<style scoped>
.chat-view { height: 100%; display: flex; flex-direction: column; }
.main-split { display: flex; height: 100%; width: 100%; }
.chat-section { flex: 3; display: flex; flex-direction: column; padding: 20px; border-right: 1px solid #e4e7ed; min-width: 0; }
.messages-container { flex: 1; overflow-y: auto; margin-bottom: 16px; padding-right: 10px; }
.message { display: flex; margin-bottom: 18px; }
.message.user { flex-direction: row-reverse; }
.avatar { width: 40px; height: 40px; border-radius: 50%; background: #2563eb; color: #fff; display: flex; align-items: center; justify-content: center; font-weight: 700; margin: 0 12px; flex-shrink: 0; }
.message.assistant .avatar { background: #16a34a; }
.avatar.sys { background: #374151; }
.content { background: #f5f7fa; padding: 12px 16px; border-radius: 8px; max-width: 86%; font-size: 14px; line-height: 1.55; color: #303133; overflow-wrap: anywhere; }
.message.user .content { background: #ecf5ff; }
.action-buttons { display: flex; justify-content: flex-end; gap: 10px; margin-top: 10px; }
.state-section { flex: 2; background: #fafafa; overflow-y: auto; min-width: 360px; }
.state-panel, .info-panel, .notice-panel { border: 1px solid #dcdfe6; border-radius: 6px; background: #fff; padding: 10px; margin-top: 10px; }
.state-panel { margin: 12px 16px 16px; }
.panel-title { font-size: 13px; font-weight: 700; color: #303133; margin-bottom: 8px; }
.skill-card { border-top: 1px solid #edf2f7; padding-top: 8px; margin-top: 8px; }
.skill-card:first-of-type { border-top: 0; padding-top: 0; }
.skill-card-head { display: flex; align-items: center; justify-content: space-between; gap: 10px; }
.skill-meta { display: flex; flex-wrap: wrap; gap: 8px; color: #606266; font-size: 12px; margin-top: 4px; }
.role-chip, .state-skill-chip { background: #ecf5ff; color: #337ecc; border-radius: 4px; padding: 2px 6px; font-size: 12px; white-space: nowrap; }
.role-chip.primary { background: #fef3c7; color: #92400e; }
.notice-panel.warning { border-color: #f3d19e; background: #fdf6ec; color: #8a5a16; }
.notice-panel.compact { margin-top: 8px; }
.notice-line { font-size: 13px; margin-top: 4px; }
.validation-item { display: grid; grid-template-columns: 70px 1fr; gap: 8px; border-left: 4px solid #909399; padding: 7px 8px; margin-top: 6px; background: #f7f8fa; }
.validation-item small { grid-column: 2; color: #909399; }
.validation-item.info { border-left-color: #409eff; }
.validation-item.warning { border-left-color: #e6a23c; background: #fdf6ec; }
.validation-item.reject { border-left-color: #f56c6c; background: #fef0f0; }
.reject-note { margin-top: 8px; color: #c45656; font-weight: 700; }
.summary-grid, .state-grid { display: grid; grid-template-columns: 92px 1fr; gap: 6px 10px; font-size: 13px; }
.summary-grid span, .state-grid span { color: #606266; }
.scenario-desc { color: #606266; font-size: 13px; margin: 10px 0 0; }
.chip-list { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }
.time-profile { margin-top: 12px; }
.json-pre, .skill-detail pre { background: #1f2937; color: #d1d5db; padding: 10px; border-radius: 4px; font-size: 12px; overflow-x: auto; }
.log-panel { font-family: Consolas, monospace; background: #1f2937; color: #e5e7eb; padding: 10px; border-radius: 5px; font-size: 12px; white-space: pre-wrap; min-width: 320px; }
.skill-detail { margin-top: 14px; border-top: 1px solid #e4e7ed; padding-top: 12px; }
.detail-columns { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.validation-rule { font-size: 13px; margin-top: 6px; }
.mt-2 { margin-top: 10px; }
:deep(.text p) { margin-top: 0; margin-bottom: 10px; }
:deep(.text p:last-child) { margin-bottom: 0; }
</style>
