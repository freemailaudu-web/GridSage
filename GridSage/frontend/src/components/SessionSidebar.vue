<template>
  <div class="sidebar-container">
    <div class="header">
      <el-button type="primary" class="new-chat-btn" @click="createNewSession">
        {{ $t('create_session') }}
      </el-button>
    </div>
    
    <div class="session-list">
      <div 
        v-for="session in sessions" 
        :key="session.id"
        :class="['session-item', { active: currentSession === session.id }]"
        @click="selectSession(session.id)"
      >
        <span class="session-title">{{ session.title || $t('unnamed_session') }}</span>
      </div>
    </div>
    
    <div class="footer-settings">
      <el-divider />
      <h4>{{ $t('model_config') }}</h4>
      <el-input v-model="modelConfig.model_name" :placeholder="$t('model_name_placeholder')" class="mb-2" @change="saveConfig"/>
      <el-input v-model="modelConfig.base_url" :placeholder="$t('base_url_placeholder')" class="mb-2" @change="saveConfig"/>
      <el-input v-model="modelConfig.api_key" type="password" :placeholder="$t('placeholder_api_key')" show-password @change="saveConfig"/>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import axios from 'axios'
import { useI18n } from 'vue-i18n'

const { t } = useI18n()

const emit = defineEmits(['session-selected'])
const props = defineProps(['currentSession'])

const sessions = ref([])
const modelConfig = ref({
  model_name: 'gpt-4o-mini',
  base_url: 'https://api.openai.com/v1',
  api_key: ''
})

onMounted(() => {
  const savedConfig = localStorage.getItem('lvgs_model_config')
  if (savedConfig) {
    modelConfig.value = JSON.parse(savedConfig)
  }
})

const saveConfig = () => {
  localStorage.setItem('lvgs_model_config', JSON.stringify(modelConfig.value))
}

const createNewSession = async () => {
  try {
    const res = await axios.post('http://127.0.0.1:8000/api/session/new')
    const sid = res.data.session_id
    sessions.value.unshift({ id: sid, title: t('unnamed_session') })
    selectSession(sid)
  } catch (error) {
    console.error("Failed to create session", error)
  }
}

const selectSession = (sid) => {
  emit('session-selected', sid)
}
</script>

<style scoped>
.sidebar-container {
  display: flex;
  flex-direction: column;
  height: 100%;
}
.header {
  padding: 16px;
}
.new-chat-btn {
  width: 100%;
}
.session-list {
  flex: 1;
  overflow-y: auto;
  padding: 0 10px;
}
.session-item {
  padding: 12px;
  margin-bottom: 8px;
  border-radius: 6px;
  cursor: pointer;
  transition: background-color 0.2s;
  font-size: 14px;
}
.session-item:hover {
  background-color: #e6e8eb;
}
.session-item.active {
  background-color: #d9ecff;
  color: #409eff;
  font-weight: 500;
}
.session-title {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  display: block;
}
.footer-settings {
  padding: 16px;
  background: #f7f9fb;
}
.footer-settings h4 {
  margin-top: 0;
  margin-bottom: 12px;
  font-size: 14px;
  color: #606266;
}
.mb-2 {
  margin-bottom: 10px;
}
</style>
