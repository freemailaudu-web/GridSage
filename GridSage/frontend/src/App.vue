<template>
  <el-container class="app-wrapper">
    <el-aside width="260px" class="sidebar">
      <SessionSidebar @session-selected="handleSessionSelected" :current-session="currentSessionId" />
    </el-aside>
    <el-main class="main-content" style="position: relative;">
      <!-- Language Switcher -->
      <div class="lang-switch" style="position: absolute; top: 12px; right: 20px; z-index: 10;">
        <el-select v-model="$i18n.locale" @change="saveLocale" size="small" style="width: 100px;">
          <el-option label="中文" value="zh"></el-option>
          <el-option label="English" value="en"></el-option>
        </el-select>
      </div>

      <ChatView v-if="currentSessionId" :session-id="currentSessionId" />
      <div v-else class="empty-state">
        <h2>LVGS Platform</h2>
        <p>Please select or create a session to start.</p>
      </div>
    </el-main>
  </el-container>
</template>

<script setup>
import { ref } from 'vue'
import SessionSidebar from './components/SessionSidebar.vue'
import ChatView from './views/ChatView.vue'

const currentSessionId = ref(null)

const handleSessionSelected = (sessionId) => {
  currentSessionId.value = sessionId
}

const saveLocale = (locale) => {
  localStorage.setItem('lvgs_locale', locale)
}
</script>

<style>
html, body {
  margin: 0;
  padding: 0;
  height: 100%;
}
.app-wrapper {
  height: 100vh;
  width: 100vw;
  overflow: hidden;
}
.sidebar {
  background-color: #f7f9fb;
  border-right: 1px solid #e4e7ed;
  display: flex;
  flex-direction: column;
}
.main-content {
  padding: 0 !important;
  display: flex;
  flex-direction: column;
  background-color: #ffffff;
}
.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  color: #909399;
}
</style>
