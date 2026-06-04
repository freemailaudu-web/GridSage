import { createI18n } from 'vue-i18n'

const messages = {
  en: {
    new_chat: '+ New Chat',
    model_config: 'Model Config',
    placeholder_api_key: 'API Key',
    send: 'Send',
    run_simulation: 'Run Simulation',
    placeholder_input: 'Describe the scenario, for example: create a high PV low-load case, or add heavy load at feeder-end nodes.',
    waiting: 'Thinking...',
    system_hello: 'Hello! I am the GridSage agent. Tell me the distribution-grid scenario you want to build.',
    req_failed: 'Request failed',
    check_api: 'Please check your API Key, backend service, or network settings.',
    task_running: 'Simulation is running in the background. Please wait...',
    sim_recv: 'Simulation command received. The backend is calculating power flow and scheduling...',
    sim_success: 'Simulation finished successfully. Estimated cost: **{cost}**.',
    train_success: 'Training finished successfully.',
    sim_error: 'Backend engine encountered an error. Log tail:\n{log}',
    create_session: '+ New Session',
    unnamed_session: 'Unnamed Session',
    model_name_placeholder: 'Model name',
    base_url_placeholder: 'Base URL',
    current_scenario: 'Current Scenario Parameters',
    node_table: 'Node Overrides',
    pv: '+ PV (kW)',
    load: '+ Load (kW)',
    ess: '+ ESS (kWh)',
    empty_text: 'No node-level perturbations',
    node: 'Bus',
    tag_pv: 'PV: x{value}',
    tag_load: 'Load: x{value}',
    algo_label: 'Algorithm',
    mode_label: 'Mode',
    mode_baseline: 'Baseline Evaluation',
    mode_train: 'Training',
    mode_evaluate: 'Evaluation',
    global_control: 'Global Controls',
    pv_multiplier: 'PV Multiplier',
    load_multiplier: 'Load Multiplier',
    ev_multiplier: 'EV Multiplier',
    switch_on: 'ON',
    switch_off: 'OFF',
    time_profile_hint: 'Time profiles enabled and varying by hour',
    mod_table_title: 'Node Modifications',
    mod_col_node: 'Node',
    mod_col_device: 'Device',
    mod_col_type: 'Change',
    mod_col_value: 'Value',
    mod_empty: 'No node-level perturbations. Current scenario is unchanged.',
    chip_nodes: 'Nodes',
    chip_lines: 'Lines',
    topo_title: 'Grid Topology - {model}',
    topo_sub_full: 'Purple = has DER | Red = user modified | Hover for details',
    topo_sub_fallback: 'Red = user modified | Hover for details',
    tooltip_node: 'Node',
    tooltip_base_devices: 'Base Devices',
    tooltip_modifications: 'Current Modifications',
    tooltip_none: 'None',
    tooltip_global_load: 'Global load multiplier applied: x{value}',
    tooltip_global_pv: 'Global PV multiplier applied: x{value}',
    tooltip_global_ev: 'Global EV multiplier applied: x{value}',
    scenario_rl_config: 'Scenario / RL Config',
    scenario_name_label: 'Scenario',
    time_profile_curves: 'Time-Varying Multipliers'
  }
}

// Compatibility note: keep the legacy storage key while normalizing old locale values to English.
const locale = 'en'
if (localStorage.getItem('lvgs_locale') !== 'en') {
  localStorage.setItem('lvgs_locale', 'en')
}

export const i18n = createI18n({
  legacy: false,
  locale,
  fallbackLocale: 'en',
  messages
})
