import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { DashboardLayout } from '@/layouts/DashboardLayout'
import { DashboardPage } from '@/pages/DashboardPage'
import { AgentMonitorPage } from '@/pages/AgentMonitorPage'
import { FactorsPage } from '@/pages/FactorsPage'
import { PipelinePage } from '@/pages/PipelinePage'
import { SettingsPage } from '@/pages/SettingsPage'

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<DashboardLayout />}>
          <Route index element={<DashboardPage />} />
          <Route path="agents" element={<AgentMonitorPage />} />
          <Route path="factors" element={<FactorsPage />} />
          <Route path="pipeline" element={<PipelinePage />} />
          <Route path="settings" element={<SettingsPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}

export default App
