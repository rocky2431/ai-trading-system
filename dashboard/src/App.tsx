import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { DashboardLayout } from '@/layouts/DashboardLayout'
import { DashboardPage } from '@/pages/DashboardPage'
import { AgentMonitorPage } from '@/pages/AgentMonitorPage'
import { FactorExplorerPage } from '@/pages/FactorExplorerPage'
import { ResearchLedgerPage } from '@/pages/ResearchLedgerPage'
import { LiveTradingPage } from '@/pages/LiveTradingPage'
import { PipelinePage } from '@/pages/PipelinePage'
import { SettingsPage } from '@/pages/SettingsPage'

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<DashboardLayout />}>
          <Route index element={<DashboardPage />} />
          <Route path="agents" element={<AgentMonitorPage />} />
          <Route path="factors" element={<FactorExplorerPage />} />
          <Route path="research" element={<ResearchLedgerPage />} />
          <Route path="trading" element={<LiveTradingPage />} />
          <Route path="pipeline" element={<PipelinePage />} />
          <Route path="settings" element={<SettingsPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}

export default App
