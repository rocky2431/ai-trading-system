import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { DashboardLayout } from '@/layouts/DashboardLayout'
import { DashboardPage } from '@/pages/DashboardPage'
import { AgentMonitorPage } from '@/pages/AgentMonitorPage'
import { MonitoringPage } from '@/pages/MonitoringPage'
import { FactorExplorerPage } from '@/pages/FactorExplorerPage'
import { FactorMiningPage } from '@/pages/FactorMiningPage'
import { StrategyWorkshopPage } from '@/pages/StrategyWorkshopPage'
import { BacktestCenterPage } from '@/pages/BacktestCenterPage'
import { ResearchLedgerPage } from '@/pages/ResearchLedgerPage'
import { ReviewQueuePage } from '@/pages/ReviewQueuePage'
import { RLTrainingPage } from '@/pages/RLTrainingPage'
import { PromptsPage } from '@/pages/PromptsPage'
import { CheckpointPage } from '@/pages/CheckpointPage'
// DISABLED: Live trading uses mock data, backend API not implemented
// import { LiveTradingPage } from '@/pages/LiveTradingPage'
import { PipelinePage } from '@/pages/PipelinePage'
import { DataCenterPage } from '@/pages/DataCenterPage'
import { SettingsPage } from '@/pages/SettingsPage'

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<DashboardLayout />}>
          <Route index element={<DashboardPage />} />
          <Route path="monitoring" element={<MonitoringPage />} />
          <Route path="agents" element={<AgentMonitorPage />} />
          <Route path="factors" element={<FactorExplorerPage />} />
          <Route path="mining" element={<FactorMiningPage />} />
          <Route path="strategies" element={<StrategyWorkshopPage />} />
          <Route path="backtest" element={<BacktestCenterPage />} />
          <Route path="research" element={<ResearchLedgerPage />} />
          <Route path="review" element={<ReviewQueuePage />} />
          <Route path="rl-training" element={<RLTrainingPage />} />
          <Route path="prompts" element={<PromptsPage />} />
          <Route path="checkpoints" element={<CheckpointPage />} />
          {/* DISABLED: Live trading route - uses mock data, backend not implemented */}
          {/* <Route path="trading" element={<LiveTradingPage />} /> */}
          <Route path="pipeline" element={<PipelinePage />} />
          <Route path="data" element={<DataCenterPage />} />
          <Route path="settings" element={<SettingsPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}

export default App
