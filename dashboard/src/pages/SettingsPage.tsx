/**
 * Settings Page - System Configuration
 */

import { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Select } from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import { Label } from '@/components/ui/label'
import { Slider } from '@/components/ui/slider'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs'
import { Badge } from '@/components/ui/badge'
import { useAppStore } from '@/store/useAppStore'
import {
  useConfigStatus,
  useAvailableModels,
  useAPIKeys,
  useTestConnections,
  useAgentConfig,
  useDataConfig,
  useFactorMiningConfig,
  useRiskControlConfig,
} from '@/hooks/useConfig'
import {
  Settings,
  Key,
  Bot,
  Database,
  TrendingUp,
  Shield,
  CheckCircle,
  XCircle,
  Loader2,
  RefreshCw,
  Eye,
  EyeOff,
} from 'lucide-react'

export function SettingsPage() {
  const { theme, toggleTheme } = useAppStore()
  const [activeTab, setActiveTab] = useState('llm')

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight flex items-center gap-2">
          <Settings className="h-8 w-8" />
          Settings
        </h1>
        <p className="text-muted-foreground">
          Configure LLM, Exchange, Agent, and System Settings
        </p>
      </div>

      <StatusOverview />

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-6">
          <TabsTrigger value="llm">
            <Key className="h-4 w-4 mr-2" />
            LLM
          </TabsTrigger>
          <TabsTrigger value="exchange">
            <TrendingUp className="h-4 w-4 mr-2" />
            Exchange
          </TabsTrigger>
          <TabsTrigger value="agents">
            <Bot className="h-4 w-4 mr-2" />
            Agents
          </TabsTrigger>
          <TabsTrigger value="data">
            <Database className="h-4 w-4 mr-2" />
            Data
          </TabsTrigger>
          <TabsTrigger value="factor">
            <TrendingUp className="h-4 w-4 mr-2" />
            Factor
          </TabsTrigger>
          <TabsTrigger value="risk">
            <Shield className="h-4 w-4 mr-2" />
            Risk
          </TabsTrigger>
        </TabsList>

        <TabsContent value="llm">
          <LLMConfigSection />
        </TabsContent>

        <TabsContent value="exchange">
          <ExchangeConfigSection />
        </TabsContent>

        <TabsContent value="agents">
          <AgentConfigSection />
        </TabsContent>

        <TabsContent value="data">
          <DataConfigSection />
        </TabsContent>

        <TabsContent value="factor">
          <FactorMiningConfigSection />
        </TabsContent>

        <TabsContent value="risk">
          <RiskControlConfigSection />
        </TabsContent>
      </Tabs>

      {/* Appearance Settings */}
      <Card>
        <CardHeader>
          <CardTitle>Appearance</CardTitle>
          <CardDescription>Customize the look and feel</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="font-medium">Theme</p>
              <p className="text-sm text-muted-foreground">
                Current: {theme}
              </p>
            </div>
            <Button variant="outline" onClick={toggleTheme}>
              Toggle Theme
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

// ============== Status Overview ==============

function StatusOverview() {
  const { status, loading, refetch } = useConfigStatus()

  if (loading) {
    return (
      <Card>
        <CardContent className="py-6">
          <div className="flex items-center justify-center">
            <Loader2 className="h-6 w-6 animate-spin mr-2" />
            Loading status...
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg">System Status</CardTitle>
          <Button variant="ghost" size="sm" onClick={refetch}>
            <RefreshCw className="h-4 w-4" />
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
          <StatusItem
            label="LLM"
            connected={status?.llm_configured ?? false}
            detail={status?.llm_provider ?? 'Not configured'}
          />
          <StatusItem
            label="Exchange"
            connected={status?.exchange_configured ?? false}
            detail={status?.exchange_id ?? 'Not configured'}
          />
          <StatusItem
            label="TimescaleDB"
            connected={status?.timescaledb_connected ?? false}
          />
          <StatusItem
            label="Redis"
            connected={status?.redis_connected ?? false}
          />
          <StatusItem
            label="Qlib"
            connected={status?.qlib_available ?? false}
          />
        </div>
      </CardContent>
    </Card>
  )
}

function StatusItem({ label, connected, detail }: { label: string; connected: boolean; detail?: string }) {
  return (
    <div className="flex items-center gap-2">
      {connected ? (
        <CheckCircle className="h-4 w-4 text-green-500" />
      ) : (
        <XCircle className="h-4 w-4 text-red-500" />
      )}
      <div>
        <p className="text-sm font-medium">{label}</p>
        {detail && <p className="text-xs text-muted-foreground">{detail}</p>}
      </div>
    </div>
  )
}

// ============== LLM Config Section ==============

function LLMConfigSection() {
  const { keys, loading, saving, saveKeys, refetch } = useAPIKeys()
  const { models } = useAvailableModels()
  const { testLLM, testingLLM, llmResult } = useTestConnections()

  const [apiKey, setApiKey] = useState('')
  const [selectedModel, setSelectedModel] = useState('')
  const [selectedEmbedding, setSelectedEmbedding] = useState('')
  const [showKey, setShowKey] = useState(false)

  const handleSave = async () => {
    const data: Record<string, string> = {}
    if (apiKey) data.api_key = apiKey
    if (selectedModel) data.model = selectedModel
    if (selectedEmbedding) data.embedding_model = selectedEmbedding
    data.provider = 'openrouter'

    const result = await saveKeys(data)
    if (result.success) {
      setApiKey('')
      refetch()
    }
  }

  const modelOptions = models?.models?.openrouter?.map(m => ({
    value: m.id,
    label: `${m.name} (${m.context_length ? `${m.context_length / 1000}K` : 'N/A'})`
  })) || []

  const embeddingOptions = models?.embedding_models?.openrouter?.map(m => ({
    value: m.id,
    label: `${m.name} (${m.dimensions}d)`
  })) || []

  if (loading) {
    return <LoadingCard title="LLM Configuration" />
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Key className="h-5 w-5" />
          LLM Configuration
        </CardTitle>
        <CardDescription>
          Configure OpenRouter API for LLM access
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Current Status */}
        <div className="p-4 bg-muted rounded-lg">
          <h4 className="font-medium mb-2">Current Configuration</h4>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <span className="text-muted-foreground">API Key:</span>{' '}
              {keys?.api_key || 'Not set'}
            </div>
            <div>
              <span className="text-muted-foreground">Model:</span>{' '}
              {keys?.model || 'Not set'}
            </div>
            <div>
              <span className="text-muted-foreground">Embedding:</span>{' '}
              {keys?.embedding_model || 'Not set'}
            </div>
          </div>
        </div>

        {/* API Key Input */}
        <div className="space-y-2">
          <Label>OpenRouter API Key</Label>
          <div className="flex gap-2">
            <div className="relative flex-1">
              <Input
                type={showKey ? 'text' : 'password'}
                placeholder="sk-or-v1-..."
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
              />
              <Button
                variant="ghost"
                size="sm"
                className="absolute right-1 top-1/2 -translate-y-1/2 h-7 w-7 p-0"
                onClick={() => setShowKey(!showKey)}
              >
                {showKey ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
              </Button>
            </div>
          </div>
          <p className="text-xs text-muted-foreground">
            Get your API key from{' '}
            <a href="https://openrouter.ai/keys" target="_blank" rel="noopener noreferrer" className="text-primary hover:underline">
              openrouter.ai/keys
            </a>
          </p>
        </div>

        {/* Model Selection */}
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <Label>Chat Model</Label>
            <Select
              options={[{ value: '', label: 'Select model...' }, ...modelOptions]}
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
            />
          </div>
          <div className="space-y-2">
            <Label>Embedding Model</Label>
            <Select
              options={[{ value: '', label: 'Select embedding...' }, ...embeddingOptions]}
              value={selectedEmbedding}
              onChange={(e) => setSelectedEmbedding(e.target.value)}
            />
          </div>
        </div>

        {/* Actions */}
        <div className="flex items-center gap-4">
          <Button onClick={handleSave} disabled={saving}>
            {saving && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
            Save Configuration
          </Button>
          <Button variant="outline" onClick={testLLM} disabled={testingLLM}>
            {testingLLM && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
            Test Connection
          </Button>
        </div>

        {/* Test Result */}
        {llmResult && (
          <div className={`p-3 rounded-lg ${llmResult.success ? 'bg-green-500/10 text-green-600' : 'bg-red-500/10 text-red-600'}`}>
            {llmResult.success ? <CheckCircle className="h-4 w-4 inline mr-2" /> : <XCircle className="h-4 w-4 inline mr-2" />}
            {llmResult.message}
          </div>
        )}
      </CardContent>
    </Card>
  )
}

// ============== Exchange Config Section ==============

function ExchangeConfigSection() {
  const { keys, loading, saving, saveKeys, refetch } = useAPIKeys()
  const { testExchange, testingExchange, exchangeResult } = useTestConnections()

  const [exchangeId, setExchangeId] = useState('binance')
  const [apiKey, setApiKey] = useState('')
  const [apiSecret, setApiSecret] = useState('')
  const [showKeys, setShowKeys] = useState(false)

  const handleSave = async () => {
    const data: Record<string, string> = {
      exchange_id: exchangeId,
    }
    if (apiKey) data.exchange_api_key = apiKey
    if (apiSecret) data.exchange_secret = apiSecret

    const result = await saveKeys(data)
    if (result.success) {
      setApiKey('')
      setApiSecret('')
      refetch()
    }
  }

  const exchangeOptions = [
    { value: 'binance', label: 'Binance' },
    { value: 'okx', label: 'OKX' },
    { value: 'bybit', label: 'Bybit' },
    { value: 'gate', label: 'Gate.io' },
  ]

  if (loading) {
    return <LoadingCard title="Exchange Configuration" />
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <TrendingUp className="h-5 w-5" />
          Exchange Configuration
        </CardTitle>
        <CardDescription>
          Configure exchange API for trading and data
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Current Status */}
        <div className="p-4 bg-muted rounded-lg">
          <h4 className="font-medium mb-2">Current Configuration</h4>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <span className="text-muted-foreground">Exchange:</span>{' '}
              {keys?.exchange_id || 'Not set'}
            </div>
            <div>
              <span className="text-muted-foreground">API Key:</span>{' '}
              {keys?.exchange_api_key || 'Not set'}
            </div>
          </div>
        </div>

        {/* Exchange Selection */}
        <div className="space-y-2">
          <Label>Exchange</Label>
          <Select
            options={exchangeOptions}
            value={exchangeId}
            onChange={(e) => setExchangeId(e.target.value)}
          />
        </div>

        {/* API Keys */}
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <Label>API Key</Label>
            <Input
              type={showKeys ? 'text' : 'password'}
              placeholder="Enter API Key"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
            />
          </div>
          <div className="space-y-2">
            <Label>API Secret</Label>
            <Input
              type={showKeys ? 'text' : 'password'}
              placeholder="Enter API Secret"
              value={apiSecret}
              onChange={(e) => setApiSecret(e.target.value)}
            />
          </div>
        </div>

        <div className="flex items-center gap-2">
          <Switch checked={showKeys} onCheckedChange={setShowKeys} />
          <Label>Show API Keys</Label>
        </div>

        {/* Actions */}
        <div className="flex items-center gap-4">
          <Button onClick={handleSave} disabled={saving}>
            {saving && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
            Save Configuration
          </Button>
          <Button variant="outline" onClick={testExchange} disabled={testingExchange}>
            {testingExchange && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
            Test Connection
          </Button>
        </div>

        {/* Test Result */}
        {exchangeResult && (
          <div className={`p-3 rounded-lg ${exchangeResult.success ? 'bg-green-500/10 text-green-600' : 'bg-red-500/10 text-red-600'}`}>
            {exchangeResult.success ? <CheckCircle className="h-4 w-4 inline mr-2" /> : <XCircle className="h-4 w-4 inline mr-2" />}
            {exchangeResult.message}
          </div>
        )}
      </CardContent>
    </Card>
  )
}

// ============== Agent Config Section ==============

function AgentConfigSection() {
  const { config, loading, saving, saveConfig } = useAgentConfig()
  const { models } = useAvailableModels()

  const modelOptions = models?.models?.openrouter?.map(m => ({
    value: m.id,
    label: m.name
  })) || []

  const handleModelChange = async (agentId: string, modelId: string) => {
    await saveConfig({ agent_id: agentId, model_id: modelId })
  }

  const handleEnabledChange = async (agentId: string, enabled: boolean) => {
    await saveConfig({ agent_id: agentId, enabled })
  }

  if (loading) {
    return <LoadingCard title="Agent Configuration" />
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Bot className="h-5 w-5" />
          Agent Configuration
        </CardTitle>
        <CardDescription>
          Configure LLM model for each IQFMP agent
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {config?.agents.map((agent) => (
            <div
              key={agent.agent_id}
              className="flex items-center justify-between p-4 border rounded-lg"
            >
              <div className="flex-1">
                <div className="flex items-center gap-2">
                  <h4 className="font-medium">{agent.agent_name}</h4>
                  <Badge variant={agent.enabled ? 'default' : 'secondary'}>
                    {agent.enabled ? 'Enabled' : 'Disabled'}
                  </Badge>
                </div>
                <p className="text-sm text-muted-foreground mt-1">
                  {agent.description}
                </p>
              </div>
              <div className="flex items-center gap-4">
                <div className="w-64">
                  <Select
                    options={[{ value: '', label: 'Select model...' }, ...modelOptions]}
                    value={agent.model_id}
                    onChange={(e) => handleModelChange(agent.agent_id, e.target.value)}
                    disabled={saving}
                  />
                </div>
                <Switch
                  checked={agent.enabled}
                  onCheckedChange={(checked) => handleEnabledChange(agent.agent_id, checked)}
                  disabled={saving}
                />
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}

// ============== Data Config Section ==============

function DataConfigSection() {
  const { config, loading, saving, saveConfig } = useDataConfig()

  const [frequency, setFrequency] = useState('')
  const [dataSource, setDataSource] = useState('')
  const [symbolsInput, setSymbolsInput] = useState('')

  const handleSave = async () => {
    const data: Record<string, unknown> = {}
    if (frequency) data.data_frequency = frequency
    if (dataSource) data.data_source = dataSource
    if (symbolsInput) {
      data.symbols = symbolsInput.split(',').map(s => s.trim().toUpperCase())
    }
    await saveConfig(data)
  }

  const frequencyOptions = config?.frequency_options.map(f => ({
    value: f.id,
    label: `${f.name} - ${f.description}`
  })) || []

  const sourceOptions = [
    { value: 'timescaledb', label: 'TimescaleDB' },
    { value: 'ccxt', label: 'CCXT (Exchange)' },
    { value: 'file', label: 'Local File' },
  ]

  if (loading) {
    return <LoadingCard title="Data Configuration" />
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Database className="h-5 w-5" />
          Data Configuration
        </CardTitle>
        <CardDescription>
          Configure data source and frequency settings
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Current Config */}
        <div className="p-4 bg-muted rounded-lg">
          <h4 className="font-medium mb-2">Current Configuration</h4>
          <div className="grid grid-cols-3 gap-4 text-sm">
            <div>
              <span className="text-muted-foreground">Frequency:</span>{' '}
              {config?.data_frequency || 'Not set'}
            </div>
            <div>
              <span className="text-muted-foreground">Source:</span>{' '}
              {config?.data_source || 'Not set'}
            </div>
            <div>
              <span className="text-muted-foreground">Symbols:</span>{' '}
              {config?.symbols.length || 0} configured
            </div>
          </div>
          {config?.symbols && config.symbols.length > 0 && (
            <div className="mt-2 flex flex-wrap gap-1">
              {config.symbols.slice(0, 10).map(s => (
                <Badge key={s} variant="outline">{s}</Badge>
              ))}
              {config.symbols.length > 10 && (
                <Badge variant="outline">+{config.symbols.length - 10} more</Badge>
              )}
            </div>
          )}
        </div>

        {/* Settings */}
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <Label>Data Frequency</Label>
            <Select
              options={[{ value: '', label: 'Select frequency...' }, ...frequencyOptions]}
              value={frequency || config?.data_frequency || ''}
              onChange={(e) => setFrequency(e.target.value)}
            />
          </div>
          <div className="space-y-2">
            <Label>Data Source</Label>
            <Select
              options={sourceOptions}
              value={dataSource || config?.data_source || ''}
              onChange={(e) => setDataSource(e.target.value)}
            />
          </div>
        </div>

        <div className="space-y-2">
          <Label>Symbols (comma-separated)</Label>
          <Input
            placeholder="BTC/USDT, ETH/USDT, SOL/USDT..."
            value={symbolsInput || config?.symbols.join(', ') || ''}
            onChange={(e) => setSymbolsInput(e.target.value)}
          />
        </div>

        <Button onClick={handleSave} disabled={saving}>
          {saving && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
          Save Configuration
        </Button>
      </CardContent>
    </Card>
  )
}

// ============== Factor Mining Config Section ==============

function FactorMiningConfigSection() {
  const { config, loading, saving, saveConfig } = useFactorMiningConfig()

  const [localConfig, setLocalConfig] = useState<{
    min_ic: number
    min_ir: number
    min_sharpe: number
    max_turnover: number
    cv_folds: number
    max_concurrent: number
  } | null>(null)

  const handleSave = async () => {
    if (!localConfig) return
    await saveConfig({
      evaluation: {
        min_ic: localConfig.min_ic,
        min_ir: localConfig.min_ir,
        min_sharpe: localConfig.min_sharpe,
        max_turnover: localConfig.max_turnover,
        cv_folds: localConfig.cv_folds,
        train_ratio: config?.evaluation.train_ratio || 0.6,
        valid_ratio: config?.evaluation.valid_ratio || 0.2,
        test_ratio: config?.evaluation.test_ratio || 0.2,
        use_dynamic_threshold: config?.evaluation.use_dynamic_threshold || true,
        deflation_rate: config?.evaluation.deflation_rate || 0.1,
      },
      max_concurrent_generation: localConfig.max_concurrent,
    })
  }

  const handleFamilyToggle = async (familyId: string, enabled: boolean) => {
    if (!config) return
    const updatedFamilies = config.factor_families.map(f =>
      f.id === familyId ? { ...f, enabled } : f
    )
    await saveConfig({ factor_families: updatedFamilies })
  }

  if (loading) {
    return <LoadingCard title="Factor Mining Configuration" />
  }

  const evaluation = config?.evaluation
  const currentConfig = localConfig || {
    min_ic: evaluation?.min_ic || 0.02,
    min_ir: evaluation?.min_ir || 0.5,
    min_sharpe: evaluation?.min_sharpe || 1.0,
    max_turnover: evaluation?.max_turnover || 0.5,
    cv_folds: evaluation?.cv_folds || 5,
    max_concurrent: config?.max_concurrent_generation || 10,
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <TrendingUp className="h-5 w-5" />
          Factor Mining Configuration
        </CardTitle>
        <CardDescription>
          Configure factor families and evaluation thresholds
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Factor Families */}
        <div>
          <h4 className="font-medium mb-3">Factor Families</h4>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {config?.factor_families.map((family) => (
              <div
                key={family.id}
                className="flex items-center justify-between p-3 border rounded-lg"
              >
                <div>
                  <p className="font-medium text-sm">{family.name}</p>
                  <p className="text-xs text-muted-foreground">{family.description}</p>
                </div>
                <Switch
                  checked={family.enabled}
                  onCheckedChange={(checked) => handleFamilyToggle(family.id, checked)}
                  disabled={saving}
                />
              </div>
            ))}
          </div>
        </div>

        {/* Evaluation Thresholds */}
        <div>
          <h4 className="font-medium mb-3">Evaluation Thresholds</h4>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            <div className="space-y-2">
              <div className="flex justify-between">
                <Label>Min IC</Label>
                <span className="text-sm text-muted-foreground">{currentConfig.min_ic}</span>
              </div>
              <Slider
                value={currentConfig.min_ic}
                onValueChange={(v) => setLocalConfig({ ...currentConfig, min_ic: v })}
                min={0}
                max={0.1}
                step={0.005}
              />
            </div>
            <div className="space-y-2">
              <div className="flex justify-between">
                <Label>Min IR</Label>
                <span className="text-sm text-muted-foreground">{currentConfig.min_ir}</span>
              </div>
              <Slider
                value={currentConfig.min_ir}
                onValueChange={(v) => setLocalConfig({ ...currentConfig, min_ir: v })}
                min={0}
                max={2}
                step={0.1}
              />
            </div>
            <div className="space-y-2">
              <div className="flex justify-between">
                <Label>Min Sharpe</Label>
                <span className="text-sm text-muted-foreground">{currentConfig.min_sharpe}</span>
              </div>
              <Slider
                value={currentConfig.min_sharpe}
                onValueChange={(v) => setLocalConfig({ ...currentConfig, min_sharpe: v })}
                min={0}
                max={3}
                step={0.1}
              />
            </div>
            <div className="space-y-2">
              <div className="flex justify-between">
                <Label>Max Turnover</Label>
                <span className="text-sm text-muted-foreground">{currentConfig.max_turnover}</span>
              </div>
              <Slider
                value={currentConfig.max_turnover}
                onValueChange={(v) => setLocalConfig({ ...currentConfig, max_turnover: v })}
                min={0}
                max={1}
                step={0.05}
              />
            </div>
            <div className="space-y-2">
              <div className="flex justify-between">
                <Label>CV Folds</Label>
                <span className="text-sm text-muted-foreground">{currentConfig.cv_folds}</span>
              </div>
              <Slider
                value={currentConfig.cv_folds}
                onValueChange={(v) => setLocalConfig({ ...currentConfig, cv_folds: v })}
                min={3}
                max={10}
                step={1}
              />
            </div>
            <div className="space-y-2">
              <div className="flex justify-between">
                <Label>Max Concurrent</Label>
                <span className="text-sm text-muted-foreground">{currentConfig.max_concurrent}</span>
              </div>
              <Slider
                value={currentConfig.max_concurrent}
                onValueChange={(v) => setLocalConfig({ ...currentConfig, max_concurrent: v })}
                min={1}
                max={20}
                step={1}
              />
            </div>
          </div>
        </div>

        <Button onClick={handleSave} disabled={saving}>
          {saving && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
          Save Evaluation Settings
        </Button>
      </CardContent>
    </Card>
  )
}

// ============== Risk Control Config Section ==============

function RiskControlConfigSection() {
  const { config, loading, saving, saveConfig } = useRiskControlConfig()

  const [localConfig, setLocalConfig] = useState<{
    max_single_loss: number
    max_daily_loss: number
    max_position: number
    max_total_position: number
    emergency_threshold: number
  } | null>(null)

  const handleSave = async () => {
    if (!localConfig) return
    await saveConfig({
      max_single_loss_pct: localConfig.max_single_loss / 100,
      max_daily_loss_pct: localConfig.max_daily_loss / 100,
      max_position_pct: localConfig.max_position / 100,
      max_total_position_pct: localConfig.max_total_position / 100,
      emergency_close_threshold: localConfig.emergency_threshold / 100,
    })
  }

  if (loading) {
    return <LoadingCard title="Risk Control Configuration" />
  }

  const riskConfig = config?.config
  const currentConfig = localConfig || {
    max_single_loss: (riskConfig?.max_single_loss_pct || 0.02) * 100,
    max_daily_loss: (riskConfig?.max_daily_loss_pct || 0.05) * 100,
    max_position: (riskConfig?.max_position_pct || 0.10) * 100,
    max_total_position: (riskConfig?.max_total_position_pct || 0.50) * 100,
    emergency_threshold: (riskConfig?.emergency_close_threshold || 0.08) * 100,
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Shield className="h-5 w-5" />
          Risk Control Configuration
        </CardTitle>
        <CardDescription>
          Configure risk management parameters
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Live Trading Status */}
        <div className={`p-4 rounded-lg ${config?.is_live_trading_enabled ? 'bg-green-500/10' : 'bg-yellow-500/10'}`}>
          <div className="flex items-center gap-2">
            {config?.is_live_trading_enabled ? (
              <CheckCircle className="h-5 w-5 text-green-500" />
            ) : (
              <Shield className="h-5 w-5 text-yellow-500" />
            )}
            <span className="font-medium">
              Live Trading: {config?.is_live_trading_enabled ? 'Enabled' : 'Disabled (Paper Mode)'}
            </span>
          </div>
        </div>

        {/* Risk Parameters */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-2">
            <div className="flex justify-between">
              <Label>Max Single Loss</Label>
              <span className="text-sm font-medium">{currentConfig.max_single_loss.toFixed(1)}%</span>
            </div>
            <Slider
              value={currentConfig.max_single_loss}
              onValueChange={(v) => setLocalConfig({ ...currentConfig, max_single_loss: v })}
              min={0.5}
              max={10}
              step={0.5}
            />
            <p className="text-xs text-muted-foreground">Maximum loss per single trade</p>
          </div>

          <div className="space-y-2">
            <div className="flex justify-between">
              <Label>Max Daily Loss</Label>
              <span className="text-sm font-medium">{currentConfig.max_daily_loss.toFixed(1)}%</span>
            </div>
            <Slider
              value={currentConfig.max_daily_loss}
              onValueChange={(v) => setLocalConfig({ ...currentConfig, max_daily_loss: v })}
              min={1}
              max={20}
              step={0.5}
            />
            <p className="text-xs text-muted-foreground">Maximum total loss per day</p>
          </div>

          <div className="space-y-2">
            <div className="flex justify-between">
              <Label>Max Position Size</Label>
              <span className="text-sm font-medium">{currentConfig.max_position.toFixed(0)}%</span>
            </div>
            <Slider
              value={currentConfig.max_position}
              onValueChange={(v) => setLocalConfig({ ...currentConfig, max_position: v })}
              min={5}
              max={50}
              step={5}
            />
            <p className="text-xs text-muted-foreground">Maximum position size per strategy</p>
          </div>

          <div className="space-y-2">
            <div className="flex justify-between">
              <Label>Max Total Position</Label>
              <span className="text-sm font-medium">{currentConfig.max_total_position.toFixed(0)}%</span>
            </div>
            <Slider
              value={currentConfig.max_total_position}
              onValueChange={(v) => setLocalConfig({ ...currentConfig, max_total_position: v })}
              min={10}
              max={100}
              step={5}
            />
            <p className="text-xs text-muted-foreground">Maximum total position across all strategies</p>
          </div>

          <div className="space-y-2 md:col-span-2">
            <div className="flex justify-between">
              <Label>Emergency Close Threshold</Label>
              <span className="text-sm font-medium text-red-500">{currentConfig.emergency_threshold.toFixed(1)}%</span>
            </div>
            <Slider
              value={currentConfig.emergency_threshold}
              onValueChange={(v) => setLocalConfig({ ...currentConfig, emergency_threshold: v })}
              min={5}
              max={25}
              step={1}
            />
            <p className="text-xs text-muted-foreground">Trigger emergency close all positions when daily loss exceeds this</p>
          </div>
        </div>

        <Button onClick={handleSave} disabled={saving}>
          {saving && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
          Save Risk Settings
        </Button>
      </CardContent>
    </Card>
  )
}

// ============== Shared Components ==============

function LoadingCard({ title }: { title: string }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex items-center justify-center py-8">
          <Loader2 className="h-6 w-6 animate-spin mr-2" />
          Loading...
        </div>
      </CardContent>
    </Card>
  )
}
