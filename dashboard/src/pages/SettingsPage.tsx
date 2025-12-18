/**
 * Settings Page - System Configuration
 */

import { useState, useEffect } from 'react'
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
  Trash2,
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
          <DataInfoSection />
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
  const { keys, loading, saving, deleting, saveKeys, deleteKeys, refetch } = useAPIKeys()
  const { testLLM, testingLLM, llmResult } = useTestConnections()

  const [apiKey, setApiKey] = useState('')
  const [showKey, setShowKey] = useState(false)
  const [deleteResult, setDeleteResult] = useState<{ success: boolean; message: string } | null>(null)

  const handleSave = async () => {
    if (!apiKey) return
    const data: Record<string, string> = {
      api_key: apiKey,
      provider: 'openrouter'
    }

    const result = await saveKeys(data)
    if (result.success) {
      setApiKey('')
      refetch()
    }
  }

  const handleDelete = async () => {
    if (!confirm('Are you sure you want to delete OpenRouter API Key configuration?')) {
      return
    }
    setDeleteResult(null)
    const result = await deleteKeys('llm')
    setDeleteResult(result)
  }

  if (loading) {
    return <LoadingCard title="LLM Configuration" />
  }

  const hasConfig = !!keys?.api_key

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Key className="h-5 w-5" />
          OpenRouter API Configuration
        </CardTitle>
        <CardDescription>
          Configure OpenRouter API Key to access various LLM models
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Current Status - 删除按钮在这里 */}
        <div className="p-4 bg-muted rounded-lg">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              {hasConfig ? (
                <CheckCircle className="h-5 w-5 text-green-500" />
              ) : (
                <XCircle className="h-5 w-5 text-red-500" />
              )}
              <div>
                <h4 className="font-medium">
                  {hasConfig ? 'API Key Configured' : 'API Key Not Configured'}
                </h4>
                {hasConfig && (
                  <p className="text-sm text-muted-foreground">
                    {keys?.api_key}
                  </p>
                )}
              </div>
            </div>
            <div className="flex items-center gap-2">
              {hasConfig && (
                <Button
                  variant="destructive"
                  size="sm"
                  onClick={handleDelete}
                  disabled={deleting}
                >
                  {deleting ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <Trash2 className="h-4 w-4" />
                  )}
                </Button>
              )}
              <Badge variant={hasConfig ? 'default' : 'secondary'}>
                {hasConfig ? 'Configured' : 'Not Configured'}
              </Badge>
            </div>
          </div>
        </div>

        {/* Delete Result */}
        {deleteResult && (
          <div className={`p-3 rounded-lg ${deleteResult.success ? 'bg-green-500/10 text-green-600' : 'bg-red-500/10 text-red-600'}`}>
            {deleteResult.success ? <CheckCircle className="h-4 w-4 inline mr-2" /> : <XCircle className="h-4 w-4 inline mr-2" />}
            {deleteResult.message}
          </div>
        )}

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
            Get API Key from{' '}
            <a href="https://openrouter.ai/keys" target="_blank" rel="noopener noreferrer" className="text-primary hover:underline">
              openrouter.ai/keys
            </a>
          </p>
        </div>

        {/* Info Box */}
        <div className="p-4 bg-blue-500/10 rounded-lg text-sm">
          <p className="font-medium text-blue-600 mb-2">Notes</p>
          <ul className="list-disc list-inside space-y-1 text-muted-foreground">
            <li>Only need to configure API Key here for verifying OpenRouter connection</li>
            <li>Each Agent's Chat Model is configured separately in the <strong>Agents</strong> tab</li>
            <li>Factor Mining LLM Model and Embedding Model are configured in the <strong>Factor</strong> tab</li>
          </ul>
        </div>

        {/* Actions */}
        <div className="flex items-center gap-4">
          <Button onClick={handleSave} disabled={saving || !apiKey}>
            {saving && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
            Save API Key
          </Button>
          <Button variant="outline" onClick={testLLM} disabled={testingLLM || !hasConfig}>
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
  const { keys, loading, saving, deleting, saveKeys, deleteKeys, refetch } = useAPIKeys()
  const { testExchange, testingExchange, exchangeResult } = useTestConnections()

  const [exchangeId, setExchangeId] = useState('binance')
  const [apiKey, setApiKey] = useState('')
  const [apiSecret, setApiSecret] = useState('')
  const [showKeys, setShowKeys] = useState(false)
  const [deleteResult, setDeleteResult] = useState<{ success: boolean; message: string } | null>(null)

  // 从已保存的配置初始化交易所选择
  useEffect(() => {
    if (keys?.exchange_id && exchangeId === 'binance') {
      setExchangeId(keys.exchange_id)
    }
  }, [keys, exchangeId])

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

  const handleDelete = async () => {
    if (!confirm('Are you sure you want to delete all Exchange configuration? This will remove the exchange ID, API key, and secret.')) {
      return
    }
    setDeleteResult(null)
    const result = await deleteKeys('exchange')
    setDeleteResult(result)
    if (result.success) {
      setExchangeId('binance')
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

  const hasConfig = keys?.exchange_id || keys?.exchange_api_key

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
        {/* Current Status - 删除按钮在这里 */}
        <div className="p-4 bg-muted rounded-lg">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              {hasConfig ? (
                <CheckCircle className="h-5 w-5 text-green-500" />
              ) : (
                <XCircle className="h-5 w-5 text-red-500" />
              )}
              <div>
                <h4 className="font-medium">
                  {hasConfig ? 'Exchange Configured' : 'Exchange Not Configured'}
                </h4>
                {hasConfig && (
                  <div className="text-sm text-muted-foreground mt-1">
                    <span>{keys?.exchange_id || 'Unknown'}</span>
                    {keys?.exchange_api_key && (
                      <span className="ml-3">{keys.exchange_api_key}</span>
                    )}
                  </div>
                )}
              </div>
            </div>
            <div className="flex items-center gap-2">
              {hasConfig && (
                <Button
                  variant="destructive"
                  size="sm"
                  onClick={handleDelete}
                  disabled={deleting}
                >
                  {deleting ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <Trash2 className="h-4 w-4" />
                  )}
                </Button>
              )}
              <Badge variant={hasConfig ? 'default' : 'secondary'}>
                {hasConfig ? 'Configured' : 'Not Configured'}
              </Badge>
            </div>
          </div>
        </div>

        {/* Delete Result */}
        {deleteResult && (
          <div className={`p-3 rounded-lg ${deleteResult.success ? 'bg-green-500/10 text-green-600' : 'bg-red-500/10 text-red-600'}`}>
            {deleteResult.success ? <CheckCircle className="h-4 w-4 inline mr-2" /> : <XCircle className="h-4 w-4 inline mr-2" />}
            {deleteResult.message}
          </div>
        )}

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

// ============== Data Info Section ==============

function DataInfoSection() {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Database className="h-5 w-5" />
          Data Management
        </CardTitle>
        <CardDescription>
          Data download and management is done in Data Center
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="p-4 bg-blue-500/10 rounded-lg">
          <h4 className="font-medium text-blue-600 mb-3">Data Management Guide</h4>
          <div className="space-y-3 text-sm text-muted-foreground">
            <p>
              <strong>Data download and management</strong> - please go to the <span className="font-medium text-foreground">Data Center</span> page:
            </p>
            <ul className="list-disc list-inside space-y-1 ml-2">
              <li>View available trading pairs and data time ranges</li>
              <li>Download historical candlestick data (spot/futures)</li>
              <li>Manage data download tasks</li>
            </ul>
            <p className="mt-4">
              <strong>When creating Factor Mining tasks</strong>, you need to select in the task creation interface:
            </p>
            <ul className="list-disc list-inside space-y-1 ml-2">
              <li>Market type (spot/futures)</li>
              <li>Trading pairs (supports multi-select)</li>
              <li>Timeframe (1m/5m/1h/1d)</li>
              <li>Data time range</li>
            </ul>
          </div>
        </div>

        <div className="flex gap-4">
          <Button variant="outline" onClick={() => window.location.href = '/data-center'}>
            <Database className="h-4 w-4 mr-2" />
            Go to Data Center
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}

// ============== Factor Mining Config Section ==============

function FactorMiningConfigSection() {
  const { config, loading, saving, saveConfig } = useFactorMiningConfig()
  const { models } = useAvailableModels()
  const { keys, saveKeys, refetch } = useAPIKeys()

  const [selectedModel, setSelectedModel] = useState('')
  const [selectedEmbedding, setSelectedEmbedding] = useState('')
  const [maxConcurrent, setMaxConcurrent] = useState(10)
  const [codeTimeout, setCodeTimeout] = useState(30)
  const [savingModel, setSavingModel] = useState(false)

  // 初始化 LLM model 和 embedding model
  useEffect(() => {
    if (keys?.model && !selectedModel) {
      setSelectedModel(keys.model)
    }
    if (keys?.embedding_model && !selectedEmbedding) {
      setSelectedEmbedding(keys.embedding_model)
    }
  }, [keys, selectedModel, selectedEmbedding])

  useEffect(() => {
    if (config) {
      setMaxConcurrent(config.max_concurrent_generation || 10)
      setCodeTimeout(config.code_execution_timeout || 30)
    }
  }, [config])

  // LLM 模型选项
  const modelOptions = models?.models?.openrouter?.map(m => ({
    value: m.id,
    label: m.name
  })) || []

  const embeddingOptions = models?.embedding_models?.openrouter?.map(m => ({
    value: m.id,
    label: `${m.name} (${m.dimensions}d)`
  })) || []

  const handleSaveModel = async () => {
    if (!selectedModel) return
    setSavingModel(true)
    await saveKeys({ model: selectedModel })
    setSavingModel(false)
    refetch()
  }

  const handleSaveEmbedding = async () => {
    if (!selectedEmbedding) return
    await saveKeys({ embedding_model: selectedEmbedding })
    refetch()
  }

  const handleSaveSystem = async () => {
    await saveConfig({
      max_concurrent_generation: maxConcurrent,
      code_execution_timeout: codeTimeout,
    })
  }

  if (loading) {
    return <LoadingCard title="Factor Mining System Configuration" />
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <TrendingUp className="h-5 w-5" />
          Factor Mining System Configuration
        </CardTitle>
        <CardDescription>
          Global system configuration, task-level configuration (factor families, evaluation thresholds, etc.) is set when creating tasks
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Info Box */}
        <div className="p-4 bg-blue-500/10 rounded-lg text-sm">
          <p className="font-medium text-blue-600 mb-2">Configuration Guide</p>
          <ul className="list-disc list-inside space-y-1 text-muted-foreground">
            <li><strong>System Configuration</strong> (this page): LLM model, Embedding model, concurrency, timeout</li>
            <li><strong>Task Configuration</strong> (when creating tasks): Factor families, evaluation thresholds, time range, dataset split</li>
          </ul>
        </div>

        {/* LLM Model Selection - 主模型 */}
        <div className="p-4 bg-muted rounded-lg">
          <h4 className="font-medium mb-3">LLM Model (主模型 - 因子代码生成)</h4>
          <p className="text-sm text-muted-foreground mb-3">
            Select the LLM model for factor code generation. Recommended: deepseek/deepseek-v3.2-speciale
          </p>
          <div className="flex gap-4 items-end">
            <div className="flex-1 space-y-2">
              <Label>Select LLM Model</Label>
              <Select
                options={[{ value: '', label: 'Select model...' }, ...modelOptions]}
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
              />
            </div>
            <Button onClick={handleSaveModel} disabled={savingModel || !selectedModel || selectedModel === keys?.model}>
              {savingModel && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
              Save
            </Button>
          </div>
          {keys?.model && (
            <p className="text-xs text-muted-foreground mt-2">
              Current: <span className="font-medium text-green-600">{keys.model}</span>
            </p>
          )}
        </div>

        {/* Embedding Model Selection */}
        <div className="p-4 bg-muted rounded-lg">
          <h4 className="font-medium mb-3">Embedding Model (Vectorization Model)</h4>
          <p className="text-sm text-muted-foreground mb-3">
            Used for vectorized representation of factor code, supports similar factor search and deduplication
          </p>
          <div className="flex gap-4 items-end">
            <div className="flex-1 space-y-2">
              <Label>Select Embedding Model</Label>
              <Select
                options={[{ value: '', label: 'Select model...' }, ...embeddingOptions]}
                value={selectedEmbedding}
                onChange={(e) => setSelectedEmbedding(e.target.value)}
              />
            </div>
            <Button onClick={handleSaveEmbedding} disabled={!selectedEmbedding || selectedEmbedding === keys?.embedding_model}>
              Save
            </Button>
          </div>
          {keys?.embedding_model && (
            <p className="text-xs text-muted-foreground mt-2">
              Current configuration: <span className="font-medium">{keys.embedding_model}</span>
            </p>
          )}
        </div>

        {/* System Settings */}
        <div className="p-4 bg-muted rounded-lg">
          <h4 className="font-medium mb-3">System Runtime Configuration</h4>
          <div className="grid grid-cols-2 gap-6">
            <div className="space-y-2">
              <div className="flex justify-between">
                <Label>Max Concurrent Generation</Label>
                <span className="text-sm font-medium">{maxConcurrent}</span>
              </div>
              <Slider
                value={maxConcurrent}
                onValueChange={(v) => setMaxConcurrent(v)}
                min={1}
                max={20}
                step={1}
              />
              <p className="text-xs text-muted-foreground">
                Maximum number of factors to generate simultaneously, affects system load
              </p>
            </div>
            <div className="space-y-2">
              <div className="flex justify-between">
                <Label>Code Execution Timeout (seconds)</Label>
                <span className="text-sm font-medium">{codeTimeout}s</span>
              </div>
              <Slider
                value={codeTimeout}
                onValueChange={(v) => setCodeTimeout(v)}
                min={10}
                max={120}
                step={5}
              />
              <p className="text-xs text-muted-foreground">
                Maximum time for executing a single factor code
              </p>
            </div>
          </div>
          <Button onClick={handleSaveSystem} disabled={saving} className="mt-4">
            {saving && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
            Save System Configuration
          </Button>
        </div>
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
