/**
 * Agent 配置对话框组件
 * 用于查看和编辑 Agent 的提示词和配置
 */

import { useState, useEffect } from 'react'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogBody,
  DialogFooter,
} from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { Switch } from '@/components/ui/switch'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Badge } from '@/components/ui/badge'
import { systemApi, type AgentConfigResponse, type AgentType } from '@/api/system'
import { Settings, MessageSquare, Code, Loader2, Save, RefreshCw } from 'lucide-react'

interface AgentConfigDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  agentType: AgentType
}

const typeLabels: Record<AgentType, string> = {
  factor_generation: 'Factor Generator',
  evaluation: 'Factor Evaluator',
  strategy: 'Strategy Builder',
  backtest: 'Backtester',
}

export function AgentConfigDialog({
  open,
  onOpenChange,
  agentType,
}: AgentConfigDialogProps) {
  const [config, setConfig] = useState<AgentConfigResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Form state
  const [activeTab, setActiveTab] = useState('prompts')
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [systemPrompt, setSystemPrompt] = useState('')
  const [userPromptTemplate, setUserPromptTemplate] = useState('')
  const [examples, setExamples] = useState('')
  const [isEnabled, setIsEnabled] = useState(true)
  const [configJson, setConfigJson] = useState('')

  // Load config when dialog opens
  useEffect(() => {
    if (open && agentType) {
      loadConfig()
    }
  }, [open, agentType])

  const loadConfig = async () => {
    setLoading(true)
    setError(null)

    try {
      const response = await systemApi.getAgentConfig(agentType)
      setConfig(response)
      setName(response.name)
      setDescription(response.description || '')
      setSystemPrompt(response.system_prompt || '')
      setUserPromptTemplate(response.user_prompt_template || '')
      setExamples(response.examples || '')
      setIsEnabled(response.is_enabled)
      setConfigJson(response.config ? JSON.stringify(response.config, null, 2) : '{}')
    } catch (err) {
      console.error('Failed to load agent config:', err)
      setError('Failed to load configuration. Click "Initialize" to create default config.')
    } finally {
      setLoading(false)
    }
  }

  const handleInitialize = async () => {
    setLoading(true)
    setError(null)

    try {
      const response = await systemApi.initAgentConfigs()
      if (response.success) {
        await loadConfig()
      } else {
        setError(response.message)
      }
    } catch (err) {
      console.error('Failed to initialize agent configs:', err)
      setError('Failed to initialize configurations')
    } finally {
      setLoading(false)
    }
  }

  const handleSave = async () => {
    setSaving(true)
    setError(null)

    try {
      // Parse config JSON
      let parsedConfig: Record<string, unknown> | undefined
      try {
        parsedConfig = configJson ? JSON.parse(configJson) : undefined
      } catch {
        setError('Invalid JSON in config settings')
        setSaving(false)
        return
      }

      const response = await systemApi.updateAgentConfig(agentType, {
        name,
        description: description || undefined,
        system_prompt: systemPrompt || undefined,
        user_prompt_template: userPromptTemplate || undefined,
        examples: examples || undefined,
        config: parsedConfig,
        is_enabled: isEnabled,
      })

      if (response.success && response.config) {
        setConfig(response.config)
      } else {
        setError(response.message)
      }
    } catch (err) {
      console.error('Failed to save agent config:', err)
      setError('Failed to save configuration')
    } finally {
      setSaving(false)
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-4xl">
        <DialogHeader onClose={() => onOpenChange(false)}>
          <div className="flex items-center gap-3">
            <Settings className="h-5 w-5 text-blue-500" />
            <DialogTitle>{typeLabels[agentType]} Configuration</DialogTitle>
            {config && (
              <Badge variant={isEnabled ? 'success' : 'secondary'}>
                {isEnabled ? 'Enabled' : 'Disabled'}
              </Badge>
            )}
          </div>
        </DialogHeader>

        <DialogBody>
          {loading ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="h-8 w-8 animate-spin text-blue-500" />
            </div>
          ) : error && !config ? (
            <div className="text-center py-12">
              <p className="text-red-500 mb-4">{error}</p>
              <Button onClick={handleInitialize} variant="outline">
                <RefreshCw className="h-4 w-4 mr-2" />
                Initialize Default Configs
              </Button>
            </div>
          ) : (
            <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
              <TabsList className="grid w-full grid-cols-3">
                <TabsTrigger value="prompts">
                  <MessageSquare className="h-4 w-4 mr-2" />
                  Prompts
                </TabsTrigger>
                <TabsTrigger value="examples">
                  <Code className="h-4 w-4 mr-2" />
                  Examples
                </TabsTrigger>
                <TabsTrigger value="settings">
                  <Settings className="h-4 w-4 mr-2" />
                  Settings
                </TabsTrigger>
              </TabsList>

              {/* Prompts Tab */}
              <TabsContent value="prompts" className="space-y-4 mt-4">
                <div className="space-y-2">
                  <Label htmlFor="system-prompt">System Prompt</Label>
                  <Textarea
                    id="system-prompt"
                    value={systemPrompt}
                    onChange={(e) => setSystemPrompt(e.target.value)}
                    placeholder="Enter system prompt..."
                    className="min-h-[200px] font-mono text-sm"
                  />
                  <p className="text-xs text-gray-500">
                    Defines the agent's role and behavior guidelines
                  </p>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="user-prompt">User Prompt Template</Label>
                  <Textarea
                    id="user-prompt"
                    value={userPromptTemplate}
                    onChange={(e) => setUserPromptTemplate(e.target.value)}
                    placeholder="Enter user prompt template..."
                    className="min-h-[150px] font-mono text-sm"
                  />
                  <p className="text-xs text-gray-500">
                    Template for user requests. Use {'{variable}'} for placeholders.
                  </p>
                </div>
              </TabsContent>

              {/* Examples Tab */}
              <TabsContent value="examples" className="space-y-4 mt-4">
                <div className="space-y-2">
                  <Label htmlFor="examples">Examples</Label>
                  <Textarea
                    id="examples"
                    value={examples}
                    onChange={(e) => setExamples(e.target.value)}
                    placeholder="Enter examples..."
                    className="min-h-[300px] font-mono text-sm"
                  />
                  <p className="text-xs text-gray-500">
                    Provide example outputs to guide the agent's responses
                  </p>
                </div>
              </TabsContent>

              {/* Settings Tab */}
              <TabsContent value="settings" className="space-y-4 mt-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="name">Name</Label>
                    <Input
                      id="name"
                      value={name}
                      onChange={(e) => setName(e.target.value)}
                      placeholder="Agent name"
                    />
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="description">Description</Label>
                    <Input
                      id="description"
                      value={description}
                      onChange={(e) => setDescription(e.target.value)}
                      placeholder="Brief description"
                    />
                  </div>
                </div>

                <div className="flex items-center justify-between p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
                  <div>
                    <Label>Enabled</Label>
                    <p className="text-xs text-gray-500">
                      Enable or disable this agent
                    </p>
                  </div>
                  <Switch
                    checked={isEnabled}
                    onCheckedChange={setIsEnabled}
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="config-json">Advanced Config (JSON)</Label>
                  <Textarea
                    id="config-json"
                    value={configJson}
                    onChange={(e) => setConfigJson(e.target.value)}
                    placeholder="{}"
                    className="min-h-[150px] font-mono text-sm"
                  />
                  <p className="text-xs text-gray-500">
                    Additional configuration in JSON format (e.g., thresholds, limits)
                  </p>
                </div>

                {config && (
                  <div className="text-xs text-gray-500 space-y-1">
                    <p>Created: {config.created_at ? new Date(config.created_at).toLocaleString() : 'N/A'}</p>
                    <p>Updated: {config.updated_at ? new Date(config.updated_at).toLocaleString() : 'N/A'}</p>
                  </div>
                )}
              </TabsContent>
            </Tabs>
          )}

          {error && config && (
            <p className="text-red-500 text-sm mt-4">{error}</p>
          )}
        </DialogBody>

        {config && (
          <DialogFooter>
            <Button variant="outline" onClick={() => onOpenChange(false)}>
              Cancel
            </Button>
            <Button onClick={handleSave} disabled={saving}>
              {saving ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Saving...
                </>
              ) : (
                <>
                  <Save className="h-4 w-4 mr-2" />
                  Save Changes
                </>
              )}
            </Button>
          </DialogFooter>
        )}
      </DialogContent>
    </Dialog>
  )
}
