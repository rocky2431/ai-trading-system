/**
 * Prompts Management Page - 提示词模板管理页面
 *
 * 功能:
 * - 查看和编辑 LLM 提示词模板
 * - 查看提示词变更历史
 * - 管理系统运行模式配置
 */

import { useState } from 'react'
import {
  FileText,
  Edit3,
  RotateCcw,
  RefreshCw,
  History,
  Settings2,
  Shield,
  Box,
  Users,
  Zap,
  Save,
  XCircle,
  Loader2,
  CheckCircle,
  AlertTriangle,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Textarea } from '@/components/ui/textarea'
import { Switch } from '@/components/ui/switch'
import { Label } from '@/components/ui/label'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@/components/ui/accordion'
import { usePrompts } from '@/hooks/usePrompts'
import { PromptTemplate, PromptHistoryEntry } from '@/api/prompts'

export function PromptsPage() {
  const {
    templates,
    history,
    systemConfig,
    defaultConfig,
    loading,
    error,
    refresh,
    updateTemplate,
    resetTemplate,
    updateSystemConfig,
    loadHistory,
  } = usePrompts()

  const [editDialogOpen, setEditDialogOpen] = useState(false)
  const [selectedTemplate, setSelectedTemplate] = useState<PromptTemplate | null>(null)
  const [editedPrompt, setEditedPrompt] = useState('')
  const [saving, setSaving] = useState(false)

  const handleEdit = (template: PromptTemplate) => {
    setSelectedTemplate(template)
    setEditedPrompt(template.system_prompt)
    setEditDialogOpen(true)
  }

  const handleSave = async () => {
    if (!selectedTemplate) return

    setSaving(true)
    try {
      await updateTemplate(selectedTemplate.agent_id, editedPrompt)
      setEditDialogOpen(false)
    } catch (err) {
      console.error('Failed to save:', err)
    } finally {
      setSaving(false)
    }
  }

  const handleReset = async (agentId: string) => {
    setSaving(true)
    try {
      await resetTemplate(agentId)
      setEditDialogOpen(false)
    } catch (err) {
      console.error('Failed to reset:', err)
    } finally {
      setSaving(false)
    }
  }

  const handleConfigChange = async (key: string, value: boolean | number) => {
    await updateSystemConfig({ [key]: value })
  }

  const formatDate = (dateStr: string) => {
    return new Date(dateStr).toLocaleString('zh-CN')
  }

  if (loading && templates.length === 0) {
    return (
      <div className="flex h-full items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-4">
        <XCircle className="h-12 w-12 text-destructive" />
        <p className="text-lg text-muted-foreground">{error.message}</p>
        <Button onClick={refresh}>重试</Button>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* 页面标题 */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">提示词与配置管理</h1>
          <p className="text-muted-foreground">
            管理 LLM 提示词模板和系统运行配置
          </p>
        </div>
        <Button variant="outline" onClick={refresh}>
          <RefreshCw className={cn('mr-2 h-4 w-4', loading && 'animate-spin')} />
          刷新
        </Button>
      </div>

      {/* 标签页 */}
      <Tabs defaultValue="templates">
        <TabsList>
          <TabsTrigger value="templates">
            <FileText className="mr-2 h-4 w-4" />
            提示词模板
          </TabsTrigger>
          <TabsTrigger value="history">
            <History className="mr-2 h-4 w-4" />
            变更历史
          </TabsTrigger>
          <TabsTrigger value="system">
            <Settings2 className="mr-2 h-4 w-4" />
            系统配置
          </TabsTrigger>
        </TabsList>

        {/* 提示词模板 */}
        <TabsContent value="templates" className="space-y-4">
          <Accordion type="single" collapsible className="w-full">
            {templates.map((template) => (
              <AccordionItem key={template.agent_id} value={template.agent_id}>
                <AccordionTrigger className="hover:no-underline">
                  <div className="flex items-center gap-3">
                    <FileText className="h-5 w-5 text-primary" />
                    <div className="text-left">
                      <div className="flex items-center gap-2">
                        <span className="font-medium">{template.agent_name}</span>
                        <Badge variant="outline" className="text-xs">
                          v{template.version}
                        </Badge>
                        {template.is_custom && (
                          <Badge variant="secondary" className="text-xs">
                            自定义
                          </Badge>
                        )}
                      </div>
                      <p className="text-sm text-muted-foreground">
                        {template.description}
                      </p>
                    </div>
                  </div>
                </AccordionTrigger>
                <AccordionContent>
                  <div className="space-y-4 pt-2">
                    <pre className="rounded-lg bg-muted p-4 text-sm overflow-auto max-h-64 whitespace-pre-wrap">
                      {template.system_prompt}
                    </pre>
                    <div className="flex gap-2">
                      <Button size="sm" onClick={() => handleEdit(template)}>
                        <Edit3 className="mr-2 h-4 w-4" />
                        编辑
                      </Button>
                      {template.is_custom && (
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => handleReset(template.agent_id)}
                        >
                          <RotateCcw className="mr-2 h-4 w-4" />
                          重置为默认
                        </Button>
                      )}
                    </div>
                  </div>
                </AccordionContent>
              </AccordionItem>
            ))}
          </Accordion>
        </TabsContent>

        {/* 变更历史 */}
        <TabsContent value="history" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>提示词变更历史</CardTitle>
              <CardDescription>记录所有提示词模板的修改</CardDescription>
            </CardHeader>
            <CardContent>
              {history.length === 0 ? (
                <p className="text-center text-muted-foreground py-8">
                  暂无变更记录
                </p>
              ) : (
                <div className="space-y-4">
                  {history.map((entry) => (
                    <div
                      key={entry.id}
                      className="flex items-start gap-4 rounded-lg border p-4"
                    >
                      <div
                        className={cn(
                          'mt-1 rounded-full p-1.5',
                          entry.change_type === 'updated' && 'bg-blue-100 text-blue-600',
                          entry.change_type === 'created' && 'bg-green-100 text-green-600',
                          entry.change_type === 'reset' && 'bg-orange-100 text-orange-600'
                        )}
                      >
                        {entry.change_type === 'updated' && <Edit3 className="h-4 w-4" />}
                        {entry.change_type === 'created' && <CheckCircle className="h-4 w-4" />}
                        {entry.change_type === 'reset' && <RotateCcw className="h-4 w-4" />}
                      </div>
                      <div className="flex-1 space-y-1">
                        <div className="flex items-center gap-2">
                          <span className="font-medium">
                            {templates.find((t) => t.agent_id === entry.agent_id)
                              ?.agent_name || entry.agent_id}
                          </span>
                          <Badge variant="outline" className="text-xs">
                            {entry.change_type === 'updated' && '已更新'}
                            {entry.change_type === 'created' && '已创建'}
                            {entry.change_type === 'reset' && '已重置'}
                          </Badge>
                        </div>
                        <p className="text-sm text-muted-foreground">
                          {formatDate(entry.changed_at)} · {entry.changed_by}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* 系统配置 */}
        <TabsContent value="system" className="space-y-4">
          {systemConfig && (
            <div className="grid gap-4 md:grid-cols-2">
              {/* 严格模式 */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Shield className="h-5 w-5" />
                    严格模式
                  </CardTitle>
                  <CardDescription>
                    控制数据持久化和存储后端要求
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <Label>PostgreSQL 严格模式</Label>
                      <p className="text-sm text-muted-foreground">
                        禁用内存存储回退
                      </p>
                    </div>
                    <Switch
                      checked={systemConfig.strict_mode_enabled}
                      onCheckedChange={(v) => handleConfigChange('strict_mode_enabled', v)}
                    />
                  </div>
                  <div className="flex items-center justify-between">
                    <div>
                      <Label>向量库严格模式</Label>
                      <p className="text-sm text-muted-foreground">
                        要求 Qdrant 连接
                      </p>
                    </div>
                    <Switch
                      checked={systemConfig.vector_strict_mode}
                      onCheckedChange={(v) => handleConfigChange('vector_strict_mode', v)}
                    />
                  </div>
                </CardContent>
              </Card>

              {/* 沙箱配置 */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Box className="h-5 w-5" />
                    沙箱执行
                  </CardTitle>
                  <CardDescription>
                    控制 LLM 生成代码的执行环境
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <Label>启用沙箱</Label>
                      <p className="text-sm text-muted-foreground">
                        在隔离环境执行代码
                      </p>
                    </div>
                    <Switch
                      checked={systemConfig.sandbox_enabled}
                      onCheckedChange={(v) => handleConfigChange('sandbox_enabled', v)}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label>执行超时（秒）</Label>
                    <Input
                      type="number"
                      value={systemConfig.sandbox_timeout_seconds}
                      onChange={(e) =>
                        handleConfigChange('sandbox_timeout_seconds', parseInt(e.target.value))
                      }
                      min={10}
                      max={600}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label>内存限制（MB）</Label>
                    <Input
                      type="number"
                      value={systemConfig.sandbox_memory_limit_mb}
                      onChange={(e) =>
                        handleConfigChange('sandbox_memory_limit_mb', parseInt(e.target.value))
                      }
                      min={128}
                      max={4096}
                    />
                  </div>
                  <div className="flex items-center justify-between">
                    <div>
                      <Label>允许网络访问</Label>
                      <p className="text-sm text-muted-foreground">
                        沙箱内网络权限
                      </p>
                    </div>
                    <Switch
                      checked={systemConfig.sandbox_network_allowed}
                      onCheckedChange={(v) => handleConfigChange('sandbox_network_allowed', v)}
                    />
                  </div>
                </CardContent>
              </Card>

              {/* 人工审核 */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Users className="h-5 w-5" />
                    人工审核
                  </CardTitle>
                  <CardDescription>
                    控制代码审核流程
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <Label>启用人工审核</Label>
                      <p className="text-sm text-muted-foreground">
                        生成代码需人工批准
                      </p>
                    </div>
                    <Switch
                      checked={systemConfig.human_review_enabled}
                      onCheckedChange={(v) => handleConfigChange('human_review_enabled', v)}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label>自动拒绝超时（秒）</Label>
                    <Input
                      type="number"
                      value={systemConfig.auto_reject_timeout_seconds}
                      onChange={(e) =>
                        handleConfigChange('auto_reject_timeout_seconds', parseInt(e.target.value))
                      }
                      min={300}
                      max={86400}
                    />
                  </div>
                </CardContent>
              </Card>

              {/* 功能开关 */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Zap className="h-5 w-5" />
                    功能开关
                  </CardTitle>
                  <CardDescription>
                    控制可选功能的启用状态
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <Label>ML 信号生成</Label>
                      <p className="text-sm text-muted-foreground">
                        使用 LightGBM 生成信号
                      </p>
                    </div>
                    <Switch
                      checked={systemConfig.ml_signal_enabled}
                      onCheckedChange={(v) => handleConfigChange('ml_signal_enabled', v)}
                    />
                  </div>
                  <div className="flex items-center justify-between">
                    <div>
                      <Label>工具上下文</Label>
                      <p className="text-sm text-muted-foreground">
                        Agent 工具调用
                      </p>
                    </div>
                    <Switch
                      checked={systemConfig.tool_context_enabled}
                      onCheckedChange={(v) => handleConfigChange('tool_context_enabled', v)}
                    />
                  </div>
                  <div className="flex items-center justify-between">
                    <div>
                      <Label>Pipeline 检查点</Label>
                      <p className="text-sm text-muted-foreground">
                        启用断点续传
                      </p>
                    </div>
                    <Switch
                      checked={systemConfig.checkpoint_enabled}
                      onCheckedChange={(v) => handleConfigChange('checkpoint_enabled', v)}
                    />
                  </div>
                </CardContent>
              </Card>
            </div>
          )}
        </TabsContent>
      </Tabs>

      {/* 编辑对话框 */}
      <Dialog open={editDialogOpen} onOpenChange={setEditDialogOpen}>
        <DialogContent className="max-w-3xl max-h-[90vh] overflow-auto">
          <DialogHeader>
            <DialogTitle>编辑提示词模板</DialogTitle>
            <DialogDescription>
              {selectedTemplate?.agent_name} - {selectedTemplate?.description}
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4">
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Badge variant="outline">ID: {selectedTemplate?.prompt_id}</Badge>
              <Badge variant="outline">版本: v{selectedTemplate?.version}</Badge>
            </div>

            <Textarea
              value={editedPrompt}
              onChange={(e) => setEditedPrompt(e.target.value)}
              className="min-h-[400px] font-mono text-sm"
              placeholder="输入系统提示词..."
            />

            {selectedTemplate?.is_custom && (
              <div className="flex items-center gap-2 text-sm text-amber-600">
                <AlertTriangle className="h-4 w-4" />
                <span>当前使用自定义提示词，点击"重置为默认"可恢复系统默认值</span>
              </div>
            )}
          </div>

          <DialogFooter className="flex-col sm:flex-row gap-2">
            {selectedTemplate?.is_custom && (
              <Button
                variant="outline"
                onClick={() => handleReset(selectedTemplate.agent_id)}
                disabled={saving}
              >
                <RotateCcw className="mr-2 h-4 w-4" />
                重置为默认
              </Button>
            )}
            <div className="flex gap-2">
              <Button variant="outline" onClick={() => setEditDialogOpen(false)}>
                取消
              </Button>
              <Button onClick={handleSave} disabled={saving}>
                {saving ? (
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                ) : (
                  <Save className="mr-2 h-4 w-4" />
                )}
                保存
              </Button>
            </div>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}
