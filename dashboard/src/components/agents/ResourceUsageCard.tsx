/**
 * 资源使用监控组件
 * 展示 CPU、内存、磁盘使用情况
 */

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import type { ResourceMetrics } from '@/types/agent'
import { Cpu, MemoryStick, HardDrive, Activity } from 'lucide-react'

interface ResourceUsageCardProps {
  resources: ResourceMetrics
}

function getUsageColor(percentage: number): string {
  if (percentage < 60) return 'text-emerald-500'
  if (percentage < 80) return 'text-amber-500'
  return 'text-red-500'
}

function getProgressColor(percentage: number): string {
  if (percentage < 60) return '[&>div]:bg-emerald-500'
  if (percentage < 80) return '[&>div]:bg-amber-500'
  return '[&>div]:bg-red-500'
}

export function ResourceUsageCard({ resources }: ResourceUsageCardProps) {
  const cpuPercentage = resources.cpu.usage
  const memoryPercentage = resources.memory.percentage
  const diskPercentage = resources.disk.percentage

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Activity className="h-5 w-5" />
          System Resources
        </CardTitle>
        <CardDescription>
          Real-time resource monitoring
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-6">
          {/* CPU */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Cpu className="h-4 w-4 text-muted-foreground" />
                <span className="text-sm font-medium">CPU</span>
              </div>
              <span className={`text-sm font-semibold ${getUsageColor(cpuPercentage)}`}>
                {cpuPercentage.toFixed(1)}%
              </span>
            </div>
            <Progress
              value={cpuPercentage}
              className={`h-2 ${getProgressColor(cpuPercentage)}`}
            />
            <div className="text-xs text-muted-foreground">
              {resources.cpu.cores} cores available
            </div>
          </div>

          {/* Memory */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <MemoryStick className="h-4 w-4 text-muted-foreground" />
                <span className="text-sm font-medium">Memory</span>
              </div>
              <span className={`text-sm font-semibold ${getUsageColor(memoryPercentage)}`}>
                {memoryPercentage.toFixed(1)}%
              </span>
            </div>
            <Progress
              value={memoryPercentage}
              className={`h-2 ${getProgressColor(memoryPercentage)}`}
            />
            <div className="text-xs text-muted-foreground">
              {resources.memory.used.toFixed(1)} GB / {resources.memory.total} GB
            </div>
          </div>

          {/* Disk */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <HardDrive className="h-4 w-4 text-muted-foreground" />
                <span className="text-sm font-medium">Disk</span>
              </div>
              <span className={`text-sm font-semibold ${getUsageColor(diskPercentage)}`}>
                {diskPercentage.toFixed(1)}%
              </span>
            </div>
            <Progress
              value={diskPercentage}
              className={`h-2 ${getProgressColor(diskPercentage)}`}
            />
            <div className="text-xs text-muted-foreground">
              {resources.disk.used} GB / {resources.disk.total} GB
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
