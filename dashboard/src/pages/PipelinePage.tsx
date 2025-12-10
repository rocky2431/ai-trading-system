import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'

export function PipelinePage() {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Pipeline</h1>
          <p className="text-muted-foreground">
            Run and monitor factor evaluation pipelines
          </p>
        </div>
        <Button>Run Pipeline</Button>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Pipeline Runs</CardTitle>
          <CardDescription>Active and completed pipeline runs</CardDescription>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">
            Pipeline list will be displayed here...
          </p>
        </CardContent>
      </Card>
    </div>
  )
}
