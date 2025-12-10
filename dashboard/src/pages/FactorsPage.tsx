import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'

export function FactorsPage() {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Factors</h1>
          <p className="text-muted-foreground">
            Manage and explore your factor library
          </p>
        </div>
        <Button>Generate Factor</Button>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Factor Library</CardTitle>
          <CardDescription>Browse and filter factors</CardDescription>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">
            Factor table will be displayed here...
          </p>
        </CardContent>
      </Card>
    </div>
  )
}
