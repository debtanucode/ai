import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts'
import type { EvalResult } from '../../types/api'

interface TrendChartProps {
  results: EvalResult[]
}

export function TrendChart({ results }: TrendChartProps) {
  const data = [...results]
    .sort((a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime())
    .map((r, i) => ({
      run: i + 1,
      composite: parseFloat((r.composite_score * 100).toFixed(1)),
      field_diff: parseFloat((r.scores.field_diff * 100).toFixed(1)),
      jaccard: parseFloat((r.scores.jaccard * 100).toFixed(1)),
    }))

  if (data.length === 0) {
    return (
      <div style={{ background: '#1e293b', borderRadius: 12, padding: 24, border: '1px solid #334155', color: '#64748b', textAlign: 'center' }}>
        No run history yet. Submit evaluations to see trends.
      </div>
    )
  }

  return (
    <div style={{ background: '#1e293b', borderRadius: 12, padding: 24, border: '1px solid #334155' }}>
      <h3 style={{ margin: '0 0 16px', color: '#e2e8f0', fontSize: 14, fontWeight: 600 }}>
        Score Trend Over Runs
      </h3>
      <ResponsiveContainer width="100%" height={240}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis dataKey="run" tick={{ fill: '#94a3b8', fontSize: 12 }} label={{ value: 'Run #', position: 'insideBottom', fill: '#64748b', dy: 8 }} />
          <YAxis domain={[0, 100]} tick={{ fill: '#94a3b8', fontSize: 12 }} unit="%" />
          <Tooltip
            contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8 }}
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            formatter={(v: any) => [v != null ? `${Number(v).toFixed(1)}%` : '—'] as any}
          />
          <Legend wrapperStyle={{ color: '#94a3b8', fontSize: 12 }} />
          <Line type="monotone" dataKey="composite" stroke="#38bdf8" strokeWidth={2} dot={false} name="Composite" />
          <Line type="monotone" dataKey="field_diff" stroke="#34d399" strokeWidth={2} dot={false} name="Field Diff" />
          <Line type="monotone" dataKey="jaccard" stroke="#a78bfa" strokeWidth={2} dot={false} name="Jaccard" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
