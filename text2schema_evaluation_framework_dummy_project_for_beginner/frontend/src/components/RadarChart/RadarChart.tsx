import {
  Radar,
  RadarChart as ReRadarChart,
  PolarGrid,
  PolarAngleAxis,
  ResponsiveContainer,
  Tooltip,
} from 'recharts'
import type { MetricScores } from '../../types/api'

interface RadarChartProps {
  scores: MetricScores
}

const METRIC_LABELS: Record<keyof MetricScores, string> = {
  jaccard: 'Jaccard',
  cosine: 'Cosine',
  levenshtein: 'Levenshtein',
  bleu: 'BLEU',
  rouge: 'ROUGE-L',
  field_diff: 'Field Diff',
  llm_judge: 'LLM Judge',
}

export function RadarChart({ scores }: RadarChartProps) {
  const data = (Object.keys(METRIC_LABELS) as (keyof MetricScores)[]).map(key => ({
    metric: METRIC_LABELS[key],
    score: parseFloat((scores[key] * 100).toFixed(1)),
  }))

  return (
    <div style={{ background: '#1e293b', borderRadius: 12, padding: 24, border: '1px solid #334155' }}>
      <h3 style={{ margin: '0 0 16px', color: '#e2e8f0', fontSize: 14, fontWeight: 600 }}>
        Metric Scores (7-axis)
      </h3>
      <ResponsiveContainer width="100%" height={300}>
        <ReRadarChart data={data}>
          <PolarGrid stroke="#334155" />
          <PolarAngleAxis dataKey="metric" tick={{ fill: '#94a3b8', fontSize: 12 }} />
          <Radar
            name="Score"
            dataKey="score"
            stroke="#38bdf8"
            fill="#38bdf8"
            fillOpacity={0.25}
          />
          <Tooltip
            contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8 }}
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            formatter={(v: any) => [v != null ? `${Number(v).toFixed(1)}%` : '—', 'Score'] as any}
          />
        </ReRadarChart>
      </ResponsiveContainer>
    </div>
  )
}
