import type { EvalResult } from '../../types/api'

interface KpiCardsProps {
  result: EvalResult
}

function KpiCard({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div style={{
      background: '#1e293b',
      border: `1px solid ${color}40`,
      borderRadius: 12,
      padding: '20px 24px',
      flex: 1,
      minWidth: 160,
    }}>
      <div style={{ color: '#94a3b8', fontSize: 12, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
        {label}
      </div>
      <div style={{ color, fontSize: 28, fontWeight: 700, marginTop: 8 }}>{value}</div>
    </div>
  )
}

export function KpiCards({ result }: KpiCardsProps) {
  const pct = (n: number) => `${(n * 100).toFixed(1)}%`
  const passColor = result.passed ? '#22c55e' : '#ef4444'

  return (
    <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
      <KpiCard label="Composite Score" value={pct(result.composite_score)} color="#38bdf8" />
      <KpiCard label="Status" value={result.passed ? 'PASS' : 'FAIL'} color={passColor} />
      <KpiCard label="Fields Matched" value={`${result.diff.matched} / ${result.diff.total_fields}`} color="#a78bfa" />
      <KpiCard label="LLM Confidence" value={result.verdict.llm_available ? pct(result.verdict.confidence) : 'N/A'} color="#fb923c" />
      <KpiCard label="Field Diff" value={pct(result.scores.field_diff)} color="#34d399" />
    </div>
  )
}
