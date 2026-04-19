import { useState, useCallback } from 'react'
import { EvalForm } from '../components/EvalForm/EvalForm'
import { KpiCards } from '../components/KpiCards/KpiCards'
import { RadarChart } from '../components/RadarChart/RadarChart'
import { FieldDiffViewer } from '../components/FieldDiffViewer/FieldDiffViewer'
import { api } from '../api/client'
import type { EvalResult, EvalRequest } from '../types/api'

export function Dashboard() {
  const [result, setResult] = useState<EvalResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleSubmit = useCallback(async (req: EvalRequest) => {
    setLoading(true)
    setError(null)
    try {
      const res = await api.evaluate(req)
      setResult(res)
    } catch (err) {
      setError(`Evaluation failed: ${err instanceof Error ? err.message : String(err)}`)
    } finally {
      setLoading(false)
    }
  }, [])

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
      <div>
        <h1 style={{ margin: '0 0 4px', fontSize: 22, fontWeight: 700, color: '#f1f5f9' }}>Dashboard</h1>
        <p style={{ margin: 0, color: '#64748b', fontSize: 14 }}>
          Compare LLM-generated JSON against golden reference using 7 complementary metrics.
        </p>
      </div>

      <EvalForm onSubmit={handleSubmit} loading={loading} />

      {error && (
        <div style={{ color: '#ef4444', background: '#7f1d1d22', padding: '12px 16px', borderRadius: 8, border: '1px solid #ef444430' }}>
          {error}
        </div>
      )}

      {result && (
        <>
          <KpiCards result={result} />
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24 }}>
            <RadarChart scores={result.scores} />
            <div style={{ background: '#1e293b', borderRadius: 12, padding: 24, border: '1px solid #334155' }}>
              <h3 style={{ margin: '0 0 12px', color: '#e2e8f0', fontSize: 14, fontWeight: 600 }}>LLM Reasoning</h3>
              <div style={{ color: '#64748b', fontSize: 12, marginBottom: 8 }}>
                Model: <span style={{ color: '#94a3b8' }}>{result.verdict.model_used || 'N/A'}</span>
                {' · '}
                Available: <span style={{ color: result.verdict.llm_available ? '#22c55e' : '#ef4444' }}>
                  {result.verdict.llm_available ? 'Yes' : 'No'}
                </span>
              </div>
              <p style={{ color: '#94a3b8', fontSize: 13, lineHeight: 1.6, margin: 0 }}>
                {result.verdict.reasoning || 'No reasoning provided (Ollama not available).'}
              </p>
            </div>
          </div>
          <FieldDiffViewer diff={result.diff} />
        </>
      )}
    </div>
  )
}
