import { useEffect, useState } from 'react'
import { TrendChart } from '../components/TrendChart/TrendChart'
import { api } from '../api/client'
import type { EvalResult } from '../types/api'

export function History() {
  const [results, setResults] = useState<EvalResult[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api.listResults(100)
      .then(setResults)
      .catch(err => setError(String(err)))
      .finally(() => setLoading(false))
  }, [])

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
      <div>
        <h1 style={{ margin: '0 0 4px', fontSize: 22, fontWeight: 700, color: '#f1f5f9' }}>History</h1>
        <p style={{ margin: 0, color: '#64748b', fontSize: 14 }}>Past evaluation runs and score trends.</p>
      </div>

      {loading && <div style={{ color: '#64748b' }}>Loading…</div>}
      {error && <div style={{ color: '#ef4444' }}>{error}</div>}

      {!loading && <TrendChart results={results} />}

      {!loading && results.length > 0 && (
        <div style={{ background: '#1e293b', borderRadius: 12, padding: 24, border: '1px solid #334155' }}>
          <h3 style={{ margin: '0 0 16px', color: '#e2e8f0', fontSize: 14, fontWeight: 600 }}>Run Log</h3>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ borderBottom: '1px solid #334155' }}>
                {['Run ID', 'Score', 'Status', 'Fields', 'Created'].map(h => (
                  <th key={h} style={{ textAlign: 'left', padding: '8px 12px', color: '#64748b', fontWeight: 600 }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {results.map(r => (
                <tr key={r.run_id} style={{ borderBottom: '1px solid #0f172a' }}>
                  <td style={{ padding: '8px 12px', color: '#94a3b8', fontFamily: 'monospace', fontSize: 11 }}>
                    {r.run_id.slice(0, 12)}…
                  </td>
                  <td style={{ padding: '8px 12px', color: '#38bdf8', fontWeight: 600 }}>
                    {(r.composite_score * 100).toFixed(1)}%
                  </td>
                  <td style={{ padding: '8px 12px' }}>
                    <span style={{
                      color: r.passed ? '#22c55e' : '#ef4444',
                      background: r.passed ? '#14532d22' : '#7f1d1d22',
                      padding: '2px 8px',
                      borderRadius: 4,
                      fontWeight: 600,
                      fontSize: 11,
                    }}>
                      {r.passed ? 'PASS' : 'FAIL'}
                    </span>
                  </td>
                  <td style={{ padding: '8px 12px', color: '#94a3b8' }}>
                    {r.diff.matched}/{r.diff.total_fields}
                  </td>
                  <td style={{ padding: '8px 12px', color: '#64748b', fontSize: 11 }}>
                    {new Date(r.created_at).toLocaleString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {!loading && results.length === 0 && (
        <div style={{ color: '#64748b', textAlign: 'center', padding: 48 }}>
          No evaluation runs yet. Use the Dashboard to run evaluations.
        </div>
      )}
    </div>
  )
}
