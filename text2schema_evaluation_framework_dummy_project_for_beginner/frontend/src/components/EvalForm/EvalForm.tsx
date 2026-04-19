import { useState } from 'react'
import type { EvalRequest } from '../../types/api'

interface EvalFormProps {
  onSubmit: (req: EvalRequest) => void
  loading: boolean
}

const EXAMPLE_GENERATED = JSON.stringify({ name: 'Alice', age: 30, role: 'admin' }, null, 2)
const EXAMPLE_GOLDEN = JSON.stringify({ name: 'Alice', age: 30, role: 'user' }, null, 2)

export function EvalForm({ onSubmit, loading }: EvalFormProps) {
  const [generated, setGenerated] = useState(EXAMPLE_GENERATED)
  const [golden, setGolden] = useState(EXAMPLE_GOLDEN)
  const [error, setError] = useState<string | null>(null)

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setError(null)
    try {
      const genParsed = JSON.parse(generated)
      const goldParsed = JSON.parse(golden)
      onSubmit({ generated: genParsed, golden: goldParsed })
    } catch (err) {
      setError(`JSON parse error: ${err instanceof Error ? err.message : String(err)}`)
    }
  }

  const textareaStyle: React.CSSProperties = {
    width: '100%',
    height: 200,
    background: '#0f172a',
    border: '1px solid #334155',
    borderRadius: 8,
    color: '#e2e8f0',
    fontFamily: 'monospace',
    fontSize: 13,
    padding: 12,
    resize: 'vertical',
    outline: 'none',
    boxSizing: 'border-box',
  }

  return (
    <form onSubmit={handleSubmit} style={{ background: '#1e293b', borderRadius: 12, padding: 24, border: '1px solid #334155' }}>
      <h3 style={{ margin: '0 0 16px', color: '#e2e8f0', fontSize: 14, fontWeight: 600 }}>Evaluate JSON</h3>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        <div>
          <label style={{ color: '#94a3b8', fontSize: 12, fontWeight: 600, display: 'block', marginBottom: 6 }}>
            Generated JSON
          </label>
          <textarea
            style={textareaStyle}
            value={generated}
            onChange={e => setGenerated(e.target.value)}
            placeholder='{"key": "value"}'
          />
        </div>
        <div>
          <label style={{ color: '#94a3b8', fontSize: 12, fontWeight: 600, display: 'block', marginBottom: 6 }}>
            Golden Reference JSON
          </label>
          <textarea
            style={textareaStyle}
            value={golden}
            onChange={e => setGolden(e.target.value)}
            placeholder='{"key": "value"}'
          />
        </div>
      </div>

      {error && (
        <div style={{ color: '#ef4444', fontSize: 12, marginTop: 8, background: '#7f1d1d22', padding: '8px 12px', borderRadius: 6 }}>
          {error}
        </div>
      )}

      <button
        type="submit"
        disabled={loading}
        style={{
          marginTop: 16,
          background: loading ? '#334155' : '#38bdf8',
          color: loading ? '#64748b' : '#0f172a',
          border: 'none',
          borderRadius: 8,
          padding: '10px 24px',
          fontWeight: 700,
          fontSize: 14,
          cursor: loading ? 'not-allowed' : 'pointer',
        }}
      >
        {loading ? 'Evaluating…' : 'Evaluate'}
      </button>
    </form>
  )
}
