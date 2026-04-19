import type { DiffResult, FieldStatus } from '../../types/api'

interface FieldDiffViewerProps {
  diff: DiffResult
}

const STATUS_COLOR: Record<FieldStatus, string> = {
  matched: '#22c55e',
  missing: '#ef4444',
  extra: '#f59e0b',
  mismatch: '#f97316',
}

const STATUS_BG: Record<FieldStatus, string> = {
  matched: '#14532d22',
  missing: '#7f1d1d22',
  extra: '#78350f22',
  mismatch: '#7c2d1222',
}

function val(v: unknown): string {
  if (v === undefined || v === null) return '—'
  return typeof v === 'object' ? JSON.stringify(v) : String(v)
}

export function FieldDiffViewer({ diff }: FieldDiffViewerProps) {
  return (
    <div style={{ background: '#1e293b', borderRadius: 12, padding: 24, border: '1px solid #334155' }}>
      <h3 style={{ margin: '0 0 4px', color: '#e2e8f0', fontSize: 14, fontWeight: 600 }}>
        Field-Level Diff
      </h3>
      <div style={{ color: '#64748b', fontSize: 12, marginBottom: 16 }}>
        {diff.matched} matched · {diff.missing} missing · {diff.extra} extra · {diff.mismatched} mismatch
      </div>

      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ borderBottom: '1px solid #334155' }}>
              {(['Path', 'Status', 'Golden Value', 'Generated Value'] as const).map(h => (
                <th key={h} style={{ textAlign: 'left', padding: '8px 12px', color: '#64748b', fontWeight: 600 }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {diff.fields.map((f, i) => (
              <tr
                key={i}
                style={{ background: STATUS_BG[f.status], borderBottom: '1px solid #1e293b' }}
              >
                <td style={{ padding: '6px 12px', color: '#e2e8f0', fontFamily: 'monospace' }}>{f.path}</td>
                <td style={{ padding: '6px 12px' }}>
                  <span style={{
                    color: STATUS_COLOR[f.status],
                    background: `${STATUS_COLOR[f.status]}20`,
                    padding: '2px 8px',
                    borderRadius: 4,
                    fontWeight: 600,
                    fontSize: 11,
                    textTransform: 'uppercase',
                  }}>
                    {f.status}
                  </span>
                </td>
                <td style={{ padding: '6px 12px', color: '#94a3b8', fontFamily: 'monospace', maxWidth: 220, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {val(f.golden_value)}
                </td>
                <td style={{ padding: '6px 12px', color: '#94a3b8', fontFamily: 'monospace', maxWidth: 220, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {val(f.generated_value)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
