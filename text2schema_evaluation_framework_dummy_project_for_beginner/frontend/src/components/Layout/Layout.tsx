import { ReactNode } from 'react'
import { Link, useLocation } from 'react-router-dom'

export function Layout({ children }: { children: ReactNode }) {
  const { pathname } = useLocation()

  return (
    <div style={{ minHeight: '100vh', background: '#0f172a', color: '#e2e8f0' }}>
      <nav style={{
        background: '#1e293b',
        borderBottom: '1px solid #334155',
        padding: '0 24px',
        display: 'flex',
        alignItems: 'center',
        height: 56,
        gap: 32,
      }}>
        <span style={{ fontWeight: 700, fontSize: 18, color: '#38bdf8' }}>SchemaEval</span>
        <Link
          to="/"
          style={{ color: pathname === '/' ? '#38bdf8' : '#94a3b8', textDecoration: 'none', fontWeight: 500 }}
        >
          Dashboard
        </Link>
        <Link
          to="/history"
          style={{ color: pathname === '/history' ? '#38bdf8' : '#94a3b8', textDecoration: 'none', fontWeight: 500 }}
        >
          History
        </Link>
      </nav>
      <main style={{ padding: '24px 32px', maxWidth: 1400, margin: '0 auto' }}>
        {children}
      </main>
    </div>
  )
}
