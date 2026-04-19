import React, { useState } from 'react'
import './App.css'
import SchemaInput from './components/SchemaInput'
import QualityBadge from './components/QualityBadge'
import OutputPanel from './components/OutputPanel'
import ERDViewer from './components/ERDViewer'
import { generateSchema, type GenerateRequest, type GenerateResponse } from './api/client'

type ActiveView = 'erd' | 'code'

export default function App() {
  const [response, setResponse] = useState<GenerateResponse | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [activeView, setActiveView] = useState<ActiveView>('erd')

  const handleSubmit = async (req: GenerateRequest) => {
    setIsLoading(true)
    setError(null)
    setResponse(null)
    try {
      const res = await generateSchema(req)
      setResponse(res)
    } catch (err: any) {
      const msg = err?.response?.data?.detail ?? err?.message ?? 'Unknown error'
      setError(typeof msg === 'string' ? msg : JSON.stringify(msg))
    } finally {
      setIsLoading(false)
    }
  }

  const erdData = response?.outputs?.erd
    ? (() => { try { return JSON.parse(response.outputs.erd) } catch { return null } })()
    : null

  const codeOutputs: Record<string, string> = {}
  if (response?.outputs) {
    Object.entries(response.outputs).forEach(([k, v]) => {
      if (k !== 'erd') codeOutputs[k] = v
    })
  }

  return (
    <div className="app">
      <header className="header">
        <h1>Text2Schema</h1>
        <span className="subtitle">Natural language → Database schema</span>
      </header>

      <div className="main-content">
        <aside className="left-panel">
          <SchemaInput onSubmit={handleSubmit} isLoading={isLoading} />
          {response?.quality && (
            <QualityBadge
              quality={response.quality}
              retryCount={response.retry_count}
              processingTimeMs={response.processing_time_ms}
              cached={response.cached}
            />
          )}
        </aside>

        <main className="right-panel">
          {error && (
            <div className="error-message">{error}</div>
          )}

          {response && (
            <>
              <div className="view-toggle">
                <button
                  className={`view-btn ${activeView === 'erd' ? 'active' : ''}`}
                  onClick={() => setActiveView('erd')}
                >
                  ERD Diagram
                </button>
                <button
                  className={`view-btn ${activeView === 'code' ? 'active' : ''}`}
                  onClick={() => setActiveView('code')}
                >
                  Schema Code
                </button>
              </div>

              {activeView === 'erd' && erdData && (
                <ERDViewer erdData={erdData} />
              )}
              {activeView === 'code' && (
                <OutputPanel outputs={codeOutputs} />
              )}
            </>
          )}

          {!response && !error && !isLoading && (
            <div style={{
              flex: 1,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: '#475569',
              flexDirection: 'column',
              gap: '0.5rem',
            }}>
              <div style={{ fontSize: '3rem' }}>⬡</div>
              <div style={{ fontSize: '1.125rem' }}>Enter a description to generate a schema</div>
            </div>
          )}

          {isLoading && (
            <div style={{
              flex: 1,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: '#94a3b8',
              flexDirection: 'column',
              gap: '1rem',
            }}>
              <div style={{
                width: '40px',
                height: '40px',
                border: '3px solid #334155',
                borderTop: '3px solid #3b82f6',
                borderRadius: '50%',
                animation: 'spin 1s linear infinite',
              }} />
              <div>Generating schema...</div>
              <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
            </div>
          )}
        </main>
      </div>
    </div>
  )
}
