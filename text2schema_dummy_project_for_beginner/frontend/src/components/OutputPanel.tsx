import React, { useState } from 'react'
import CodeMirror from '@uiw/react-codemirror'
import { sql } from '@codemirror/lang-sql'
import { oneDark } from '@codemirror/theme-one-dark'

interface Props {
  outputs: Record<string, string>
}

export default function OutputPanel({ outputs }: Props) {
  const tabs = Object.keys(outputs).filter(k => k !== 'erd')
  const [activeTab, setActiveTab] = useState(tabs[0] || '')
  const [copied, setCopied] = useState(false)

  if (tabs.length === 0) {
    return (
      <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#475569' }}>
        No output available
      </div>
    )
  }

  const currentContent = outputs[activeTab] || ''

  const handleCopy = async () => {
    await navigator.clipboard.writeText(currentContent)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      <div style={{
        display: 'flex',
        gap: '0.5rem',
        padding: '0.75rem 1rem',
        borderBottom: '1px solid #334155',
        background: '#1e293b',
        alignItems: 'center',
      }}>
        {tabs.map(tab => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            style={{
              padding: '0.375rem 0.75rem',
              borderRadius: '6px',
              border: 'none',
              background: activeTab === tab ? '#3b82f6' : 'transparent',
              color: activeTab === tab ? 'white' : '#94a3b8',
              cursor: 'pointer',
              fontSize: '0.8rem',
              fontWeight: activeTab === tab ? 600 : 400,
              textTransform: 'uppercase',
              letterSpacing: '0.05em',
            }}
          >
            {tab}
          </button>
        ))}
        <div style={{ flex: 1 }} />
        <button
          onClick={handleCopy}
          style={{
            padding: '0.375rem 0.75rem',
            borderRadius: '6px',
            border: '1px solid #334155',
            background: 'transparent',
            color: copied ? '#86efac' : '#94a3b8',
            cursor: 'pointer',
            fontSize: '0.8rem',
          }}
        >
          {copied ? 'Copied!' : 'Copy'}
        </button>
      </div>
      <div style={{ flex: 1, overflow: 'auto' }}>
        <CodeMirror
          value={currentContent}
          extensions={[sql()]}
          theme={oneDark}
          editable={false}
          basicSetup={{ lineNumbers: true, foldGutter: true }}
          style={{ fontSize: '13px', height: '100%' }}
        />
      </div>
    </div>
  )
}
