import React, { useState } from 'react'
import type { GenerateRequest } from '../api/client'

interface Props {
  onSubmit: (req: GenerateRequest) => void
  isLoading: boolean
}

const DIALECTS = [
  { id: 'postgresql', label: 'PostgreSQL' },
  { id: 'mysql', label: 'MySQL' },
  { id: 'mongodb', label: 'MongoDB' },
  { id: 'dynamodb', label: 'DynamoDB' },
]

const FORMATS = [
  { id: 'sql', label: 'SQL DDL' },
  { id: 'nosql', label: 'NoSQL Schema' },
  { id: 'all', label: 'All Formats' },
]

export default function SchemaInput({ onSubmit, isLoading }: Props) {
  const [description, setDescription] = useState('')
  const [targetDb, setTargetDb] = useState<GenerateRequest['target_db']>('postgresql')
  const [outputFormat, setOutputFormat] = useState<GenerateRequest['output_format']>('sql')

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (description.length < 10) return
    onSubmit({ description, target_db: targetDb, output_format: outputFormat, use_cache: true })
  }

  return (
    <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
      <div>
        <label style={{ display: 'block', fontSize: '0.875rem', color: '#94a3b8', marginBottom: '0.5rem' }}>
          Description
        </label>
        <textarea
          value={description}
          onChange={e => setDescription(e.target.value)}
          placeholder="Describe your database schema... (e.g., 'An e-commerce platform with users, products, orders, and reviews')"
          style={{
            width: '100%',
            minHeight: '120px',
            background: '#0f172a',
            border: '1px solid #334155',
            borderRadius: '8px',
            padding: '0.75rem',
            color: '#e2e8f0',
            fontSize: '0.875rem',
            resize: 'vertical',
            fontFamily: 'inherit',
          }}
          required
          minLength={10}
        />
        {description.length > 0 && description.length < 10 && (
          <p style={{ color: '#f87171', fontSize: '0.75rem', marginTop: '0.25rem' }}>
            Description must be at least 10 characters
          </p>
        )}
      </div>

      <div>
        <label style={{ display: 'block', fontSize: '0.875rem', color: '#94a3b8', marginBottom: '0.5rem' }}>
          Target Database
        </label>
        <select
          value={targetDb}
          onChange={e => setTargetDb(e.target.value as GenerateRequest['target_db'])}
          style={{
            width: '100%',
            background: '#0f172a',
            border: '1px solid #334155',
            borderRadius: '8px',
            padding: '0.625rem 0.75rem',
            color: '#e2e8f0',
            fontSize: '0.875rem',
          }}
        >
          {DIALECTS.map(d => (
            <option key={d.id} value={d.id}>{d.label}</option>
          ))}
        </select>
      </div>

      <div>
        <label style={{ display: 'block', fontSize: '0.875rem', color: '#94a3b8', marginBottom: '0.5rem' }}>
          Output Format
        </label>
        <select
          value={outputFormat}
          onChange={e => setOutputFormat(e.target.value as GenerateRequest['output_format'])}
          style={{
            width: '100%',
            background: '#0f172a',
            border: '1px solid #334155',
            borderRadius: '8px',
            padding: '0.625rem 0.75rem',
            color: '#e2e8f0',
            fontSize: '0.875rem',
          }}
        >
          {FORMATS.map(f => (
            <option key={f.id} value={f.id}>{f.label}</option>
          ))}
        </select>
      </div>

      <button
        type="submit"
        disabled={isLoading || description.length < 10}
        style={{
          background: isLoading || description.length < 10 ? '#374151' : '#3b82f6',
          color: isLoading || description.length < 10 ? '#9ca3af' : 'white',
          border: 'none',
          borderRadius: '8px',
          padding: '0.75rem',
          fontSize: '0.875rem',
          fontWeight: 600,
          cursor: isLoading || description.length < 10 ? 'not-allowed' : 'pointer',
          transition: 'background 0.2s',
        }}
      >
        {isLoading ? 'Generating...' : 'Generate Schema'}
      </button>
    </form>
  )
}
