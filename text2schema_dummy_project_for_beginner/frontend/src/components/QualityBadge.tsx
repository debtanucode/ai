import React from 'react'
import type { QualityScore } from '../api/client'

interface ScoreBarProps {
  label: string
  value: number
}

function ScoreBar({ label, value }: ScoreBarProps) {
  const color = value >= 0.8 ? '#22c55e' : value >= 0.6 ? '#eab308' : '#ef4444'
  const pct = Math.round(value * 100)
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.75rem' }}>
      <span style={{ width: '90px', color: '#94a3b8', flexShrink: 0 }}>{label}</span>
      <div style={{ flex: 1, background: '#1e293b', borderRadius: '4px', height: '6px', overflow: 'hidden' }}>
        <div style={{ width: `${pct}%`, height: '100%', background: color, borderRadius: '4px', transition: 'width 0.5s' }} />
      </div>
      <span style={{ width: '36px', textAlign: 'right', color }}>{pct}%</span>
    </div>
  )
}

interface Props {
  quality: QualityScore
  retryCount: number
  processingTimeMs: number
  cached: boolean
}

export default function QualityBadge({ quality, retryCount, processingTimeMs, cached }: Props) {
  const compositeColor = quality.composite >= 0.8 ? '#22c55e' : quality.composite >= 0.6 ? '#eab308' : '#ef4444'
  return (
    <div style={{
      background: '#0f172a',
      border: `1px solid ${quality.passed ? '#166534' : '#7f1d1d'}`,
      borderRadius: '8px',
      padding: '1rem',
      display: 'flex',
      flexDirection: 'column',
      gap: '0.75rem',
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <span style={{ fontSize: '1.25rem', fontWeight: 700, color: compositeColor }}>
            {Math.round(quality.composite * 100)}%
          </span>
          <span style={{
            fontSize: '0.75rem',
            padding: '0.125rem 0.5rem',
            borderRadius: '9999px',
            background: quality.passed ? '#14532d' : '#450a0a',
            color: quality.passed ? '#86efac' : '#fca5a5',
          }}>
            {quality.passed ? 'PASSED' : 'FAILED'}
          </span>
        </div>
        <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
          {cached && (
            <span style={{ fontSize: '0.7rem', padding: '0.125rem 0.375rem', background: '#1e3a5f', color: '#93c5fd', borderRadius: '4px' }}>
              CACHED
            </span>
          )}
          {retryCount > 0 && (
            <span style={{ fontSize: '0.7rem', padding: '0.125rem 0.375rem', background: '#451a03', color: '#fdba74', borderRadius: '4px' }}>
              {retryCount} retries
            </span>
          )}
          <span style={{ fontSize: '0.7rem', color: '#64748b' }}>{(processingTimeMs / 1000).toFixed(1)}s</span>
        </div>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.375rem' }}>
        <ScoreBar label="Syntax" value={quality.syntax} />
        <ScoreBar label="Integrity" value={quality.integrity} />
        <ScoreBar label="Naming" value={quality.naming} />
        <ScoreBar label="Completeness" value={quality.completeness} />
      </div>
    </div>
  )
}
