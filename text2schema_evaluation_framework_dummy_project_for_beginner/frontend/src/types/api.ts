// TypeScript mirrors of all Pydantic models

export type FieldStatus = 'matched' | 'missing' | 'extra' | 'mismatch'

export interface FieldDiff {
  path: string
  status: FieldStatus
  golden_value?: unknown
  generated_value?: unknown
}

export interface DiffResult {
  fields: FieldDiff[]
  total_fields: number
  matched: number
  missing: number
  extra: number
  mismatched: number
}

export interface SemanticVerdict {
  llm_available: boolean
  score: number
  confidence: number
  reasoning: string
  model_used: string
}

export interface MetricScores {
  jaccard: number
  cosine: number
  levenshtein: number
  bleu: number
  rouge: number
  field_diff: number
  llm_judge: number
}

export interface EvalResult {
  run_id: string
  composite_score: number
  passed: boolean
  scores: MetricScores
  diff: DiffResult
  verdict: SemanticVerdict
  generated: Record<string, unknown>
  golden: Record<string, unknown>
  tags: string[]
  created_at: string
}

export interface BatchResult {
  results: EvalResult[]
  total: number
  passed: number
  failed: number
  avg_composite_score: number
}

export interface MetricConfig {
  jaccard?: number
  cosine?: number
  levenshtein?: number
  bleu?: number
  rouge?: number
  field_diff?: number
  llm_judge?: number
}

export interface EvalRequest {
  generated: Record<string, unknown>
  golden: Record<string, unknown>
  metric_config?: MetricConfig
  pass_threshold?: number
  run_id?: string
  tags?: string[]
}

export interface HealthStatus {
  status: string
  ollama: {
    available: boolean
    detail: string
    url: string
  }
  model: string
}

// WebSocket message types
export type WsMessage =
  | { type: 'started'; run_id?: string }
  | { type: 'result'; data: EvalResult }
  | { type: 'error'; message: string }
