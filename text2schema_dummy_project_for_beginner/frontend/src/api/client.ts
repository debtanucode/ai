import axios from 'axios'

const BASE_URL = import.meta.env.VITE_API_URL ?? ''

const api = axios.create({
  baseURL: `${BASE_URL}/api`,
  timeout: 120000,
  headers: {
    'Content-Type': 'application/json',
  },
})

export interface ForeignKey {
  references_table: string
  references_column: string
  on_delete: string
  on_update: string
}

export interface Column {
  name: string
  type: string
  nullable: boolean
  primary_key: boolean
  foreign_key: ForeignKey | null
  unique: boolean
  index: boolean
  default: string | null
  comment: string | null
}

export interface Index {
  name: string
  columns: string[]
  unique: boolean
  index_type: string
}

export interface Table {
  name: string
  columns: Column[]
  indexes: Index[]
  constraints: Array<{ name: string; type: string; expression: string }>
  comment: string | null
}

export interface SchemaDefinition {
  tables: Table[]
  target_db: string
  version: string
  description: string | null
}

export interface QualityScore {
  syntax: number
  integrity: number
  naming: number
  completeness: number
  composite: number
  passed: boolean
}

export interface GenerateRequest {
  description: string
  target_db: 'postgresql' | 'mysql' | 'mongodb' | 'dynamodb'
  output_format: 'sql' | 'nosql' | 'erd' | 'all'
  use_cache?: boolean
  conversation_history?: Array<{ role: string; content: string }>
}

export interface GenerateResponse {
  schema: SchemaDefinition | null
  quality: QualityScore | null
  retry_count: number
  outputs: Record<string, string>
  cached: boolean
  processing_time_ms: number
}

export interface Dialect {
  id: string
  label: string
  output: string
}

export async function generateSchema(request: GenerateRequest): Promise<GenerateResponse> {
  const response = await api.post<GenerateResponse>('/generate', request)
  return response.data
}

export async function getHealth(): Promise<{ status: string }> {
  const response = await api.get<{ status: string }>('/health')
  return response.data
}

export async function getDialects(): Promise<Dialect[]> {
  const response = await api.get<Dialect[]>('/dialects')
  return response.data
}
