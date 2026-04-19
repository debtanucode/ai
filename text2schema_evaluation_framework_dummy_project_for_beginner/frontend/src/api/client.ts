import axios from 'axios'
import type { EvalRequest, EvalResult, BatchResult, HealthStatus } from '../types/api'

const http = axios.create({ baseURL: '/api' })

export const api = {
  health: (): Promise<HealthStatus> =>
    http.get<HealthStatus>('/health').then(r => r.data),

  evaluate: (req: EvalRequest): Promise<EvalResult> =>
    http.post<EvalResult>('/evaluate', req).then(r => r.data),

  evaluateBatch: (reqs: EvalRequest[]): Promise<BatchResult> =>
    http.post<BatchResult>('/evaluate/batch', reqs).then(r => r.data),

  listResults: (limit = 50): Promise<EvalResult[]> =>
    http.get<EvalResult[]>(`/results?limit=${limit}`).then(r => r.data),

  getResult: (runId: string): Promise<EvalResult> =>
    http.get<EvalResult>(`/results/${runId}`).then(r => r.data),
}
