import { useCallback, useEffect, useRef, useState } from 'react'
import type { WsMessage } from '../types/api'

export type WsStatus = 'connecting' | 'open' | 'closed' | 'error'

export function useWebSocket(url = 'ws://localhost:8000/ws/evaluate') {
  const ws = useRef<WebSocket | null>(null)
  const [status, setStatus] = useState<WsStatus>('closed')
  const [lastMessage, setLastMessage] = useState<WsMessage | null>(null)

  const connect = useCallback(() => {
    if (ws.current && ws.current.readyState !== WebSocket.CLOSED) return
    setStatus('connecting')
    const socket = new WebSocket(url)

    socket.onopen = () => setStatus('open')
    socket.onclose = () => setStatus('closed')
    socket.onerror = () => setStatus('error')
    socket.onmessage = (event) => {
      try {
        const msg: WsMessage = JSON.parse(event.data)
        setLastMessage(msg)
      } catch {
        // ignore parse errors
      }
    }
    ws.current = socket
  }, [url])

  const send = useCallback((data: unknown) => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify(data))
    }
  }, [])

  const disconnect = useCallback(() => {
    ws.current?.close()
    ws.current = null
  }, [])

  useEffect(() => () => { ws.current?.close() }, [])

  return { status, lastMessage, connect, send, disconnect }
}
