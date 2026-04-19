import React, { useCallback } from 'react'
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  Handle,
  Position,
  NodeTypes,
} from 'reactflow'
import 'reactflow/dist/style.css'

interface ERDColumn {
  name: string
  type: string
  primary_key: boolean
  nullable: boolean
  foreign_key: { references_table: string; references_column: string } | null
}

interface TableNodeData {
  label: string
  columns: ERDColumn[]
}

function TableNode({ data }: { data: TableNodeData }) {
  return (
    <div style={{
      background: '#1e293b',
      border: '1px solid #334155',
      borderRadius: '8px',
      minWidth: '200px',
      boxShadow: '0 4px 6px -1px rgba(0,0,0,0.3)',
      fontSize: '12px',
    }}>
      <div style={{
        background: '#3b82f6',
        color: 'white',
        padding: '6px 12px',
        borderRadius: '8px 8px 0 0',
        fontWeight: 700,
        fontSize: '13px',
      }}>
        {data.label}
      </div>
      {data.columns.map((col, i) => (
        <div key={col.name} style={{
          position: 'relative',
          padding: '4px 12px',
          borderBottom: i < data.columns.length - 1 ? '1px solid #1e293b' : 'none',
          display: 'flex',
          alignItems: 'center',
          gap: '6px',
          background: i % 2 === 0 ? '#0f172a' : '#1e293b',
        }}>
          <Handle
            type="source"
            position={Position.Right}
            id={col.name}
            style={{ right: -8, background: '#3b82f6', width: 8, height: 8 }}
          />
          <Handle
            type="target"
            position={Position.Left}
            id={col.name}
            style={{ left: -8, background: '#64748b', width: 8, height: 8 }}
          />
          {col.primary_key && (
            <span style={{ color: '#fbbf24', fontSize: '10px' }}>PK</span>
          )}
          {col.foreign_key && !col.primary_key && (
            <span style={{ color: '#a78bfa', fontSize: '10px' }}>FK</span>
          )}
          <span style={{ color: '#e2e8f0', flex: 1 }}>{col.name}</span>
          <span style={{ color: '#64748b', fontSize: '11px' }}>{col.type}</span>
        </div>
      ))}
    </div>
  )
}

const nodeTypes: NodeTypes = { tableNode: TableNode }

interface ERDData {
  nodes: any[]
  edges: any[]
}

interface Props {
  erdData: ERDData
}

export default function ERDViewer({ erdData }: Props) {
  return (
    <div style={{ flex: 1, height: '100%' }}>
      <ReactFlow
        nodes={erdData.nodes}
        edges={erdData.edges}
        nodeTypes={nodeTypes}
        fitView
        attributionPosition="bottom-left"
      >
        <Background color="#334155" gap={20} />
        <Controls />
        <MiniMap
          nodeColor="#1e293b"
          maskColor="rgba(15, 23, 42, 0.7)"
        />
      </ReactFlow>
    </div>
  )
}
