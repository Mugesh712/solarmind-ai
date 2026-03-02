import { useMemo } from 'react'

export default function AttentionMap({ panel }) {
    // Generate simulated ViT attention map data
    const attentionData = useMemo(() => {
        const gridSize = 14  // ViT-Base/16 produces 14x14 patch grid for 224x224 input
        const grid = []

        const hasDefect = panel && panel.defect !== 'normal' && panel.defect !== 'Clean'
        // Primary attention center (where the defect is)
        const cx = hasDefect ? (Math.random() * 6 + 4) : 7
        const cy = hasDefect ? (Math.random() * 6 + 4) : 7
        const spread = hasDefect ? (Math.random() * 2 + 2.5) : 8

        for (let row = 0; row < gridSize; row++) {
            for (let col = 0; col < gridSize; col++) {
                const dist = Math.sqrt((row - cy) ** 2 + (col - cx) ** 2)
                let intensity = hasDefect
                    ? Math.max(0.05, Math.exp(-(dist * dist) / (2 * spread * spread)))
                    : Math.random() * 0.15 + 0.03

                // Add some noise
                intensity += (Math.random() - 0.5) * 0.08
                intensity = Math.max(0, Math.min(1, intensity))

                grid.push({
                    row, col,
                    intensity: parseFloat(intensity.toFixed(3)),
                })
            }
        }
        return { grid, gridSize }
    }, [panel?.id])

    const getHeatColor = (intensity) => {
        if (intensity > 0.8) return `rgba(239, 68, 68, ${intensity * 0.9})`   // Red
        if (intensity > 0.6) return `rgba(251, 146, 60, ${intensity * 0.85})`  // Orange
        if (intensity > 0.4) return `rgba(250, 204, 21, ${intensity * 0.8})`   // Yellow
        if (intensity > 0.2) return `rgba(59, 130, 246, ${intensity * 0.7})`   // Blue
        return `rgba(59, 130, 246, ${intensity * 0.4})`                         // Dim blue
    }

    const maxIntensity = Math.max(...attentionData.grid.map(c => c.intensity))
    const defectLabel = panel?.defect?.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()) || 'Normal'
    const confidence = panel?.confidence ? (panel.confidence * 100).toFixed(1) : '—'

    return (
        <div className="card">
            <div className="card-header">
                <span className="card-title">🔬 ViT Attention Map</span>
                <span style={{
                    fontSize: '0.68rem',
                    padding: '2px 8px',
                    borderRadius: 'var(--radius-pill)',
                    background: 'rgba(139,92,246,0.12)',
                    color: 'var(--accent-purple, #a78bfa)',
                    fontWeight: 600,
                }}>
                    14×14 Patch Grid
                </span>
            </div>

            {/* Model Info */}
            <div style={{ display: 'flex', gap: 12, marginTop: 12, marginBottom: 14 }}>
                <div style={{
                    flex: 1, padding: '8px 10px', borderRadius: 'var(--radius-sm)',
                    background: 'var(--bg-primary)', border: '1px solid var(--border-color)',
                    textAlign: 'center',
                }}>
                    <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', marginBottom: 2 }}>Model</div>
                    <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>ViT-B/16</div>
                </div>
                <div style={{
                    flex: 1, padding: '8px 10px', borderRadius: 'var(--radius-sm)',
                    background: 'var(--bg-primary)', border: '1px solid var(--border-color)',
                    textAlign: 'center',
                }}>
                    <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', marginBottom: 2 }}>Prediction</div>
                    <div style={{ fontSize: '0.75rem', fontWeight: 600, color: (panel?.defect !== 'normal' && panel?.defect !== 'Clean') ? 'var(--accent-red)' : 'var(--accent-green)' }}>{defectLabel}</div>
                </div>
                <div style={{
                    flex: 1, padding: '8px 10px', borderRadius: 'var(--radius-sm)',
                    background: 'var(--bg-primary)', border: '1px solid var(--border-color)',
                    textAlign: 'center',
                }}>
                    <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', marginBottom: 2 }}>Confidence</div>
                    <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--accent-blue)' }}>{confidence}%</div>
                </div>
            </div>

            {/* Attention Grid */}
            <div style={{
                display: 'grid',
                gridTemplateColumns: `repeat(${attentionData.gridSize}, 1fr)`,
                gap: 1,
                borderRadius: 'var(--radius-sm)',
                overflow: 'hidden',
                border: '1px solid var(--border-color)',
                background: '#0a0e17',
            }}>
                {attentionData.grid.map((cell, i) => (
                    <div
                        key={i}
                        title={`Patch (${cell.row}, ${cell.col}): ${(cell.intensity * 100).toFixed(0)}%`}
                        style={{
                            aspectRatio: '1',
                            background: getHeatColor(cell.intensity),
                            transition: 'background 0.3s ease',
                            cursor: 'crosshair',
                        }}
                    />
                ))}
            </div>

            {/* Color Scale Legend */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 10 }}>
                <span style={{ fontSize: '0.6rem', color: 'var(--text-muted)' }}>Low</span>
                <div style={{
                    flex: 1, height: 8, borderRadius: 4,
                    background: 'linear-gradient(to right, rgba(59,130,246,0.2), rgba(59,130,246,0.6), rgba(250,204,21,0.7), rgba(251,146,60,0.8), rgba(239,68,68,0.9))',
                }} />
                <span style={{ fontSize: '0.6rem', color: 'var(--text-muted)' }}>High</span>
            </div>

            {/* Stats */}
            <div style={{
                display: 'flex', gap: 8, marginTop: 10, fontSize: '0.65rem', color: 'var(--text-muted)',
                justifyContent: 'space-between',
            }}>
                <span>Peak attention: <strong style={{ color: 'var(--text-primary)' }}>{(maxIntensity * 100).toFixed(0)}%</strong></span>
                <span>Patches: <strong style={{ color: 'var(--text-primary)' }}>196</strong></span>
                <span>Embed dim: <strong style={{ color: 'var(--text-primary)' }}>768</strong></span>
            </div>
        </div>
    )
}
