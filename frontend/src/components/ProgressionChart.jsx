import { useMemo } from 'react'

export default function ProgressionChart({ panels }) {
    const chartData = useMemo(() => {
        if (!panels || panels.length === 0) return []

        // Group defect counts over simulated time periods
        const months = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar']
        const defectTypes = ['micro_crack', 'hotspot', 'dust_soiling']

        return months.map((month, i) => {
            const factor = 0.6 + i * 0.08
            const counts = {}
            defectTypes.forEach(type => {
                const base = panels.filter(p => p.defect === type).length
                counts[type] = Math.round(base * factor * (0.85 + Math.random() * 0.3))
            })
            const total = Object.values(counts).reduce((a, b) => a + b, 0)
            return { month, ...counts, total }
        })
    }, [panels])

    const maxTotal = Math.max(...chartData.map(d => d.total), 1)

    const typeColors = {
        micro_crack: { bar: 'var(--accent-red)', bg: 'rgba(248,113,113,0.15)' },
        hotspot: { bar: 'var(--accent-orange)', bg: 'rgba(251,146,60,0.15)' },
        dust_soiling: { bar: 'var(--accent-yellow)', bg: 'rgba(250,204,21,0.15)' },
    }

    const typeLabels = {
        micro_crack: 'Micro Crack',
        hotspot: 'Hotspot',
        dust_soiling: 'Dust Soiling',
    }

    // Simulated trend stats
    const currentMonth = chartData[chartData.length - 1]
    const prevMonth = chartData[chartData.length - 2]
    const trendPct = currentMonth && prevMonth
        ? (((currentMonth.total - prevMonth.total) / prevMonth.total) * 100).toFixed(1)
        : 0

    return (
        <div className="card">
            <div className="card-header">
                <span className="card-title">📈 Defect Progression Trends</span>
                <span style={{
                    fontSize: '0.72rem',
                    padding: '3px 10px',
                    borderRadius: 'var(--radius-pill)',
                    background: trendPct > 0 ? 'rgba(248,113,113,0.12)' : 'rgba(52,211,153,0.12)',
                    color: trendPct > 0 ? 'var(--accent-red)' : 'var(--accent-green)',
                    fontWeight: 600,
                }}>
                    {trendPct > 0 ? '↑' : '↓'} {Math.abs(trendPct)}% vs last month
                </span>
            </div>

            {/* Stacked Bar Chart */}
            <div style={{ display: 'flex', alignItems: 'flex-end', gap: 8, height: 140, padding: '0 4px', marginTop: 16 }}>
                {chartData.map((data, i) => (
                    <div key={i} style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
                        <div style={{
                            width: '100%',
                            display: 'flex',
                            flexDirection: 'column-reverse',
                            height: `${(data.total / maxTotal) * 100}%`,
                            minHeight: 8,
                            borderRadius: '4px 4px 0 0',
                            overflow: 'hidden',
                            transition: 'height 0.5s ease',
                        }}>
                            {Object.keys(typeColors).map(type => {
                                const count = data[type] || 0
                                const pct = data.total > 0 ? (count / data.total) * 100 : 0
                                return (
                                    <div
                                        key={type}
                                        title={`${typeLabels[type]}: ${count}`}
                                        style={{
                                            height: `${pct}%`,
                                            background: typeColors[type].bar,
                                            minHeight: count > 0 ? 3 : 0,
                                            opacity: 0.85,
                                            transition: 'height 0.4s ease',
                                        }}
                                    />
                                )
                            })}
                        </div>
                        <span style={{ fontSize: '0.6rem', color: 'var(--text-muted)', marginTop: 4 }}>{data.month}</span>
                    </div>
                ))}
            </div>

            {/* Legend */}
            <div style={{ display: 'flex', gap: 16, marginTop: 16, justifyContent: 'center' }}>
                {Object.entries(typeLabels).map(([key, label]) => (
                    <div key={key} style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '0.68rem', color: 'var(--text-secondary)' }}>
                        <div style={{
                            width: 10, height: 10, borderRadius: 2,
                            background: typeColors[key].bar,
                        }} />
                        {label}
                    </div>
                ))}
            </div>

            {/* Monthly Totals Row */}
            <div style={{ display: 'flex', gap: 8, marginTop: 12, padding: '8px 0', borderTop: '1px solid var(--border-color)' }}>
                {chartData.map((data, i) => (
                    <div key={i} style={{ flex: 1, textAlign: 'center' }}>
                        <div style={{ fontSize: '0.82rem', fontWeight: 700, color: 'var(--text-primary)' }}>{data.total}</div>
                        <div style={{ fontSize: '0.58rem', color: 'var(--text-muted)' }}>defects</div>
                    </div>
                ))}
            </div>
        </div>
    )
}
