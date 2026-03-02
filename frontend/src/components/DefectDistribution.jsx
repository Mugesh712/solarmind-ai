export default function DefectDistribution({ panels }) {
    const counts = { normal: 0, micro_crack: 0, hotspot: 0, dust_soiling: 0 }
    panels.forEach(p => { counts[p.defect] = (counts[p.defect] || 0) + 1 })
    const total = panels.length

    const bars = [
        { label: 'Normal', key: 'normal', color: 'normal' },
        { label: 'Micro-Crack', key: 'micro_crack', color: 'micro_crack' },
        { label: 'Hotspot', key: 'hotspot', color: 'hotspot' },
        { label: 'Dust/Soiling', key: 'dust_soiling', color: 'dust_soiling' },
    ]

    return (
        <div className="card">
            <div className="card-header">
                <span className="card-title">🔍 Defect Distribution</span>
            </div>
            <div className="defect-dist">
                {bars.map(bar => {
                    const count = counts[bar.key] || 0
                    const pct = total > 0 ? (count / total * 100) : 0
                    return (
                        <div key={bar.key} className="defect-bar-row">
                            <span className="defect-label">{bar.label}</span>
                            <div className="defect-bar-track">
                                <div
                                    className={`defect-bar-fill ${bar.color}`}
                                    style={{ width: `${Math.max(pct, 2)}%` }}
                                >
                                    {pct > 5 ? `${count} (${pct.toFixed(1)}%)` : ''}
                                </div>
                            </div>
                        </div>
                    )
                })}
            </div>
        </div>
    )
}
