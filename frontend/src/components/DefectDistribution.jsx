export default function DefectDistribution({ panels }) {
    // Count all defect types, merging old names with new dataset names
    const rawCounts = {}
    panels.forEach(p => { rawCounts[p.defect] = (rawCounts[p.defect] || 0) + 1 })

    // Merge: 'Clean' + 'normal' → Healthy, and map dataset class names
    const mergedCounts = {
        healthy: (rawCounts['normal'] || 0) + (rawCounts['Clean'] || 0),
        'Bird-drop': rawCounts['Bird-drop'] || 0,
        'Dusty': (rawCounts['Dusty'] || 0) + (rawCounts['dust_soiling'] || 0),
        'Electrical-damage': rawCounts['Electrical-damage'] || 0,
        'Physical-Damage': (rawCounts['Physical-Damage'] || 0) + (rawCounts['micro_crack'] || 0),
        'Snow-Covered': (rawCounts['Snow-Covered'] || 0) + (rawCounts['hotspot'] || 0),
    }

    const total = panels.length

    const bars = [
        { label: 'Healthy', key: 'healthy', color: 'normal' },
        { label: 'Bird-drop', key: 'Bird-drop', color: 'hotspot' },
        { label: 'Dusty', key: 'Dusty', color: 'dust_soiling' },
        { label: 'Electrical Dmg', key: 'Electrical-damage', color: 'micro_crack' },
        { label: 'Physical Dmg', key: 'Physical-Damage', color: 'micro_crack' },
        { label: 'Snow-Covered', key: 'Snow-Covered', color: 'normal' },
    ]

    return (
        <div className="card">
            <div className="card-header">
                <span className="card-title">🔍 Defect Distribution</span>
            </div>
            <div className="defect-dist">
                {bars.map(bar => {
                    const count = mergedCounts[bar.key] || 0
                    const pct = total > 0 ? (count / total * 100) : 0
                    return (
                        <div key={bar.key} className="defect-bar-row">
                            <span className="defect-label">{bar.label}</span>
                            <div className="defect-bar-track">
                                <div
                                    className={`defect-bar-fill ${bar.color}`}
                                    style={{ width: `${Math.max(pct, count > 0 ? 2 : 0)}%` }}
                                >
                                    {pct > 5 ? `${count} (${pct.toFixed(1)}%)` : ''}
                                </div>
                            </div>
                            <span className="defect-count">{count} ({pct.toFixed(1)}%)</span>
                        </div>
                    )
                })}
            </div>
        </div>
    )
}
