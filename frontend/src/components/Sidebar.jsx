export default function Sidebar({ activePage, onNavigate, apiConnected }) {
    const navItems = [
        { id: 'dashboard', icon: '📊', label: 'Dashboard' },
        { id: 'panels', icon: '🔲', label: 'Panel Map' },
        { id: 'defects', icon: '🔍', label: 'Defect Detection' },
        { id: 'simulator', icon: '🎛️', label: 'Simulator' },
        { id: 'comparison', icon: '📊', label: 'Model Comparison' },
    ]

    return (
        <aside className="sidebar">
            <div className="sidebar-header">
                <div className="sidebar-logo">
                    <div className="logo-icon">☀️</div>
                    <div>
                        <h1>SolarMind AI</h1>
                        <div className="subtitle">Predictive Maintenance</div>
                    </div>
                </div>
            </div>

            <nav className="sidebar-nav">
                {navItems.map(item => (
                    <div
                        key={item.id}
                        className={`nav-item ${activePage === item.id ? 'active' : ''}`}
                        onClick={() => onNavigate(item.id)}
                    >
                        <span className="icon">{item.icon}</span>
                        <span>{item.label}</span>
                    </div>
                ))}
            </nav>

            <div className="sidebar-footer">
                <div className="status-indicator">
                    <span className="status-dot" style={apiConnected ? {} : { background: 'var(--accent-yellow)', boxShadow: '0 0 8px var(--accent-yellow)' }}></span>
                    <span>{apiConnected ? 'API Connected • TRL-8' : 'Demo Mode • Offline'}</span>
                </div>
                <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)', marginTop: 8 }}>
                    Edge Node: Jetson Orin NX
                </div>
                <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)', marginTop: 2 }}>
                    Model: ViT+Swin Ensemble v2.1
                </div>
            </div>
        </aside>
    )
}

