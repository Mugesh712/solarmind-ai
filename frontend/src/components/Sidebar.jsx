export default function Sidebar({ activePage, onNavigate }) {
    const navItems = [
        { id: 'dashboard', icon: '📊', label: 'Dashboard' },
        { id: 'panels', icon: '🔲', label: 'Panel Map' },
        { id: 'defects', icon: '🔍', label: 'Defect Detection' },
        { id: 'recommendations', icon: '🛠️', label: 'Recommendations' },
        { id: 'forecasting', icon: '📈', label: 'Forecasting' },
        { id: 'federation', icon: '🌐', label: 'Federated Learning' },
        { id: 'model', icon: '🤖', label: 'AI Model Info' },
        { id: 'settings', icon: '⚙️', label: 'Settings' },
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
                    <span className="status-dot"></span>
                    <span>System Online • TRL-8</span>
                </div>
                <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)', marginTop: 8 }}>
                    Edge Node: Jetson Orin NX
                </div>
                <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)', marginTop: 2 }}>
                    Model: ViT-Base/16 v2.1
                </div>
            </div>
        </aside>
    )
}
