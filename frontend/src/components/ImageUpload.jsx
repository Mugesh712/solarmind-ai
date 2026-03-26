import { useState, useRef } from 'react';

const API_URL = 'http://localhost:8000';

export default function ImageUpload() {
    const [file, setFile] = useState(null);
    const [preview, setPreview] = useState(null);
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);
    const [dragOver, setDragOver] = useState(false);
    const [downloading, setDownloading] = useState(false);
    const [emailAddress, setEmailAddress] = useState('');
    const [sendingEmail, setSendingEmail] = useState(false);
    const [emailSuccess, setEmailSuccess] = useState(null);
    const fileRef = useRef(null);

    const handleFile = (f) => {
        if (!f) return;
        const validTypes = ['image/jpeg', 'image/png', 'image/bmp'];
        if (!validTypes.includes(f.type)) {
            setError('Please upload a JPG, PNG, or BMP image.');
            return;
        }
        if (f.size > 10 * 1024 * 1024) {
            setError('File too large. Max 10MB.');
            return;
        }
        setFile(f);
        setError(null);
        setResult(null);
        const reader = new FileReader();
        reader.onload = (e) => setPreview(e.target.result);
        reader.readAsDataURL(f);
    };

    const handleDrop = (e) => {
        e.preventDefault();
        setDragOver(false);
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    };

    const handleAnalyze = async () => {
        if (!file) return;
        setLoading(true);
        setError(null);
        try {
            const formData = new FormData();
            formData.append('file', file);
            const res = await fetch(`${API_URL}/api/analyze`, {
                method: 'POST',
                body: formData,
            });
            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || 'Analysis failed');
            }
            const data = await res.json();
            setResult(data);
        } catch (err) {
            setError(err.message || 'Failed to analyze image');
        } finally {
            setLoading(false);
        }
    };

    const handleDownloadReport = async () => {
        if (!result) return;
        setDownloading(true);
        try {
            const res = await fetch(`${API_URL}/api/report/download`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    report_title: result.analysis?.report_title || 'Solar Panel Defect Analysis Report',
                    report_date: result.analysis?.report_date || '',
                    predicted_class: result.classification?.predicted_class || '',
                    confidence: result.classification?.confidence || 0,
                    analysis_text: result.analysis?.analysis || '',
                    panel_id: result.analysis?.panel_id || '',
                    model_type: result.classification?.model_type || '',
                    source: result.analysis?.source || '',
                    filename: result.filename || '',
                }),
            });
            if (!res.ok) throw new Error('Download failed');
            const blob = await res.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            const disposition = res.headers.get('Content-Disposition') || '';
            const match = disposition.match(/filename="(.+?)"/);
            a.download = match ? match[1] : 'solarmind_report.pdf';
            document.body.appendChild(a);
            a.click();
            a.remove();
            window.URL.revokeObjectURL(url);
        } catch (err) {
            setError('Failed to download report. Please try again.');
        } finally {
            setDownloading(false);
        }
    };

    const handleEmailReport = async () => {
        if (!result || !emailAddress.trim()) return;
        setSendingEmail(true);
        setEmailSuccess(null);
        try {
            const res = await fetch(`${API_URL}/api/report/email`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    recipient_email: emailAddress.trim(),
                    report_title: result.analysis?.report_title || 'Solar Panel Defect Analysis Report',
                    report_date: result.analysis?.report_date || '',
                    predicted_class: result.classification?.predicted_class || '',
                    confidence: result.classification?.confidence || 0,
                    analysis_text: result.analysis?.analysis || '',
                    panel_id: result.analysis?.panel_id || '',
                    model_type: result.classification?.model_type || '',
                    source: result.analysis?.source || '',
                    filename: result.filename || '',
                }),
            });
            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || 'Failed to send email');
            }
            setEmailSuccess(`Report sent successfully to ${emailAddress.trim()}`);
            setEmailAddress('');
        } catch (err) {
            setError(err.message || 'Failed to send email. Please try again.');
        } finally {
            setSendingEmail(false);
        }
    };

    const handleReset = () => {
        setFile(null);
        setPreview(null);
        setResult(null);
        setError(null);
        setEmailAddress('');
        setEmailSuccess(null);
    };

    const getSeverityFromText = (text) => {
        if (!text) return { level: 'Unknown', color: '#6b7280' };
        const lower = text.toLowerCase();
        if (lower.includes('critical')) return { level: 'Critical', color: '#ef4444' };
        if (lower.includes('high')) return { level: 'High', color: '#f97316' };
        if (lower.includes('medium')) return { level: 'Medium', color: '#eab308' };
        if (lower.includes('low')) return { level: 'Low', color: '#22c55e' };
        return { level: 'Unknown', color: '#6b7280' };
    };

    const getClassIcon = (cls) => {
        const icons = {
            'Bird-drop': '🐦',
            'Clean': '✅',
            'Dusty': '🌫️',
            'Electrical-damage': '⚡',
            'Physical-Damage': '💥',
            'Snow-Covered': '❄️',
        };
        return icons[cls] || '🔍';
    };

    const getPriorityFromText = (text) => {
        if (!text) return null;
        const lower = text.toLowerCase();
        if (lower.includes('p1-immediate') || lower.includes('p1')) return { label: 'P1 — Immediate', color: '#ef4444' };
        if (lower.includes('p2-urgent') || lower.includes('p2')) return { label: 'P2 — Urgent', color: '#f97316' };
        if (lower.includes('p3-scheduled') || lower.includes('p3')) return { label: 'P3 — Scheduled', color: '#eab308' };
        if (lower.includes('p4-monitor') || lower.includes('p4')) return { label: 'P4 — Monitor', color: '#22c55e' };
        return null;
    };

    // Parse analysis text into sections for structured display
    const parseReportSections = (analysisText) => {
        if (!analysisText) return [];
        const sections = [];
        const sectionIcons = {
            'executive summary': '📋',
            'defect classification': '🏷️',
            'detailed technical': '🔬',
            'estimated panel lifetime': '⏳',
            'energy loss': '⚡',
            'root cause': '🔎',
            'recommended corrective': '🔧',
            'preventive maintenance': '🛡️',
            'safety considerations': '⚠️',
            'conclusion': '🎯',
        };

        // Split by numbered sections (1. **Title**: ...)
        const lines = analysisText.split('\n');
        let currentSection = null;
        let currentContent = [];

        for (const line of lines) {
            const sectionMatch = line.match(/^\s*(\d+)\.\s*\*\*(.*?)\*\*:?\s*(.*)/);
            if (sectionMatch) {
                if (currentSection) {
                    sections.push({ ...currentSection, content: currentContent.join('\n').trim() });
                }
                const num = sectionMatch[1];
                const title = sectionMatch[2];
                const rest = sectionMatch[3] || '';
                let icon = '📄';
                for (const [key, ico] of Object.entries(sectionIcons)) {
                    if (title.toLowerCase().includes(key)) { icon = ico; break; }
                }
                currentSection = { num, title, icon };
                currentContent = rest ? [rest] : [];
            } else if (currentSection) {
                currentContent.push(line);
            }
        }
        if (currentSection) {
            sections.push({ ...currentSection, content: currentContent.join('\n').trim() });
        }

        return sections;
    };

    const renderMarkdownLine = (line, i) => {
        // Render a line with bold markdown converted to <strong>
        const html = line
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/^- /, '• ');
        return <p key={i} dangerouslySetInnerHTML={{ __html: html }} />;
    };

    return (
        <div className="image-upload-card">
            <div className="card-header">
                <h3>🔬 Panel Defect Analyzer</h3>
                <p className="subtitle">Upload a solar panel image for AI-powered defect analysis</p>
                <p className="dataset-badge">
                    📊 Powered by <a href="https://www.kaggle.com/datasets/alicjalena/pv-panel-defect-dataset" target="_blank" rel="noreferrer">PV Panel Defect Dataset</a> + <strong>Sarvam AI</strong>
                </p>
            </div>

            {/* Upload Zone */}
            {!result && (
                <div
                    className={`upload-zone ${dragOver ? 'drag-over' : ''} ${preview ? 'has-preview' : ''}`}
                    onDrop={handleDrop}
                    onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                    onDragLeave={() => setDragOver(false)}
                    onClick={() => !preview && fileRef.current?.click()}
                >
                    <input
                        ref={fileRef}
                        type="file"
                        accept=".jpg,.jpeg,.png,.bmp"
                        style={{ display: 'none' }}
                        onChange={(e) => handleFile(e.target.files?.[0])}
                    />
                    {preview ? (
                        <div className="preview-container">
                            <img src={preview} alt="Preview" className="preview-image" />
                            <div className="preview-info">
                                <span className="filename">{file?.name}</span>
                                <span className="filesize">{(file?.size / 1024).toFixed(1)} KB</span>
                            </div>
                        </div>
                    ) : (
                        <div className="upload-placeholder">
                            <span className="upload-icon">📤</span>
                            <p>Drag & drop a solar panel image here</p>
                            <p className="upload-hint">or click to browse (JPG, PNG, BMP — Max 10MB)</p>
                        </div>
                    )}
                </div>
            )}

            {/* Action Buttons */}
            {!result && (
                <div className="upload-actions">
                    {preview && (
                        <>
                            <button
                                className="btn-analyze"
                                onClick={handleAnalyze}
                                disabled={loading}
                            >
                                {loading ? (
                                    <><span className="spinner"></span> Analyzing...</>
                                ) : (
                                    '🔍 Analyze Defect'
                                )}
                            </button>
                            <button className="btn-reset" onClick={handleReset}>Clear</button>
                        </>
                    )}
                </div>
            )}

            {/* Error */}
            {error && (
                <div className="upload-error">⚠️ {error}</div>
            )}

            {/* Results */}
            {result && (
                <div className="analysis-results">
                    {/* Not a Solar Panel Warning */}
                    {result.classification?.is_solar_panel === false ? (
                        <div className="not-solar-panel-warning">
                            <div className="warning-icon-large">🚫</div>
                            <h4 style={{ color: '#f59e0b', margin: '12px 0 8px' }}>No Solar Panel Detected</h4>
                            <p style={{ color: '#94a3b8', fontSize: '14px', lineHeight: '1.6', margin: '0 0 16px' }}>
                                The uploaded image does not appear to contain a solar panel.
                                Please upload a clear photo of a solar panel for accurate defect analysis.
                            </p>
                            <div style={{
                                background: 'rgba(245, 158, 11, 0.1)',
                                border: '1px solid rgba(245, 158, 11, 0.2)',
                                borderRadius: '8px',
                                padding: '12px 16px',
                                fontSize: '13px',
                                color: '#cbd5e1',
                                textAlign: 'left',
                            }}>
                                <strong style={{ color: '#f59e0b' }}>💡 Tips for best results:</strong>
                                <ul style={{ margin: '8px 0 0', paddingLeft: '20px', lineHeight: '1.8' }}>
                                    <li>Use a close-up photo of the solar panel surface</li>
                                    <li>Ensure the panel is clearly visible in the frame</li>
                                    <li>Avoid photos of unrelated objects, people, or landscapes</li>
                                </ul>
                            </div>
                            <button className="btn-reset" onClick={handleReset} style={{ marginTop: '16px', width: '100%' }}>
                                🔄 Try Another Image
                            </button>
                        </div>
                    ) : (
                        <>
                            {/* Report Header */}
                            <div className="analysis-report">
                                <div className="report-header">
                                    <div className="report-header-top">
                                        <div className="report-title-block">
                                            <h4 className="report-title">
                                                📊 {result.analysis?.report_title || 'Solar Panel Defect Analysis Report'}
                                            </h4>
                                            <span className="report-date">
                                                🕐 {result.analysis?.report_date || new Date().toLocaleString()}
                                            </span>
                                        </div>
                                        {(() => {
                                            const severity = getSeverityFromText(result.analysis?.analysis);
                                            const priority = getPriorityFromText(result.analysis?.analysis);
                                            return (
                                                <div className="report-badges">
                                                    <span className="report-severity-badge" style={{ backgroundColor: `${severity.color}22`, color: severity.color, borderColor: `${severity.color}44` }}>
                                                        {severity.level} Severity
                                                    </span>
                                                    {priority && (
                                                        <span className="report-priority-badge" style={{ backgroundColor: `${priority.color}22`, color: priority.color, borderColor: `${priority.color}44` }}>
                                                            {priority.label}
                                                        </span>
                                                    )}
                                                </div>
                                            );
                                        })()}
                                    </div>
                                    <div className="report-classification-strip">
                                        <span className="class-icon-large">{getClassIcon(result.classification?.predicted_class)}</span>
                                        <div className="report-class-info">
                                            <span className="report-class-name">{result.classification?.predicted_class}</span>
                                            <span className="report-confidence">
                                                {(result.classification?.confidence * 100).toFixed(1)}% confidence
                                                <span className="report-model-tag">
                                                    {result.classification?.mode === 'real' ? '🤖 ViT+Swin Ensemble' : '🔮 Analysis'}
                                                </span>
                                            </span>
                                        </div>
                                    </div>
                                </div>

                                {/* Probability Bars (compact) */}
                                <div className="probability-bars compact">
                                    <h5>Class Probabilities</h5>
                                    {result.classification?.probabilities &&
                                        Object.entries(result.classification.probabilities)
                                            .sort(([, a], [, b]) => b - a)
                                            .map(([cls, prob]) => (
                                                <div key={cls} className="prob-bar-row">
                                                    <span className="prob-label">
                                                        {getClassIcon(cls)} {cls}
                                                    </span>
                                                    <div className="prob-bar-track">
                                                        <div
                                                            className="prob-bar-fill"
                                                            style={{
                                                                width: `${Math.max(1, prob * 100)}%`,
                                                                backgroundColor:
                                                                    cls === result.classification.predicted_class
                                                                        ? '#3b82f6'
                                                                        : '#475569',
                                                            }}
                                                        ></div>
                                                    </div>
                                                    <span className="prob-value">{(prob * 100).toFixed(1)}%</span>
                                                </div>
                                            ))}
                                </div>

                                {/* Structured Report Sections */}
                                <div className="report-body">
                                    <div className="report-body-header">
                                        <h5>
                                            {result.analysis?.source === 'sarvam-ai' ? '🧠 Sarvam AI Detailed Analysis' : '📋 Detailed Analysis Report'}
                                        </h5>
                                        <span className="analysis-source-tag">
                                            {result.analysis?.source === 'sarvam-ai' ? 'Sarvam AI (sarvam-m)' : 'Built-in Analysis'}
                                        </span>
                                    </div>

                                    {(() => {
                                        const sections = parseReportSections(result.analysis?.analysis);
                                        if (sections.length > 0) {
                                            return (
                                                <div className="report-sections">
                                                    {sections.map((section, idx) => (
                                                        <div key={idx} className="report-section">
                                                            <div className="report-section-header">
                                                                <span className="report-section-icon">{section.icon}</span>
                                                                <span className="report-section-num">{section.num}.</span>
                                                                <span className="report-section-title">{section.title}</span>
                                                            </div>
                                                            <div className="report-section-content">
                                                                {section.content.split('\n').map((line, j) => {
                                                                    if (!line.trim()) return null;
                                                                    return renderMarkdownLine(line, j);
                                                                })}
                                                            </div>
                                                        </div>
                                                    ))}
                                                </div>
                                            );
                                        }
                                        // Fallback: render as plain text
                                        return (
                                            <div className="analysis-text">
                                                {result.analysis?.analysis?.split('\n').map((line, i) => (
                                                    renderMarkdownLine(line, i)
                                                ))}
                                            </div>
                                        );
                                    })()}
                                </div>

                                {/* Download & Email Actions */}
                                <button
                                    className="report-download-btn"
                                    onClick={handleDownloadReport}
                                    disabled={downloading}
                                >
                                    {downloading ? (
                                        <><span className="spinner"></span> Generating PDF...</>
                                    ) : (
                                        '📥 Download Full Report (PDF)'
                                    )}
                                </button>

                                {/* Email Report Section */}
                                <div className="report-email-section">
                                    <div className="report-email-header">
                                        <span>📧</span>
                                        <span>Send Report to Email</span>
                                    </div>
                                    <div className="report-email-input-row">
                                        <input
                                            type="email"
                                            className="report-email-input"
                                            placeholder="Enter email address..."
                                            value={emailAddress}
                                            onChange={(e) => { setEmailAddress(e.target.value); setEmailSuccess(null); }}
                                            onKeyDown={(e) => e.key === 'Enter' && handleEmailReport()}
                                            disabled={sendingEmail}
                                        />
                                        <button
                                            className="report-email-btn"
                                            onClick={handleEmailReport}
                                            disabled={sendingEmail || !emailAddress.trim()}
                                        >
                                            {sendingEmail ? (
                                                <><span className="spinner"></span> Sending...</>
                                            ) : (
                                                '📤 Send'
                                            )}
                                        </button>
                                    </div>
                                    {emailSuccess && (
                                        <div className="report-email-success">
                                            ✅ {emailSuccess}
                                        </div>
                                    )}
                                </div>
                            </div>

                            <button className="btn-reset" onClick={handleReset} style={{ marginTop: '16px', width: '100%' }}>
                                🔄 Analyze Another Image
                            </button>
                        </>
                    )}
                </div>
            )}
        </div>
    );
}
