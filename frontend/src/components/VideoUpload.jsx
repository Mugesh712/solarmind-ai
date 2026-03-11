import { useState, useRef } from 'react';

const API_URL = 'http://localhost:8000';

export default function VideoUpload() {
    const [file, setFile] = useState(null);
    const [preview, setPreview] = useState(null);
    const [loading, setLoading] = useState(false);
    const [progress, setProgress] = useState('');
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);
    const [dragOver, setDragOver] = useState(false);
    const [expandedFrame, setExpandedFrame] = useState(null);
    const fileRef = useRef(null);

    const handleFile = (f) => {
        if (!f) return;
        const validTypes = ['video/mp4', 'video/avi', 'video/quicktime', 'video/x-matroska', 'video/webm'];
        const validExts = ['.mp4', '.avi', '.mov', '.mkv', '.webm'];
        const hasValidExt = validExts.some(ext => f.name.toLowerCase().endsWith(ext));
        if (!validTypes.includes(f.type) && !hasValidExt) {
            setError('Please upload an MP4, AVI, MOV, MKV, or WebM video.');
            return;
        }
        if (f.size > 100 * 1024 * 1024) {
            setError('File too large. Max 100MB.');
            return;
        }
        setFile(f);
        setError(null);
        setResult(null);
        setPreview(URL.createObjectURL(f));
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
        setProgress('Uploading video...');
        try {
            const formData = new FormData();
            formData.append('file', file);
            setProgress('Extracting frames & analyzing...');
            const res = await fetch(`${API_URL}/api/analyze/video`, {
                method: 'POST',
                body: formData,
            });
            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || 'Video analysis failed');
            }
            const data = await res.json();
            setResult(data);
            setProgress('');
        } catch (err) {
            setError(err.message || 'Failed to analyze video');
        } finally {
            setLoading(false);
        }
    };

    const handleReset = () => {
        setFile(null);
        if (preview) URL.revokeObjectURL(preview);
        setPreview(null);
        setResult(null);
        setError(null);
        setProgress('');
        setExpandedFrame(null);
    };

    const getClassIcon = (cls) => {
        const icons = {
            'Bird-drop': '🐦',
            'Clean': '✅',
            'Dusty': '🌫️',
            'Electrical-damage': '⚡',
            'Physical-Damage': '💥',
            'Snow-Covered': '❄️',
            'Not a Solar Panel': '🚫',
        };
        return icons[cls] || '🔍';
    };

    const getDefectColor = (cls) => {
        const colors = {
            'Bird-drop': '#f97316',
            'Clean': '#22c55e',
            'Dusty': '#eab308',
            'Electrical-damage': '#ef4444',
            'Physical-Damage': '#f87171',
            'Snow-Covered': '#60a5fa',
            'Not a Solar Panel': '#6b7280',
        };
        return colors[cls] || '#6b7280';
    };

    const formatTime = (sec) => {
        const m = Math.floor(sec / 60);
        const s = Math.floor(sec % 60);
        return `${m}:${s.toString().padStart(2, '0')}`;
    };

    return (
        <div className="video-upload-card card">
            <div className="card-header" style={{ marginBottom: 0 }}>
                <span className="card-title">🎬 Video Defect Analyzer</span>
                <span className="card-badge blue">Drone / Camera</span>
            </div>
            <p className="video-upload-subtitle">
                Upload a solar panel video — frames are extracted automatically and analyzed for defects
            </p>

            {/* Upload Zone */}
            {!result && (
                <div
                    className={`video-upload-zone ${dragOver ? 'drag-over' : ''} ${preview ? 'has-preview' : ''}`}
                    onDrop={handleDrop}
                    onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                    onDragLeave={() => setDragOver(false)}
                    onClick={() => !preview && fileRef.current?.click()}
                >
                    <input
                        ref={fileRef}
                        type="file"
                        accept=".mp4,.avi,.mov,.mkv,.webm"
                        style={{ display: 'none' }}
                        onChange={(e) => handleFile(e.target.files?.[0])}
                    />
                    {preview ? (
                        <div className="video-preview-container">
                            <video src={preview} className="video-preview-player" controls muted />
                            <div className="video-preview-info">
                                <span className="filename">{file?.name}</span>
                                <span className="filesize">{(file?.size / (1024 * 1024)).toFixed(1)} MB</span>
                            </div>
                        </div>
                    ) : (
                        <div className="upload-placeholder">
                            <span className="upload-icon">🎥</span>
                            <p>Drag & drop a solar panel video here</p>
                            <p className="upload-hint">or click to browse (MP4, AVI, MOV, MKV, WebM — Max 100MB)</p>
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
                                    <><span className="spinner"></span> {progress || 'Analyzing...'}</>
                                ) : (
                                    '🎬 Analyze Video'
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
                <div className="video-results">
                    {/* Full video non-panel warning */}
                    {result.summary?.no_solar_panel ? (
                        <div className="not-solar-panel-warning" style={{ textAlign: 'center', padding: '32px 24px' }}>
                            <div className="warning-icon-large">🚫</div>
                            <h4 style={{ color: '#f59e0b', margin: '12px 0 8px' }}>No Solar Panel Detected</h4>
                            <p style={{ color: '#94a3b8', fontSize: '14px', lineHeight: '1.6', margin: '0 0 16px' }}>
                                None of the {result.total_frames_analyzed} analyzed frames appear to contain a solar panel.
                                Please upload a video that shows solar panels for defect analysis.
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
                                    <li>Use drone or camera footage of solar panel installations</li>
                                    <li>Ensure solar panels are clearly visible in the video</li>
                                    <li>Avoid videos of unrelated scenes</li>
                                </ul>
                            </div>
                            <button className="btn-reset" onClick={handleReset} style={{ marginTop: '16px', width: '100%' }}>
                                🎬 Try Another Video
                            </button>
                        </div>
                    ) : (
                        <>
                    {/* Summary Card */}
                    <div className="video-summary">
                        <div className="video-summary-stats">
                            <div className="video-stat">
                                <span className="video-stat-value">{result.total_frames_analyzed}</span>
                                <span className="video-stat-label">Frames Analyzed</span>
                            </div>
                            <div className="video-stat">
                                <span className="video-stat-value">{formatTime(result.video_duration_sec)}</span>
                                <span className="video-stat-label">Duration</span>
                            </div>
                            <div className="video-stat">
                                <span className="video-stat-value" style={{ color: getDefectColor(result.summary?.dominant_defect) }}>
                                    {getClassIcon(result.summary?.dominant_defect)} {result.summary?.dominant_defect}
                                </span>
                                <span className="video-stat-label">Dominant Defect</span>
                            </div>
                            <div className="video-stat">
                                <span className="video-stat-value" style={{ color: result.summary?.defect_rate_pct > 50 ? '#ef4444' : result.summary?.defect_rate_pct > 0 ? '#eab308' : '#22c55e' }}>
                                    {result.summary?.defect_rate_pct}%
                                </span>
                                <span className="video-stat-label">Defect Rate</span>
                            </div>
                        </div>

                        {/* Non-panel frames notice */}
                        {result.summary?.non_panel_frames > 0 && (
                            <div style={{
                                background: 'rgba(107, 114, 128, 0.15)',
                                border: '1px solid rgba(107, 114, 128, 0.3)',
                                borderRadius: '8px',
                                padding: '8px 12px',
                                margin: '12px 0 0',
                                fontSize: '13px',
                                color: '#94a3b8',
                            }}>
                                🚫 {result.summary.non_panel_frames} of {result.total_frames_analyzed} frames did not contain a solar panel
                            </div>
                        )}

                        {/* Defect Distribution Bar */}
                        {result.summary?.defect_distribution && (
                            <div className="video-defect-distribution">
                                <div className="video-defect-bar">
                                    {Object.entries(result.summary.defect_distribution).map(([cls, count]) => (
                                        <div
                                            key={cls}
                                            className="video-defect-segment"
                                            style={{
                                                width: `${(count / result.total_frames_analyzed) * 100}%`,
                                                backgroundColor: getDefectColor(cls),
                                            }}
                                            title={`${cls}: ${count} frames`}
                                        />
                                    ))}
                                </div>
                                <div className="video-defect-legend">
                                    {Object.entries(result.summary.defect_distribution).map(([cls, count]) => (
                                        <span key={cls} className="video-legend-item">
                                            <span className="video-legend-dot" style={{ backgroundColor: getDefectColor(cls) }} />
                                            {cls}: {count}
                                        </span>
                                    ))}
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Frame-by-Frame Results */}
                    <div className="frame-results-header">
                        <h5>📸 Frame-by-Frame Analysis</h5>
                        <span className="frame-count-badge">{result.frames?.length} frames</span>
                    </div>
                    <div className="frame-results-grid">
                        {result.frames?.map((frame, i) => (
                            <div
                                key={i}
                                className={`frame-card ${expandedFrame === i ? 'expanded' : ''}`}
                                onClick={() => setExpandedFrame(expandedFrame === i ? null : i)}
                            >
                                <div className="frame-card-top">
                                    <img src={frame.thumbnail} alt={`Frame ${i}`} className="frame-thumbnail" />
                                    <div className="frame-card-overlay">
                                        <span className="frame-timestamp">{formatTime(frame.timestamp_sec)}</span>
                                    </div>
                                </div>
                                <div className="frame-card-body">
                                    <div className="frame-defect-label" style={{ color: getDefectColor(frame.classification?.predicted_class) }}>
                                        {getClassIcon(frame.classification?.predicted_class)} {frame.classification?.predicted_class}
                                    </div>
                                    <div className="frame-confidence">
                                        {((frame.classification?.confidence || 0) * 100).toFixed(1)}%
                                    </div>
                                </div>

                                {/* Expanded: Sarvam AI Analysis */}
                                {expandedFrame === i && frame.analysis?.analysis && (
                                    <div className="frame-analysis-expanded">
                                        <div className="frame-analysis-title">
                                            {frame.analysis?.source === 'sarvam-ai' ? '🧠 Sarvam AI' : '📋 Report'}
                                        </div>
                                        <div className="frame-analysis-text">
                                            {frame.analysis.analysis.split('\n').map((line, j) => (
                                                <p key={j} dangerouslySetInnerHTML={{ __html: line.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') }} />
                                            ))}
                                        </div>
                                    </div>
                                )}
                            </div>
                        ))}
                    </div>

                    <button className="btn-reset" onClick={handleReset} style={{ marginTop: '16px', width: '100%' }}>
                        🎬 Analyze Another Video
                    </button>
                        </>
                    )}
                </div>
            )}
        </div>
    );
}
