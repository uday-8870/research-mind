import { useState, useRef, useEffect } from 'react'
import {
  Search, Zap, BookOpen, Brain, CheckCircle,
  Clock, AlertTriangle, ExternalLink, ChevronDown,
  Activity, Layers, Terminal, ArrowRight
} from 'lucide-react'

// ── Types ────────────────────────────────────────────────────────
interface EvalScores {
  faithfulness: number
  coverage: number
  hallucination_rate: number
  quality_score: number
  reasoning: string
  flagged_claims: string[]
}

interface Source {
  url: string
  title: string
  score: number
}

interface ResearchResult {
  task_id: string
  query: string
  sub_queries: string[]
  report: string
  sources: Source[]
  eval_scores: EvalScores | null
  status: string
}

type Phase = 'idle' | 'searching' | 'reading' | 'synthesizing' | 'evaluating' | 'done' | 'error'

const PHASE_LABELS: Record<Phase, string> = {
  idle: 'Ready',
  searching: 'Decomposing query & searching web...',
  reading: 'Reading & chunking documents...',
  synthesizing: 'Synthesizing research report...',
  evaluating: 'Running eval pipeline...',
  done: 'Research complete',
  error: 'Error occurred',
}

const PHASE_PROGRESS: Record<Phase, number> = {
  idle: 0, searching: 25, reading: 50, synthesizing: 75, evaluating: 90, done: 100, error: 0,
}

// ── API ──────────────────────────────────────────────────────────
const API = '/api'

async function startResearch(query: string, depth: string): Promise<string> {
  const res = await fetch(`${API}/research`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, depth }),
  })
  if (!res.ok) throw new Error(await res.text())
  const data = await res.json()
  return data.task_id
}

async function pollStatus(taskId: string): Promise<{ status: string; progress: number }> {
  const res = await fetch(`${API}/research/${taskId}/status`)
  if (!res.ok) throw new Error('Status check failed')
  return res.json()
}

async function getResult(taskId: string): Promise<ResearchResult> {
  const res = await fetch(`${API}/research/${taskId}`)
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

// ── Score Bar ────────────────────────────────────────────────────
function ScoreBar({ label, value, invert = false, color }: {
  label: string; value: number; invert?: boolean; color: string
}) {
  const display = invert ? 1 - value : value
  return (
    <div style={{ marginBottom: 12 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
        <span style={{ fontSize: 12, color: 'var(--text-muted)', fontFamily: 'JetBrains Mono' }}>{label}</span>
        <span style={{ fontSize: 12, fontFamily: 'JetBrains Mono', color }}>
          {invert ? `${((1 - value) * 100).toFixed(0)}% clean` : `${(value * 100).toFixed(0)}%`}
        </span>
      </div>
      <div style={{ height: 4, background: 'var(--bg-3)', borderRadius: 2, overflow: 'hidden' }}>
        <div style={{
          height: '100%', width: `${display * 100}%`, background: color,
          borderRadius: 2, transition: 'width 1s ease',
          boxShadow: `0 0 8px ${color}`,
        }} />
      </div>
    </div>
  )
}

// ── Agent Step ───────────────────────────────────────────────────
function AgentStep({ icon: Icon, label, active, done }: {
  icon: any; label: string; active: boolean; done: boolean
}) {
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 10, padding: '8px 12px',
      borderRadius: 8, background: active ? 'rgba(88,166,255,0.08)' : 'transparent',
      border: `1px solid ${active ? 'rgba(88,166,255,0.3)' : 'transparent'}`,
      transition: 'all 0.3s',
    }}>
      <div style={{
        width: 28, height: 28, borderRadius: 6,
        background: done ? 'rgba(63,185,80,0.15)' : active ? 'rgba(88,166,255,0.15)' : 'var(--bg-3)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        animation: active ? 'pulse-glow 2s infinite' : 'none',
      }}>
        <Icon size={14} color={done ? 'var(--accent-2)' : active ? 'var(--accent)' : 'var(--text-dim)'} />
      </div>
      <span style={{
        fontSize: 13, fontFamily: 'JetBrains Mono',
        color: done ? 'var(--accent-2)' : active ? 'var(--text)' : 'var(--text-dim)',
      }}>{label}</span>
      {done && <CheckCircle size={12} color='var(--accent-2)' style={{ marginLeft: 'auto' }} />}
      {active && (
        <div style={{
          marginLeft: 'auto', width: 6, height: 6, borderRadius: '50%',
          background: 'var(--accent)', animation: 'blink 1s infinite',
        }} />
      )}
    </div>
  )
}

// ── Main App ─────────────────────────────────────────────────────
export default function App() {
  const [query, setQuery] = useState('')
  const [depth, setDepth] = useState('deep')
  const [phase, setPhase] = useState<Phase>('idle')
  const [result, setResult] = useState<ResearchResult | null>(null)
  const [error, setError] = useState('')
  const [showSources, setShowSources] = useState(false)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const pollRef = useRef<any>(null)

  const isRunning = ['searching', 'reading', 'synthesizing', 'evaluating'].includes(phase)
  const progress = PHASE_PROGRESS[phase]

  const phaseOrder: Phase[] = ['searching', 'reading', 'synthesizing', 'evaluating']
  const phaseIndex = phaseOrder.indexOf(phase)

  async function handleSubmit() {
    if (!query.trim() || isRunning) return
    setError('')
    setResult(null)
    setPhase('searching')

    try {
      const taskId = await startResearch(query, depth)

      // Poll for completion
      pollRef.current = setInterval(async () => {
        try {
          const status = await pollStatus(taskId)
          const s = status.status as Phase
          if (['searching', 'reading', 'synthesizing', 'evaluating', 'done', 'failed'].includes(s)) {
            if (s !== 'failed') setPhase(s as Phase)
          }
          if (s === 'done') {
            clearInterval(pollRef.current)
            const res = await getResult(taskId)
            setResult(res)
            setPhase('done')
          } else if (s === 'failed') {
            clearInterval(pollRef.current)
            setPhase('error')
            setError('Research task failed. Check API logs.')
          }
        } catch (e: any) {
          clearInterval(pollRef.current)
          setPhase('error')
          setError(e.message)
        }
      }, 2000)
    } catch (e: any) {
      setPhase('error')
      setError(e.message)
    }
  }

  useEffect(() => () => clearInterval(pollRef.current), [])

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* Grid background */}
      <div style={{
        position: 'fixed', inset: 0, pointerEvents: 'none', zIndex: 0,
        backgroundImage: `linear-gradient(var(--border) 1px, transparent 1px),
          linear-gradient(90deg, var(--border) 1px, transparent 1px)`,
        backgroundSize: '40px 40px',
        opacity: 0.3,
      }} />

      {/* Glow blob */}
      <div style={{
        position: 'fixed', top: -200, left: '50%', transform: 'translateX(-50%)',
        width: 800, height: 600, borderRadius: '50%',
        background: 'radial-gradient(ellipse, rgba(88,166,255,0.06) 0%, transparent 70%)',
        pointerEvents: 'none', zIndex: 0,
      }} />

      <div style={{ position: 'relative', zIndex: 1, maxWidth: 900, margin: '0 auto', padding: '40px 24px', width: '100%' }}>

        {/* Header */}
        <header style={{ textAlign: 'center', marginBottom: 56, animation: 'fadeUp 0.6s ease' }}>
          <div style={{
            display: 'inline-flex', alignItems: 'center', gap: 8, marginBottom: 20,
            padding: '4px 12px', borderRadius: 20, border: '1px solid var(--border-glow)',
            background: 'rgba(88,166,255,0.06)', fontSize: 11,
            fontFamily: 'JetBrains Mono', color: 'var(--accent)',
          }}>
            <Activity size={10} />
            MULTI-AGENT RESEARCH SYSTEM v1.0
          </div>

          <h1 style={{
            fontSize: 'clamp(40px, 6vw, 72px)', fontWeight: 800, letterSpacing: '-2px',
            lineHeight: 1.05, marginBottom: 16,
            background: 'linear-gradient(135deg, #e6edf3 0%, #7d8590 100%)',
            WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent',
          }}>
            ResearchMind
          </h1>

          <p style={{ fontSize: 16, color: 'var(--text-muted)', maxWidth: 480, margin: '0 auto', lineHeight: 1.7 }}>
            Autonomous multi-agent research with built-in eval pipeline.
            Ask anything. Get sourced, scored, reliable answers.
          </p>
        </header>

        {/* Search Box */}
        <div style={{
          background: 'var(--bg-2)', border: '1px solid var(--border-glow)',
          borderRadius: 16, padding: 20, marginBottom: 24,
          animation: 'fadeUp 0.6s ease 0.1s both',
          boxShadow: '0 4px 24px rgba(0,0,0,0.4)',
        }}>
          <textarea
            ref={textareaRef}
            value={query}
            onChange={e => setQuery(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSubmit() } }}
            placeholder="What do you want to research? e.g. 'How do large language models handle multi-step reasoning?'"
            disabled={isRunning}
            rows={3}
            style={{
              width: '100%', background: 'transparent', border: 'none', outline: 'none',
              color: 'var(--text)', fontSize: 16, fontFamily: 'Inter', lineHeight: 1.6,
              resize: 'none', marginBottom: 16,
            }}
          />

          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            {/* Depth selector */}
            <div style={{ display: 'flex', gap: 6 }}>
              {['quick', 'standard', 'deep'].map(d => (
                <button key={d} onClick={() => setDepth(d)}
                  style={{
                    padding: '5px 12px', borderRadius: 6, fontSize: 12,
                    fontFamily: 'JetBrains Mono', cursor: 'pointer', border: '1px solid',
                    borderColor: depth === d ? 'var(--accent)' : 'var(--border)',
                    background: depth === d ? 'rgba(88,166,255,0.1)' : 'transparent',
                    color: depth === d ? 'var(--accent)' : 'var(--text-muted)',
                    transition: 'all 0.2s',
                  }}>
                  {d}
                </button>
              ))}
            </div>

            <div style={{ flex: 1 }} />

            <button onClick={handleSubmit} disabled={!query.trim() || isRunning}
              style={{
                display: 'flex', alignItems: 'center', gap: 8,
                padding: '10px 22px', borderRadius: 8, border: 'none', cursor: 'pointer',
                background: isRunning ? 'var(--bg-3)' : 'var(--accent)',
                color: isRunning ? 'var(--text-muted)' : '#0d1117',
                fontSize: 14, fontWeight: 600, fontFamily: 'Syne',
                transition: 'all 0.2s', opacity: !query.trim() ? 0.5 : 1,
              }}>
              {isRunning
                ? <><div style={{ width: 14, height: 14, border: '2px solid var(--text-dim)', borderTopColor: 'var(--text-muted)', borderRadius: '50%', animation: 'spin 0.8s linear infinite' }} /> Running</>
                : <><Search size={14} /> Research</>
              }
            </button>
          </div>
        </div>

        {/* Progress Panel */}
        {isRunning && (
          <div style={{
            background: 'var(--bg-2)', border: '1px solid var(--border)',
            borderRadius: 16, padding: 24, marginBottom: 24,
            animation: 'fadeUp 0.4s ease',
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
              <div style={{ fontFamily: 'JetBrains Mono', fontSize: 12, color: 'var(--accent)' }}>
                {PHASE_LABELS[phase]}
              </div>
              <div style={{ fontFamily: 'JetBrains Mono', fontSize: 12, color: 'var(--text-muted)' }}>
                {progress}%
              </div>
            </div>

            {/* Progress bar */}
            <div style={{ height: 3, background: 'var(--bg-3)', borderRadius: 2, marginBottom: 20, overflow: 'hidden' }}>
              <div style={{
                height: '100%', width: `${progress}%`,
                background: 'linear-gradient(90deg, var(--accent), var(--accent-purple))',
                borderRadius: 2, transition: 'width 0.5s ease',
                boxShadow: '0 0 12px rgba(88,166,255,0.5)',
              }} />
            </div>

            {/* Agent steps */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              {[
                { icon: Search, label: 'search_agent', phase: 'searching' as Phase },
                { icon: BookOpen, label: 'reader_agent', phase: 'reading' as Phase },
                { icon: Layers, label: 'synthesis_agent', phase: 'synthesizing' as Phase },
                { icon: Brain, label: 'critic_agent', phase: 'evaluating' as Phase },
              ].map((step, i) => (
                <AgentStep
                  key={step.label}
                  icon={step.icon}
                  label={step.label}
                  active={phase === step.phase}
                  done={phaseIndex > i}
                />
              ))}
            </div>
          </div>
        )}

        {/* Error */}
        {phase === 'error' && (
          <div style={{
            background: 'rgba(247,129,102,0.08)', border: '1px solid rgba(247,129,102,0.3)',
            borderRadius: 12, padding: 16, marginBottom: 24, display: 'flex', gap: 12, alignItems: 'flex-start',
            animation: 'fadeUp 0.4s ease',
          }}>
            <AlertTriangle size={16} color='var(--accent-warm)' style={{ marginTop: 2, flexShrink: 0 }} />
            <div>
              <div style={{ fontSize: 14, fontWeight: 600, color: 'var(--accent-warm)', marginBottom: 4 }}>Research Failed</div>
              <div style={{ fontSize: 13, color: 'var(--text-muted)', fontFamily: 'JetBrains Mono' }}>{error || 'Unknown error. Check that your API keys are configured.'}</div>
            </div>
          </div>
        )}

        {/* Result */}
        {result && phase === 'done' && (
          <div style={{ animation: 'fadeUp 0.5s ease' }}>

            {/* Sub-queries */}
            <div style={{
              display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 20,
            }}>
              {result.sub_queries.map((q, i) => (
                <span key={i} style={{
                  fontSize: 11, fontFamily: 'JetBrains Mono', padding: '4px 10px',
                  borderRadius: 20, border: '1px solid var(--border-glow)',
                  color: 'var(--text-muted)', background: 'var(--bg-2)',
                }}>
                  {q.length > 50 ? q.slice(0, 50) + '…' : q}
                </span>
              ))}
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 280px', gap: 20, alignItems: 'start' }}>

              {/* Report */}
              <div style={{
                background: 'var(--bg-2)', border: '1px solid var(--border)',
                borderRadius: 16, padding: 32,
              }}>
                <div style={{
                  fontSize: 11, fontFamily: 'JetBrains Mono', color: 'var(--text-dim)',
                  marginBottom: 20, display: 'flex', alignItems: 'center', gap: 8,
                }}>
                  <Terminal size={11} />
                  RESEARCH_REPORT.md
                </div>
                <div style={{
                  fontSize: 15, lineHeight: 1.8, color: 'var(--text)',
                  fontFamily: 'Inter',
                }}>
                  <ReportRenderer content={result.report} />
                </div>
              </div>

              {/* Sidebar */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>

                {/* Eval Scores */}
                {result.eval_scores && (
                  <div style={{
                    background: 'var(--bg-2)', border: '1px solid var(--border)',
                    borderRadius: 16, padding: 20,
                  }}>
                    <div style={{
                      fontSize: 11, fontFamily: 'JetBrains Mono', color: 'var(--text-dim)',
                      marginBottom: 16, display: 'flex', alignItems: 'center', gap: 8,
                    }}>
                      <Activity size={11} />
                      EVAL_SCORES
                    </div>

                    {/* Quality badge */}
                    <div style={{
                      textAlign: 'center', padding: '16px 0', marginBottom: 16,
                      borderBottom: '1px solid var(--border)',
                    }}>
                      <div style={{
                        fontSize: 42, fontFamily: 'Syne', fontWeight: 800,
                        color: result.eval_scores.quality_score >= 8 ? 'var(--accent-2)' :
                               result.eval_scores.quality_score >= 6 ? 'var(--accent)' : 'var(--accent-warm)',
                      }}>
                        {result.eval_scores.quality_score.toFixed(1)}
                      </div>
                      <div style={{ fontSize: 11, fontFamily: 'JetBrains Mono', color: 'var(--text-dim)' }}>
                        / 10 quality score
                      </div>
                    </div>

                    <ScoreBar
                      label="faithfulness"
                      value={result.eval_scores.faithfulness}
                      color="var(--accent)"
                    />
                    <ScoreBar
                      label="coverage"
                      value={result.eval_scores.coverage}
                      color="var(--accent-2)"
                    />
                    <ScoreBar
                      label="hallucination"
                      value={result.eval_scores.hallucination_rate}
                      invert={true}
                      color="var(--accent-purple)"
                    />

                    {result.eval_scores.flagged_claims.length > 0 && (
                      <div style={{ marginTop: 12, padding: 10, borderRadius: 8, background: 'rgba(247,129,102,0.08)' }}>
                        <div style={{ fontSize: 11, fontFamily: 'JetBrains Mono', color: 'var(--accent-warm)', marginBottom: 6 }}>
                          FLAGGED CLAIMS
                        </div>
                        {result.eval_scores.flagged_claims.slice(0, 2).map((c, i) => (
                          <div key={i} style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>
                            · {c.length > 60 ? c.slice(0, 60) + '…' : c}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}

                {/* Sources */}
                <div style={{
                  background: 'var(--bg-2)', border: '1px solid var(--border)',
                  borderRadius: 16, padding: 20,
                }}>
                  <button
                    onClick={() => setShowSources(!showSources)}
                    style={{
                      width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                      background: 'none', border: 'none', cursor: 'pointer',
                      marginBottom: showSources ? 12 : 0,
                    }}>
                    <div style={{ fontSize: 11, fontFamily: 'JetBrains Mono', color: 'var(--text-dim)', display: 'flex', alignItems: 'center', gap: 8 }}>
                      <BookOpen size={11} />
                      SOURCES ({result.sources.length})
                    </div>
                    <ChevronDown size={12} color='var(--text-dim)'
                      style={{ transform: showSources ? 'rotate(180deg)' : 'none', transition: '0.2s' }} />
                  </button>

                  {showSources && result.sources.map((s, i) => (
                    <a key={i} href={s.url} target="_blank" rel="noopener noreferrer"
                      style={{
                        display: 'block', padding: '8px 0',
                        borderTop: i > 0 ? '1px solid var(--border)' : 'none',
                        textDecoration: 'none',
                      }}>
                      <div style={{ fontSize: 12, color: 'var(--text)', marginBottom: 2, display: 'flex', gap: 4, alignItems: 'flex-start' }}>
                        <ExternalLink size={10} color='var(--text-dim)' style={{ marginTop: 2, flexShrink: 0 }} />
                        {(s.title || s.url).slice(0, 45) + ((s.title || s.url).length > 45 ? '…' : '')}
                      </div>
                      <div style={{ fontSize: 10, fontFamily: 'JetBrains Mono', color: 'var(--text-dim)' }}>
                        relevance: {(s.score * 100).toFixed(0)}%
                      </div>
                    </a>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Demo hint when idle */}
        {phase === 'idle' && (
          <div style={{ animation: 'fadeUp 0.6s ease 0.3s both' }}>
            <div style={{ fontSize: 12, fontFamily: 'JetBrains Mono', color: 'var(--text-dim)', marginBottom: 12, textAlign: 'center' }}>
              EXAMPLE QUERIES
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, justifyContent: 'center' }}>
              {[
                'How do transformer attention mechanisms work?',
                'What is the state of nuclear fusion energy in 2025?',
                'Latest advances in CRISPR gene editing therapies',
              ].map(q => (
                <button key={q} onClick={() => setQuery(q)}
                  style={{
                    padding: '8px 16px', borderRadius: 8, border: '1px solid var(--border)',
                    background: 'var(--bg-2)', color: 'var(--text-muted)', fontSize: 13,
                    cursor: 'pointer', fontFamily: 'Inter', transition: 'all 0.2s',
                    display: 'flex', alignItems: 'center', gap: 6,
                  }}>
                  <ArrowRight size={12} />
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

// ── Simple Markdown Renderer ─────────────────────────────────────
function ReportRenderer({ content }: { content: string }) {
  const lines = content.split('\n')
  return (
    <div>
      {lines.map((line, i) => {
        if (line.startsWith('# ')) return <h1 key={i} style={{ fontSize: 22, fontWeight: 800, marginBottom: 16, marginTop: i > 0 ? 32 : 0, fontFamily: 'Syne', color: 'var(--text)' }}>{line.slice(2)}</h1>
        if (line.startsWith('## ')) return <h2 key={i} style={{ fontSize: 17, fontWeight: 700, marginBottom: 10, marginTop: 28, fontFamily: 'Syne', color: 'var(--text)' }}>{line.slice(3)}</h2>
        if (line.startsWith('### ')) return <h3 key={i} style={{ fontSize: 15, fontWeight: 600, marginBottom: 8, marginTop: 20, fontFamily: 'Syne', color: 'var(--text)' }}>{line.slice(4)}</h3>
        if (line.startsWith('- ') || line.startsWith('* ')) return (
          <div key={i} style={{ display: 'flex', gap: 8, marginBottom: 6 }}>
            <span style={{ color: 'var(--accent)', marginTop: 2 }}>·</span>
            <span style={{ color: 'var(--text)', fontSize: 14 }}>{line.slice(2)}</span>
          </div>
        )
        if (line.startsWith('> ')) return <blockquote key={i} style={{ borderLeft: '3px solid var(--accent)', paddingLeft: 16, margin: '12px 0', color: 'var(--text-muted)', fontStyle: 'italic' }}>{line.slice(2)}</blockquote>
        if (line === '') return <div key={i} style={{ height: 8 }} />
        return <p key={i} style={{ marginBottom: 4, fontSize: 14, color: 'var(--text)', lineHeight: 1.75 }}>{line}</p>
      })}
    </div>
  )
}
