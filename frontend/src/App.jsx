import { useEffect, useMemo, useState } from 'react'
import {
  Activity, ArrowDownUp, ArrowRight, CalendarDays, CheckCircle2,
  ChevronDown, CircleAlert, Clock3, Code2, RefreshCw, Shield,
  Signal, Sparkles, Trophy, Zap,
} from 'lucide-react'

const API_BASE = (import.meta.env.VITE_API_URL || '').replace(/\/$/, '')

async function api(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    ...options,
  })
  const body = await response.json().catch(() => ({}))
  if (!response.ok) {
    throw new Error(body.detail || `Request failed (${response.status})`)
  }
  return body
}

function formatKickoff(value) {
  if (!value) return 'Time unavailable'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return date.toLocaleString([], { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' })
}

function TeamCrest({ name, logos = {}, className = '' }) {
  const [failed, setFailed] = useState(false)
  const src = logos[name]
  useEffect(() => setFailed(false), [src])
  return (
    <span className={`team-crest ${className}`} aria-label={`${name || 'Team'} crest`}>
      {src && !failed ? (
        <img src={src} alt="" loading="lazy" onError={() => setFailed(true)} />
      ) : (
        <Shield size={19} aria-hidden="true" />
      )}
    </span>
  )
}

function FormDots({ form = [] }) {
  if (!form.length) return <span className="muted small">Not enough recent results</span>
  return (
    <div className="form-dots">
      {form.map((item, index) => (
        <span className={`form-dot ${item.toLowerCase()}`} key={`${item}-${index}`} title={item}>
          {item}
        </span>
      ))}
    </div>
  )
}

function ProbabilityRow({ label, value, tone }) {
  const percentage = Math.max(0, Math.min(100, Number(value || 0) * 100))
  return (
    <div className="prob-row">
      <div className="prob-heading">
        <span>{label}</span>
        <strong>{percentage.toFixed(1)}%</strong>
      </div>
      <div className="track">
        <div className={`fill ${tone}`} style={{ width: `${percentage}%` }} />
      </div>
    </div>
  )
}

export default function App() {
  const [health, setHealth] = useState(null)
  const [teams, setTeams] = useState([])
  const [fixtures, setFixtures] = useState([])
  const [fixtureUpdated, setFixtureUpdated] = useState(null)
  const [home, setHome] = useState('')
  const [away, setAway] = useState('')
  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(true)
  const [predicting, setPredicting] = useState(false)
  const [error, setError] = useState('')
  const [fixtureError, setFixtureError] = useState('')
  const [logos, setLogos] = useState({})

  const homeTeam = useMemo(() => teams.find(team => team.name === home), [teams, home])
  const awayTeam = useMemo(() => teams.find(team => team.name === away), [teams, away])

  async function loadDashboard() {
    setLoading(true)
    setError('')
    try {
      const [healthData, teamData, fixtureData] = await Promise.all([
        api('/api/health'),
        api('/api/teams'),
        api('/api/fixtures'),
      ])
      setHealth(healthData)
      setTeams(teamData.teams || [])
      setFixtures(fixtureData.events || [])
      setFixtureUpdated(fixtureData.updated_utc || null)
      setHome(current => current && teamData.teams.some(team => team.name === current)
        ? current
        : (teamData.teams.find(team => team.name.includes('Seattle'))?.name || teamData.teams[0]?.name || ''))
      setAway(current => current && teamData.teams.some(team => team.name === current)
        ? current
        : (teamData.teams.find(team => team.name.includes('Inter Miami'))?.name ||
          teamData.teams.find(team => team.name !== (teamData.teams[0]?.name))?.name || ''))
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadDashboard()
    // Team crests are optional: missing permissions/network must never break predictions.
    api('/api/team-logos').then(data => setLogos(data.logos || {})).catch(() => setLogos({}))
  }, [])

  useEffect(() => {
    setPrediction(null)
  }, [home, away])

  async function runPrediction(event) {
    event.preventDefault()
    if (!home || !away || home === away) return
    setPredicting(true)
    setError('')
    setPrediction(null)
    try {
      const result = await api('/api/predict', {
        method: 'POST',
        body: JSON.stringify({ home_team: home, away_team: away }),
      })
      setPrediction(result)
    } catch (err) {
      setError(err.message)
    } finally {
      setPredicting(false)
    }
  }

  const modelReady = Boolean(health?.model_loaded)
  const upcoming = fixtures.filter(item => item.state !== 'final').slice(0, 8)
  const completed = fixtures.filter(item => item.state === 'final').slice(0, 4)

  return (
    <div className="app-shell">
      <div className="ambient ambient-one" />
      <div className="ambient ambient-two" />
      <header className="topbar">
        <a className="brand" href="#" aria-label="MLS Match Predictor home">
          <span className="brand-mark"><img src="/mls-predict-logo.png" alt="MLS Predict logo" /></span>
          <span>MLS<span className="brand-light">PREDICT</span></span>
        </a>
        <div className="topbar-right">
          <span className={`connection ${health?.history_loaded ? 'online' : 'offline'}`}>
            <span className="pulse" />
            {health?.history_loaded ? 'DATA CONNECTED' : 'CONNECTING'}
          </span>
          <a className="icon-link" href="https://github.com/calkarawoh/MLS-Match-Predictor" target="_blank" rel="noreferrer" aria-label="GitHub repository">
            <Code2 size={19} />
          </a>
        </div>
      </header>

      <main className="main-content">
        <section className="hero">
          <div className="eyebrow"><span className="eyebrow-line" /> FOOTBALL INTELLIGENCE / 2026</div>
          <div className="hero-row">
            <div>
              <h1>Predict the<br /><span>unpredictable.</span></h1>
              <p className="hero-copy">MLS match insights powered by historical results and machine learning.</p>
            </div>
            <div className="hero-orb"><Trophy size={43} strokeWidth={1.25} /></div>
          </div>
          <div className="hero-meta">
            <span><Signal size={14} /> REAL MATCH HISTORY</span>
            <span><Zap size={14} /> RANDOM FOREST</span>
            <span><Activity size={14} /> PRE-MATCH ESTIMATES</span>
          </div>
        </section>

        {error && (
          <div className="alert error-alert">
            <CircleAlert size={18} />
            <div><strong>Something needs attention</strong><span>{error}</span></div>
            <button className="alert-action" onClick={loadDashboard}>Retry</button>
          </div>
        )}

        <section className="section-grid" id="matchup">
          <div className="panel matchup-panel">
            <div className="panel-head">
              <div><span className="section-kicker">01 / MATCH CENTRE</span><h2>Build your matchup</h2></div>
              <span className="live-tag"><span className="pulse" /> PREDICT</span>
            </div>
            <form onSubmit={runPrediction}>
              <div className="team-selectors">
                <label className="team-field">
                  <span className="field-label">HOME TEAM</span>
                  <span className="select-wrap select-wrap-overlay">
                    <TeamCrest name={home} logos={logos} />
                    <span className="selected-team-name" aria-hidden="true">{home || 'Choose home team'}</span>
                    <ChevronDown size={16} aria-hidden="true" />
                    <select aria-label="Home team" value={home} onChange={event => setHome(event.target.value)} disabled={loading || !teams.length}>
                      {teams.map(team => <option value={team.name} key={team.name}>{team.name}</option>)}
                    </select>
                  </span>
                </label>
                <div className="versus"><span>VS</span><ArrowDownUp size={15} /></div>
                <label className="team-field">
                  <span className="field-label">AWAY TEAM</span>
                  <span className="select-wrap select-wrap-overlay">
                    <TeamCrest name={away} logos={logos} />
                    <span className="selected-team-name" aria-hidden="true">{away || 'Choose away team'}</span>
                    <ChevronDown size={16} aria-hidden="true" />
                    <select aria-label="Away team" value={away} onChange={event => setAway(event.target.value)} disabled={loading || !teams.length}>
                      {teams.map(team => <option value={team.name} key={team.name}>{team.name}</option>)}
                    </select>
                  </span>
                </label>
              </div>
              <button className="predict-button" type="submit" disabled={loading || predicting || !modelReady || !home || !away || home === away}>
                {predicting ? <><RefreshCw className="spin" size={18} /> ANALYZING MATCH</> : <><Sparkles size={18} /> PREDICT MATCH <ArrowRight size={18} /></>}
              </button>
              {!modelReady && !loading && (
                <p className="model-warning">The API is connected, but the trained real-MLS model is not loaded. Train it locally with <code>python real_model.py train</code>.</p>
              )}
            </form>
          </div>

          <div className="panel status-panel">
            <div className="panel-head compact">
              <div><span className="section-kicker">SYSTEM STATUS</span><h2>Model health</h2></div>
              <Activity size={19} className="muted-icon" />
            </div>
            <div className="status-line"><span>Historical data</span><strong className={health?.history_loaded ? 'good' : 'bad'}>{health?.history_loaded ? 'Connected' : 'Unavailable'}</strong></div>
            <div className="status-line"><span>Prediction model</span><strong className={modelReady ? 'good' : 'bad'}>{modelReady ? 'Ready' : 'Not loaded'}</strong></div>
            <div className="status-line"><span>Completed matches</span><strong>{health?.matches_loaded?.toLocaleString?.() ?? '—'}</strong></div>
            <div className="status-line"><span>Score feed checked</span><strong>{health?.last_scoreboard_check_utc ? formatKickoff(health.last_scoreboard_check_utc) : 'Not recorded'}</strong></div>
            <button className="text-button" onClick={loadDashboard}><RefreshCw size={14} /> Refresh dashboard data</button>
          </div>
        </section>

        <section className="panel prediction-panel" id="predictions">
          <div className="panel-head">
            <div><span className="section-kicker">02 / MODEL OUTPUT</span><h2>Outcome probabilities</h2></div>
            <span className="subtle-chip">THREE-WAY RESULT</span>
          </div>
          {!prediction ? (
            <div className="empty-prediction">
              <div className="empty-icon"><Activity size={24} /></div>
              <div><strong>Your prediction appears here</strong><span>Select two different teams and run the model to see estimated outcomes.</span></div>
            </div>
          ) : (
            <>
              <div className="prediction-result">
                <div><span className="section-kicker">MODEL'S TOP OUTCOME</span><strong>{prediction.top_prediction}</strong></div>
                <span className="result-check"><CheckCircle2 size={18} /> CALCULATED</span>
              </div>
              <div className="probability-list">
                <ProbabilityRow label={`${home} · HOME WIN`} value={prediction.probabilities.home_win} tone="home" />
                <ProbabilityRow label="DRAW" value={prediction.probabilities.draw} tone="draw" />
                <ProbabilityRow label={`${away} · AWAY WIN`} value={prediction.probabilities.away_win} tone="away" />
              </div>
              <p className="model-note">{prediction.note}</p>
            </>
          )}
        </section>

        <section className="form-section" id="form">
          <div className="section-title-row"><div><span className="section-kicker">03 / TEAM INTELLIGENCE</span><h2>Recent form</h2></div><span className="subtle-chip">LAST FIVE COMPLETED</span></div>
          <div className="form-grid">
            {[{team: homeTeam, label: 'HOME FORM'}, {team: awayTeam, label: 'AWAY FORM'}].map((item, index) => (
              <div className="panel form-card" key={item.label}>
                <div className="form-card-top"><span>{item.label}</span><TeamCrest name={item.team?.name} logos={logos} /></div>
                <h3>{item.team?.name || 'Select a team'}</h3>
                <FormDots form={item.team?.form || []} />
                <div className="stat-grid">
                  <div><strong>{item.team?.goals_for_per_match?.toFixed(2) ?? '—'}</strong><span>GF / MATCH</span></div>
                  <div><strong>{item.team?.goals_against_per_match?.toFixed(2) ?? '—'}</strong><span>GA / MATCH</span></div>
                  <div><strong>{item.team?.points_per_match?.toFixed(2) ?? '—'}</strong><span>PTS / MATCH</span></div>
                </div>
              </div>
            ))}
          </div>
        </section>

        <section className="panel fixtures-panel" id="fixtures">
          <div className="panel-head">
            <div><span className="section-kicker">04 / MATCH SCHEDULE</span><h2>Fixtures & recent scores</h2></div>
            <div className="fixture-updated"><Clock3 size={14} /> {fixtureUpdated ? `Updated ${formatKickoff(fixtureUpdated)}` : 'No saved update'}</div>
          </div>
          {!fixtures.length ? (
            <div className="empty-fixtures">No fixtures are available in the saved scoreboard snapshot.</div>
          ) : (
            <div className="fixture-list">
              {upcoming.map((event, index) => (
                <div className="fixture-row" key={event.event_id || `${event.date}-${index}`}>
                  <div className="fixture-date"><strong>{formatKickoff(event.date)}</strong><span>{event.state === 'live' ? 'IN PROGRESS' : 'SCHEDULED'} · LOCAL TIME</span></div>
                  <div className="fixture-teams"><span className="fixture-team home"><span>{event.home_team}</span><TeamCrest name={event.home_team} logos={logos} /></span><b>{event.state === 'live' ? `${event.home_goals ?? '—'} : ${event.away_goals ?? '—'}` : 'VS'}</b><span className="fixture-team away"><TeamCrest name={event.away_team} logos={logos} /><span>{event.away_team}</span></span></div>
                  <span className={`fixture-status ${event.state}`}>{event.state}</span>
                </div>
              ))}
              {completed.map((event, index) => (
                <div className="fixture-row completed" key={event.event_id || `final-${index}`}>
                  <div className="fixture-date"><strong>{formatKickoff(event.date)}</strong><span>FINAL · LOCAL TIME</span></div>
                  <div className="fixture-teams"><span className="fixture-team home"><span>{event.home_team}</span><TeamCrest name={event.home_team} logos={logos} /></span><b>{event.home_goals} : {event.away_goals}</b><span className="fixture-team away"><TeamCrest name={event.away_team} logos={logos} /><span>{event.away_team}</span></span></div>
                  <span className="fixture-status final">FINAL</span>
                </div>
              ))}
            </div>
          )}
          <p className="feed-disclaimer">Fixture and score information comes from a saved snapshot of an unofficial ESPN public feed. Refresh timing and coverage are not guaranteed.</p>
        </section>

        <footer className="footer">
          <div className="brand small-brand"><span className="brand-mark"><img src="/mls-predict-logo.png" alt="" /></span><span>MLS<span className="brand-light">PREDICT</span></span></div>
          <span>Built for football analytics · Not betting advice</span>
          <a href="https://github.com/calkarawoh/MLS-Match-Predictor" target="_blank" rel="noreferrer">VIEW SOURCE <ArrowRight size={13} /></a>
        </footer>
        <nav className="mobile-nav" aria-label="Mobile quick navigation">
          <a href="#matchup"><Shield size={19}/><span>Match</span></a>
          <a href="#predictions"><Activity size={19}/><span>Predict</span></a>
          <a href="#form"><Trophy size={19}/><span>Form</span></a>
          <a href="#fixtures"><CalendarDays size={19}/><span>Fixtures</span></a>
        </nav>
      </main>
    </div>
  )
}
