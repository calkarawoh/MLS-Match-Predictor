"""Unofficial ESPN MLS score feed. No API key; endpoints may change.

Completed games -> data/mls_real_matches.csv. Fixtures/scores -> data/mls_live_snapshot.json.
No fabricated possession, shots or injuries are added.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

BASE = 'https://site.api.espn.com/apis/site/v2/sports/soccer/usa.1'
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'data'
HISTORY = DATA_DIR / 'mls_real_matches.csv'
SNAPSHOT = DATA_DIR / 'mls_live_snapshot.json'
COLUMNS = ['event_id', 'date', 'season', 'home_team', 'away_team', 'home_goals', 'away_goals']
HEADERS = {'User-Agent': 'MLSResearchDashboard/1.0 (educational project; respectful requests)'}


def fetch_json(session: requests.Session, endpoint: str, params: dict | None = None) -> dict:
    try:
        response = session.get(f'{BASE}/{endpoint}', params=params, timeout=25, headers=HEADERS)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError('Expected a JSON object')
        return payload
    except (requests.RequestException, ValueError) as error:
        raise RuntimeError(f'Could not fetch ESPN {endpoint}: {error}') from error


def parse_event(event: dict) -> dict | None:
    """Parse ESPN scoreboard/schedule event. Returns None for unrecognized shape."""
    comps = event.get('competitions') or []
    if not comps:
        return None
    comp = comps[0]
    teams = {entry.get('homeAway'): entry for entry in comp.get('competitors', [])}
    if 'home' not in teams or 'away' not in teams:
        return None
    home, away = teams['home'], teams['away']
    team_name = lambda entry: (entry.get('team') or {}).get('displayName') or (entry.get('team') or {}).get('name')
    if not team_name(home) or not team_name(away):
        return None
    status = (comp.get('status') or event.get('status') or {})
    status_type = status.get('type') or {}
    state = status_type.get('state', '')
    completed = status_type.get('completed', False)
    date = event.get('date') or comp.get('date')
    if not date:
        return None
    try:
        timestamp = datetime.fromisoformat(date.replace('Z', '+00:00'))
    except ValueError:
        return None
    def score(entry: dict) -> int | None:
        raw = entry.get('score')
        if isinstance(raw, dict):
            raw = raw.get('value', raw.get('displayValue'))
        try:
            return int(float(raw))
        except (ValueError, TypeError):
            return None
    hg, ag = score(home), score(away)
    # ESPN may put 0-0 placeholders on fixtures that have not kicked off.
    # A score belongs only to an in-progress or completed game.
    if state not in ('in', 'post') and not completed:
        hg, ag = None, None
    if state == 'post' and hg is not None and ag is not None:
        completed = True
    # Do not treat a postponed/cancelled status as an upcoming fixture.
    name = str(status_type.get('name') or '').upper()
    if any(x in name for x in ('POSTPONED', 'CANCELED', 'CANCELLED')):
        return None
    return {
        'event_id': str(event.get('id', '')),
        'date': timestamp.isoformat(),
        'season': timestamp.year,
        'home_team': team_name(home), 'away_team': team_name(away),
        'home_goals': hg, 'away_goals': ag,
        'state': 'final' if completed else ('live' if state == 'in' else 'scheduled'),
        'status': status_type.get('shortDetail') or status_type.get('description') or '',
    }


def fetch_scoreboard(session: requests.Session, dates: str, limit: int = 2000) -> list[dict]:
    payload = fetch_json(session, 'scoreboard', {'dates': dates, 'limit': limit})
    return [parsed for event in payload.get('events', []) if (parsed := parse_event(event))]


def fetch_teams(session: requests.Session) -> list[str]:
    payload = fetch_json(session, 'teams', {'limit': 100})
    leagues = (payload.get('sports') or [{}])[0].get('leagues') or [{}]
    wrappers = leagues[0].get('teams') or []
    ids = []
    for wrapper in wrappers:
        team = wrapper.get('team') or wrapper
        if team.get('id'):
            ids.append(str(team['id']))
    return sorted(set(ids))


def year_events(session: requests.Session, year: int) -> list[dict]:
    """Prefer one calendar-year scoreboard; fallback to team schedules if sparse.

    Schedules avoid reliance on undocumented ESPN date-range support. Results
    are deduplicated by event ID because each game is on two team schedules.
    """
    try:
        events = fetch_scoreboard(session, str(year))
        finished = sum(e['state'] == 'final' for e in events)
        if finished >= 120:
            return events
        print(f'{year}: season scoreboard returned only {finished} finals; trying team schedules')
    except RuntimeError as error:
        print(f'{year}: season scoreboard unavailable ({error}); trying team schedules')
        events = []
    ids = fetch_teams(session)
    if not ids:
        raise RuntimeError('No team IDs returned. ESPN may have changed its API.')
    by_id = {e['event_id']: e for e in events if e['event_id']}
    errors = []
    for idx, team_id in enumerate(ids, 1):
        try:
            payload = fetch_json(session, f'teams/{team_id}/schedule', {'season': year, 'limit': 100})
            for event in payload.get('events', []):
                parsed = parse_event(event)
                if parsed and parsed['season'] == year and parsed['event_id']:
                    by_id[parsed['event_id']] = parsed
        except RuntimeError as error:
            errors.append(str(error))
        if idx % 10 == 0:
            print(f'  {idx}/{len(ids)} teams checked')
        time.sleep(0.12)
    finals = sum(x['state'] == 'final' for x in by_id.values())
    if finals < 60:
        raise RuntimeError(f'Only {finals} completed matches retrieved for {year}. Refusing to train on an incomplete season. First error: {errors[:1]}')
    if errors:
        print(f'Warning: {len(errors)} team schedules failed; coverage may be incomplete.')
    return list(by_id.values())


def save_history(records: list[dict]) -> int:
    DATA_DIR.mkdir(exist_ok=True)
    rows = [{k: x.get(k) for k in COLUMNS} for x in records
            if x.get('state') == 'final' and x.get('home_goals') is not None and x.get('away_goals') is not None]
    frame = pd.DataFrame(rows, columns=COLUMNS)
    if HISTORY.exists():
        prev = pd.read_csv(HISTORY, dtype={'event_id': str})
        frame = pd.concat([prev, frame], ignore_index=True)
    frame = frame.drop_duplicates('event_id', keep='last').sort_values(['date', 'event_id'])
    frame.to_csv(HISTORY, index=False)
    return len(frame)


def sync_history(start: int = 2024, end: int | None = None) -> int:
    end = end or datetime.now(timezone.utc).year
    all_records = []
    with requests.Session() as session:
        for year in range(start, end + 1):
            events = year_events(session, year)
            completed = [e for e in events if e['state'] == 'final']
            print(f'{year}: {len(completed)} completed MLS games loaded')
            all_records.extend(completed)
            time.sleep(0.25)
    count = save_history(all_records)
    print(f'Saved {count} total real completed matches: {HISTORY}')
    return count


def refresh_live(days_ahead: int = 10, days_behind: int = 2) -> list[dict]:
    """Recent results, live events and upcoming fixtures. Run periodically; not a push feed."""
    now = datetime.now(timezone.utc)
    records = {}
    errors = []
    with requests.Session() as session:
        for offset in range(-days_behind, days_ahead + 1):
            day = (now + timedelta(days=offset)).strftime('%Y%m%d')
            try:
                for event in fetch_scoreboard(session, day, 100):
                    records[event['event_id']] = event
            except RuntimeError as error:
                errors.append(str(error))
            time.sleep(0.08)
    if not records and errors:
        raise RuntimeError(f'Live feed unavailable. {errors[0]}')
    if errors:
        print(f'Warning: {len(errors)} days could not be checked; fixture list may be incomplete')
    events = sorted(records.values(), key=lambda e: e['date'])
    DATA_DIR.mkdir(exist_ok=True)
    SNAPSHOT.write_text(json.dumps({'updated_utc': now.isoformat(), 'events': events}, indent=2), encoding='utf-8')
    if any(e['state'] == 'final' for e in events):
        print(f'History now contains {save_history(events)} completed results')
    print(f'Refreshed {len(events)} MLS fixtures/results at {now.isoformat()}')
    return events


def load_snapshot() -> dict:
    if SNAPSHOT.exists():
        return json.loads(SNAPSHOT.read_text(encoding='utf-8'))
    return {'updated_utc': None, 'events': []}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pull real MLS data from an unofficial ESPN feed')
    parser.add_argument('action', choices=['sync', 'refresh'])
    parser.add_argument('--start', type=int, default=2024)
    parser.add_argument('--end', type=int)
    args = parser.parse_args()
    if args.action == 'sync':
        sync_history(args.start, args.end)
    else:
        refresh_live()
