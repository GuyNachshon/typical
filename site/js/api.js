// decide() talks to the local FastAPI server when live, and falls back to a
// precomputed replay (or null) everywhere else - including window.TYPICAL_STATIC
// pages served with no backend at all.
let replaysPromise = null;
let _mode = 'static';
let healthPromise = null;

function loadReplays() {
  if (!replaysPromise) {
    replaysPromise = fetch('data/replays.json')
      .then((res) => (res.ok ? res.json() : {}))
      .catch(() => ({}));
  }
  return replaysPromise;
}

// Small deterministic string hash (djb2-ish) - only needs to be stable, not secure.
export function hashKey(state, queries) {
  const s = state + JSON.stringify(queries);
  let h = 5381;
  for (let i = 0; i < s.length; i++) h = ((h << 5) + h + s.charCodeAt(i)) | 0;
  return (h >>> 0).toString(36);
}

// Every probability on the site prints through here. Two places, leading zero dropped — except
// where two places would lie: .998 must not read as 1.00, because the difference between certain
// and nearly certain is the only thing a calibrated model is selling.
export function fmtProb(p) {
  const s = p.toFixed(2);
  // two places is the reading width; the only reason to spend a third is that two of them would
  // print a certainty the model did not have, in either direction. Below a thousandth even three
  // places round to .000, which reads as impossible rather than unlikely -- so say the bound.
  if (s === '1.00' && p < 1) return p.toFixed(3).replace(/^0/, '');
  if (s === '0.00' && p > 0) return p < 0.0005 ? '<.001' : p.toFixed(3).replace(/^0/, '');
  return s.replace(/^0/, '');
}

// ponytail: one in-flight request at a time -- the local MPS server serializes anyway, and
// concurrent fetches from several screens only add queueing latency (and once crashed Metal).
let chain = Promise.resolve();
function queued(fn) {
  const run = chain.then(fn, fn);
  chain = run.catch(() => {});
  return run;
}

export async function decide(state, queries, { model = 'typical-small' } = {}) {
  const isStatic = typeof window !== 'undefined' && window.TYPICAL_STATIC;
  if (healthPromise) await healthPromise; // first decide() may race the health probe
  if (!isStatic && _mode === 'live') {
    try {
      const res = await queued(() =>
        fetch('/api/decide', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ model, state, queries }),
        })
      );
      if (res.ok) return await res.json();
    } catch {
      // fall through to replay
    }
  }
  const replays = await loadReplays();
  const key = hashKey(state, queries);
  return replays[key] ?? null;
}

export function mode() {
  return _mode;
}

export function probeHealth() {
  healthPromise = _probeHealth();
  return healthPromise;
}

async function _probeHealth() {
  try {
    const res = await fetch('/api/health');
    _mode = res.ok ? 'live' : 'static';
  } catch {
    _mode = 'static';
  }
  if (typeof document !== 'undefined') {
    document.querySelectorAll('[data-mode-pill]').forEach((pill) => {
      pill.textContent = _mode;
      pill.classList.toggle('mode-live', _mode === 'live');
      pill.classList.toggle('mode-static', _mode !== 'live');
    });
  }
  return _mode;
}

if (typeof window !== 'undefined') {
  window.addEventListener('DOMContentLoaded', () => probeHealth());
}

function selfTest() {
  console.assert(hashKey('a', [{ q: 1 }]) === hashKey('a', [{ q: 1 }]), 'hashKey is deterministic');
  console.assert(hashKey('a', [{ q: 1 }]) !== hashKey('b', [{ q: 1 }]), 'hashKey differs on different state');
  console.assert(hashKey('a', [{ q: 1 }]) !== hashKey('a', [{ q: 2 }]), 'hashKey differs on different queries');
  console.assert(fmtProb(0.5874) === '.59' && fmtProb(0.083) === '.08', 'two places, no leading zero');
  console.assert(fmtProb(0.998) === '.998', 'near-certain never prints as certain');
  console.assert(fmtProb(1) === '1.00' && fmtProb(0) === '.00', 'and the actual ends are left alone');
  console.assert(fmtProb(0.0013) === '.001', 'nor near-zero as zero');
  console.log('api.js self-test OK');
  return true;
}

if (typeof process !== 'undefined' && import.meta.url === `file://${process.argv[1]}`) {
  selfTest();
}

export { selfTest };
