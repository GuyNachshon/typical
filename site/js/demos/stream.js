// Hero decision stream: cycles 5 item types (support -> team/wants, alert -> page/ticket/
// ignore, agent step -> next tool of 6, ad observation -> is-ad/brand, policy -> yes/no)
// through ctx.heatTicker. Live mode calls ctx.decide() for real, every ~600ms. Static mode
// replays scripts/record_docs_demos.py's precomputed "recorded" decision for the same pool
// item, at the same cadence - never a client-fabricated number either way.
const TICK_MS = 600;
const TYPES = ['support', 'alert', 'tool', 'ad', 'policy'];
const SEED_ROWS = 6; // push a few rows immediately so first paint isn't an empty ticker

let poolPromise = null;
function loadPool() {
  if (!poolPromise) {
    poolPromise = fetch('data/demos/stream.json')
      .then((res) => (res.ok ? res.json() : null))
      .catch(() => null);
  }
  return poolPromise;
}

function truncate(text, n = 88) {
  return text.length > n ? `${text.slice(0, n)}…` : text;
}

// Server result(s) for one pool item -> ticker chips. Mirrors record_docs_demos.py's
// decisions_for() so live rows and recorded rows render identically.
function decisionsFor(kind, results) {
  if (kind === 'support') {
    const [team, wants] = results;
    return [
      { label: team.argmax, p: team.probs[team.argmax] },
      { label: wants.argmax, p: wants.probs[wants.argmax] },
    ];
  }
  if (kind === 'alert' || kind === 'tool') {
    const r = results[0];
    return [{ label: r.argmax, p: r.probs[r.argmax] }];
  }
  if (kind === 'ad') {
    const [isAd, brand] = results;
    const out = [{ label: isAd.argmax, p: isAd.probs[isAd.argmax] }];
    if (isAd.argmax === 'advertisement') out.push({ label: brand.argmax, p: brand.probs[brand.argmax] });
    return out;
  }
  // policy
  const r = results[0];
  return [{ label: 'yes', p: r.probs.yes }];
}

export async function mount(el, ctx) {
  const pool = await loadPool();
  if (!pool) {
    el.textContent = 'stream offline — data/demos/stream.json missing';
    return;
  }
  const ticker = ctx.heatTicker(el, { maxRows: 60 });
  const cursor = { support: 0, alert: 0, tool: 0, ad: 0, policy: 0 };
  let typeIdx = 0;

  async function tick() {
    const kind = TYPES[typeIdx % TYPES.length];
    typeIdx += 1;
    const bucket = pool[kind];
    const items = bucket.items;
    if (!items || !items.length) return;
    const i = cursor[kind] % items.length;
    cursor[kind] += 1;
    const item = items[i];

    if (ctx.mode() === 'live') {
      const queries = item.queries || bucket.queries;
      const res = await ctx.decide(item.text, queries);
      if (!res) return; // server hiccup this tick - just skip it, no fabricated row
      ticker.push({ text: truncate(item.text), decisions: decisionsFor(kind, res.results), ms: res.ms });
    } else {
      const rec = bucket.recorded && bucket.recorded[i];
      if (!rec) return;
      ticker.push({ text: truncate(item.text), decisions: rec.decisions, ms: rec.ms });
    }
  }

  for (let i = 0; i < SEED_ROWS; i++) await tick();
  setInterval(tick, TICK_MS);
}
