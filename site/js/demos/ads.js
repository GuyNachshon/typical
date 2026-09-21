// 06 ADS - a feed of 40 observations, each scored three ways in one decide() call: is-ad
// (2-way Choice), brand (Choice K=20, cue criteria in the question), category (Choice K=8).
// Brand/category rows are only *displayed* when is-ad wins - the cells are still computed
// every time (one bundled request), matching how the probe measured its ~590ms/observation.
//
// TEMPLATE (mirrored verbatim in scripts/record_data_demos.py's ads_queries()):
//   state = observation text, verbatim
//   query[0] is-ad  = {type:'choice', question:AD_Q, labels:['advertisement','not_an_advertisement']}
//     AD_Q (exact) = "Is this text an advertisement or not? Each option's criterion is listed;
//       pick the single best match.\nadvertisement: promotes a product, service, or brand so the
//       reader will buy or use it  not_an_advertisement: news, a personal message, a recipe, a
//       notice, a job post, a support request, or other text that does not sell anything"
//   query[1] brand  = {type:'choice', question:BRAND_Q, labels:[...20 brand names]}
//     BRAND_Q = "Which brand is this observation advertising? Each option's criterion is
//       listed; pick the single best match.\n" + "{brand}: {cue}" pairs joined by two spaces
//   query[2] category = {type:'choice', question:CAT_Q, labels:[...8 category ids]}
//     CAT_Q = "Which product category does this observation belong to? Each option's
//       criterion is listed; pick the single best match.\n" + "{id}: {desc}" pairs joined by
//       two spaces

const AD_Q =
  "Is this text an advertisement or not? Each option's criterion is listed; pick the single best match.\n" +
  'advertisement: promotes a product, service, or brand so the reader will buy or use it  ' +
  'not_an_advertisement: news, a personal message, a recipe, a notice, a job post, a support request, or other text that does not sell anything';

function brandQuestion(brands) {
  const pairs = Object.entries(brands).map(([name, b]) => `${name}: ${b.cue}`);
  return "Which brand is this observation advertising? Each option's criterion is listed; pick the single best match.\n" + pairs.join('  ');
}

function categoryQuestion(categories) {
  const pairs = categories.map((c) => `${c.id}: ${c.desc}`);
  return "Which product category does this observation belong to? Each option's criterion is listed; pick the single best match.\n" + pairs.join('  ');
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

function row(label, p, isWinner) {
  const r = el('div', 'ex-row');
  r.appendChild(el('span', 'ex-row-text', label));
  const bar = el('span', 'ex-bar');
  const fill = el('span', 'ex-bar-fill' + (isWinner ? ' is-winner' : ''));
  fill.style.width = `${(p * 100).toFixed(1)}%`;
  bar.appendChild(fill);
  r.appendChild(bar);
  r.appendChild(el('span', 'ex-num', p.toFixed(2)));
  return r;
}

export async function mount(el0, ctx) {
  const res = await fetch('data/demos/ads.json');
  const pool = res.ok ? await res.json() : null;
  if (!pool) {
    el0.innerHTML = '<p class="ex-muted">ads pool unavailable.</p>';
    return;
  }
  const brandNames = Object.keys(pool.brands);
  const catIds = pool.categories.map((c) => c.id);
  const brandQ = brandQuestion(pool.brands);
  const catQ = categoryQuestion(pool.categories);

  el0.innerHTML = '';
  el0.className = 'exhibit ads-exhibit';

  const tally = el('p', 'ads-tally', 'scoring 40 observations…');
  el0.appendChild(tally);

  const feed = el('div', 'ads-register');
  el0.appendChild(feed);

  const VISIBLE = 6;
  const showMore = document.createElement('button');
  showMore.type = 'button';
  showMore.className = 'ex-ghost';
  showMore.textContent = `Show all ${pool.observations.length} →`;
  showMore.hidden = pool.observations.length <= VISIBLE;
  el0.appendChild(showMore);
  let expanded = false;
  showMore.addEventListener('click', () => {
    expanded = true;
    showMore.hidden = true;
    pending.forEach((item) => feed.appendChild(item));
    pending.length = 0;
  });
  const pending = [];

  const questions = document.createElement('details');
  questions.className = 'ex-details';
  const summary = el('summary', null, 'the prompt');
  const pre = el('pre', null, `is-ad:\n${AD_Q}\n\nbrand:\n${brandQ}\n\ncategory:\n${catQ}`);
  questions.append(summary, pre);
  el0.appendChild(questions);

  const caption = el(
    'p',
    'ex-caption',
    "40 observations, 20 brands: brand .94 and category .94 with cue rules in the prompt (chance .05 / .125); is-it-an-ad .88 vs .80 always-yes. The cues are the program — without them brand drops to .75."
  );
  el0.appendChild(caption);

  let adCorrect = 0,
    adTotal = 0;
  let brandCorrect = 0,
    brandTotal = 0;
  let catCorrect = 0,
    catTotal = 0;

  function updateTally() {
    const adAcc = adTotal ? (adCorrect / adTotal).toFixed(2) : '—';
    const brandAcc = brandTotal ? (brandCorrect / brandTotal).toFixed(2) : '—';
    const catAcc = catTotal ? (catCorrect / catTotal).toFixed(2) : '—';
    tally.textContent = `is-ad ${adCorrect}/${adTotal} (${adAcc}) · brand ${brandCorrect}/${brandTotal} (${brandAcc}) · category ${catCorrect}/${catTotal} (${catAcc})`;
  }
  updateTally();

  for (let idx = 0; idx < pool.observations.length; idx++) {
    const obs = pool.observations[idx];
    const item = el('div', 'ads-item');
    item.appendChild(el('p', 'ads-text', obs.text));
    const decisions = el('div', 'ads-decisions');
    item.appendChild(decisions);
    const gold = el('span', 'ads-gold', `gold: ${obs.isAd ? `${obs.brand} · advertisement` : 'not an advertisement'}`);
    item.appendChild(gold);
    if (expanded || idx < VISIBLE) feed.appendChild(item);
    else pending.push(item);

    const out = await ctx.decide(obs.text, [
      { type: 'choice', question: AD_Q, labels: ['advertisement', 'not_an_advertisement'] },
      { type: 'choice', question: brandQ, labels: brandNames },
      { type: 'choice', question: catQ, labels: catIds },
    ]);
    const [adR, brandR, catR] = out?.results ?? [];
    const isAdPred = adR?.argmax === 'advertisement';

    adTotal++;
    if (isAdPred === obs.isAd) adCorrect++;

    decisions.appendChild(row(`IS-AD · ${adR?.argmax ?? '?'}`, adR?.probs?.[adR.argmax] ?? 0, true));
    if (isAdPred) {
      decisions.appendChild(row(`BRAND · ${brandR?.argmax ?? '?'}`, brandR?.probs?.[brandR.argmax] ?? 0, true));
      decisions.appendChild(row(`CATEGORY · ${catR?.argmax ?? '?'}`, catR?.probs?.[catR.argmax] ?? 0, true));
    }

    if (obs.isAd) {
      brandTotal++;
      if (brandR?.argmax === obs.brand) brandCorrect++;
      catTotal++;
      if (catR?.argmax === pool.brands[obs.brand]?.category) catCorrect++;
    }

    updateTally();
    if (ctx.mode() !== 'live') await sleep(Math.min(out?.ms ?? 15, 35));
  }
}
