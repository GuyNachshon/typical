// release.js — the launch post. Reuses shell.js (loaded as its own <script type="module"> tag,
// unchanged, right before this one) for everything it already does: the hero and card DOOM films,
// the drive/snake game screens, and the cost chart on #chart-g2 via mountResults -> costBar off
// site/data/models.json. This page reuses those same element ids on purpose so shell.js's boot()
// mounts them with no page-specific code here.
//
// The one chart shell.js doesn't own is the read-path diagram under "One state, many decisions" —
// mount it directly.
import { mountFlow } from './arch.js';

document.addEventListener('DOMContentLoaded', () => {
  const el = document.getElementById('chart-arch');
  if (el) mountFlow(el);
});

// The whole field on one benchmark. Every row is the same 231 public ids, so the table is
// comparable by construction — which is also why it is not flattering: we sit mid-field, and a
// classifier a quarter of typical-small's size is two rows above it. Printing the field we lose in
// is worth more than printing the one we win.
async function mountField() {
  const host = document.getElementById('jev-field');
  if (!host) return;
  const doc = await fetch('data/jevbench-field.json').then((r) => (r.ok ? r.json() : null)).catch(() => null);
  if (!doc?.rows?.length) return;

  const table = document.createElement('table');
  table.className = 'register field';
  const head = ['model', 'kind', 'standard', 'hard'];
  table.innerHTML = `<thead><tr>${head.map((h) => `<th>${h}</th>`).join('')}</tr></thead>`;
  const body = document.createElement('tbody');
  doc.rows.forEach((r) => {
    const tr = document.createElement('tr');
    if (r.ours) tr.className = 'is-ours';
    const name = r.size ? `${r.name} · ${r.size}` : r.name;
    [name, r.note ? `${r.kind} · ${r.note}` : r.kind, fmt(r.std), fmt(r.hard)].forEach((v, i) => {
      const td = document.createElement('td');
      td.textContent = v;
      td.dataset.label = head[i];
      tr.appendChild(td);
    });
    body.appendChild(tr);
  });
  // chance sits in the table rather than in a footnote: a row you cannot beat is a row
  const chance = document.createElement('tr');
  chance.className = 'is-chance';
  [['guessing', 'majority baseline', fmt(doc.chance.std), fmt(doc.chance.hard)]][0].forEach((v, i) => {
    const td = document.createElement('td');
    td.textContent = v;
    td.dataset.label = head[i];
    chance.appendChild(td);
  });
  body.appendChild(chance);
  table.appendChild(body);
  host.replaceChildren(table);

  const note = document.createElement('p');
  note.className = 'note mt-18';
  note.textContent = `${doc.what} ${doc.caveats.join(' ')}`;
  host.appendChild(note);
}

const fmt = (v) => (v == null ? '—' : v.toFixed(3).replace(/^0/, ''));

document.addEventListener('DOMContentLoaded', mountField);
