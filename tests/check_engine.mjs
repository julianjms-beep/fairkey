// The browser engine must give exactly LightGBM's answers. Run: node tests/check_engine.mjs
import fs from "node:fs";
await import("../web/fk-model.js");
const M = globalThis.FKModel;
const model = JSON.parse(fs.readFileSync(new URL("../web/data/model.json", import.meta.url)));
const fx = JSON.parse(fs.readFileSync(new URL("./fixtures.json", import.meta.url)));
const nF = fx.x[0].length;
let maxRaw = 0, maxShap = 0, maxBase = 0, maxReason = 0, sumErr = 0;
fx.x.forEach((row, i) => {
  const x = row.map(v => (v === null ? NaN : v));
  const r = M.raw(model.risk, x);
  maxRaw = Math.max(maxRaw, Math.abs(r - fx.raw[i]));
  const { phi, base } = M.explain(model.risk, x, nF);
  phi.forEach((v, j) => (maxShap = Math.max(maxShap, Math.abs(v - fx.contrib[i][j]))));
  maxBase = Math.max(maxBase, Math.abs(base - fx.contrib[i][nF]));
  sumErr = Math.max(sumErr, Math.abs(base + phi.reduce((a, b) => a + b, 0) - r));
  const pr = M.multi(model.reasons, x);
  pr.forEach((v, k) => (maxReason = Math.max(maxReason, Math.abs(v - fx.reasons[i][k]))));
});
const ok1 = maxRaw < 1e-5 && maxShap < 1e-5 && maxBase < 1e-5 && maxReason < 1e-6 && sumErr < 1e-9;
console.log(`${fx.x.length} applications · max |raw| ${maxRaw.toExponential(2)} · max |SHAP| ${maxShap.toExponential(2)} · `
  + `max |base| ${maxBase.toExponential(2)} · max |reason p| ${maxReason.toExponential(2)} · SHAP sums ${sumErr.toExponential(2)} · ${ok1 ? "PASS" : "FAIL"}`);

// ---- "no lender chosen": every lender of a state at once (rawCats, leavesCats) and the exact SHAP of their average
const meta = JSON.parse(fs.readFileSync(new URL("../web/data/meta.json", import.meta.url)));
const li = meta.features.indexOf("lender"), nc = meta.cats.lender.length;
const lw = meta.states.FL.lw, wv = new Float64Array(nc);
meta.cats.lender.forEach((l, i) => (wv[i] = lw[l] || 0));
const tot = wv.reduce((a, b) => a + b, 0); wv.forEach((v, i) => (wv[i] = v / tot));
const cats = [...wv.keys()].filter(i => wv[i] > 0);
let eCats = 0, eLeaves = 0, eMix = 0;
fx.x.slice(0, 60).forEach((row, i) => {
  const x = row.map(v => (v === null ? NaN : v)); x[li] = NaN;
  const rc = M.rawCats(model.risk, x, li, nc, cats);
  let avg = 0;
  for (const c of cats) { const y = x.slice(); y[li] = c; const r = M.raw(model.risk, y); eCats = Math.max(eCats, Math.abs(r - rc[c])); avg += wv[c] * r; }
  if (i < 10) {
    const lc = M.leavesCats(model.risk, x, li, nc, cats);
    for (const c of cats.slice(0, 8)) { const y = x.slice(); y[li] = c; M.leaves(model.risk, y).forEach((v, k) => (eLeaves = Math.max(eLeaves, Math.abs(v - lc[k * nc + c])))); }
  }
  const e = M.explainMix(model.risk, x, nF, li, wv, "FL");
  eMix = Math.max(eMix, Math.abs(e.base + e.phi.reduce((a, b) => a + b, 0) - avg));
});
// brute force: exact Shapley values over the other inputs (2^10 coalitions) for two applications
function nextNode(t, n, x) {
  const f = t.f[n], v = x[f], th = t.t[n];
  if (th < 0) { if (!(v >= 0) || Number.isNaN(v)) return t.r[n]; return t.cs[-th - 1].includes(Math.trunc(v)) ? t.l[n] : t.r[n]; }
  let fv = v; const mt = t.mt[n];
  if (Number.isNaN(fv) && mt !== 2) fv = 0;
  if ((mt === 1 && Math.abs(fv) <= 1e-35) || (mt === 2 && Number.isNaN(fv))) return t.dl[n] ? t.l[n] : t.r[n];
  return fv <= th ? t.l[n] : t.r[n];
}
function ev(t, n, x, inS, set) {        // E[tree | inputs in S known], lender = the weighted mix
  if (t.f[n] === -1) { let m = 0; for (const c of set) m += wv[c]; return t.v[n] * m; }
  if (t.f[n] === li) {
    const L = [], R = [], th = t.t[n];
    for (const c of set) ((th < 0 ? t.cs[-th - 1].includes(c) : c <= th) ? L : R).push(c);
    return (L.length ? ev(t, t.l[n], x, inS, L) : 0) + (R.length ? ev(t, t.r[n], x, inS, R) : 0);
  }
  if (inS[t.f[n]]) return ev(t, nextNode(t, n, x), x, inS, set);
  const cn = t.c[n] || 1;
  return (t.c[t.l[n]] / cn) * ev(t, t.l[n], x, inS, set) + (t.c[t.r[n]] / cn) * ev(t, t.r[n], x, inS, set);
}
const players = meta.features.map((_, j) => j).filter(j => j !== li), P = players.length;
const fact = k => (k <= 1 ? 1 : k * fact(k - 1));
let eBrute = 0;
for (const row of fx.x.slice(0, 2)) {
  const x = row.map(v => (v === null ? NaN : v)); x[li] = NaN;
  const val = new Float64Array(1 << P);
  for (let m = 0; m < (1 << P); m++) {
    const inS = new Array(nF).fill(false); players.forEach((j, b) => { if (m & (1 << b)) inS[j] = true; });
    for (const t of model.risk.trees) val[m] += ev(t, 0, x, inS, cats);
  }
  const e = M.explainMix(model.risk, x, nF, li, wv, "FL");
  players.forEach((j, b) => {
    let phi = 0;
    for (let m = 0; m < (1 << P); m++) if (!(m & (1 << b))) {
      let k = 0; for (let q = m; q; q &= q - 1) k++;
      phi += fact(k) * fact(P - k - 1) / fact(P) * (val[m | (1 << b)] - val[m]);
    }
    eBrute = Math.max(eBrute, Math.abs(phi - e.phi[j]));
  });
  eBrute = Math.max(eBrute, Math.abs(val[0] - e.base), Math.abs(e.phi[li]));
}
const ok2 = eCats < 1e-9 && eLeaves < 1e-12 && eMix < 1e-9 && eBrute < 1e-9;
console.log(`all lenders at once (Florida weights) · rawCats ${eCats.toExponential(2)} · leavesCats ${eLeaves.toExponential(2)} · `
  + `mix SHAP sums ${eMix.toExponential(2)} · vs brute-force Shapley ${eBrute.toExponential(2)} · ${ok2 ? "PASS" : "FAIL"}`);
process.exit(ok1 && ok2 ? 0 : 1);
