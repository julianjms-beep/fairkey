/* FairKey — the page. Everything is computed here, on your device: the risk model, TreeSHAP, the reason model,
   the what-if search, the safe-zone map and the lender comparison. No data is sent anywhere. */
(() => {
  "use strict";
  const $ = s => document.querySelector(s), $$ = s => [...document.querySelectorAll(s)];
  const M = window.FKModel;
  let META, MODEL, EV, FAIR, LANG = "en";
  const ORIG = new Map();
  const S = { income: 85000, price: 350000, downPct: 10, debts: 450, loanType: 1, term: 30, state: "FL", lender: "",
    coapp: false, manuf: false, rate: 6.5, tax: null };

  // ---------------------------------------------------------------- language + formatting
  const D = () => (window.I18N[LANG] && window.I18N[LANG].dyn) || {};
  const t = (k, v) => {
    let s = D()[k] ?? window.I18N.en.dyn[k] ?? k;
    if (v) s = s.replace(/\{(\w+)\}/g, (_, x) => (v[x] ?? ""));
    return s;
  };
  const loc = () => (LANG === "es" ? "es-US" : "en-US");
  // the site always opens in English; Spanish is one tap away (or a link with ?lang=es)
  const startLang = () => (new URLSearchParams(location.search).get("lang") === "es" ? "es" : "en");
  const nf = (v, d = 0) => new Intl.NumberFormat(loc(), { maximumFractionDigits: d, minimumFractionDigits: d }).format(v);
  const usd = v => new Intl.NumberFormat(loc(), { style: "currency", currency: "USD", maximumFractionDigits: 0 }).format(v);
  const pc = (p, d = 0) => nf(100 * p, d) + "%";
  const pts = v => (v > 0 ? "+" : v < 0 ? "−" : "") + nf(Math.abs(v), Math.abs(v) < 10 ? 1 : 0);
  const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  function applyLang(l) {
    LANG = l;
    document.documentElement.lang = l;
    $$(".lang button").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.lang === l)));
    const st = (window.I18N[l] && window.I18N[l].static) || {};
    $$("[data-i18n]").forEach(el => {
      const k = el.dataset.i18n;
      if (!ORIG.has(el)) ORIG.set(el, el.innerHTML);
      el.innerHTML = l === "en" ? ORIG.get(el) : (st[k] ?? ORIG.get(el));
    });
    if (META) {
      const low = EV.bands_check.find(r => r.band === "low" && r.year === "2025");
      const FILL = { nRisk: nf(MODEL.risk.trees.length), nReason: nf(MODEL.reasons.trees.length), nFeat: META.features.length,
        lowDen: nf(100 * low.denied, 0), nM: nf(EV.n.complete / 1e6, 1), nAll: nf(EV.n.all_raw / 1e6, 1),
        nTest: nf(EV.n.test / 1e6, 1), nLenders: Object.keys(META.lenders).length };
      $$("[data-fill]").forEach(el => { el.innerHTML = el.innerHTML.replace(/\{(\w+)\}/g, (m, k) => (FILL[k] ?? m)); });
    }
    document.title = l === "es" ? "FairKey — ¿Te dirá que sí el banco?" : "FairKey — Will the bank say yes?";
    $("#say").placeholder = t("say.ph");
    if (META) { fillSelects(); renderPersonas(); renderStatic(); update(); heroRelabel(); }
  }

  // ---------------------------------------------------------------- the application → model inputs
  const idx = name => META.features.indexOf(name);
  function derive(s) {
    const down = s.price * s.downPct / 100;
    const base = Math.max(s.price - down, 1000);
    // financed fees, as lenders report them: FHA upfront insurance, VA funding fee, USDA guarantee fee
    const fee = s.loanType === 2 ? 0.0175 : s.loanType === 3 ? (s.downPct >= 10 ? 0.0125 : s.downPct >= 5 ? 0.015 : 0.0215)
      : s.loanType === 4 ? 0.01 : 0;
    const loan = base * (1 + fee);
    const ltv = loan / s.price * 100;
    const r = s.rate / 100 / 12, n = s.term * 12;
    const pi = r > 0 ? loan * r / (1 - Math.pow(1 + r, -n)) : loan / n;
    // taxes + insurance: the state's typical property-tax rate (Census ACS 2024) + its average home insurance (NAIC 2023)
    const sr = (META && META.states[s.state]) || {};
    const autoTi = s.price * ((sr.tax_rate ?? 1.0) + (sr.ins_rate ?? 0.6)) / 100 / 12;
    const ti = s.tax != null ? s.tax : autoTi;
    const mi = s.loanType === 2 ? loan * 0.0055 / 12 : s.loanType === 4 ? loan * 0.0035 / 12
      : (s.loanType === 1 && ltv > 80 ? loan * 0.005 / 12 : 0);
    const housing = pi + ti + mi;
    const dti = (housing + s.debts) / (s.income / 12) * 100;
    return { down, loan, ltv, pi, ti, mi, housing, dti, autoTi };
  }
  // HMDA reports DTI in bands outside 36–49%, loan and price as the middle of a $10,000 bucket, income in $1,000s
  function hmdaDti(v) {
    if (v < 20) return 15; if (v < 30) return 25; if (v < 36) return 33;
    const r = Math.round(v);
    if (r <= 49) return Math.max(36, r);
    return v <= 60 ? 55 : 65;
  }
  const bucket = v => Math.floor(v / 10000) * 10 + 5;
  function vector(s) {
    const d = derive(s);
    const f = {
      income_k: Math.max(1, Math.round(s.income / 1000)), loan_k: bucket(d.loan), price_k: bucket(s.price),
      ltv: Math.round(d.ltv * 1000) / 1000, dti: hmdaDti(d.dti), term_years: s.term,
      coapp: +s.coapp, manufactured: +s.manuf,
      loan_type: META.cats.loan_type.indexOf(s.loanType),
      state: META.cats.state ? META.cats.state.indexOf(s.state) : NaN,
      lender: s.lender ? META.cats.lender.indexOf(s.lender) : NaN,
    };
    f.loan_to_income = bucket(d.loan) / f.income_k;
    return META.features.map(k => f[k]);
  }

  const cal = raw => M.sigmoid(META.platt[0] * raw + META.platt[1]);
  // lender weights per state: where people in that state actually apply (2022–23). All lenders for the risk,
  // the biggest ones for the explanations (TreeSHAP is the slow part)
  // The engine walks each tree once for all of them (FKModel.rawCats) and explains their average exactly
  // (FKModel.explainMix), instead of repeating everything once per lender.
  const LWC = {};
  function lwFor(st) {
    if (LWC[st]) return LWC[st];
    const cats = META.cats.lender, sh = (META.states[st] || {}).lw || META.lender_share;
    let all = cats.map((l, i) => [i, sh[l] || 0]).filter(w => w[1] > 0);
    const tot = all.reduce((a, w) => a + w[1], 0);
    all = all.map(([i, w]) => [i, w / tot]);
    const wv = new Float64Array(cats.length);
    for (const [i, w] of all) wv[i] = w;
    return (LWC[st] = { all, wv, idx: all.map(a => a[0]) });
  }
  const NC = () => META.cats.lender.length;
  // the named lenders that handle at least 0.4% of a state's applications (the others are "smaller lenders" there)
  const stateLenders = st => {
    const lw = (META.states[st] || {}).lw || META.lender_share;
    return Object.keys(META.lenders).filter(k => (lw[k] || 0) >= 0.004);
  };
  /** Calibrated chance of denial. With no lender chosen: averaged over the lenders people in that state actually use. */
  function riskX(x, st = S.state) {
    const li = idx("lender");
    if (li < 0 || !Number.isNaN(x[li])) return cal(M.raw(MODEL.risk, x));
    const W = lwFor(st), r = M.rawCats(MODEL.risk, x, li, NC(), W.idx);
    let p = 0;
    for (const [i, w] of W.all) p += w * cal(r[i]);
    return p;
  }
  const risk = s => riskX(vector(s), s.state);

  // why: exact TreeSHAP. With no lender chosen it explains the average over your state's lenders, so the lender
  // itself is not one of the pushes (it isn't something you entered).
  function explainX(x) {
    const li = idx("lender"), n = META.features.length;
    if (li < 0 || !Number.isNaN(x[li])) return M.explain(MODEL.risk, x, n);
    return M.explainMix(MODEL.risk, x, n, li, lwFor(S.state).wv, S.state);
  }
  function reasonsX(x) {
    const li = idx("lender"), K = META.reasons.length;
    if (li < 0 || !Number.isNaN(x[li])) return M.multi(MODEL.reasons, x);
    const W = lwFor(S.state), rk = Array.from({ length: K }, (_, k) => M.rawCats(MODEL.reasons, x, li, NC(), W.idx, k));
    const out = new Array(K).fill(0);
    for (const [i, w] of W.all) M.softmax(rk.map(r => r[i])).forEach((v, k) => (out[k] += w * v));
    return out;
  }
  const band = p => (p < META.bands.low ? "low" : p < META.bands.high ? "medium" : "high");
  const color = p => (p < META.bands.low ? "var(--good)" : p < META.bands.high ? "var(--mid)" : "var(--bad)");
  const ink = p => (p < META.bands.low ? "var(--good-ink)" : p < META.bands.high ? "var(--mid-ink)" : "var(--bad-ink)");   // for text

  // ---------------------------------------------------------------- form
  const parseMoney = v => Number(String(v).replace(/[^0-9.]/g, "")) || 0;
  // state names: Spanish where they differ
  const ST_ES = { DC: "Distrito de Columbia", HI: "Hawái", LA: "Luisiana", MI: "Míchigan", MO: "Misuri", MS: "Misisipi",
    NC: "Carolina del Norte", ND: "Dakota del Norte", NJ: "Nueva Jersey", NM: "Nuevo México", NY: "Nueva York",
    OR: "Oregón", PA: "Pensilvania", SC: "Carolina del Sur", SD: "Dakota del Sur", WV: "Virginia Occidental" };
  const stName = k => (LANG === "es" && ST_ES[k]) || (META.states[k] || {}).name || k;
  function fillSelects() {
    const sEl = $("#state");
    sEl.innerHTML = Object.keys(META.states).sort((a, b) => stName(a).localeCompare(stName(b), loc()))
      .map(k => `<option value="${k}">${esc(stName(k))}</option>`).join("");
    sEl.value = S.state;
    const l = $("#lender");
    const keys = stateLenders(S.state);
    if (S.lender && S.lender !== "other" && META.lenders[S.lender] && !keys.includes(S.lender)) keys.push(S.lender);
    l.innerHTML = `<option value="">${esc(t("lender.any", { st: stName(S.state) }))}</option>` +
      keys.map(k => [k, META.lenders[k]]).sort((a, b) => a[1].localeCompare(b[1]))
        .map(([k, v]) => `<option value="${k}">${esc(v)}</option>`).join("") +
      `<option value="other">${esc(t("lender.other", { n: Object.keys(META.lenders).length }))}</option>`;
    l.value = S.lender;
  }
  function syncForm() {
    $("#income").value = nf(S.income); $("#incomeR").value = S.income;
    $("#price").value = nf(S.price); $("#priceR").value = Math.min(S.price, 1500000);
    $("#debts").value = nf(S.debts); $("#debtsR").value = Math.min(S.debts, 5000);
    $("#downR").value = S.downPct;
    $$("#loanType button").forEach(b => b.setAttribute("aria-pressed", String(+b.dataset.v === S.loanType)));
    $$("#term button").forEach(b => b.setAttribute("aria-pressed", String(+b.dataset.v === S.term)));
    $("#coapp").checked = S.coapp; $("#manuf").checked = S.manuf;
    $("#state").value = S.state;
    $("#lender").value = S.lender;
  }
  function bindForm() {
    const pair = (num, rng, key, max) => {
      $(num).addEventListener("input", e => { S[key] = Math.min(parseMoney(e.target.value), max); $(rng).value = S[key]; update(); });
      $(num).addEventListener("blur", e => { e.target.value = nf(S[key]); });
      $(rng).addEventListener("input", e => { S[key] = +e.target.value; $(num).value = nf(S[key]); update(); });
    };
    pair("#income", "#incomeR", "income", 5e6);
    pair("#price", "#priceR", "price", 2e7);
    pair("#debts", "#debtsR", "debts", 1e5);
    $("#downR").addEventListener("input", e => { S.downPct = +e.target.value; update(); });
    $$("#loanType button").forEach(b => b.addEventListener("click", () => { S.loanType = +b.dataset.v; syncForm(); update(); }));
    $$("#term button").forEach(b => b.addEventListener("click", () => { S.term = +b.dataset.v; syncForm(); update(); }));
    ["coapp", "manuf"].forEach(k => $("#" + k).addEventListener("change", e => { S[k] = e.target.checked; update(); }));
    $("#state").addEventListener("change", e => { S.state = e.target.value; markPersona(-1); fillSelects(); update(); });
    $("#lender").addEventListener("change", e => { S.lender = e.target.value; update(); });
    $("#rate").addEventListener("input", e => { const v = parseFloat(e.target.value.replace(",", ".")); if (v >= 0 && v < 25) { S.rate = v; update(); } });
    $("#tax").addEventListener("input", e => { const v = e.target.value.trim(); S.tax = v === "" ? null : parseMoney(v); update(); });
    $$(".lang button").forEach(b => b.addEventListener("click", () => applyLang(b.dataset.lang)));
  }

  // ---------------------------------------------------------------- result
  const FEAT_NAME = k => t("feat." + k);
  function featValue(k, s, d) {
    switch (k) {
      case "dti": return nf(d.dti, 0) + "%";
      case "ltv": return nf(d.ltv, 1) + "%";
      case "loan_type": return t("lt." + s.loanType);
      case "lender": return s.lender ? (s.lender === "other" ? t("lender.otherShort") : META.lenders[s.lender]) : t("lender.anyShort");
      case "price_k": return usd(s.price);
      case "loan_k": return usd(d.loan);
      case "income_k": return usd(s.income);
      case "loan_to_income": return nf(d.loan / s.income, 1) + "×";
      case "term_years": return s.term + " " + t("years");
      case "state": return stName(s.state);
      case "coapp": case "manufactured":
        return t(({ coapp: s.coapp, manufactured: s.manuf })[k] ? "yes" : "no");
      default: return "";
    }
  }

  function bars(el, rows, opts = {}) {
    const max = opts.max || Math.max(1e-9, ...rows.map(r => Math.abs(r.v)));
    el.innerHTML = rows.map(r => {
      const w = Math.min(100, Math.abs(r.v) / max * (opts.diverge ? 50 : 100));
      const left = opts.diverge ? (r.v >= 0 ? 50 : 50 - w) : 0;
      const cls = opts.diverge ? (r.v >= 0 ? "up" : "down") : (r.me ? "plain me" : "plain");
      return `<div class="bar ${cls}"><span class="nm" title="${esc(r.title || r.name)}">${r.name}</span>` +
        `<span class="tr">${opts.diverge ? '<span class="mid"></span>' : ""}<i style="left:${left}%;width:${w}%${r.color ? `;background:${r.color}` : ""}"></i></span>` +
        `<span class="vl">${r.label}</span></div>`;
    }).join("");
  }

  let lastP = null, lastActs = [];
  function update() {
    const d = derive(S);
    $("#downOut").textContent = `${nf(S.downPct, S.downPct % 1 ? 1 : 0)}% · ${usd(d.down)}`;
    const dtiWarn = d.dti > 50 ? " warn" : "";
    $("#derived").innerHTML =
      `<div><b>${usd(d.loan)}</b><span>${t("d.loan")}</span></div>` +
      `<div><b>${nf(d.ltv, 1)}%</b><span>${t("d.ltv")}</span></div>` +
      `<div class="${dtiWarn}"><b>${nf(d.dti, 0)}%</b><span>${t("d.dti", { pay: usd(d.housing) })}</span></div>`;
    $("#tax").placeholder = t("tax.auto", { v: nf(Math.round(d.autoTi)) });

    const x = vector(S);
    const p = riskX(x);
    lastP = p;
    renderVerdict(p);
    scheduleSlow();
  }

  // the heavier parts (explanations, what-if, map) run just after you stop moving the slider
  let slowTimer = null;
  function scheduleSlow() { clearTimeout(slowTimer); slowTimer = setTimeout(slow, 90); }
  function slow() {
    const d = derive(S), x = vector(S), p = lastP;
    // ---- why (TreeSHAP)
    const { phi, base } = explainX(x);
    const a = META.platt[0];
    const p0 = cal(base);
    const sum = phi.reduce((u, v) => u + v, 0);
    const slope = Math.abs(sum) > 0.02 && Math.sign(p - p0) === Math.sign(sum) ? (p - p0) / sum : a * p * (1 - p);
    const rows = META.features.map((k, j) => ({ k, v: 100 * phi[j] * slope }));
    rows.sort((u, v) => Math.abs(v.v) - Math.abs(u.v));
    const top = rows.slice(0, 6), restV = rows.slice(6).reduce((u, r) => u + r.v, 0);
    const shapRows = top.map(r => ({ name: `${FEAT_NAME(r.k)} <small>${esc(featValue(r.k, S, d))}</small>`, title: FEAT_NAME(r.k),
      v: r.v, label: pts(r.v) }));
    if (Math.abs(restV) >= 0.05) shapRows.push({ name: t("feat.rest"), v: restV, label: pts(restV) });
    bars($("#shap"), shapRows, { diverge: true });
    $("#shapFoot").innerHTML = t("shap.foot", { p0: pc(p0, 1), p: pc(p, p < .1 ? 1 : 0) });

    // ---- reasons
    const rp = reasonsX(x);
    const rr = META.reasons.map((k, i) => ({ k, v: rp[i] })).sort((u, v) => v.v - u.v);
    bars($("#reasons"), rr.map((r, i) => ({ name: t("reason." + r.k), v: r.v, label: pc(r.v), me: i === 0 })), { max: 1 });

    // ---- what would change it
    const acts = whatIf(p);
    lastActs = acts;
    $("#actions").innerHTML = acts.length ? acts.map((a2, i) =>
      `<div class="act${a2.combo ? " combo" : ""}" role="button" tabindex="0" data-i="${i}">${a2.combo ? `<span class="tag">${t("act.combo")}</span>` : ""}` +
      `<span class="what">${a2.text}${a2.note ? `<small class="note">${a2.note}</small>` : ""}</span><span class="ft"><span class="from">${pc(p, p < .1 ? 1 : 0)}</span>` +
      `<span class="to">${pc(a2.p, a2.p < .1 ? 1 : 0)}</span><span class="gain">${pts(100 * (a2.p - p))} ${t("ptsShort")}</span></span></div>`).join("")
      : `<p class="empty">${t("act.none")}</p>`;

    // ---- summary in plain words
    const topUp = rows.find(r => r.v > 0.3);
    // the summary suggests a change to the application itself; switching lender stays in the cards, with its caveat
    const best = acts.find(a2 => !a2.combo && a2.group !== "lender");
    const sAvg = META.states[S.state].denied;
    let sum1 = t("sum.p", { p: pc(p, p < .1 ? 1 : 0), cmp: t(p > sAvg * 1.15 ? "sum.above" : p < sAvg * 0.85 ? "sum.below" : "sum.near"),
      avg: pc(sAvg, 1), st: stName(S.state) });
    if (topUp) sum1 += " " + t("sum.top", { f: FEAT_NAME(topUp.k).toLowerCase(), v: esc(featValue(topUp.k, S, d)) });
    sum1 += " " + t("sum.reason", { r: t("reason." + rr[0].k).toLowerCase() });
    if (best && best.p < p - 0.005) sum1 += " " + t("sum.best", { a: lowerFirst(best.text.replace(/<[^>]+>/g, "")), p: pc(best.p, best.p < .1 ? 1 : 0) });
    $("#summary").innerHTML = sum1;

    renderLenders(x);
    renderSame(p);
    renderReality(p);
    // the map and the tree-by-tree view are the heaviest: drawn when their tab is open (or opened later)
    stale.zone = stale.ai = true;
    drawVisible();
  }
  const stale = { zone: true, ai: true };
  function drawVisible() {
    if (curTab === "map" && stale.zone) { stale.zone = false; drawZone(vector(S), lastP); }
    if (curTab === "ai" && stale.ai) { stale.ai = false; renderAI(vector(S)); }
  }

  const lowerFirst = v => v.charAt(0).toLowerCase() + v.slice(1);
  function whatIf(p0) {
    const cand = [];
    const add = (id, s2, text, group) => cand.push({ id, s: s2, text, group, p: risk(s2) });
    for (const k of [5, 10]) if (S.downPct + k <= 60)
      add("down" + k, { ...S, downPct: S.downPct + k }, t("act.down", { pct: k, usd: usd(S.price * k / 100) }), "down");
    if (S.debts >= 100) {
      add("debtHalf", { ...S, debts: S.debts / 2 }, t("act.debt", { usd: usd(S.debts / 2) }), "debt");
      add("debtZero", { ...S, debts: 0 }, t("act.debtAll", { usd: usd(S.debts) }), "debt");
    }
    for (const k of [10, 20]) add("price" + k, { ...S, price: S.price * (1 - k / 100) }, t("act.price", { pct: k, usd: usd(S.price * (1 - k / 100)) }), "price");
    if (S.loanType !== 1 && S.downPct >= 3) add("conv", { ...S, loanType: 1 }, t("act.type", { t: t("lt.1") }), "type");
    if (S.loanType !== 2 && S.downPct >= 3.5) add("fha", { ...S, loanType: 2 }, t("act.type", { t: "FHA" }), "type");
    if (S.term !== 30) add("term30", { ...S, term: 30 }, t("act.term"), "term");
    if (idx("lender") >= 0) {
      const ls = lenderRisks(vector(S));
      if (ls.length && ls[0].k !== S.lender) {
        add("lender", { ...S, lender: ls[0].k }, t("act.lender", { l: esc(ls[0].name) }), "lender");
        cand[cand.length - 1].note = t("act.lenderNote");
      }
    }
    const good = cand.filter(c => c.p < p0 - 0.003).sort((u, v) => u.p - v.p);
    const seen = new Set(), out = [];
    for (const c of good) { if (seen.has(c.group)) continue; seen.add(c.group); out.push(c); if (out.length === 3) break; }
    if (out.length >= 2) {
      const s2 = { ...out[0].s };
      const o1 = out[1].s;
      ["downPct", "debts", "price", "loanType", "term", "lender"].forEach(k => { if (o1[k] !== S[k]) s2[k] = o1[k]; });
      const pc2 = risk(s2);
      if (pc2 < out[0].p - 0.003) out.push({ combo: true, s: s2, p: pc2, text: `${out[0].text} + ${lowerFirst(out[1].text)}` });
    }
    return out;
  }

  function lenderRisks(x) {
    const li = idx("lender");
    if (li < 0) return [];
    const keys = stateLenders(S.state);
    if (S.lender && META.lenders[S.lender] && !keys.includes(S.lender)) keys.push(S.lender);
    const ids = keys.map(k => META.cats.lender.indexOf(k)), r = M.rawCats(MODEL.risk, x, li, NC(), ids);
    return keys.map((k, j) => ({ k, name: META.lenders[k], p: cal(r[ids[j]]) })).sort((a, b) => a.p - b.p);
  }
  function renderLenders(x) {
    const ls = lenderRisks(x);
    $("#lendersCap").innerHTML = t("r.lenderscap", { st: stName(S.state) });
    if (!ls.length) { $("#lenders").innerHTML = ""; return; }
    const rows = ls.slice(0, 6);
    if (S.lender && S.lender !== "other" && !rows.some(r => r.k === S.lender)) rows.push(ls.find(r => r.k === S.lender));
    const y = x.slice(); y[idx("lender")] = META.cats.lender.indexOf("other");
    const other = cal(M.raw(MODEL.risk, y));
    const mx = Math.max(...rows.map(r => r.p), other, 0.05);
    bars($("#lenders"), [...rows.map(r => ({ name: esc(r.name), v: r.p, label: pc(r.p, r.p < .1 ? 1 : 0), me: r.k === S.lender, color: color(r.p) })),
      { name: `<i>${esc(t("lender.otherShort"))}</i>`, v: other, label: pc(other, other < .1 ? 1 : 0), color: color(other) }], { max: mx * 1.05 });
  }

  function renderSame(p) {
    const cr = FAIR && FAIR.calibration_race;
    if (!cr) return;
    const groups = ["White", "Asian", "Hispanic", "Black"].filter(g => cr[g]);
    const pick = g => cr[g].reduce((a, b) => (Math.abs(b.pred - p) < Math.abs(a.pred - p) ? b : a));
    const ref = pick("White");
    $("#sameCap").innerHTML = t("same.cap", { p: pc(ref.pred, 0) });
    const rows = groups.map(g => { const b = pick(g); return { name: t("g." + g), v: b.actual, label: pc(b.actual, 0), color: g === "Black" ? "var(--bad)" : g === "Hispanic" ? "var(--mid)" : "var(--teal)" }; });
    bars($("#sameBars"), rows, { max: Math.max(...rows.map(r => r.v)) * 1.1 });
  }

  // ---------------------------------------------------------------- safe-zone map
  const ZW = 36, ZH = 24;
  let zoneData = null;
  function riskColor(p) {
    const stops = [[0, [42, 157, 143]], [META.bands.low, [140, 190, 130]], [(META.bands.low + META.bands.high) / 2, [233, 160, 59]],
      [META.bands.high, [220, 110, 80]], [0.5, [209, 73, 91]], [1, [150, 30, 50]]];
    for (let i = 1; i < stops.length; i++) if (p <= stops[i][0]) {
      const [a, ca] = stops[i - 1], [b, cb] = stops[i], f = (p - a) / (b - a);
      return ca.map((c, j) => Math.round(c + (cb[j] - c) * f));
    }
    return stops[stops.length - 1][1];
  }
  function blur(g, w, h) {
    const k = [1, 4, 6, 4, 1], a = new Float32Array(g.length), b = new Float32Array(g.length);
    const at = (v, i, n) => Math.min(n - 1, Math.max(0, i));
    for (let j = 0; j < h; j++) for (let i = 0; i < w; i++) {
      let s = 0; for (let d = -2; d <= 2; d++) s += k[d + 2] * g[j * w + at(0, i + d, w)]; a[j * w + i] = s / 16;
    }
    for (let j = 0; j < h; j++) for (let i = 0; i < w; i++) {
      let s = 0; for (let d = -2; d <= 2; d++) s += k[d + 2] * a[at(0, j + d, h) * w + i]; b[j * w + i] = s / 16;
    }
    return b;
  }
  function drawZone(x, p) {
    const cv = $("#zone"), ctx = cv.getContext("2d");
    const pMin = Math.max(60000, S.price * 0.4), pMax = Math.max(S.price * 1.8, pMin + 100000), dMax = 40;
    // with no lender chosen: shift each cell by this application's lender average (log-odds), so the map stays fast
    const li = idx("lender");
    const avg = li >= 0 && Number.isNaN(x[li]);
    let shift = 0;
    const other = META.cats.lender.indexOf("other");
    if (avg) {
      const y = x.slice(); y[li] = other;
      const lo = q => Math.log(q / (1 - q));
      shift = lo(Math.min(0.999, Math.max(0.001, p))) - lo(cal(M.raw(MODEL.risk, y)));
    }
    const grid = new Float32Array(ZW * ZH);
    for (let j = 0; j < ZH; j++) for (let i = 0; i < ZW; i++) {
      const price = pMin + (pMax - pMin) * (i + 0.5) / ZW, dp = dMax * (1 - (j + 0.5) / ZH);
      const y = vector({ ...S, price, downPct: dp });
      if (avg) y[li] = other;
      const z = META.platt[0] * M.raw(MODEL.risk, y) + META.platt[1] + shift;
      grid[j * ZW + i] = 1 / (1 + Math.exp(-z));
    }
    zoneData = { pMin, pMax, dMax, grid };
    const off = document.createElement("canvas");
    off.width = ZW; off.height = ZH;
    const oc = off.getContext("2d"), img = oc.createImageData(ZW, ZH);
    // soften the tree model's step edges for display only (the tooltip still reads the exact grid)
    const soft = blur(grid, ZW, ZH);
    soft.forEach((v, k) => { const c = riskColor(v); img.data.set([c[0], c[1], c[2], 255], k * 4); });
    oc.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.clearRect(0, 0, cv.width, cv.height);
    ctx.drawImage(off, 0, 0, cv.width, cv.height);
    ctx.font = "600 22px Inter, sans-serif";
    ctx.fillStyle = "rgba(255,255,255,.92)";
    ctx.textBaseline = "bottom";
    ctx.textAlign = "center";
    const kfmt = v => "$" + (v >= 1e6 ? nf(v / 1e6, 2) + "M" : nf(v / 1000, 0) + "k");
    for (const f of [0.15, 0.5, 0.85]) ctx.fillText(kfmt(pMin + (pMax - pMin) * f), f * cv.width, cv.height - 8);
    ctx.textBaseline = "middle";
    ctx.textAlign = "left";
    for (const dp of [10, 20, 30]) ctx.fillText(dp + "%", 12, cv.height * (1 - dp / dMax));
    const cx = (S.price - pMin) / (pMax - pMin) * cv.width, cy = (1 - Math.min(S.downPct, dMax) / dMax) * cv.height;
    ctx.lineWidth = 7; ctx.strokeStyle = "#fff";
    ctx.beginPath(); ctx.arc(cx, cy, 16, 0, Math.PI * 2); ctx.stroke();
    ctx.lineWidth = 3.5; ctx.strokeStyle = "#0d1b2a";
    ctx.beginPath(); ctx.arc(cx, cy, 16, 0, Math.PI * 2); ctx.stroke();
  }
  function bindZone() {
    const cv = $("#zone"), tip = $("#zoneTip");
    const frac = e => { const r = cv.getBoundingClientRect(); return [(e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height]; };
    cv.addEventListener("mousemove", e => {
      if (!zoneData) return;
      const [fx, fy] = frac(e);
      const i = Math.min(ZW - 1, Math.max(0, Math.floor(fx * ZW))), j = Math.min(ZH - 1, Math.max(0, Math.floor(fy * ZH)));
      const price = zoneData.pMin + (zoneData.pMax - zoneData.pMin) * fx, dp = zoneData.dMax * (1 - fy);
      tip.hidden = false;
      tip.style.left = (fx * 100) + "%"; tip.style.top = (fy * 100) + "%";
      tip.textContent = `${usd(price)} · ${nf(dp, 0)}% ${t("zone.down")} → ${pc(zoneData.grid[j * ZW + i], 0)}`;
    });
    cv.addEventListener("mouseleave", () => (tip.hidden = true));
    cv.addEventListener("click", e => {
      if (!zoneData) return;
      const [fx, fy] = frac(e);
      S.price = Math.round((zoneData.pMin + (zoneData.pMax - zoneData.pMin) * fx) / 5000) * 5000;
      S.downPct = Math.max(0, Math.round(zoneData.dMax * (1 - fy) * 2) / 2);
      syncForm(); update();
    });
  }

  // ---------------------------------------------------------------- evidence + audit
  function axes(W, H, m, xt, yt, xl, yl, sx = 1, sy = 1) {
    const X = v => m.l + v / sx * (W - m.l - m.r), Y = v => H - m.b - v / sy * (H - m.t - m.b);
    let s = "";
    for (const v of yt) s += `<line class="grid" x1="${m.l}" x2="${W - m.r}" y1="${Y(v)}" y2="${Y(v)}"/><text x="${m.l - 8}" y="${Y(v) + 4}" text-anchor="end">${Math.round(v * 100)}%</text>`;
    for (const v of xt) s += `<text x="${X(v)}" y="${H - m.b + 18}" text-anchor="middle">${Math.round(v * 100)}%</text>`;
    s += `<line class="ax" x1="${m.l}" x2="${W - m.r}" y1="${H - m.b}" y2="${H - m.b}"/><line class="ax" x1="${m.l}" x2="${m.l}" y1="${m.t}" y2="${H - m.b}"/>`;
    s += `<text x="${(W + m.l - m.r) / 2}" y="${H - 4}" text-anchor="middle">${xl}</text>`;
    s += `<text transform="translate(12 ${(H - m.b + m.t) / 2}) rotate(-90)" text-anchor="middle">${yl}</text>`;
    return s;
  }
  function renderStatic() {
    const m = EV.metrics, op = EV.operating;
    const big = v => (v >= 1e6 ? nf(v / 1e6, v >= 1e7 ? 1 : 2) + "M" : nf(v / 1000, 0) + "k");
    $("#hsN").textContent = big(EV.n.complete);
    $("#hsT").textContent = big(EV.n.test);
    const calErr = EV.calibration.reduce((a, c) => a + c.n * Math.abs(c.pred - c.actual), 0) / EV.calibration.reduce((a, c) => a + c.n, 0);
    $("#kpis").innerHTML = [
      [nf(m.fairkey.ap, 3), t("k1.vs", { lr: nf(m.logistic.ap, 3), dti: nf(m.dti.ap, 3) }), t("k1")],
      [`${op.per1000.fk_false} <small style="font-size:.5em;color:var(--muted)">vs ${op.per1000.rule_false}</small>`, t("k2.vs", { rec: pc(op.rule.recall, 0) }), t("k2")],
      [nf(100 * calErr, 1) + " " + t("ptsShort"), t("k3.vs"), t("k3")],
      [pc(EV.reasons.top1, 0), t("k4.vs", { maj: pc(EV.reasons.majority, 0) }), t("k4")],
    ].map(([b, vs, p]) => `<div class="kpi"><b>${b}</b><div class="vs">${vs}</div><p>${p}</p></div>`).join("");
    $("#accText").innerHTML = t("acc", { acc: pc(m.accuracy_all_approve, 0) });

    const W = 460, H = 300, mg = { l: 48, r: 14, t: 10, b: 40 };
    const X = v => mg.l + v * (W - mg.l - mg.r), Y = v => H - mg.b - v * (H - mg.t - mg.b);
    const line = (p2, c, w = 3, dash = "") => `<polyline fill="none" stroke="${c}" stroke-width="${w}" stroke-linejoin="round" ${dash ? `stroke-dasharray="${dash}"` : ""} points="${p2.map(([r, q]) => `${X(r).toFixed(1)},${Y(q).toFixed(1)}`).join(" ")}"/>`;
    let s = axes(W, H, mg, [0, .25, .5, .75, 1], [0, .25, .5, .75, 1], t("ax.recall"), t("ax.precision"));
    s += `<line x1="${X(0)}" x2="${X(1)}" y1="${Y(m.base_rate)}" y2="${Y(m.base_rate)}" stroke="#bbb" stroke-dasharray="4 4"/><text x="${X(0) + 8}" y="${Y(m.base_rate) - 6}" text-anchor="start">${t("ax.chance")}</text>`;
    s += line(EV.pr.dti, "#9aa5b1", 2.5, "6 4") + line(EV.pr.logistic, "var(--mid)", 2.5) + line(EV.pr.fairkey, "var(--teal)", 3.5);
    const fr = op.fairkey_same_recall;
    s += `<line x1="${X(op.rule.recall)}" x2="${X(op.rule.recall)}" y1="${Y(op.rule.precision) - 8}" y2="${Y(fr.precision) + 8}" stroke="var(--ink)" stroke-width="1.5" stroke-dasharray="3 3"/>`;
    s += `<circle cx="${X(op.rule.recall)}" cy="${Y(op.rule.precision)}" r="7" fill="var(--bad)" stroke="#fff" stroke-width="2"/>`;
    s += `<circle cx="${X(fr.recall)}" cy="${Y(fr.precision)}" r="7" fill="var(--teal)" stroke="#fff" stroke-width="2"/>`;
    s += `<text x="${X(op.rule.recall) - 10}" y="${Y(op.rule.precision) + 22}" text-anchor="end" style="fill:var(--bad-ink);font-weight:600">${t("ax.rule")}</text>`;
    $("#prChart").innerHTML = s;
    $("#prLegend").innerHTML = `<span><i style="background:var(--teal)"></i>FairKey (${nf(m.fairkey.ap, 3)})</span><span><i style="background:var(--mid)"></i>${t("lg.lr")} (${nf(m.logistic.ap, 3)})</span><span><i style="background:#9aa5b1"></i>${t("lg.dti")} (${nf(m.dti.ap, 3)})</span>`;

    let c = axes(W, H, mg, [0, .25, .5, .75, 1], [0, .25, .5, .75, 1], t("ax.pred"), t("ax.actual"));
    c += `<line x1="${X(0)}" y1="${Y(0)}" x2="${X(1)}" y2="${Y(1)}" stroke="#bbb" stroke-dasharray="4 4"/>`;
    const mxn = Math.max(...EV.calibration.map(b => b.n));
    c += EV.calibration.map(b => `<circle cx="${X(b.pred)}" cy="${Y(b.actual)}" r="${4 + 9 * Math.sqrt(b.n / mxn)}" fill="var(--teal)" fill-opacity=".75" stroke="#fff" stroke-width="1.5"><title>${t("cal.tip", { p: pc(b.pred, 1), a: pc(b.actual, 1), n: nf(b.n) })}</title></circle>`).join("");
    $("#calChart").innerHTML = c;
    const bc = EV.bands_check;
    $("#bandsTable").innerHTML = `<table><thead><tr><th>${t("bt.band")}</th><th>${t("bt.share")}</th><th>${t("bt.d24")}</th><th>${t("bt.d25")}</th></tr></thead><tbody>` +
      ["low", "medium", "high"].map(b => {
        const r24 = bc.find(r => r.band === b && r.year === "2024"), r25 = bc.find(r => r.band === b && r.year === "2025");
        return `<tr><td><span class="chip ${b}">${t("b." + b)}</span></td><td>${pc(r25.share, 0)}</td><td>${pc(r24.denied, 1)}</td><td><b>${pc(r25.denied, 1)}</b></td></tr>`;
      }).join("") + "</tbody></table>";

    if (FAIR) { renderAudit(currentTab); renderAuditStatic(); renderStates(); }
  }

  let currentTab = "race";
  function renderAudit(k) {
    currentTab = k;
    $$("#auditTabs button").forEach(b => b.setAttribute("aria-selected", String(b.dataset.k === k)));
    $("#auditTable").innerHTML = `<table><thead><tr><th>${t("at.group")}</th><th>${t("at.n")}</th><th>${t("at.obs")}</th><th>${t("at.exp")}</th><th>${t("at.ratio")}</th><th>${t("at.ff")}</th><th>${t("at.caught")}</th></tr></thead><tbody>` +
      FAIR[k].map(r => {
        const sig = r.ratio_ci[0] > 1 ? "hi" : r.ratio_ci[1] < 1 ? "lo" : "";
        return `<tr><td>${esc(t("g." + r.group))}</td><td>${nf(r.n)}</td><td>${pc(r.observed, 1)}</td><td>${pc(r.expected, 1)}</td>` +
          `<td><span class="ratio ${sig}">×${nf(r.ratio, 2)}</span> <span class="ci">${nf(r.ratio_ci[0], 2)}–${nf(r.ratio_ci[1], 2)}</span></td>` +
          `<td>${pc(r.false_flag, 1)}</td><td>${pc(r.caught, 0)}</td></tr>`;
      }).join("") + "</tbody></table>";
  }

  function renderAuditStatic() {
    const W = 460, H = 300, mg = { l: 48, r: 14, t: 10, b: 40 };
    const cr = FAIR.calibration_race;
    const top = Math.max(...Object.values(cr).flat().map(b => Math.max(b.pred, b.actual)));
    const step = top > 0.6 ? 0.25 : top > 0.3 ? 0.2 : 0.1;
    const mx = Math.min(1, Math.ceil(top / step - 1e-9) * step);
    const X = v => mg.l + v / mx * (W - mg.l - mg.r), Y = v => H - mg.b - v / mx * (H - mg.t - mg.b);
    const ticks = []; for (let v = 0; v <= mx + 1e-9; v += step) ticks.push(v);
    let s = axes(W, H, mg, ticks, ticks, t("ax.pred"), t("ax.actual"), mx, mx);
    s += `<line x1="${X(0)}" y1="${Y(0)}" x2="${X(mx)}" y2="${Y(mx)}" stroke="#4a6078" stroke-dasharray="4 4"/>`;
    const col = { White: "#9fb3c8", Asian: "#7fd6c9", Hispanic: "#e9a03b", Black: "#ff8a98" };
    for (const [g, pa] of Object.entries(cr)) {
      s += `<polyline fill="none" stroke="${col[g]}" stroke-width="2.5" points="${pa.map(b => `${X(b.pred).toFixed(1)},${Y(b.actual).toFixed(1)}`).join(" ")}"/>`;
      s += pa.map(b => `<circle cx="${X(b.pred)}" cy="${Y(b.actual)}" r="3.5" fill="${col[g]}"><title>${t("g." + g)}: ${t("cal.tip", { p: pc(b.pred, 1), a: pc(b.actual, 1), n: nf(b.n) })}</title></circle>`).join("");
    }
    $("#groupCal").innerHTML = s;
    $("#groupLegend").innerHTML = Object.keys(cr).map(g => `<span><i style="background:${col[g]}"></i>${t("g." + g)}</span>`).join("");

    const r = Object.fromEntries(FAIR.race.map(x => [x.group, x]));
    const bw = r.Black.ratio / r.White.ratio;
    $("#finding").innerHTML = `<div class="big">×${nf(bw, 2)}</div><div><p>${t("find1", { b: pc(r.Black.observed, 1), be: pc(r.Black.expected, 1), w: pc(r.White.observed, 1), we: pc(r.White.expected, 1), x: nf(bw, 2) })}</p><p>${t("find2")}</p></div>`;

    const px = FAIR.proxy;
    if (!px || !FAIR.neighbourhood) return;      // still running (audit.py writes these last)
    bars($("#proxy"), Object.keys(px).map(k => ({ name: t("px." + k), v: px[k].auc - 0.5, label: nf(px[k].auc, 3), me: k === "shipped" })), { max: 0.5 });
    if (px.finances && px.shipped && px["shipped+state"] && px["shipped+county"])
      $("#proxyNote").innerHTML = t("px.note", { f: nf(px.finances.auc, 3), l: "+" + nf(px.shipped.auc - px.finances.auc, 3),
        s: "+" + nf(px["shipped+state"].auc - px.shipped.auc, 3), c: "+" + nf(px["shipped+county"].auc - px.shipped.auc, 3) });

    const nb = FAIR.neighbourhood;
    const vers = Object.keys(nb).filter(k => nb[k] && typeof nb[k] === "object" && nb[k].ap);
    let h = `<div class="nb-row head"><span>${t("nb.version")}</span><span>PR-AUC</span><span>${t("nb.ff")}</span></div>`;
    h += vers.map(k => `<div class="nb-row"><span><b>${t("px." + k)}</b>${k === "shipped" ? `<span class="ship">${t("nb.ship")}</span>` : ""}</span><span>${nf(nb[k].ap, 3)}</span><span>${pc(nb[k].false_flag.Black, 1)} · ${pc(nb[k].false_flag.Hispanic, 1)} · ${pc(nb[k].false_flag.White, 1)}</span></div>`).join("");
    const tw = vers.map(k => nb[k].twin).find(Boolean);
    const sh = nb.shipped, ss = nb["shipped+state"], sc = nb["shipped+county"], ne = nb.neighbourhood;
    const fi = nb.finances, pxs = FAIR.proxy;
    const sg = v => (v >= 0 ? "+" : "−") + nf(Math.abs(v), 3);
    const vv = { twin: tw ? (tw.pts >= 0 ? "+" : "−") + nf(Math.abs(tw.pts), 2) : "0",
      lap: fi ? nf(sh.ap - fi.ap, 3) : "", pf: pxs.finances ? nf(pxs.finances.auc, 3) : "", ps: pxs.shipped ? nf(pxs.shipped.auc, 3) : "",
      bf: fi ? pc(fi.false_flag.Black, 1) : "", bs: pc(sh.false_flag.Black, 1),
      sap: ss ? nf(ss.ap - sh.ap, 3) : "", pst: pxs["shipped+state"] ? nf(pxs["shipped+state"].auc, 3) : "", bst: ss ? pc(ss.false_flag.Black, 1) : "",
      dap: sc ? nf(sc.ap - sh.ap, 3) : "", pcty: pxs["shipped+county"] ? nf(pxs["shipped+county"].auc, 3) : "", bcty: sc ? pc(sc.false_flag.Black, 1) : "",
      nap: ne && sc ? sg(ne.ap - sc.ap) : "" };
    h += `<p class="nb-verdict">${t("nb.verdict", vv)}</p>`;
    $("#nb").innerHTML = h;
  }

  // ---------------------------------------------------------------- the audit, state by state (a tile map: one square per state)
  const TILE = { ME: [0, 10], WI: [1, 5], VT: [1, 9], NH: [1, 10],
    WA: [2, 0], ID: [2, 1], MT: [2, 2], ND: [2, 3], MN: [2, 4], IL: [2, 5], MI: [2, 6], NY: [2, 8], MA: [2, 9],
    OR: [3, 0], NV: [3, 1], WY: [3, 2], SD: [3, 3], IA: [3, 4], IN: [3, 5], OH: [3, 6], PA: [3, 7], NJ: [3, 8], CT: [3, 9], RI: [3, 10],
    CA: [4, 0], UT: [4, 1], CO: [4, 2], NE: [4, 3], MO: [4, 4], KY: [4, 5], WV: [4, 6], VA: [4, 7], MD: [4, 8], DE: [4, 9],
    AZ: [5, 1], NM: [5, 2], KS: [5, 3], AR: [5, 4], TN: [5, 5], NC: [5, 6], SC: [5, 7], DC: [5, 8],
    OK: [6, 3], LA: [6, 4], MS: [6, 5], AL: [6, 6], GA: [6, 7], HI: [7, 0], AK: [7, 1], TX: [7, 3], FL: [7, 8] };
  let stG = "black", stSel = null;
  const RC = { mid: [58, 80, 104], hi: [255, 107, 125], lo: [95, 196, 181] };
  function ratioCol(r) {
    const mix = (a, b, f) => a.map((c, i) => Math.round(c + (b[i] - c) * f));
    const c = r >= 1 ? mix(RC.mid, RC.hi, Math.min(1, (r - 1) / 0.6)) : mix(RC.mid, RC.lo, Math.min(1, (1 - r) / 0.4));
    return `rgb(${c.join(",")})`;
  }
  function renderStates() {
    const BS = FAIR && FAIR.states;
    if (!BS) { $("#stCard").hidden = true; return; }
    $$("#stTabs button").forEach(b => b.setAttribute("aria-selected", String(b.dataset.g === stG)));
    const T = 50, G = 4;
    let s = "";
    for (const [k, [r, c]] of Object.entries(TILE)) {
      const v = BS[k] && BS[k][stG], x = c * (T + G), y = r * (T + G);
      const sig = v && (v.ci[0] > 1 || v.ci[1] < 1);
      s += `<g class="tile${v ? "" : " na"}${k === stSel ? " sel" : ""}" data-s="${k}" tabindex="0" role="button" aria-label="${esc(stName(k))}${v ? ": ×" + nf(v.ratio, 2) : ""}">` +
        `<rect x="${x}" y="${y}" width="${T}" height="${T}" rx="7" fill="${v ? ratioCol(v.ratio) : "#22364b"}" fill-opacity="${v && !sig ? 0.5 : 1}"/>` +
        `<text class="ab" x="${x + T / 2}" y="${y + 22}">${k}</text>` +
        `<text class="rv" x="${x + T / 2}" y="${y + 39}">${v ? "×" + nf(v.ratio, 1) : "·"}</text></g>`;
    }
    $("#stMap").innerHTML = s;
    const vals = Object.entries(BS).filter(([, r]) => r[stG]).map(([k, r]) => [k, r[stG]]).sort((a, b) => b[1].ratio - a[1].ratio);
    if (vals.length) {
      const hi = vals[0], lo = vals[vals.length - 1];
      $("#stNote").innerHTML = t("st.note", { m: vals.length, a: vals.filter(([, v]) => v.ratio > 1).length,
        c: vals.filter(([, v]) => v.ci[0] > 1).length, b: vals.filter(([, v]) => v.ci[1] < 1).length,
        hi: esc(stName(hi[0])), hiv: nf(hi[1].ratio, 2), lo: esc(stName(lo[0])), lov: nf(lo[1].ratio, 2) });
    }
    const g = `linear-gradient(90deg, ${ratioCol(0.6)}, ${ratioCol(1)}, ${ratioCol(1.3)}, ${ratioCol(1.6)})`;
    $("#stKey").innerHTML = `<div>${t("st.key")}</div><div class="grad" style="background:${g}"></div>` +
      `<div class="lb"><span>×0.6</span><span>×1</span><span>×1.3</span><span>×1.6+</span></div>` +
      `<div class="fx"><span><i style="background:${ratioCol(1.4)};opacity:.5"></i>${t("st.faint")}</span><span><i style="background:#22364b"></i>${t("st.na")}</span></div>`;
    stInfo(stSel || S.state);
  }
  function stInfo(k) {
    const r = FAIR.states[k];
    if (!r) { $("#stInfo").innerHTML = ""; return; }
    const v = r[stG], G = stG === "black" ? "Black" : "Hispanic";
    let h = `<h4>${esc(stName(k))}</h4><div class="sub">${t("st.n", { n: nf(r.n) })}</div>`;
    if (!v) h += `<p>${t("st.few", { g: t("g." + G).toLowerCase(), n: nf(r["n_" + stG] || 0) })}</p>`;
    else {
      const w = r.white;
      h += `<div class="big">×${nf(v.ratio, 2)}<small>${t("st.ci", { a: nf(v.ci[0], 2), b: nf(v.ci[1], 2) })}</small></div>` +
        `<div class="ln"><span>${t("st.line", { g: t("g." + G) })}</span><span>${pc(v.obs, 1)} · ${t("st.exp")} ${pc(v.exp, 1)}</span>` +
        `<span>${t("st.line", { g: t("g.White") })}</span><span>${pc(w.obs, 1)} · ${t("st.exp")} ${pc(w.exp, 1)}</span></div>` +
        `<p>${t(v.ci[0] > 1 ? "st.sigHi" : v.ci[1] < 1 ? "st.sigLo" : "st.sigNo")}</p>`;
    }
    $("#stInfo").innerHTML = h;
  }
  function bindStates() {
    $$("#stTabs button").forEach(b => b.addEventListener("click", () => { stG = b.dataset.g; renderStates(); }));
    const pick = e => { const g = e.target.closest(".tile"); if (!g) return; stSel = g.dataset.s; renderStates(); };
    $("#stMap").addEventListener("click", pick);
    $("#stMap").addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(e); } });
    $("#stMap").addEventListener("pointerover", e => {
      const g = e.target.closest(".tile");
      if (g && e.pointerType === "mouse") stInfo(g.dataset.s);
    });
    $("#stMap").addEventListener("pointerleave", () => FAIR && stInfo(stSel || S.state));
  }

  // ---------------------------------------------------------------- the verdict: a number, and "N in 100"
  let pShown = null, pRaf = 0;
  function showP(p) {
    const from = pShown == null ? p : pShown, t0 = performance.now();
    cancelAnimationFrame(pRaf);
    const step = now => {
      const k = Math.min(1, (now - t0) / 450), e = 1 - Math.pow(1 - k, 3), v = from + (p - from) * e;
      pShown = v;
      $("#pBig").textContent = pc(v, v < 0.1 ? 1 : 0);
      if (k < 1) pRaf = requestAnimationFrame(step);
    };
    pRaf = requestAnimationFrame(step);
  }
  function buildIcons() {
    const house = '<svg viewBox="0 0 20 20"><path d="M10 1.6 19 9v9.4H1V9z"/></svg>';
    $("#icons").innerHTML = Array.from({ length: 100 }, () => house).join("");
    [...$("#icons").children].forEach((el, i) => (el.style.transitionDelay = `${(i % 10) * 12 + Math.floor(i / 10) * 18}ms`));
  }
  function renderVerdict(p) {
    const b = band(p);
    $("#verdict").style.setProperty("--vcol", color(p));
    $("#verdict").style.setProperty("--vink", ink(p));
    showP(p);
    const chip = $("#bandChip");
    chip.className = "chip " + b;
    chip.textContent = t("b." + b);
    const n = Math.round(p * 100);
    $("#vFreq").innerHTML = n < 1 ? t("freq.lt1") : t("freq", { n: `<b>${n}</b>` });
    [...$("#icons").children].forEach((el, i) => el.classList.toggle("on", i < n));
    $("#mbar").style.setProperty("--vcol", color(p));
    $("#mbP").textContent = pc(p, p < 0.1 ? 1 : 0);
    $("#mbB").textContent = t("b." + b);
    const chk = EV.bands_check.find(r => r.band === b && r.year === "2025");
    $("#bandText").innerHTML = t("bandText", { st: stName(S.state), avg: pc(META.states[S.state].denied, 1), act: pc(chk.denied, 0) });
  }

  // ---------------------------------------------------------------- reality check: the most similar real applications
  let SMP = null;
  function lookAlikes(s) {
    if (!SMP) return null;
    const d = derive(s), inc = Math.log(Math.max(1, s.income / 1000)), pr = Math.log(Math.max(1, s.price / 1000));
    const ltv = d.ltv, dti = hmdaDti(d.dti), n = SMP.n, manuf = +s.manuf, coapp = +s.coapp;
    const sti = SMP.states ? SMP.states.indexOf(s.state) : -1;
    // same loan type, same kind of home and the same debt-to-income band (lenders' limits make DTI a cliff, not a slope);
    // if fewer than 40 real applications match, the DTI band is widened. Applications from your own state come first.
    const pick = tol => {
      const dist = new Float64Array(n);
      let pool = 0;
      for (let i = 0; i < n; i++) {
        const hard = SMP.type[i] === s.loanType && SMP.manuf[i] === manuf && Math.abs(SMP.dti[i] - dti) <= tol;
        if (hard) pool++;
        const a = (Math.log(Math.max(1, SMP.income[i])) - inc) / 0.25, b = (Math.log(SMP.price[i]) - pr) / 0.25;
        const c = (SMP.ltv[i] / 10 - ltv) / 5;
        dist[i] = a * a + b * b + c * c + (SMP.coapp[i] !== coapp ? 1 : 0) + (SMP.term[i] !== s.term ? 1 : 0) +
          (sti >= 0 && SMP.st[i] !== sti ? 1.5 : 0) + (hard ? 0 : 1e6);
      }
      return { dist, pool };
    };
    let r = pick(1);
    if (r.pool < 40) r = pick(8);
    if (r.pool < 40) return null;
    const k = Math.min(200, r.pool), dist = r.dist;
    const order = new Uint32Array(n).map((_, i) => i).sort((u, v) => dist[u] - dist[v]);
    let den = 0, same = 0;
    for (let j = 0; j < k; j++) { den += SMP.denied[order[j]]; if (sti >= 0 && SMP.st[order[j]] === sti) same++; }
    return { k, den, same, rate: den / k };
  }
  function renderReality(p) {
    const r = lookAlikes(S);
    if (!r) { $("#reality").innerHTML = ""; return; }
    const mx = Math.max(p, r.rate, 0.05) * 1.15;
    $("#reality").innerHTML =
      `<svg viewBox="0 0 46 46" aria-hidden="true"><circle cx="23" cy="23" r="21" fill="#dff0f0"/><path d="M14 24l6 6 12-13" fill="none" stroke="#16697a" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/></svg>` +
      `<div><span class="rk">${t("real.k")}</span><div>${t("real.t", { d: `<b>${r.den}</b>`, k: r.k, p: pc(r.rate, 0), s: r.same, st: esc(stName(S.state)) })}</div>` +
      `<div class="cmp"><span>FairKey</span><i style="width:${(100 * p / mx).toFixed(1)}%"></i><em>${pc(p, p < .1 ? 1 : 0)}</em>` +
      `<span>${t("real.lbl")}</span><i class="r" style="width:${(100 * r.rate / mx).toFixed(1)}%"></i><em>${pc(r.rate, 0)}</em></div></div>`;
  }

  // ---------------------------------------------------------------- detail tabs
  let curTab = "why";
  function setTab(p) {
    curTab = p;
    $$("#dtabs button").forEach(b => b.setAttribute("aria-selected", String(b.dataset.p === p)));
    $$(".dpanel").forEach(d => (d.hidden = d.dataset.p !== p));
    const wasStale = stale.ai;
    drawVisible();
    if (p === "ai" && !wasStale) playAI();
  }

  // ---------------------------------------------------------------- inside the AI: the trees adding up
  let AI = null, aiRaf = 0, aiSel = null;
  function computeAI(x) {
    const li = idx("lender"), T = MODEL.risk.trees.length;
    const avg = li >= 0 && Number.isNaN(x[li]);
    const curve = new Float64Array(T), push = new Float64Array(T);
    if (avg) {                                   // every lender of the state in one walk per tree
      const W = lwFor(S.state), nc = NC(), lv = M.leavesCats(MODEL.risk, x, li, nc, W.idx), cum = new Float64Array(nc);
      for (let k = 0; k < T; k++) for (const [i, w] of W.all) {
        const v = lv[k * nc + i];
        cum[i] += v; curve[k] += w * cal(cum[i]); push[k] += w * v;
      }
    } else {
      const lv = M.leaves(MODEL.risk, x);
      let c = 0;
      for (let k = 0; k < T; k++) { c += lv[k]; curve[k] = cal(c); push[k] = lv[k]; }
    }
    const xp = x.slice();
    if (avg) xp[li] = META.cats.lender.indexOf("other");
    let top = 1;
    for (let k = 1; k < T; k++) if (Math.abs(push[k]) > Math.abs(push[top])) top = k;
    return { curve, push, T, xp, avg, top };
  }
  let AW = 640;   // follows the chart's real width, so 1 unit = 1 pixel and the labels stay readable on phones
  const AH = 230, AM = { l: 40, r: 12, t: 14, b: 64 };
  const aiX = k => AM.l + (k / (AI.T - 1)) * (AW - AM.l - AM.r);
  function aiMax() { return Math.max(0.1, ...AI.curve) * 1.15; }
  function drawAI(upto) {
    const svgEl = $("#aiChart"), cw = Math.round(svgEl.clientWidth);
    if (cw > 0 && cw !== AW) { AW = Math.max(300, cw); svgEl.setAttribute("viewBox", `0 0 ${AW} ${AH}`); }
    const mx = aiMax(), H1 = AH - AM.b, Y = v => H1 - (v / mx) * (H1 - AM.t);
    const n = Math.max(1, Math.round(upto));
    let s = "";
    const stp = mx <= 0.12 ? 0.02 : mx <= 0.3 ? 0.05 : mx <= 0.6 ? 0.1 : 0.2;
    for (let v = stp; v < mx - 1e-9; v += stp)
      s += `<line x1="${AM.l}" x2="${AW - AM.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="#1d3045"/><text x="${AM.l - 6}" y="${Y(v) + 4}" text-anchor="end">${Math.round(v * 100)}%</text>`;
    // vote bars: scaled to the typical big vote (95th percentile), so one huge early tree doesn't flatten the rest
    const absV = Array.from(AI.push, Math.abs).sort((a, b) => a - b), pm = absV[Math.floor(absV.length * 0.95)] || 1e-9, base = AH - 34;
    for (let k = 0; k < n; k++) {
      const h = Math.max(0.6, Math.min(14, Math.abs(AI.push[k]) / pm * 11));
      s += `<rect x="${(aiX(k) - 0.8).toFixed(1)}" y="${(AI.push[k] > 0 ? base - h : base).toFixed(1)}" width="1.6" height="${h.toFixed(1)}" fill="${AI.push[k] > 0 ? "#ff6b7d" : "#5fc4b5"}"/>`;
    }
    s += `<text x="${AM.l}" y="${AH - 2}">${t("ai.t1")}</text><text x="${AW - AM.r}" y="${AH - 2}" text-anchor="end">${t("ai.tN", { n: AI.T })}</text>`;
    s += `<text x="${AM.l}" y="${base - 19}" fill="#6f859a">${t("ai.push")}</text>`;
    let d = "";
    for (let k = 0; k < n; k++) d += (k ? "L" : "M") + aiX(k).toFixed(1) + "," + Y(AI.curve[k]).toFixed(1);
    s += `<path d="${d}" fill="none" stroke="#f4c95d" stroke-width="2.4" stroke-linejoin="round"/>`;
    const last = n - 1;
    s += `<circle cx="${aiX(last)}" cy="${Y(AI.curve[last])}" r="5" fill="#f4c95d" stroke="#0d1b2a" stroke-width="2"/>`;
    if (aiSel != null && aiSel < n) s += `<line x1="${aiX(aiSel)}" x2="${aiX(aiSel)}" y1="${AM.t}" y2="${base + 15}" stroke="#fff" stroke-opacity=".55" stroke-dasharray="3 3"/>`;
    $("#aiChart").innerHTML = s;
    $("#aiTree").textContent = nf(n);
    $("#aiP").textContent = pc(AI.curve[last], AI.curve[last] < .1 ? 1 : 0);
  }
  function playAI() {
    if (!AI) return;
    cancelAnimationFrame(aiRaf);
    const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
    const t0 = performance.now(), D = reduce ? 1 : 2400;
    const step = now => {
      const k = Math.min(1, (now - t0) / D), e = 1 - Math.pow(1 - k, 2);
      drawAI(1 + e * (AI.T - 1));
      if (k < 1) aiRaf = requestAnimationFrame(step);
    };
    aiRaf = requestAnimationFrame(step);
  }
  const fmtThr = (k, v) => {
    switch (k) {
      case "income_k": case "loan_k": case "price_k": return usd(v * 1000);
      case "ltv": case "dti": return nf(v, Math.abs(v - Math.round(v)) < 0.01 ? 0 : 1) + "%";
      case "loan_to_income": return nf(v, 2) + "×";
      case "term_years": return nf(v, 0) + " " + t("years");
      default: return nf(v, 2);
    }
  };
  const catName = (k, v) => k === "loan_type" ? t("lt." + v) : k === "lender" ? (v === "other" ? t("lender.otherShort") : META.lenders[v] || v) : String(v);
  function renderTree(k) {
    aiSel = k;
    const tree = MODEL.risk.trees[k], pth = M.path(tree, AI.xp);
    const rows = pth.steps.map(st => {
      const f = META.features[st.f], val = AI.xp[st.f];
      let q, you, yes;
      if (st.set) {
        const names = st.set.map(i => catName(f, META.cats[f][i]));
        you = catName(f, META.cats[f][val]);
        yes = st.left;
        q = f === "lender" ? t("q.lender", { n: names.length, v: esc(names.slice(0, 2).join(", ")) + (names.length > 2 ? "…" : "") })
          : t("q.cat", { f: FEAT_NAME(f).toLowerCase(), F: FEAT_NAME(f), v: names.join(t("or")) });
      } else if (f === "coapp" || f === "manufactured") {
        q = t("q." + f); yes = !st.left; you = null;
      } else {
        q = t("q.num", { f: FEAT_NAME(f).toLowerCase(), F: FEAT_NAME(f), v: fmtThr(f, st.th) });
        you = (f === "price_k" || f === "loan_k" ? "≈ " : "") + fmtThr(f, val);
        yes = st.left;
      }
      return `<li><span>${q}${you != null ? ` <small>(${t("q.yours")}: ${esc(you)})</small>` : ""}</span><span class="ans">${t(yes ? "yes" : "no")}</span></li>`;
    });
    const v = pth.value;
    rows.push(`<li class="leaf"><span>${t(v > 0 ? "ai.up" : "ai.down")}</span><span class="ans ${v > 0 ? "up" : "down"}">${v > 0 ? "+" : "−"}${nf(Math.abs(v), 3)}</span></li>`);
    $("#aiTreeTitle").innerHTML = t(k === AI.top ? "ai.treeTop" : "ai.tree", { k: k + 1, n: AI.T });
    $("#aiPath").innerHTML = rows.join("");
    $("#aiNote").innerHTML = t(AI.avg ? "ai.noteAvg" : "ai.note", { st: stName(S.state) });
  }
  function renderAI(x) {
    AI = computeAI(x);
    $("#aiN").textContent = nf(AI.T);
    $("#aiCap").innerHTML = t("ai.cap", { n: nf(AI.T), p: pc(lastP, lastP < .1 ? 1 : 0) });
    renderTree(AI.top);
    if (curTab === "ai") playAI(); else drawAI(AI.T);
  }
  function bindAI() {
    const svg = $("#aiChart");
    const pick = e => {
      if (!AI) return;
      const r = svg.getBoundingClientRect(), fx = (e.clientX - r.left) / r.width * AW;
      const k = Math.round((fx - AM.l) / (AW - AM.l - AM.r) * (AI.T - 1));
      if (k >= 0 && k < AI.T) { cancelAnimationFrame(aiRaf); renderTree(k); drawAI(AI.T); }
    };
    svg.addEventListener("click", pick);
    svg.addEventListener("pointermove", e => { if (e.pointerType === "mouse" && e.buttons === 0) pick(e); });
    $("#aiPlay").addEventListener("click", () => { renderTree(AI.top); playAI(); });
    let rt = null;
    addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(() => { if (AI && curTab === "ai") drawAI(AI.T); }, 120); });
  }

  // ---------------------------------------------------------------- examples, smooth changes and the demo
  const PERSONAS = [
    { id: "kim", av: "K", col: "#2a9d8f", s: { state: "CA", income: 160000, price: 520000, downPct: 20, debts: 600, loanType: 1, term: 30, coapp: true, manuf: false } },
    { id: "robert", av: "R", col: "#16697a", s: { state: "OH", income: 48000, price: 260000, downPct: 40, debts: 150, loanType: 1, term: 30, coapp: false, manuf: false } },
    { id: "luis", av: "L", col: "#3d5a80", s: { state: "TX", income: 74000, price: 330000, downPct: 0, debts: 300, loanType: 3, term: 30, coapp: false, manuf: false } },
    { id: "dale", av: "D", col: "#a0701c", s: { state: "AL", income: 45000, price: 120000, downPct: 5, debts: 250, loanType: 2, term: 30, coapp: false, manuf: true } },
    { id: "maria", av: "M", col: "#c9971d", s: { state: "FL", income: 62000, price: 285000, downPct: 3.5, debts: 380, loanType: 2, term: 30, coapp: false, manuf: false } },
    { id: "tasha", av: "T", col: "#d1495b", s: { state: "IL", income: 60000, price: 290000, downPct: 5, debts: 450, loanType: 1, term: 30, coapp: false, manuf: false } },
  ];
  function renderPersonas() {
    $("#personas").innerHTML = PERSONAS.map((P, i) => {
      const p = risk({ ...S, ...P.s, lender: "" }), b = band(p);
      return `<button type="button" class="pp" data-i="${i}" aria-pressed="false"><span class="av" style="background:${P.col}">${P.av}</span>` +
        `<span><b>${t("pp." + P.id)} <em>· ${esc(stName(P.s.state))}</em></b><small>${t("pp." + P.id + ".d")}</small></span><span class="pr ${b}">${pc(p, 0)}</span></button>`;
    }).join("");
    $$(".pp").forEach(b => b.addEventListener("click", e => { if (e.isTrusted) stopDemo(); applyPersona(+b.dataset.i); }));
  }
  const markPersona = i => $$(".pp").forEach(b => b.setAttribute("aria-pressed", String(+b.dataset.i === i)));
  function applyPersona(i) {
    markPersona(i);
    tweenTo({ ...PERSONAS[i].s, lender: "" });
  }
  let twRaf = 0;
  function tweenTo(target) {
    const keys = ["income", "price", "downPct", "debts"].filter(k => target[k] != null);
    const from = Object.fromEntries(keys.map(k => [k, S[k]]));
    const newState = target.state != null && target.state !== S.state;
    for (const k of ["state", "loanType", "term", "coapp", "manuf", "lender", "rate"]) if (target[k] != null) S[k] = target[k];
    if (newState || target.lender != null) fillSelects();
    cancelAnimationFrame(twRaf);
    const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
    const t0 = performance.now(), D = reduce ? 1 : 750;
    const step = now => {
      const k = Math.min(1, (now - t0) / D), e = k < .5 ? 4 * k * k * k : 1 - Math.pow(-2 * k + 2, 3) / 2;
      for (const key of keys) S[key] = from[key] + (target[key] - from[key]) * e;
      if (k === 1) for (const key of keys) S[key] = target[key];
      syncForm(); update();
      if (k < 1) twRaf = requestAnimationFrame(step);
    };
    twRaf = requestAnimationFrame(step);
  }
  let demoT = null;
  function startDemo() {
    stopDemo();
    $("#check").scrollIntoView({ behavior: "smooth" });
    $("#demoPill").hidden = false;
    const tabs = ["why", "change", "map", "lenders", "ai"];
    let k = 0;
    const next = () => {
      applyPersona(k % PERSONAS.length);
      setTab(tabs[k % tabs.length]);
      k++;
      demoT = setTimeout(next, 5200);
    };
    demoT = setTimeout(next, 900);
  }
  function stopDemo() { clearTimeout(demoT); demoT = null; $("#demoPill").hidden = true; }

  // ---------------------------------------------------------------- "tell it in your own words" (rules, not a language model)
  function parseSay(raw) {
    const txt = " " + raw.toLowerCase().replace(/[’']/g, "'").replace(/\s+/g, " ") + " ";
    const out = {};
    if (/\bfha\b/.test(txt)) out.loanType = 2;
    else if (/\bva\b|veteran|veterano|military|militar|\barmy\b|\bnavy\b|ej[eé]rcito/.test(txt)) out.loanType = 3;
    else if (/\busda\b|\brural\b/.test(txt)) out.loanType = 4;
    else if (/conventional|convencional/.test(txt)) out.loanType = 1;
    const tm = txt.match(/\b(15|20|30)[\s-]*(?:years?|yrs?|años|anos)\b/);
    if (tm) out.term = +tm[1];
    if (/\bwe\b|\bour\b|my (?:wife|husband|partner|fianc)|\bcouple\b|two incomes|ganamos|cobramos|nuestr|queremos|mi (?:mujer|marido|esposa|esposo|novia|novio|pareja)|entre los dos|\bjuntos\b/.test(txt)) out.coapp = true;
    else if (/\bi\b|\bmy\b|\bgano\b|\bcobro\b|\bquiero\b/.test(txt)) out.coapp = false;
    if (/mobile home|manufactured|trailer|prefabricad|casa m[oó]vil|caravana/.test(txt)) out.manuf = true;
    const stHit = findState(txt);
    if (stHit) out.state = stHit;
    const noDown = /nothing down|no down|zero down|\b0 down|sin entrada|sin enganche/.test(txt);
    if (noDown) out.downPct = 0;
    const re = /\$?\s?(\d{1,3}(?:[.,]\d{3})+|\d+(?:[.,]\d+)?)\s*(k\b|mil\b|thousand|grand|m\b|million|mill[oó]n(?:es)?)?\s*(%|percent|por ?ciento)?/g;
    const toks = [];
    let m;
    while ((m = re.exec(txt))) {
      if (!m[1]) continue;
      let v = /^\d{1,3}([.,]\d{3})+$/.test(m[1]) ? +m[1].replace(/[.,]/g, "") : parseFloat(m[1].replace(",", "."));
      if (m[2]) v *= /^m/.test(m[2]) && !/^mil\b/.test(m[2]) ? 1e6 : 1e3;
      toks.push({ v, pct: !!m[3], s: m.index, e: m.index + m[0].length });
    }
    const KW = {
      income: /earn|make|income|salar|paid|\bgan[oa]|ganamos|cobr|ingres|sueldo|a year|per year|al año|anual|annual|yearly/,
      price: /house|home|condo|apartment|place|price|cost|worth|\bbuy|casa|vivienda|piso|precio|cuesta|comprar|valor/,
      down: /down|deposit|entrada|enganche|inicial|ahorrad|saved/,
      debts: /debt|\bcar\b|student|card|month|\/mo|deuda|coche|carro|pr[eé]stamo|tarjeta|al mes|mensual|cuota/,
    };
    const taken = new Set(noDown ? ["down"] : []);
    toks.forEach((tk, i) => {
      // context = the words around the number, cut at commas, full stops and "and"/"y" (one clause each side)
      const CUT = /[,;.](?:\s|$)|\s(?:and|but|y|pero)(?:\s|$)/g;
      let before = txt.slice(Math.max(i ? toks[i - 1].e : 0, tk.s - 40), tk.s), after = txt.slice(tk.e, Math.min(i + 1 < toks.length ? toks[i + 1].s : txt.length, tk.e + 30));
      const bc = [...before.matchAll(CUT)].pop();
      if (bc) before = before.slice(bc.index + bc[0].length);
      const ac = after.search(CUT);
      if (ac >= 0) after = after.slice(0, ac);
      if (/^\s*-?\s*(?:years?|yrs?|años|anos)\b/.test(after)) return;
      if (tk.pct) {
        if (/rate|inter[eé]s|tipo/.test(before + after)) { if (tk.v > 0 && tk.v < 25) out.rate = tk.v; }
        else if (!("downPct" in out) || out.downPct === 0) { out.downPct = Math.min(60, tk.v); taken.add("down"); }
        return;
      }
      if (tk.v < 20) return;
      const sc = {};
      for (const [f, r] of Object.entries(KW)) sc[f] = (r.test(after) ? 2 : 0) + (r.test(before.slice(-24)) ? 3 : r.test(before) ? 1 : 0);
      if (tk.v < 10000) { sc.price -= 9; sc.income -= 9; }
      if (tk.v > 15000) sc.debts -= 9;
      let best = null, bs = 0;
      for (const f of ["down", "debts", "income", "price"]) if (!taken.has(f) && sc[f] > bs) { bs = sc[f]; best = f; }
      if (!best) best = tk.v >= 150000 && !taken.has("price") ? "price" : tk.v >= 15000 && !taken.has("income") ? "income" : tk.v < 10000 && !taken.has("debts") ? "debts" : null;
      if (!best) return;
      taken.add(best);
      if (best === "down") out.downAmt = tk.v; else out[best] = tk.v;
    });
    if (out.downAmt != null) { out.downPct = Math.min(60, out.downAmt / (out.price || S.price) * 100); delete out.downAmt; }
    return out;
  }
  // full state names in English or Spanish, longest first ("West Virginia" before "Virginia")
  let ST_RE = null;
  function findState(txt) {
    if (!ST_RE) {
      const names = [];
      for (const [k, v] of Object.entries(META.states)) {
        names.push([v.name.toLowerCase(), k]);
        if (ST_ES[k]) names.push([ST_ES[k].toLowerCase(), k]);
      }
      names.push(["washington dc", "DC"], ["washington d.c.", "DC"], ["hawaii", "HI"], ["nueva york", "NY"], ["tejas", "TX"],
        ["florída", "FL"], ["california", "CA"], ["nuevo mexico", "NM"], ["michigan", "MI"], ["oregon", "OR"], ["misisipi", "MS"]);
      names.sort((a, b) => b[0].length - a[0].length);
      ST_RE = names.map(([n, k]) => [new RegExp("(?:^|[^a-záéíóúñ])" + n.replace(/[.]/g, "\\.") + "(?![a-záéíóúñ])"), k]);
    }
    for (const [re, k] of ST_RE) if (re.test(txt)) return k;
    return null;
  }
  function runSay() {
    const txt = $("#say").value.trim();
    if (!txt) return;
    stopDemo();
    const o = parseSay(txt), got = [];
    if (o.income != null) got.push(t("say.inc", { v: usd(o.income) }));
    if (o.price != null) got.push(t("say.price", { v: usd(o.price) }));
    if (o.downPct != null) got.push(t("say.down", { v: nf(o.downPct, o.downPct % 1 ? 1 : 0) }));
    if (o.debts != null) got.push(t("say.debts", { v: usd(o.debts) }));
    if (o.loanType != null) got.push(t("lt." + o.loanType));
    if (o.term != null) got.push(t("say.term", { v: o.term }));
    if (o.coapp) got.push(t("say.coapp"));
    if (o.manuf) got.push(t("f.manuf"));
    if (o.rate != null) got.push(t("say.rate", { v: nf(o.rate, 2) }));
    if (o.state != null) got.push(esc(stName(o.state)));
    $("#sayGot").innerHTML = got.length ? `<span>${t("say.got")}</span>` + got.map(g => `<span>${g}</span>`).join("")
      : `<span class="miss">${t("say.none")}</span>`;
    if (!got.length) return;
    markPersona(-1);
    tweenTo({ ...o, lender: S.lender });
    $("#form").classList.remove("flash"); void $("#form").offsetWidth; $("#form").classList.add("flash");
  }

  // ---------------------------------------------------------------- hero: real applications become a house, then get sorted
  function heroViz() {
    const cv = $("#heroCanvas");
    if (!cv || !SMP) return;
    const ctx = cv.getContext("2d");
    const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
    let W = 0, H = 0, dots = [], seed = 11;
    const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
    function inHouse(x, y) {   // unit square, y down
      if (x > 0.44 && x < 0.56 && y > 0.7) return false;                                    // door
      if (y >= 0.46 && y <= 0.96 && x >= 0.17 && x <= 0.83) return true;                     // walls
      if (x > 0.66 && x < 0.75 && y > 0.14 && y < 0.36) return true;                          // chimney
      return y < 0.47 && y > 0.07 && Math.abs(x - 0.5) < (y - 0.07) * 1.03;                   // roof
    }
    function layout() {
      const r = cv.getBoundingClientRect();
      if (!r.width) return;
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      W = r.width; H = r.height;
      cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const N = Math.min(SMP.n, W > 480 ? 1600 : 1000);
      // house targets on a gently jittered grid
      const side = Math.min(W, H) * 0.98, ox = (W - side) / 2, oy = (H - side) / 2 - H * 0.02;
      let sp = 0.05, pts = [];
      for (let it = 0; it < 30; it++) {
        pts = [];
        for (let y = 0.05; y < 1; y += sp) for (let x = 0.05; x < 1; x += sp) if (inHouse(x, y)) pts.push([x, y]);
        if (pts.length >= N && pts.length < N * 1.06) break;
        sp *= Math.sqrt(pts.length / N) * 0.995;
      }
      seed = 11;
      const house = pts.slice(0, N).map(([x, y]) => [ox + (x + (rnd() - .5) * sp * .3) * side, oy + (y + (rnd() - .5) * sp * .3) * side]);
      for (let i = house.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [house[i], house[j]] = [house[j], house[i]]; }
      // sorted targets: a rectangle, columns left (low predicted risk) to right (high)
      const rw = W * 0.94, rh = H * 0.6, cell = Math.sqrt(rw * rh / N), cols = Math.floor(rw / cell), rows = Math.ceil(N / cols);
      const cx0 = (W - cols * cell) / 2 + cell / 2, cy0 = H * 0.16 + (rh + rows * 0 - rows * cell) / 2 + rows * cell - cell / 2;
      const idx = Array.from({ length: N }, (_, i) => i).sort((a, b) => SMP.p[a] - SMP.p[b] || a - b);
      const sorted = new Array(N);
      idx.forEach((i, rank) => { const c = Math.floor(rank / rows), r2 = rank % rows; sorted[i] = [cx0 + c * cell, cy0 - r2 * cell]; });
      const rad = Math.max(1.3, Math.min(sp * side, cell) * 0.36);
      dots = Array.from({ length: N }, (_, i) => ({
        h: house[i] || [W / 2, H / 2], s: sorted[i], den: SMP.denied[i] === 1,
        r0: [rnd() * W, rnd() * H], x: 0, y: 0, rad,
      }));
      dots.forEach(d => { [d.x, d.y] = reduce ? d.s : d.r0; });
      geom = { cy0, rows, cell, cols, cx0 };
    }
    let geom = null;
    const COL = { ok: "rgba(95,196,181,.85)", no: "rgba(255,107,125,.95)" };
    function draw(labels) {
      ctx.clearRect(0, 0, W, H);
      ctx.fillStyle = COL.ok;
      ctx.beginPath();
      for (const d of dots) if (!d.den) { ctx.moveTo(d.x + d.rad, d.y); ctx.arc(d.x, d.y, d.rad, 0, 6.283); }
      ctx.fill();
      ctx.fillStyle = COL.no;
      ctx.beginPath();
      for (const d of dots) if (d.den) { ctx.moveTo(d.x + d.rad * 1.15, d.y); ctx.arc(d.x, d.y, d.rad * 1.15, 0, 6.283); }
      ctx.fill();
      if (labels > 0 && geom) {
        ctx.globalAlpha = labels;
        const y = geom.cy0 + geom.cell * 1.6, x0 = geom.cx0 - geom.cell / 2, x1 = x0 + geom.cols * geom.cell;
        ctx.strokeStyle = "rgba(201,214,226,.5)"; ctx.lineWidth = 1.2;
        ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x1, y); ctx.lineTo(x1 - 7, y - 4); ctx.moveTo(x1, y); ctx.lineTo(x1 - 7, y + 4); ctx.stroke();
        ctx.font = "600 12px Inter, sans-serif"; ctx.fillStyle = "#a9bbcc"; ctx.textBaseline = "top";
        ctx.textAlign = "left"; ctx.fillText(t("hero.low"), x0, y + 7);
        ctx.textAlign = "right"; ctx.fillText(t("hero.high"), x1, y + 7);
        ctx.globalAlpha = 1;
      }
    }
    // timeline: fly in → house → sorted → house …
    const PH = [["in", 1700], ["house", 2600], ["toS", 1900], ["sorted", 4600], ["toH", 1700]];
    const cap = $("#heroCap");
    let phase = 0, pStart = performance.now(), visible = true, raf = 0, lastCap = "";
    const setCap = k => { if (k === lastCap) return; lastCap = k; cap.style.opacity = 0; setTimeout(() => { cap.innerHTML = t(k); cap.style.opacity = 1; }, 250); };
    const ease = k => (k < .5 ? 4 * k * k * k : 1 - Math.pow(-2 * k + 2, 3) / 2);
    function frame(now) {
      raf = 0;
      if (!visible) return;
      let [name, dur] = PH[phase];
      let k = (now - pStart) / dur;
      if (k >= 1) { phase = (phase + 1) % PH.length; if (phase === 0) phase = 1; pStart = now; [name, dur] = PH[phase]; k = 0; }
      const from = name === "in" ? "r0" : name === "toS" ? "h" : name === "toH" ? "s" : null;
      const to = name === "in" || name === "toH" || name === "house" ? "h" : "s";
      let labels = 0;
      if (from) {
        for (const d of dots) {
          const delay = (d[to][0] / W) * 0.35, q = ease(Math.min(1, Math.max(0, (k - delay) / 0.65)));
          d.x = d[from][0] + (d[to][0] - d[from][0]) * q; d.y = d[from][1] + (d[to][1] - d[from][1]) * q;
        }
        labels = name === "toS" ? Math.max(0, k * 2 - 1) : name === "toH" ? Math.max(0, 1 - k * 2) : 0;
      } else {
        for (const d of dots) [d.x, d.y] = d[to];
        labels = name === "sorted" ? 1 : 0;
      }
      setCap(name === "in" || name === "house" || name === "toH" ? "hero.c1" : "hero.c2");
      draw(labels);
      raf = requestAnimationFrame(frame);
    }
    layout();
    if (reduce) { draw(1); cap.innerHTML = t("hero.c2"); heroRelabel = () => { cap.innerHTML = t("hero.c2"); draw(1); }; return; }
    heroRelabel = () => { lastCap = ""; };
    new IntersectionObserver(es => {
      visible = es[0].isIntersecting;
      if (visible && !raf) raf = requestAnimationFrame(frame);
    }).observe(cv);
    let rt = null;
    addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(() => { layout(); phase = 1; pStart = performance.now(); }, 150); });
    raf = requestAnimationFrame(frame);
  }
  let heroRelabel = () => {};

  // phones: show the live answer while the form is on screen but the verdict is not
  function mobileBar() {
    if (!("IntersectionObserver" in window)) return;
    let formIn = false, verdictIn = false;
    const set = () => $("#mbar").classList.toggle("show", formIn && !verdictIn);
    new IntersectionObserver(es => { formIn = es[0].isIntersecting; set(); }).observe($("#form"));
    new IntersectionObserver(es => { verdictIn = es[0].isIntersecting; set(); }, { threshold: 0.35 }).observe($("#verdict"));
  }

  // ---------------------------------------------------------------- reveal on scroll
  function reveal() {
    const els = $$("#kpis, #trust .card, #fair .card, #fair .finding, #how .card, #how .pipeline li");
    if (!("IntersectionObserver" in window)) return;
    els.forEach((el, i) => { el.classList.add("reveal"); el.style.transitionDelay = `${(i % 3) * 70}ms`; });
    const io = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); } }), { threshold: 0.08 });
    els.forEach(el => io.observe(el));
  }

  // ---------------------------------------------------------------- boot
  async function boot() {
    const get = u => fetch(u).then(r => { if (!r.ok) throw new Error(u); return r.json(); });
    document.documentElement.classList.add("js");
    try {                                        // say what is loading, in the visitor's language, before the data arrives
      const l0 = startLang();
      const msg = ((window.I18N[l0] || {}).dyn || {})["load.msg"];
      if (msg) $("#loadMsg").textContent = msg;
    } catch (e) { /* keep the English text */ }
    [META, MODEL, EV, FAIR, SMP] = await Promise.all([get("data/meta.json"), get("data/model.json"), get("data/evidence.json"),
      get("data/fairness.json").catch(() => null), get("data/sample.json").catch(() => null)]);
    buildIcons();
    bindForm();
    bindZone();
    bindAI();
    bindStates();
    $$("#dtabs button").forEach(b => b.addEventListener("click", e => { if (e.isTrusted) stopDemo(); setTab(b.dataset.p); }));
    $("#actions").addEventListener("click", e => {
      const el = e.target.closest(".act"); if (!el) return;
      const a2 = lastActs[+el.dataset.i]; if (!a2) return;
      stopDemo(); markPersona(-1); tweenTo({ ...a2.s });
    });
    $("#actions").addEventListener("keydown", e => { if (e.key === "Enter" && e.target.closest(".act")) e.target.click(); });
    setTab("why");
    $("#demoBtn").addEventListener("click", startDemo);
    $("#demoStop").addEventListener("click", stopDemo);
    $("#sayBtn").addEventListener("click", runSay);
    $("#say").addEventListener("keydown", e => { if (e.key === "Enter") { e.preventDefault(); runSay(); } });
    $("#form").addEventListener("pointerdown", e => { if (e.isTrusted && demoT) stopDemo(); });
    $$("#auditTabs button").forEach(b => b.addEventListener("click", () => renderAudit(b.dataset.k)));
    const l = startLang();
    fillSelects();
    syncForm();
    applyLang(l);
    $("#loading").classList.add("done");
    heroViz();
    reveal();
    mobileBar();
    if (/[?&]demo\b/.test(location.search)) startDemo();
  }
  boot().catch(e => {
    console.error(e);
    $("#loading").innerHTML = `<p style="padding:20px;text-align:center">Could not load the model. ${esc(e.message)}</p>`;
  });
})();
