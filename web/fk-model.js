/* FairKey model engine — runs the trained LightGBM trees in your browser, so your numbers never leave your device.
   predict: walks every tree exactly like LightGBM does.
   explain: exact TreeSHAP (Lundberg et al., 2018) — how much each answer pushed the risk up or down.
   Checked against LightGBM's own predictions and SHAP values in tests/check_engine.mjs. */
(function (g) {
  "use strict";

  function next(t, n, x) {
    const f = t.f[n], v = x[f], th = t.t[n];
    if (th < 0) {                                   // categorical split: left if the category is in the set
      if (!(v >= 0) || Number.isNaN(v)) return t.r[n];
      return t.cs[-th - 1].includes(Math.trunc(v)) ? t.l[n] : t.r[n];
    }
    let fv = v;
    const mt = t.mt[n];
    if (Number.isNaN(fv) && mt !== 2) fv = 0;
    if ((mt === 1 && Math.abs(fv) <= 1e-35) || (mt === 2 && Number.isNaN(fv))) return t.dl[n] ? t.l[n] : t.r[n];
    return fv <= th ? t.l[n] : t.r[n];
  }

  function leafValue(t, x) {
    let n = 0;
    while (t.f[n] !== -1) n = next(t, n, x);
    return t.v[n];
  }

  /** Raw score (log-odds) for class `cls` of a model {num_class, trees}. */
  function raw(model, x, cls = 0) {
    const K = model.num_class || 1;
    let s = 0;
    for (let i = cls; i < model.trees.length; i += K) s += leafValue(model.trees[i], x);
    return s;
  }

  function softmax(zs) {
    const m = Math.max(...zs), e = zs.map(z => Math.exp(z - m)), s = e.reduce((a, b) => a + b, 0);
    return e.map(v => v / s);
  }

  function multi(model, x) {
    const K = model.num_class;
    return softmax(Array.from({ length: K }, (_, k) => raw(model, x, k)));
  }

  // ---------------------------------------------------------------- TreeSHAP (path-dependent, exact)
  // Same layout as the reference C++ (shap/tree_shap.h): one flat buffer, each depth copies its parent's path.
  const SZ = 8192;
  const PF = new Int32Array(SZ), PZ = new Float64Array(SZ), PO = new Float64Array(SZ), PW = new Float64Array(SZ);

  function extend(o, d, zf, of, fi) {
    PF[o + d] = fi; PZ[o + d] = zf; PO[o + d] = of; PW[o + d] = d === 0 ? 1 : 0;
    for (let i = d - 1; i >= 0; i--) {
      PW[o + i + 1] += of * PW[o + i] * (i + 1) / (d + 1);
      PW[o + i] = zf * PW[o + i] * (d - i) / (d + 1);
    }
  }

  function unwind(o, d, pi) {
    const of = PO[o + pi], zf = PZ[o + pi];
    let nxt = PW[o + d];
    for (let i = d - 1; i >= 0; i--) {
      if (of !== 0) {
        const tmp = PW[o + i];
        PW[o + i] = nxt * (d + 1) / ((i + 1) * of);
        nxt = tmp - PW[o + i] * zf * (d - i) / (d + 1);
      } else {
        PW[o + i] = PW[o + i] * (d + 1) / (zf * (d - i));
      }
    }
    for (let i = pi; i < d; i++) { PF[o + i] = PF[o + i + 1]; PZ[o + i] = PZ[o + i + 1]; PO[o + i] = PO[o + i + 1]; }
  }

  function unwoundSum(o, d, pi) {
    const of = PO[o + pi], zf = PZ[o + pi];
    let nxt = PW[o + d], total = 0;
    for (let i = d - 1; i >= 0; i--) {
      if (of !== 0) {
        const tmp = nxt * (d + 1) / ((i + 1) * of);
        total += tmp;
        nxt = PW[o + i] - tmp * zf * ((d - i) / (d + 1));
      } else if (zf !== 0) {
        total += (PW[o + i] / zf) / ((d - i) / (d + 1));
      }
    }
    return total;
  }

  function shapTree(t, x, phi) {
    function rec(n, d, po, zf, of, fi) {
      const o = po + d + 1;
      for (let i = 0; i < d; i++) { PF[o + i] = PF[po + i]; PZ[o + i] = PZ[po + i]; PO[o + i] = PO[po + i]; PW[o + i] = PW[po + i]; }
      extend(o, d, zf, of, fi);
      if (t.f[n] === -1) {
        const v = t.v[n];
        for (let i = 1; i <= d; i++) phi[PF[o + i]] += unwoundSum(o, d, i) * (PO[o + i] - PZ[o + i]) * v;
        return;
      }
      const hot = next(t, n, x), cold = hot === t.l[n] ? t.r[n] : t.l[n];
      const cn = t.c[n] || 1, sf = t.f[n];
      let iz = 1, io = 1, pi = 0;
      for (; pi <= d; pi++) if (PF[o + pi] === sf) break;
      if (pi !== d + 1) { iz = PZ[o + pi]; io = PO[o + pi]; unwind(o, d, pi); d -= 1; }
      rec(hot, d + 1, o, (t.c[hot] / cn) * iz, io, sf);
      rec(cold, d + 1, o, (t.c[cold] / cn) * iz, 0, sf);
    }
    rec(0, 0, 0, 1, 1, -1);
  }

  function expectedTree(t) {
    if (t._e !== undefined) return t._e;
    let s = 0;
    for (let n = 0; n < t.f.length; n++) if (t.f[n] === -1) s += t.v[n] * t.c[n];
    return (t._e = t.f[0] === -1 ? t.v[0] : s / (t.c[0] || 1));
  }

  /** SHAP values in log-odds for class `cls`: {phi: [per feature], base} with base + sum(phi) = raw score. */
  function explain(model, x, nFeatures, cls = 0) {
    const K = model.num_class || 1, phi = new Array(nFeatures).fill(0);
    let base = 0;
    for (let i = cls; i < model.trees.length; i += K) {
      const t = model.trees[i];
      if (t.f[0] === -1) { base += t.v[0]; continue; }
      shapTree(t, x, phi);
      base += expectedTree(t);
    }
    return { phi, base };
  }

  /** The questions one tree asks about x, in order, and the leaf it lands on (for showing the model "thinking"). */
  function path(t, x) {
    const steps = [];
    let n = 0;
    while (t.f[n] !== -1) {
      const m = next(t, n, x), th = t.t[n];
      steps.push({ f: t.f[n], th: th < 0 ? null : th, set: th < 0 ? t.cs[-th - 1] : null, left: m === t.l[n] });
      n = m;
    }
    return { steps, value: t.v[n] };
  }

  /** Leaf value of every tree for x (class 0), so a page can show the score building up tree by tree. */
  function leaves(model, x) {
    return model.trees.map(t => leafValue(t, x));
  }

  // ---------------------------------------------------------------- every category of one feature at once
  // When you haven't picked a lender, the page needs the model for every lender in your state. Most of each tree
  // is the same for all of them, so instead of walking every tree once per lender, we walk it once and only split
  // the group of lenders at the nodes that ask about the lender. A group is a 64-bit set (two 32-bit halves).
  function catMask(t, n) {                             // which categories go LEFT at node n (cached on the tree)
    if (!t._ml) { t._ml = new Int32Array(t.f.length); t._mh = new Int32Array(t.f.length); t._mk = new Uint8Array(t.f.length); }
    if (!t._mk[n]) {
      let lo = 0, hi = 0;
      const th = t.t[n];
      if (th < 0) { for (const c of t.cs[-th - 1]) if (c < 32) lo |= 1 << c; else if (c < 64) hi |= 1 << (c - 32); }
      else for (let c = 0; c < 64 && c <= th; c++) if (c < 32) lo |= 1 << c; else hi |= 1 << (c - 32);
      t._ml[n] = lo; t._mh[n] = hi; t._mk[n] = 1;
    }
  }
  const forBits = (lo, hi, fn) => {
    for (let m = lo; m; m &= m - 1) fn(31 - Math.clz32(m & -m));
    for (let m = hi; m; m &= m - 1) fn(63 - Math.clz32(m & -m));
  };
  const toMask = cats => { let lo = 0, hi = 0; for (const c of cats) if (c < 32) lo |= 1 << c; else hi |= 1 << (c - 32); return [lo, hi]; };
  function walkCats(t, n, x, mf, lo, hi, out) {
    while (t.f[n] !== -1) {
      if (t.f[n] === mf) {
        catMask(t, n);
        const Ll = lo & t._ml[n], Lh = hi & t._mh[n], Rl = lo & ~t._ml[n], Rh = hi & ~t._mh[n];
        const goL = (Ll | Lh) !== 0, goR = (Rl | Rh) !== 0;
        if (goL && goR) { walkCats(t, t.l[n], x, mf, Ll, Lh, out); walkCats(t, t.r[n], x, mf, Rl, Rh, out); return; }
        n = goL ? t.l[n] : t.r[n];
      } else n = next(t, n, x);
    }
    const v = t.v[n];
    forBits(lo, hi, c => (out[c] += v));
  }
  /** Raw score for every category in `cats` of feature `mf` (the rest of x fixed): out[c] = raw(x with x[mf] = c). */
  function rawCats(model, x, mf, nCat, cats, cls = 0) {
    const K = model.num_class || 1, out = new Float64Array(nCat), [lo, hi] = toMask(cats);
    for (let i = cls; i < model.trees.length; i += K) walkCats(model.trees[i], 0, x, mf, lo, hi, out);
    return out;
  }
  /** Leaf value of every tree for every category: out[k * nCat + c]. */
  function leavesCats(model, x, mf, nCat, cats) {
    const T = model.trees.length, out = new Float64Array(T * nCat), tmp = new Float64Array(nCat), [lo, hi] = toMask(cats);
    for (let k = 0; k < T; k++) {
      tmp.fill(0);
      walkCats(model.trees[k], 0, x, mf, lo, hi, tmp);
      for (const c of cats) out[k * nCat + c] = tmp[c];
    }
    return out;
  }

  // Exact TreeSHAP of the AVERAGE model over the categories of feature `mf`, g(x) = Σ w[c] · f(x with x[mf] = c).
  // `mf` is not a player: at its nodes we follow both sides, keeping the set of categories that can still reach each
  // leaf; a leaf counts with the total weight of its categories. One walk per tree instead of one per category.
  // Result: base + Σ phi = Σ w[c] · raw(x with c), and phi[mf] = 0.
  function shapTreeMix(t, x, phi, mf, wv, lo0, hi0) {
    const mass = (lo, hi) => { let s = 0; forBits(lo, hi, c => (s += wv[c])); return s; };
    function visit(n, o, d, lo, hi) {
      if (t.f[n] === -1) {
        const v = t.v[n] * mass(lo, hi);
        if (v !== 0) for (let i = 1; i <= d; i++) phi[PF[o + i]] += unwoundSum(o, d, i) * (PO[o + i] - PZ[o + i]) * v;
        return;
      }
      const sf = t.f[n];
      if (sf === mf) {
        catMask(t, n);
        const Ll = lo & t._ml[n], Lh = hi & t._mh[n], Rl = lo & ~t._ml[n], Rh = hi & ~t._mh[n];
        if (Ll | Lh) pass(t.l[n], o, d, Ll, Lh);       // each side gets its own copy of the path:
        if (Rl | Rh) pass(t.r[n], o, d, Rl, Rh);       // the other side's subtree may unwind it
        return;
      }
      const hot = next(t, n, x), cold = hot === t.l[n] ? t.r[n] : t.l[n];
      const cn = t.c[n] || 1;
      let iz = 1, io = 1, pi = 0;
      for (; pi <= d; pi++) if (PF[o + pi] === sf) break;
      if (pi !== d + 1) { iz = PZ[o + pi]; io = PO[o + pi]; unwind(o, d, pi); d -= 1; }
      edge(hot, d + 1, o, (t.c[hot] / cn) * iz, io, sf, lo, hi);
      edge(cold, d + 1, o, (t.c[cold] / cn) * iz, 0, sf, lo, hi);
    }
    function edge(n, d, po, zf, of, fi, lo, hi) {      // copy the parent's path and add this split
      const o = po + d + 1;
      for (let i = 0; i < d; i++) { PF[o + i] = PF[po + i]; PZ[o + i] = PZ[po + i]; PO[o + i] = PO[po + i]; PW[o + i] = PW[po + i]; }
      extend(o, d, zf, of, fi);
      visit(n, o, d, lo, hi);
    }
    function pass(n, po, d, lo, hi) {                  // same path, fresh copy
      const o = po + d + 1;
      for (let i = 0; i <= d; i++) { PF[o + i] = PF[po + i]; PZ[o + i] = PZ[po + i]; PO[o + i] = PO[po + i]; PW[o + i] = PW[po + i]; }
      visit(n, o, d, lo, hi);
    }
    edge(0, 0, 0, 1, 1, -1, lo0, hi0);
  }
  function expectedMix(t, n, mf, wv, lo, hi) {
    if (t.f[n] === -1) { let s = 0; forBits(lo, hi, c => (s += wv[c])); return t.v[n] * s; }
    if (t.f[n] === mf) {
      catMask(t, n);
      const Ll = lo & t._ml[n], Lh = hi & t._mh[n], Rl = lo & ~t._ml[n], Rh = hi & ~t._mh[n];
      return ((Ll | Lh) ? expectedMix(t, t.l[n], mf, wv, Ll, Lh) : 0) + ((Rl | Rh) ? expectedMix(t, t.r[n], mf, wv, Rl, Rh) : 0);
    }
    const cn = t.c[n] || 1;
    return (t.c[t.l[n]] / cn) * expectedMix(t, t.l[n], mf, wv, lo, hi) + (t.c[t.r[n]] / cn) * expectedMix(t, t.r[n], mf, wv, lo, hi);
  }
  /** SHAP values of the weighted average over the categories of feature `mf` (weights `w` per category, summing to 1,
   *  at most 64 categories). `key` (optional) caches the base value, which depends only on the weights. */
  function explainMix(model, x, nFeatures, mf, w, key, cls = 0) {
    const K = model.num_class || 1, phi = new Array(nFeatures).fill(0);
    const cats = []; for (let c = 0; c < w.length; c++) if (w[c] > 0) cats.push(c);
    const [lo, hi] = toMask(cats);
    const cache = model._mixBase || (model._mixBase = {}), ck = key != null ? key + "|" + cls : null;
    const known = ck != null && cache[ck] !== undefined;
    let base = 0;
    for (let i = cls; i < model.trees.length; i += K) {
      const t = model.trees[i];
      if (t.f[0] === -1) { if (!known) base += t.v[0]; continue; }
      shapTreeMix(t, x, phi, mf, w, lo, hi);
      if (!known) base += expectedMix(t, 0, mf, w, lo, hi);
    }
    if (ck != null) { if (known) base = cache[ck]; else cache[ck] = base; }
    return { phi, base };
  }

  g.FKModel = { raw, multi, explain, path, leaves, rawCats, leavesCats, explainMix, softmax, sigmoid: z => 1 / (1 + Math.exp(-z)) };
})(typeof window !== "undefined" ? window : globalThis);
