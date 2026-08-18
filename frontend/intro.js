/* GTP intro / loading animation.
 *
 * Self-contained: builds its own overlay element, animates it, then removes
 * itself from the DOM. Does not read or depend on any element/class already
 * on the host page (that page's layout is being rewritten concurrently) —
 * the only thing it assumes is that <body> exists and that intro.css is
 * linked (for the CSS custom-property tokens it reads, with literal
 * fallbacks baked into intro.css in case a token name ever changes).
 *
 * Plays on every load (no sessionStorage gate — product decision: the
 * animation is the point, not a one-time flourish) and takes ~4s end to
 * end, so it MUST be interruptible: any click/tap/keypress skips straight
 * to a quick fade-out. Never traps the user.
 *
 * Road geometry: 13 simplified corridor polylines derived from
 * frontend/corridors.geojson (Douglas-Peucker simplified from ~3800 raw
 * points down to 239, projected equirectangular-ish onto a 700x960 viewBox).
 * Baked in as a literal rather than fetched at runtime, so the intro has
 * zero network dependency and can never be slowed down or broken by a
 * missing/slow corridors.geojson.
 */
(function () {
  'use strict';

  var VIEWBOX = '0 0 700 960';
  var ROADS = [{"id":0,"cls":"highway","d":"M469.4 249.0 L467.1 242.7 L472.1 253.1 L453.2 247.5 L447.3 249.2 L413.3 294.8 L392.9 337.3 L382.6 349.9 L361.9 373.7 L316.9 400.4 L305.7 430.5 L307.8 441.2 L306.7 460.8 L305.6 464.3 L302.9 466.6 L299.7 467.4 L285.1 465.1 L280.7 465.5 L276.6 467.8 L269.5 476.2 L272.0 476.9 L270.1 478.7 L270.7 480.6 L273.8 481.0 L274.9 483.6 L282.2 487.8 L284.7 487.0 L285.6 489.1"},{"id":1,"cls":"arterial","d":"M537.6 200.3 L536.1 190.1 L537.3 188.4 L573.7 182.5 L575.7 181.5 L577.0 175.8 L579.6 177.8 L576.7 178.8 L576.1 182.0"},{"id":2,"cls":"arterial","d":"M576.1 182.0 L574.9 187.5 L574.9 203.8 L577.1 218.6 L595.0 280.5 L585.8 280.4 L585.6 291.8 L583.2 292.4"},{"id":3,"cls":"arterial","d":"M469.4 249.0 L466.9 242.7 L471.3 252.2 L475.2 279.8 L490.9 339.1 L493.2 352.0 L495.3 356.2 L496.0 354.9"},{"id":4,"cls":"expressway","d":"M512.1 43.3 L506.9 47.8 L503.5 42.1 L499.9 40.6 L476.8 71.1 L471.0 87.5 L426.9 144.1 L393.7 180.0 L364.9 209.1 L359.0 216.8 L353.2 227.1 L346.6 244.8 L344.3 262.9 L345.0 273.7 L349.4 300.4 L351.1 305.1 L353.9 307.4 L363.4 308.1 L375.1 315.4 L390.5 327.5 L398.5 331.0 L399.9 330.0 L399.0 328.1 L397.7 328.2 L395.9 331.2 L396.4 333.9 L398.2 333.3 L398.1 331.9 L388.8 327.0 L384.8 323.4 L383.0 325.8"},{"id":5,"cls":"arterial","d":"M583.2 292.4 L585.6 291.8 L585.6 293.7 L570.2 293.6 L570.0 308.0 L548.2 315.0 L490.9 339.6 L493.7 353.6 L500.4 363.9 L493.1 352.5 L484.7 353.6 L484.5 360.2"},{"id":6,"cls":"arterial","d":"M670.0 158.6 L643.2 171.9 L624.5 187.6 L620.8 187.4 L615.2 183.8 L603.7 182.4 L598.8 180.3 L585.9 181.9 L579.3 178.2 L576.8 178.7 L576.3 181.1 L574.6 182.7 L538.1 188.8 L536.5 190.4 L537.6 200.3"},{"id":7,"cls":"arterial","d":"M484.5 360.2 L484.7 353.6 L493.1 352.5 L489.5 339.7 L467.0 341.1 L424.9 339.8 L404.5 334.6 L390.6 328.0 L384.8 323.4 L383.0 325.8"},{"id":8,"cls":"expressway","d":"M124.1 609.0 L120.6 607.8 L125.1 597.3 L160.1 574.2 L171.7 559.6 L124.6 537.3 L99.3 518.9 L54.6 495.0 L38.6 488.5 L30.0 482.3"},{"id":9,"cls":"expressway","d":"M600.8 622.6 L605.9 591.5 L605.2 585.1 L583.5 550.6 L575.7 546.5 L530.8 539.5 L528.1 540.6 L527.2 562.1 L528.9 615.2 L533.1 633.9 L537.3 640.8 L537.8 643.7 L525.0 679.3 L517.1 697.9 L529.3 738.7 L539.0 756.1 L537.4 768.8 L534.5 777.2 L535.7 792.8 L530.7 847.9 L529.1 857.3 L526.4 862.5 L526.0 869.0 L520.3 874.1 L510.2 877.1 L499.0 884.7 L491.0 887.7 L479.7 888.8 L465.6 894.6 L460.6 897.7 L454.9 904.2 L445.9 907.2 L437.1 912.5 L429.5 919.8 L427.3 918.5"},{"id":10,"cls":"highway","d":"M525.4 637.1 L526.5 640.8 L529.5 642.5 L528.2 646.0 L525.6 647.3 L523.7 657.3 L523.4 653.2 L521.9 651.4 L520.5 652.2 L517.8 645.9 L513.3 650.4 L479.4 665.8 L456.6 670.4 L450.3 674.5 L415.3 689.5 L374.9 696.9 L342.3 705.0 L330.1 709.0 L327.5 714.1 L324.1 713.6 L325.1 712.8 L327.0 713.6"},{"id":11,"cls":"highway","d":"M566.4 103.2 L558.0 97.2 L551.2 111.6 L556.5 114.1 L570.9 124.5 L572.3 123.9 L572.9 118.8 L581.6 105.1 L573.7 119.3 L574.1 123.9 L576.7 131.6 L576.0 136.2 L563.7 152.7 L534.5 186.9 L511.7 207.8 L485.4 240.6 L484.8 239.8 L498.7 223.8"},{"id":12,"cls":"highway","d":"M392.2 233.8 L394.3 231.1 L391.4 229.3 L381.1 228.4 L380.3 221.3 L365.3 210.5 L364.1 210.7 L356.1 222.3 L347.0 245.0 L344.7 262.0 L346.8 282.8 L339.2 287.5 L332.7 286.4 L328.4 287.1 L310.0 269.8 L292.5 276.0 L277.9 286.6 L259.2 294.2 L242.5 310.7 L198.5 339.7 L185.4 350.7 L156.9 366.2 L146.6 374.2 L128.3 383.3 L110.9 397.6 L89.7 410.0 L77.0 422.4 L59.2 451.5 L58.1 455.9 L56.0 455.8 L49.9 464.0 L31.6 483.7 L30.0 482.3"}];

  // Timing budget (ms). Total lifetime target: ~4s of deliberate, watchable
  // draw (product decision — this used to be ~1.2s, changed to be slower
  // and to actually be watched), with a generous hard-cap failsafe well
  // past that so a stalled browser can never make this hang. Anything that
  // makes the sequence THIS long has to be skippable — see attachSkip().
  //
  // All 13 corridors draw simultaneously (no stagger, no per-path delay) —
  // one blank-to-fully-drawn sweep on a fixed shared duration, so every path
  // starts and finishes at the same instant regardless of its own length.
  // That "same instant" finish is what FADE_START_MS is anchored after,
  // below, so the app is never revealed mid-stroke.
  var DRAW_MS = 3000;        // shared draw duration for every road, set inline (see run()) — not in intro.css
  // WORD_DELAY_MS (2.4s) and the overlay's normal fade duration (550) are
  // also encoded in intro.css's transition-duration / transition-delay
  // values for .gtp-intro-word / #gtp-intro — keep both files in sync.
  var FADE_START_MS = DRAW_MS + 400; // whole-overlay fade-out begins only once every road has finished, plus a short hold
  var FADE_MS = 550;         // matches intro.css #gtp-intro transition-duration
  var FAST_FADE_MS = 220;    // matches intro.css .gtp-intro-skip — used for skip / reduced-motion / fallback
  var HARD_CAP_MS = 6000;    // absolute failsafe: overlay is gone by here no matter what
  var REDUCED_HOLD_MS = 160;

  function prefersReducedMotion() {
    try {
      return !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
    } catch (e) {
      return false;
    }
  }

  function build() {
    var root = document.createElement('div');
    root.id = 'gtp-intro';
    root.setAttribute('aria-hidden', 'true');

    var svgNS = 'http://www.w3.org/2000/svg';
    var svg = document.createElementNS(svgNS, 'svg');
    svg.setAttribute('class', 'gtp-intro-svg');
    svg.setAttribute('viewBox', VIEWBOX);
    svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
    for (var i = 0; i < ROADS.length; i++) {
      var r = ROADS[i];
      var p = document.createElementNS(svgNS, 'path');
      p.setAttribute('class', 'gtp-road gtp-road-' + r.cls);
      p.setAttribute('d', r.d);
      svg.appendChild(p);
    }

    var wrap = document.createElement('div');
    wrap.className = 'gtp-intro-svg-wrap';
    wrap.appendChild(svg);

    var word = document.createElement('div');
    word.className = 'gtp-intro-word';
    var title = document.createElement('span');
    title.className = 'gtp-intro-title';
    title.textContent = 'GTP';
    var sub = document.createElement('span');
    sub.className = 'gtp-intro-sub';
    sub.textContent = 'Gurugram Traffic Intelligence';
    word.appendChild(title);
    word.appendChild(sub);

    var inner = document.createElement('div');
    inner.className = 'gtp-intro-inner';
    inner.appendChild(wrap);
    inner.appendChild(word);
    root.appendChild(inner);

    var hint = document.createElement('div');
    hint.className = 'gtp-intro-hint';
    hint.textContent = 'click or press any key to skip';
    root.appendChild(hint);

    document.body.appendChild(root);
    return root;
  }

  function remove(root) {
    if (root && root.parentNode) root.parentNode.removeChild(root);
  }

  // Any click/tap/keypress on the overlay skips straight to a quick fade —
  // required because this plays in full on every page load (no "seen it
  // already" gate), so a returning visitor (or anyone reloading repeatedly
  // while developing against this page) must always have an instant way out.
  function attachSkip(root, onSkip) {
    var fired = false;
    function handler() {
      if (fired) return;
      fired = true;
      cleanup();
      onSkip();
    }
    function cleanup() {
      root.removeEventListener('click', handler);
      root.removeEventListener('touchstart', handler);
      document.removeEventListener('keydown', handler);
    }
    root.addEventListener('click', handler);
    root.addEventListener('touchstart', handler, { passive: true });
    document.addEventListener('keydown', handler);
    return cleanup;
  }

  function run() {
    var root = build();
    var removed = false;
    function safeRemove() {
      if (removed) return;
      removed = true;
      remove(root);
    }

    // Absolute safety net: whatever else happens (an exception mid-setup, a
    // stalled transition in some odd browser), the overlay is force-removed
    // by this point regardless. This is what makes the intro "never block",
    // set comfortably above the ~4s intended runtime rather than fighting it.
    setTimeout(safeRemove, HARD_CAP_MS);

    var finished = false;
    var skipCleanup = null;
    function finish(fast) {
      if (finished) return;
      finished = true;
      if (skipCleanup) skipCleanup();
      if (fast) root.classList.add('gtp-intro-skip');
      root.classList.add('gtp-intro-out'); // pointer-events:none applies immediately, see intro.css
      setTimeout(safeRemove, (fast ? FAST_FADE_MS : FADE_MS) + 60);
    }

    skipCleanup = attachSkip(root, function () { finish(true); });

    if (prefersReducedMotion()) {
      root.classList.add('gtp-intro-reduced');
      // No drawing, no motion — roads/word render fully-formed on the first
      // frame (see intro.css), so this is just a brief, near-instant fade.
      // This matters more now than it used to: the full-motion path takes
      // ~4s on every single load, so honoring this setting is the single
      // most consequential accessibility behavior in the whole feature.
      requestAnimationFrame(function () {
        setTimeout(function () { finish(true); }, REDUCED_HOLD_MS);
      });
      return;
    }

    try {
      var paths = root.querySelectorAll('.gtp-road');
      var transitionStr = 'stroke-dashoffset ' + (DRAW_MS / 1000) + 's cubic-bezier(.4,0,.2,1)';
      for (var i = 0; i < paths.length; i++) {
        var len = paths[i].getTotalLength();
        paths[i].style.strokeDasharray = len;
        paths[i].style.strokeDashoffset = len; // hidden: whole stroke pushed into the "gap" of the dash pattern
        // The transition itself is set inline too, deliberately — NOT via a
        // CSS class rule. A class rule setting stroke-dashoffset can never
        // win a cascade fight against this inline stroke-dashoffset (inline
        // beats any stylesheet selector short of !important), so the class
        // approach silently never animates: the property just sits at
        // whatever the inline value says, forever. Keeping both the start
        // and end values inline sidesteps that entirely.
        paths[i].style.transition = transitionStr;
      }
      // Force a reflow so the "hidden" (offset === length) state above is
      // committed as its own style/layout pass before the flip below.
      // eslint-disable-next-line no-unused-expressions
      root.getBoundingClientRect();
      // Double rAF, not single: a single rAF callback can still land inside
      // the SAME committed frame as the "hidden" state set above (the forced
      // reflow guarantees a layout pass, but not that a frame was actually
      // painted with stroke-dashoffset === length before the value changes
      // again) — when that happens the browser has nothing to transition
      // FROM and just jumps straight to the end value, i.e. the roads render
      // fully-drawn instantly with no visible draw-in at all. Waiting a
      // second rAF guarantees the hidden state was genuinely painted first.
      requestAnimationFrame(function () {
        requestAnimationFrame(function () {
          for (var j = 0; j < paths.length; j++) {
            paths[j].style.strokeDashoffset = '0'; // draws in: hidden -> fully visible, all 13 roads together
          }
          root.classList.add('gtp-intro-draw'); // drives the word/hint CSS fades only — never touches stroke-dashoffset
          // Anchored here (actual draw start), not at run() entry: if the
          // browser's first rAF is delayed — a backgrounded tab, or realistic
          // main-thread contention from the host page's own script (this runs
          // alongside a MapLibre GL bundle load) — the fade-out still waits
          // for the draw to actually get its full on-screen duration instead
          // of firing on a wall-clock schedule that assumed rAF was prompt.
          // FADE_START_MS is DRAW_MS + a hold, so the app is only ever
          // revealed strictly after every road has finished drawing.
          setTimeout(function () { finish(false); }, FADE_START_MS);
        });
      });
    } catch (e) {
      // getTotalLength() or similar unsupported/failing — degrade to a
      // static reveal (roads render solid, no draw-in) rather than crash.
      root.classList.add('gtp-intro-reduced');
      requestAnimationFrame(function () {
        setTimeout(function () { finish(true); }, REDUCED_HOLD_MS);
      });
    }
  }

  function safeRun() {
    try {
      run();
    } catch (e) {
      // Never let the intro itself become the reason the page looks broken.
      var stray = document.getElementById('gtp-intro');
      if (stray && stray.parentNode) stray.parentNode.removeChild(stray);
      if (window.console && console.error) console.error('[gtp-intro]', e);
    }
  }

  if (document.body) {
    safeRun();
  } else {
    document.addEventListener('DOMContentLoaded', safeRun);
  }
})();
