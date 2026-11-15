/* Site behaviour: theme toggle, mobile menu, tabs, copy buttons, "on this page" highlighting,
   search, figure placeholders and the home-page chart readout. No libraries needed. */
(function () {
  "use strict";

  var root = document.documentElement;

  function store(key, value) {
    try {
      if (value === undefined) return window.localStorage.getItem(key);
      window.localStorage.setItem(key, value);
    } catch (e) { return null; }
    return null;
  }

  /* ---------- Theme ---------- */
  function currentTheme() {
    var set = root.getAttribute("data-theme");
    if (set === "dark" || set === "light") return set;
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }
  var themeBtn = document.querySelector(".theme-toggle");
  if (themeBtn) {
    var label = function () {
      themeBtn.setAttribute("aria-label", currentTheme() === "dark" ? "Switch to light theme" : "Switch to dark theme");
    };
    label();
    themeBtn.addEventListener("click", function () {
      var next = currentTheme() === "dark" ? "light" : "dark";
      root.setAttribute("data-theme", next);
      store("msa-theme", next);
      label();
    });
  }

  /* ---------- Mobile menu ---------- */
  var navBtn = document.querySelector(".nav-toggle");
  var backdrop = document.querySelector(".backdrop");
  function setNav(open) {
    document.body.classList.toggle("nav-open", open);
    if (navBtn) navBtn.setAttribute("aria-expanded", open ? "true" : "false");
  }
  if (navBtn) navBtn.addEventListener("click", function () { setNav(!document.body.classList.contains("nav-open")); });
  if (backdrop) backdrop.addEventListener("click", function () { setNav(false); });
  document.addEventListener("keydown", function (e) { if (e.key === "Escape") setNav(false); });
  var current = document.querySelector('.nav-list a[aria-current="page"]');
  if (current && current.scrollIntoView) current.scrollIntoView({ block: "nearest" });

  /* ---------- Figure placeholders ---------- */
  document.querySelectorAll(".figure-slot img").forEach(function (img) {
    var mark = function () { img.closest(".figure-slot").classList.add("is-empty"); };
    if (img.complete && img.naturalWidth === 0) mark();
    img.addEventListener("error", mark);
  });

  /* ---------- Tabs ---------- */
  document.querySelectorAll(".tabbed-set").forEach(function (set, n) {
    var labels = set.querySelectorAll(".tabbed-labels label");
    var blocks = set.querySelectorAll(".tabbed-content > .tabbed-block");
    if (!labels.length || labels.length !== blocks.length) return;
    var list = document.createElement("div");
    list.className = "tablist";
    list.setAttribute("role", "tablist");
    var buttons = [];
    labels.forEach(function (lab, i) {
      var b = document.createElement("button");
      b.type = "button";
      b.id = "tab-" + n + "-" + i;
      b.textContent = lab.textContent;
      b.setAttribute("role", "tab");
      b.setAttribute("aria-controls", "panel-" + n + "-" + i);
      blocks[i].id = "panel-" + n + "-" + i;
      blocks[i].setAttribute("role", "tabpanel");
      blocks[i].setAttribute("aria-labelledby", b.id);
      blocks[i].classList.add("tabpanel");
      b.addEventListener("click", function () { select(i); });
      b.addEventListener("keydown", function (e) {
        if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
          var j = (i + (e.key === "ArrowRight" ? 1 : -1) + buttons.length) % buttons.length;
          select(j); buttons[j].focus();
        }
      });
      buttons.push(b);
      list.appendChild(b);
    });
    function select(i) {
      buttons.forEach(function (b, k) {
        b.setAttribute("aria-selected", k === i ? "true" : "false");
        b.tabIndex = k === i ? 0 : -1;
        blocks[k].hidden = k !== i;
      });
    }
    set.insertBefore(list, set.querySelector(".tabbed-content"));
    set.classList.add("tabs", "is-enhanced");
    select(0);
  });

  /* ---------- Copy buttons on code blocks ---------- */
  document.querySelectorAll(".article pre").forEach(function (pre) {
    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "copy-btn";
    btn.textContent = "Copy";
    btn.addEventListener("click", function () {
      var text = pre.querySelector("code") ? pre.querySelector("code").innerText : pre.innerText;
      var done = function () { btn.textContent = "Copied"; setTimeout(function () { btn.textContent = "Copy"; }, 1500); };
      var fallback = function () {
        var range = document.createRange();
        range.selectNodeContents(pre.querySelector("code") || pre);
        var sel = window.getSelection(); sel.removeAllRanges(); sel.addRange(range);
        btn.textContent = "Press Ctrl+C";
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done, fallback);
      } else { fallback(); }
    });
    pre.appendChild(btn);
  });

  /* ---------- "On this page" highlighting ---------- */
  var tocLinks = Array.prototype.slice.call(document.querySelectorAll(".toc a"));
  if (tocLinks.length && "IntersectionObserver" in window) {
    var byId = {};
    tocLinks.forEach(function (a) { byId[decodeURIComponent(a.hash.slice(1))] = a; });
    var visible = {};
    var obs = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) { visible[en.target.id] = en.isIntersecting; });
      var heads = Object.keys(byId).map(function (id) { return document.getElementById(id); }).filter(Boolean);
      var active = null;
      for (var i = 0; i < heads.length; i++) {
        if (heads[i].getBoundingClientRect().top < window.innerHeight * 0.35) active = heads[i];
      }
      if (!active && heads.length) active = heads[0];
      tocLinks.forEach(function (a) { a.classList.toggle("is-active", active && byId[active.id] === a); });
    }, { rootMargin: "0px 0px -60% 0px", threshold: [0, 1] });
    Object.keys(byId).forEach(function (id) { var h = document.getElementById(id); if (h) obs.observe(h); });
  }

  /* ---------- Search ---------- */
  var input = document.getElementById("site-search");
  var results = document.getElementById("search-results");
  var base = document.body.getAttribute("data-root") || "";
  var index = window.SEARCH_INDEX || [];
  function escapeHtml(s) { return s.replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); }
  function highlight(text, terms) {
    var out = escapeHtml(text);
    terms.forEach(function (t) {
      if (!t) return;
      var re = new RegExp("(" + t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + ")", "gi");
      out = out.replace(re, "<mark>$1</mark>");
    });
    return out;
  }
  function snippet(text, terms) {
    var lower = text.toLowerCase();
    var at = -1;
    for (var i = 0; i < terms.length && at < 0; i++) at = lower.indexOf(terms[i]);
    if (at < 0) return text.slice(0, 140) + (text.length > 140 ? "…" : "");
    var start = Math.max(0, at - 50);
    return (start > 0 ? "…" : "") + text.slice(start, start + 150) + (start + 150 < text.length ? "…" : "");
  }
  function runSearch(q) {
    var terms = q.toLowerCase().split(/\s+/).filter(function (t) { return t.length > 1; });
    if (!terms.length) { results.hidden = true; results.innerHTML = ""; return; }
    var scored = [];
    index.forEach(function (p) {
      var title = p.title.toLowerCase(), heads = p.headings.join(" ").toLowerCase(), body = p.text.toLowerCase();
      var score = 0, all = true;
      terms.forEach(function (t) {
        var s = 0;
        if (title.indexOf(t) >= 0) s += 10;
        if (heads.indexOf(t) >= 0) s += 4;
        if (body.indexOf(t) >= 0) s += 1 + Math.min(5, body.split(t).length - 1) * 0.3;
        if (!s) all = false;
        score += s;
      });
      if (all && score) scored.push({ p: p, score: score });
    });
    scored.sort(function (a, b) { return b.score - a.score; });
    if (!scored.length) {
      results.innerHTML = '<li class="r-empty">No pages match “' + escapeHtml(q) + '”.</li>';
    } else {
      results.innerHTML = scored.slice(0, 8).map(function (r, i) {
        return '<li><a href="' + base + r.p.url + '"' + (i === 0 ? ' class="is-active"' : "") + '>' +
          '<span class="r-section">' + escapeHtml(r.p.section) + '</span>' +
          '<span class="r-title">' + highlight(r.p.title, terms) + '</span>' +
          '<span class="r-snippet">' + highlight(snippet(r.p.text, terms), terms) + '</span></a></li>';
      }).join("");
    }
    results.hidden = false;
  }
  if (input && results) {
    input.addEventListener("input", function () { runSearch(input.value.trim()); });
    input.addEventListener("focus", function () { if (input.value.trim()) runSearch(input.value.trim()); });
    input.addEventListener("keydown", function (e) {
      var links = Array.prototype.slice.call(results.querySelectorAll("a"));
      var at = links.findIndex(function (a) { return a.classList.contains("is-active"); });
      if (e.key === "ArrowDown" || e.key === "ArrowUp") {
        e.preventDefault();
        if (!links.length) return;
        var next = e.key === "ArrowDown" ? Math.min(links.length - 1, at + 1) : Math.max(0, at - 1);
        links.forEach(function (a, k) { a.classList.toggle("is-active", k === next); });
        links[next].scrollIntoView({ block: "nearest" });
      } else if (e.key === "Enter" && links.length) {
        e.preventDefault();
        window.location.href = links[Math.max(0, at)].href;
      } else if (e.key === "Escape") {
        results.hidden = true; input.blur();
      }
    });
    document.addEventListener("click", function (e) {
      if (!e.target.closest(".search")) results.hidden = true;
    });
    document.addEventListener("keydown", function (e) {
      if (e.key === "/" && document.activeElement !== input && !/input|textarea/i.test(document.activeElement.tagName)) {
        e.preventDefault(); input.focus();
      }
    });
  }

  /* ---------- Home chart readout ---------- */
  var readout = document.querySelector(".axes-readout");
  if (readout) {
    var initial = readout.innerHTML;
    document.querySelectorAll(".axes .pt").forEach(function (pt) {
      var show = function () {
        readout.innerHTML = "<b>" + pt.getAttribute("data-name") + "</b> · group gap " + pt.getAttribute("data-gap") +
          " · SAS " + pt.getAttribute("data-sas") + " · AUROC " + pt.getAttribute("data-auroc");
      };
      pt.addEventListener("mouseenter", show);
      pt.addEventListener("focus", show);
      pt.addEventListener("click", show);
      pt.addEventListener("mouseleave", function () { readout.innerHTML = initial; });
      pt.addEventListener("blur", function () { readout.innerHTML = initial; });
    });
  }
})();
