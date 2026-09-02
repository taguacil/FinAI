/* FinAI web — shared client helpers: Plotly dark theme, formatters, charts. */

const FINAI = (() => {
  const ACCENT = '#C3C9D0';  /* brushed silver */
  /* Monochrome metallic ramp — reads as an institutional chart, not a rainbow. */
  const COLORWAY = ['#C7CDD4', '#9AA4AF', '#74808C', '#B6AEA3', '#4C555F', '#8A8F98', '#A9B0B8', '#5E666F', '#D2D7DC', '#7A828B'];
  const BORDER = '#262A30';
  const GRID = 'rgba(38,42,48,0.5)';  /* lighter grid for a calmer, minimal look */
  const TEXT = '#E6E8EB';
  const MUTED = '#868D96';

  const baseLayout = (over = {}) => ({
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: TEXT, family: 'Inter, sans-serif', size: 12 },
    colorway: COLORWAY,
    margin: { t: 10, r: 12, b: 36, l: 52 },
    xaxis: { gridcolor: GRID, zerolinecolor: GRID, linecolor: BORDER, tickcolor: BORDER },
    yaxis: { gridcolor: GRID, zerolinecolor: GRID, linecolor: BORDER, tickcolor: BORDER },
    legend: { bgcolor: 'rgba(0,0,0,0)', font: { color: MUTED } },
    hoverlabel: { bgcolor: '#14161A', bordercolor: BORDER, font: { color: TEXT } },
    ...over,
  });

  const config = { displayModeBar: false, responsive: true };

  // Raw InstrumentType key -> display label (mirrors the backend map, used for chips).
  const ASSET_CLASS_LABELS = {
    stock: 'Equities', etf: 'ETFs', bond: 'Bonds', crypto: 'Crypto',
    cash: 'Cash', mutual_fund: 'Mutual Funds', option: 'Options', future: 'Futures',
  };
  const assetClassLabel = (k) => ASSET_CLASS_LABELS[k] || (k ? k.charAt(0).toUpperCase() + k.slice(1) : '—');

  // Fixed asset-class colours: muted, low-saturation hues chosen so stacked
  // bands stay distinguishable while keeping the restrained, professional feel.
  const ASSET_COLORS = {
    Equities: '#8FA9C4', Bonds: '#84A99A', ETFs: '#C2A97E', 'Mutual Funds': '#B08C97',
    Cash: '#5E666F', Crypto: '#9A8FB5', Options: '#7FA8A0', Futures: '#A9926F',
  };

  const fmtMoney = (v, ccy = 'USD') => {
    if (v === null || v === undefined || isNaN(v)) return '—';
    try {
      return new Intl.NumberFormat('en-US', { style: 'currency', currency: ccy, maximumFractionDigits: 0 }).format(v);
    } catch (_) {
      return `${ccy} ${Math.round(v).toLocaleString()}`;
    }
  };
  const fmtNum = (v, d = 2) => (v === null || v === undefined || isNaN(v)) ? '—' : Number(v).toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });
  const fmtPct = (v, d = 2) => (v === null || v === undefined || isNaN(v)) ? '—' : `${v >= 0 ? '+' : ''}${Number(v).toFixed(d)}%`;
  const signClass = (v) => (v === null || v === undefined || isNaN(v)) ? '' : (v >= 0 ? 'pos' : 'neg');

  function equityCurve(el, dates, values, ccy) {
    if (!dates || !dates.length) { el.innerHTML = '<p class="text-muted text-sm p-4">No history yet.</p>'; return; }
    const trace = {
      x: dates, y: values, type: 'scatter', mode: 'lines',
      line: { color: ACCENT, width: 1.6, shape: 'spline', smoothing: 0.4 },
      fill: 'tozeroy', fillcolor: 'rgba(195,201,208,0.05)',
      hovertemplate: `%{x}<br>${ccy} %{y:,.0f}<extra></extra>`,
    };
    const ymin = Math.min(...values), ymax = Math.max(...values);
    const pad = (ymax - ymin) * 0.08 || ymax * 0.05;
    Plotly.newPlot(el, [trace], baseLayout({
      yaxis: { gridcolor: GRID, zerolinecolor: GRID, range: [ymin - pad, ymax + pad], tickprefix: '' },
      xaxis: { gridcolor: 'rgba(0,0,0,0)', linecolor: BORDER },
    }), config);
  }

  function donut(el, labels, values) {
    if (!labels || !labels.length) { el.innerHTML = '<p class="text-muted text-sm p-4">No positions.</p>'; return; }
    // Prefer fixed asset-class colours when labels are known classes; else fall back.
    const colors = labels.map((l, i) => ASSET_COLORS[l] || COLORWAY[i % COLORWAY.length]);
    const trace = {
      labels, values, type: 'pie', hole: 0.62, sort: true, direction: 'clockwise',
      marker: { colors, line: { color: '#0A0B0D', width: 2 } },
      textinfo: 'none',
      hovertemplate: '%{label}<br>%{value:,.0f} (%{percent})<extra></extra>',
    };
    Plotly.newPlot(el, [trace], baseLayout({
      margin: { t: 8, r: 8, b: 8, l: 8 },
      showlegend: true,
      legend: { orientation: 'v', x: 1, y: 0.5, font: { color: MUTED, size: 11 } },
    }), config);
  }

  const noData = (el, msg) => { el.innerHTML = `<p class="text-muted text-sm p-4">${msg || 'No data.'}</p>`; };

  // Multi-line chart. series = [{ name, y, color?, dash?, width? }]; x shared.
  function lineMulti(el, x, series, opts = {}) {
    if (!x || !x.length || !series || !series.length) { noData(el, opts.empty); return; }
    const traces = series.map((s, i) => ({
      x, y: s.y, name: s.name, type: 'scatter', mode: 'lines',
      line: { color: s.color || COLORWAY[i % COLORWAY.length], width: s.width || 1.6, dash: s.dash || 'solid', shape: 'spline', smoothing: 0.3 },
      hovertemplate: `%{x}<br>${s.name}: %{y:${opts.hoverfmt || ',.2f'}}<extra></extra>`,
    }));
    Plotly.newPlot(el, traces, baseLayout({
      showlegend: series.length > 1,
      legend: { orientation: 'h', y: 1.12, x: 0, font: { color: MUTED, size: 11 } },
      xaxis: { gridcolor: 'rgba(0,0,0,0)', linecolor: BORDER },
      yaxis: { gridcolor: GRID, zerolinecolor: GRID, ticksuffix: opts.ysuffix || '' },
      margin: { t: 24, r: 12, b: 36, l: 52 },
    }), config);
  }

  // Stacked (cumulating) area chart — bands sum to the total at each x.
  // series = [{ label, values, color? }]; x shared. Uses fixed asset-class
  // colours when the label is a known class.
  function stackArea(el, x, series, opts = {}) {
    if (!x || !x.length || !series || !series.length) { noData(el, opts.empty); return; }
    const ccy = opts.ccy || 'USD';
    const traces = series.map((s, i) => ({
      x, y: s.values, name: s.label, type: 'scatter', mode: 'lines',
      stackgroup: 'one',
      line: { width: 0.8, color: s.color || ASSET_COLORS[s.label] || COLORWAY[i % COLORWAY.length] },
      fillcolor: s.color || ASSET_COLORS[s.label] || COLORWAY[i % COLORWAY.length],
      hovertemplate: `%{x}<br>${s.label}: ${ccy} %{y:,.0f}<extra></extra>`,
    }));
    Plotly.newPlot(el, traces, baseLayout({
      showlegend: true,
      legend: { orientation: 'h', y: 1.12, x: 0, font: { color: MUTED, size: 11 } },
      xaxis: { gridcolor: 'rgba(0,0,0,0)', linecolor: BORDER },
      yaxis: { gridcolor: GRID, zerolinecolor: GRID, tickprefix: '' },
      margin: { t: 24, r: 12, b: 36, l: 60 },
    }), config);
  }

  // Filled drawdown area (values are <= 0, e.g. percent below peak).
  function drawdown(el, x, values, opts = {}) {
    if (!x || !x.length) { noData(el, opts.empty); return; }
    const trace = {
      x, y: values, type: 'scatter', mode: 'lines',
      line: { color: '#D07B6B', width: 1.2 },
      fill: 'tozeroy', fillcolor: 'rgba(208,123,107,0.12)',
      hovertemplate: `%{x}<br>%{y:.2f}%<extra></extra>`,
    };
    Plotly.newPlot(el, [trace], baseLayout({
      xaxis: { gridcolor: 'rgba(0,0,0,0)', linecolor: BORDER },
      yaxis: { gridcolor: GRID, zerolinecolor: GRID, ticksuffix: '%' },
    }), config);
  }

  // Histogram (returns distribution).
  function histogram(el, values, opts = {}) {
    if (!values || !values.length) { noData(el, opts.empty); return; }
    const trace = {
      x: values, type: 'histogram', marker: { color: ACCENT, line: { color: '#0A0B0D', width: 1 } },
      opacity: 0.85, nbinsx: opts.bins || 40,
      hovertemplate: `%{x}<br>%{y} periods<extra></extra>`,
    };
    Plotly.newPlot(el, [trace], baseLayout({
      bargap: 0.02,
      xaxis: { gridcolor: 'rgba(0,0,0,0)', linecolor: BORDER, ticksuffix: opts.xsuffix || '' },
      yaxis: { gridcolor: GRID, zerolinecolor: GRID },
    }), config);
  }

  // Horizontal bars (e.g. optimizer target weights). rows = [{label, value}].
  function barsH(el, rows, opts = {}) {
    if (!rows || !rows.length) { noData(el, opts.empty); return; }
    const labels = rows.map(r => r.label), values = rows.map(r => r.value);
    const trace = {
      x: values, y: labels, type: 'bar', orientation: 'h',
      marker: { color: ACCENT }, hovertemplate: `%{y}: %{x:${opts.hoverfmt || '.2f'}}${opts.suffix || ''}<extra></extra>`,
    };
    Plotly.newPlot(el, [trace], baseLayout({
      margin: { t: 8, r: 12, b: 32, l: Math.max(80, opts.leftMargin || 0) },
      xaxis: { gridcolor: GRID, zerolinecolor: GRID, ticksuffix: opts.suffix || '' },
      yaxis: { gridcolor: 'rgba(0,0,0,0)', autorange: 'reversed' },
    }), config);
  }

  // Scatter (e.g. efficient frontier: points of {x: risk, y: return}), optional highlight.
  function scatter(el, pts, opts = {}) {
    if (!pts || !pts.length) { noData(el, opts.empty); return; }
    const traces = [{
      x: pts.map(p => p.x), y: pts.map(p => p.y), type: 'scatter', mode: 'lines+markers',
      line: { color: ACCENT, width: 1.4 }, marker: { color: ACCENT, size: 5 },
      name: opts.name || 'Frontier',
      hovertemplate: `Risk %{x:.2f}${opts.suffix || ''}<br>Return %{y:.2f}${opts.suffix || ''}<extra></extra>`,
    }];
    if (opts.highlight) {
      traces.push({
        x: [opts.highlight.x], y: [opts.highlight.y], type: 'scatter', mode: 'markers',
        marker: { color: '#5CB27F', size: 11, symbol: 'star' }, name: opts.highlight.name || 'Selected',
        hovertemplate: `%{fullData.name}<br>Risk %{x:.2f}${opts.suffix || ''}<br>Return %{y:.2f}${opts.suffix || ''}<extra></extra>`,
      });
    }
    Plotly.newPlot(el, traces, baseLayout({
      showlegend: !!opts.highlight,
      legend: { orientation: 'h', y: 1.12, x: 0, font: { color: MUTED, size: 11 } },
      xaxis: { title: { text: opts.xtitle || 'Risk', font: { color: MUTED, size: 11 } }, gridcolor: GRID, zerolinecolor: GRID, ticksuffix: opts.suffix || '' },
      yaxis: { title: { text: opts.ytitle || 'Return', font: { color: MUTED, size: 11 } }, gridcolor: GRID, zerolinecolor: GRID, ticksuffix: opts.suffix || '' },
      margin: { t: 24, r: 12, b: 44, l: 56 },
    }), config);
  }

  // Percentile fan chart (Monte Carlo). bands = {p5,p25,p50,p75,p95} arrays over x.
  function fanChart(el, x, bands, ccy, opts = {}) {
    if (!x || !x.length || !bands || !bands.p50) { noData(el, opts.empty); return; }
    const band = (lo, hi, color) => ([
      { x, y: hi, type: 'scatter', mode: 'lines', line: { width: 0 }, hoverinfo: 'skip', showlegend: false },
      { x, y: lo, type: 'scatter', mode: 'lines', line: { width: 0 }, fill: 'tonexty', fillcolor: color, hoverinfo: 'skip', showlegend: false },
    ]);
    const traces = [];
    if (bands.p5 && bands.p95) traces.push(...band(bands.p5, bands.p95, 'rgba(195,201,208,0.08)'));
    if (bands.p25 && bands.p75) traces.push(...band(bands.p25, bands.p75, 'rgba(195,201,208,0.16)'));
    traces.push({
      x, y: bands.p50, type: 'scatter', mode: 'lines', name: 'Median',
      line: { color: ACCENT, width: 1.8 },
      hovertemplate: `%{x}<br>${ccy || ''} %{y:,.0f}<extra></extra>`,
    });
    Plotly.newPlot(el, traces, baseLayout({
      showlegend: false,
      xaxis: { gridcolor: 'rgba(0,0,0,0)', linecolor: BORDER, title: { text: opts.xtitle || '', font: { color: MUTED, size: 11 } } },
      yaxis: { gridcolor: GRID, zerolinecolor: GRID },
      margin: { t: 12, r: 12, b: 40, l: 60 },
    }), config);
  }

  return {
    baseLayout, config, fmtMoney, fmtNum, fmtPct, signClass, assetClassLabel,
    equityCurve, donut, lineMulti, stackArea, drawdown, histogram, barsH, scatter, fanChart,
    COLORWAY, ACCENT,
  };
})();
