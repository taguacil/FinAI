/* FinAI web — shared client helpers: Plotly dark theme, formatters, charts. */

const FINAI = (() => {
  const ACCENT = '#2DD4BF';
  const COLORWAY = ['#2DD4BF', '#6366F1', '#F59E0B', '#22C55E', '#F87171', '#38BDF8', '#A78BFA', '#FB7185', '#34D399', '#FBBF24'];
  const BORDER = '#1E2A3D';
  const GRID = 'rgba(30,42,61,0.5)';  /* lighter grid for a calmer, minimal look */
  const TEXT = '#E6EDF3';
  const MUTED = '#8B98A5';

  const baseLayout = (over = {}) => ({
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: TEXT, family: 'Inter, sans-serif', size: 12 },
    colorway: COLORWAY,
    margin: { t: 10, r: 12, b: 36, l: 52 },
    xaxis: { gridcolor: GRID, zerolinecolor: GRID, linecolor: BORDER, tickcolor: BORDER },
    yaxis: { gridcolor: GRID, zerolinecolor: GRID, linecolor: BORDER, tickcolor: BORDER },
    legend: { bgcolor: 'rgba(0,0,0,0)', font: { color: MUTED } },
    hoverlabel: { bgcolor: '#131A2A', bordercolor: BORDER, font: { color: TEXT } },
    ...over,
  });

  const config = { displayModeBar: false, responsive: true };

  // Raw InstrumentType key -> display label (mirrors the backend map, used for chips).
  const ASSET_CLASS_LABELS = {
    stock: 'Equities', etf: 'ETFs', bond: 'Bonds', crypto: 'Crypto',
    cash: 'Cash', mutual_fund: 'Mutual Funds', option: 'Options', future: 'Futures',
  };
  const assetClassLabel = (k) => ASSET_CLASS_LABELS[k] || (k ? k.charAt(0).toUpperCase() + k.slice(1) : '—');

  // Fixed, institutional colours per asset class for a consistent read across views.
  const ASSET_COLORS = {
    Equities: '#2DD4BF', ETFs: '#38BDF8', Bonds: '#6366F1', Crypto: '#F59E0B',
    Cash: '#64748B', 'Mutual Funds': '#A78BFA', Options: '#FB7185', Futures: '#34D399',
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
      fill: 'tozeroy', fillcolor: 'rgba(45,212,191,0.04)',
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
      marker: { colors, line: { color: '#0B0F1A', width: 2 } },
      textinfo: 'none',
      hovertemplate: '%{label}<br>%{value:,.0f} (%{percent})<extra></extra>',
    };
    Plotly.newPlot(el, [trace], baseLayout({
      margin: { t: 8, r: 8, b: 8, l: 8 },
      showlegend: true,
      legend: { orientation: 'v', x: 1, y: 0.5, font: { color: MUTED, size: 11 } },
    }), config);
  }

  return { baseLayout, config, fmtMoney, fmtNum, fmtPct, signClass, assetClassLabel, equityCurve, donut, COLORWAY, ACCENT };
})();
