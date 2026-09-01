---
name: stock-daily-digest
description: Generate a daily portfolio digest with YTD performance, P&L %, technical signals (MA crossovers), and relevant market news. Use when a scheduled cron job or the user asks for a daily portfolio summary.
---

# Stock Daily Digest Skill

Use this skill when a cron job or the user wants a **daily portfolio digest** with performance metrics, technical signals, and market news matched to portfolio positions.

## Workflow

All interactions use `mcporter call` bash commands to talk to the FinAI MCP server.

### Prerequisites

- `mcporter` CLI installed and on PATH
- FinAI MCP server configured as `finai` in mcporter config
- Check status with: `mcporter list` (should show `finai (N tools, X.Xs)`)

### 1. Select Portfolio

```bash
mcporter call finai.select_portfolio portfolio_id=d97d941e-f87f-4b00-828b-2d533824d26d --output json
```

### 2. Refresh Data (best-effort)

```bash
mcporter call finai.refresh_data --output json
```

If this fails/times out, continue with cached data and note: `⚠️ Using cached data`

### 3. Get Portfolio Summary

```bash
mcporter call finai.get_portfolio_summary --output json
```

Extract from response:
- `total_value` — portfolio total in USD
- For each position: `quantity`, `price`, `pnl_pct`, `instrument_type`
- Calculate weight: `weight_pct = (quantity × price) / total_value × 100`

### 4. Get YTD Performance

```bash
mcporter call finai.get_ytd_performance --output json
```

Use the returned `ytd_return` for each position directly.

### 5. Filter Positions

Include only: **stock**, **etf**, **crypto**
Exclude: bonds, cash, mutual_fund, fixed-income

### 6. Get MA Signals

For each filtered symbol:

```bash
mcporter call finai.get_moving_average_signal symbol=SYM short_period=9 long_period=26 --output json
mcporter call finai.get_moving_average_signal symbol=SYM short_period=50 long_period=200 --output json
```

Map each result to a signal indicator:
- `▲` bullish (positive MA difference)
- `▼` bearish (negative MA difference)

Display both signals separately: `9/26` column and `50/200` column.

### 7. Categorize Positions

| Category        | Criteria                                      |
|-----------------|-----------------------------------------------|
| TOP PERFORMERS  | P&L % > +15%                                  |
| STEADY          | 0% ≤ P&L % ≤ +15%                             |
| LAGGARDS        | P&L % < 0%                                    |
| WATCH LIST      | Signal = `▼▼` OR (YTD < 0% AND weight > 5%)  |

Sort each category by **P&L % descending**.

### 8. Fetch Relevant News

Use WebSearch to find recent news for portfolio positions:

1. **Top movers** — search for the top 3 performers and bottom 3 laggards:
   ```
   WebSearch: "{SYMBOL} stock news today"
   ```

2. **Watch list items** — search for any positions on the watch list

3. **Match patterns** — correlate news to portfolio performance:
   - Earnings reports → price moves
   - Sector trends → multiple positions affected
   - Macro events → broad portfolio impact
   - Company-specific news → individual position moves

Keep news concise: 1-2 sentences per relevant item.

### 9. Generate Recommendations

Based on signals, news, and portfolio state, provide actionable recommendations:

| Condition | Recommendation Type |
|-----------|---------------------|
| `▼▼` signal + negative news | Consider reducing position |
| `▲▲` signal + positive news | Hold or consider adding |
| High weight + `▼▼` signal | Rebalance warning |
| YTD < -20% + no catalyst | Review thesis |
| `▲▲` signal + underweight | Potential opportunity |

**Guidelines for recommendations:**
- Be specific: name the position and action
- Cite the signal and/or news driving the recommendation
- Distinguish between "consider" (suggestion) vs "warning" (risk alert)
- Max 3-5 position-level recommendations
- Include 1 portfolio-level observation if relevant

---

## Output Format

The skill MUST produce **two layers of output** each time it runs:

1. A **Compact View** (short summary) at the top of the response
2. The **Full Digest** in the detailed table format
3. The **Full Digest must also be written to a markdown file** in the workspace

### 1. Compact View (summary)

Place this at the top of the tool response before the full digest:

```markdown
### Portfolio Snapshot · {date}{cached_flag}

- Value: ${total_value}
- Total P&L: {+/-}{total_pnl_pct}%
- YTD: {+/-}{portfolio_ytd}%

Top movers (P&L %):
- {SYMBOL}: {pnl}%
- {SYMBOL}: {pnl}%
- {SYMBOL}: {pnl}%

Watch list:
- {SYMBOL}: {short_reason}
- {SYMBOL}: {short_reason}

Key actions:
- {concise_action_1}
- {concise_action_2}
- {concise_action_3}
```

Where:
- `cached_flag` is ` · ⚠️ Using cached data` when refresh_data had errors, otherwise empty
- Top movers: pick ~3 largest positive P&L % from TOP PERFORMERS
- Watch list: list up to ~3 watch list symbols with very short reasons
- Key actions: 3–5 bullets summarizing the most important recommendations

### 2. Full Digest (detailed)

After the Compact View, include the full digest in this format:

```markdown
┌─────────────────────────────────────────────────────────────┐
│  PORTFOLIO DIGEST · {date}{cached_flag}                     │
└─────────────────────────────────────────────────────────────┘

  Value      ${total_value}
  P&L        {+/-}{total_pnl_pct}%
  YTD        {+/-}{portfolio_ytd}%

─────────────────────────────────────────────────────────────

▲ TOP PERFORMERS

  Symbol      Type     Weight    P&L %    YTD %   9/26   50/200
  ─────────────────────────────────────────────────────────────
  {SYMBOL}    {type}   {wt}%    {pnl}%   {ytd}%    ▲       ▲
  ...

─────────────────────────────────────────────────────────────

◆ STEADY

  Symbol      Type     Weight    P&L %    YTD %   9/26   50/200
  ─────────────────────────────────────────────────────────────
  {SYMBOL}    {type}   {wt}%    {pnl}%   {ytd}%    ▲       ▼
  ...

─────────────────────────────────────────────────────────────

▼ LAGGARDS

  Symbol      Type     Weight    P&L %    YTD %   9/26   50/200
  ─────────────────────────────────────────────────────────────
  {SYMBOL}    {type}   {wt}%    {pnl}%   {ytd}%    ▼       ▼
  ...

─────────────────────────────────────────────────────────────

⚠ WATCH LIST

  Symbol      Type     Weight    P&L %    YTD %   9/26   50/200   Reason
  ────────────────────────────────────────────────────────────────────
  {SYMBOL}    {type}   {wt}%    {pnl}%   {ytd}%    ▼       ▼      {reason}
  ...

─────────────────────────────────────────────────────────────

Signal legend:  ▲ bullish (short MA > long MA)   ▼ bearish (short MA < long MA)

─────────────────────────────────────────────────────────────

📰 MARKET NEWS & PATTERNS

  {SYMBOL}: {brief news headline or event} → {impact on position}
  {SYMBOL}: {brief news headline or event} → {impact on position}
  ...

  Patterns detected:
  • {pattern observation, e.g., "Tech sector rally lifting AAPL, MSFT, GOOGL"}
  • {pattern observation, e.g., "Earnings miss driving TSLA weakness"}

─────────────────────────────────────────────────────────────

💡 RECOMMENDATIONS

  {Action recommendation based on signals and news}
  • {SYMBOL}: {recommendation + reasoning}
  • {SYMBOL}: {recommendation + reasoning}
  ...

  Portfolio-level:
  • {Overall portfolio recommendation if applicable}

─────────────────────────────────────────────────────────────

SUMMARY
• {n} top performers, {n} steady, {n} laggards
• {n} positions on watch list
• Overall bias: {bullish/neutral/bearish}
```

### 3. Markdown File Output

In addition to returning the Compact View + Full Digest in the tool response formatted in markdown, the skill MUST also:

- Write the **Full Digest** (including the header box, tables, news, recommendations, summary) to a markdown file:
  - Path: `digest/portfolio-digest-{YYYY-MM-DD}.md`
  - Location: `digest/` folder in the workspace root (create the folder if it does not exist)
- Overwrite the file if it already exists for that date.

This allows external tools (editor, automation, etc.) to pick up the daily digest from disk.

---

## Guidelines

1. **Minimize derivation** — use MCP values directly when available:
   - P&L %, YTD % → from MCP response
   - Weight % → calculate as `(quantity × price) / total_value × 100`
2. Always return **both**: Compact View (summary) + Full Digest in the response.
3. Also write the Full Digest to `portfolio-digest-{YYYY-MM-DD}.md` as described above.
4. Use `N/A` for unavailable metrics.
5. If cached data is used, add `⚠️ Using cached data` after the date and in `cached_flag`.

## Error Handling

- MCP unavailable → `Digest generation failed: FinAI MCP server unavailable.`
- Refresh fails but summary works → proceed with cached data notice
- Missing MA signals → show `--` in signal column and display a warning with the reason it is missing.
