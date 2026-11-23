# Progress log — FlyRank ML track

A short record of each assignment: what I built, what the numbers were, and what I decided. My own words come from my notebooks; the receipts are copied from the notebook outputs.

## Decisions log

- **Lane (decided 24 Nov 2026):** freestyle direction Growth / Recovery / Momentum Prediction, in a staged form with ideas from a seasonality-aware refresh queue built in. I started with Lane 2 (Refresh / Content Opportunity Scoring) as a provisional choice, and I also looked at AI Referral Opportunity. I chose this direction because it turns my own observations (the wait before judging an article, and seasonality) into design features, it gives a forecast from prior and future windows instead of an association check, and it can grow as months of data are added.
- **Frozen baseline (ML-07, Lane 2 on the starter data):** the STALE_VISIBLE refresh rule. Precision@10 / @20 / @50 = 0.600 / 0.450 / 0.440 against a base rate of 0.542. It stays as my submitted record. I still need to build a new baseline for the freestyle lane, and to check how the track treats the lane switch for Week 5, which asks for the same pages as the Week 4 baseline.
- **Validation:** hold out whole clients, not random pages. The ML-01 held-out comparison used a random page split, so a client-level test is still to do.
- **Data conventions:** rate columns are percentages (ctr 0.76 means 0.76%); `avg_position = 0` means no data; `trend_direction` and `trend_pct` are never features (they make the label).
- **Working note on timing:** articles usually get a few months after their initial post or refresh before their trend is judged, because of Google search algorithm updates. I plan to keep this waiting period configurable by action.

---

## ML-01 — Run the Starter Notebooks

- **Notebooks:** `notebooks/01_first_look_and_discovery.ipynb`, `notebooks/02_your_first_readable_model.ipynb` (committed 29 Oct 2026).
- **Receipts, notebook 01:** hand-written rule Precision@50 0.240 (about 12 of the top 50) vs random forest 0.740 (about 37 of the top 50), roughly 3.1x. Top-50 declining rate on the full data 0.340.
- **My discovery (notebook 01):** This analysis found essentially no linear relationship (correlation = 0.001) between keyword search volume and page impressions. Filtering out zero-impression pages made no difference, as all 30,000 pages in the sample had at least one impression. In this sample, search volume does not appear to track the impressions a page gets. These findings are observed and directional, based on a single anonymized dataset, and should not be generalized to Google’s algorithms or broader search behavior.
- **Receipts, notebook 02:** hand rule Precision@20 0.900 and @50 0.680; depth-2 tree 0.550 and 0.600; leaky tree 1.000. Depth-4 tree 0.680; swapped features 0.720 (first split `avg_position`). On a held-out split, the rule and both depth-4 trees all scored 0.660; the rule flagged only 17 of 30,000 pages (4 in the test set).
- **My experiment (notebook 02):** This analysis compared a hand-written rule with decision trees for finding declining web pages. The hand rule, while precise for the 17 of 30,000 pages it flagged, left most pages unranked and guessed for the rest. Decision trees ranked every page, and a deeper tree improved Precision@50 in-sample (from 0.600 to 0.680).

After swapping impressions_90d for engagement_rate, the first split became avg_position. On held-out pages, however, the rule and both depth-4 trees scored 0.660, so the swapped features’ in-sample edge (0.720) disappeared. With only 50 pages and a single split, small differences may simply be noise. The key lesson is to evaluate results honestly and state clearly what my experiment actually demonstrates.

## Week 1 — Research question (`work/notebooks/w01_research_question.ipynb`)

**My lane and why**

I am choosing Lane 2, Refresh / Content Opportunity Scoring. It builds on the hand rule and model comparison I already ran in notebooks 01 and 02, and its output (a ranked review queue) is something an editor can act on directly. This is a provisional choice. By Week 4 I will compare it with the AI Referral freestyle direction, which connects to my interest in generative-engine optimization.

**The question**

Unit of analysis: one page. Output: a ranked list of pages with a short reason for each.
Action: the editor reviews the top pages and decides whether to refresh, leave or monitor each one.
Cost of a wrong call: if the ranking is wrong, editor hours go to pages that did not need work, or a page that keeps losing visibility is missed.
Why data or ML can help, and why it is not just “train a model”: no person can read 30,000 pages, and a simple rule flags only 17 of them. A ranking can cover every page, but it must be judged against that rule and checked on pages it has not seen.

**The data**

In the starter dataset, there are 30,000 pages across 32 clients. 54.2% of pages are marked as declining, which is more than an editor can review. A simple stale-and-visible rule (180+ days since update and 500+ impressions) flags only 17 pages, so most pages go unranked. In notebook 01, a learned model scored 0.740 on Precision@50 against 0.240 for a hand rule, in one run.

**Careful words**

I can say: which signals are observed more often among declining pages in this sample, and that a ranking can help editors prioritize. These are directional, decision-support findings.
I cannot say: that a refresh will cause recovery, how Google’s algorithm works, or anything about AI citations or AI visibility. The “declining” label is a proxy based on recent data, not a confirmed future outcome. I will keep client data out of anything public.

## ML-03 — Frame Your Lane as an ML Task (`work/notebooks/w02_ml_task_framing.ipynb`, committed 10 Nov 2026)

**Task type**

The main goal is to produce a ranked list of pages so editors know which ones to review first. Even though the model is trained using a yes/no label for whether a page is declining, the output is an ordered list that prioritizes editorial attention. For every page that ranks near the top, the editor chooses whether to update, add new content, safeguard, remove unnecessary material, or simply keep an eye on it.

**Target or proxy**

The model uses the is_declining_label, which is created from trend_direction == “down”—this is based on the percentage change in performance in the current window. Because the label comes from the outcome variable (trend_direction and trend_pct), these cannot be used as features to avoid leakage. The label is only a proxy for actual decline and may not be perfect.

**Success metric**

“Good” means the model achieves a higher Precision@50 (the fraction of the top 50 pages labelled as declining) than both the base rate (54.2% of all pages) and the stale (180+ days) and visible (500+ impressions) rule, measured on clients the model has not seen before (not just new pages from a client already in the dataset).

**Unit of analysis**

Each of the initial rows represents a single page, and pages from the same client can appear multiple times. This is why I plan to validate performance using entire clients. The target variable, is_declining_label, is set to 1 when the trend_direction is “down.”

**Why ML**

It’s challenging to combine factors like age, impressions, position, and CTR in a single rule, since setting all the cutoffs and weights by hand is both subjective and limiting. Using a model allows these signals to be integrated, providing a ranking for every page without manual thresholds. However, there’s not yet clear evidence that the model outperforms the rule. In notebook 02, both the rule and the tree-based model achieved a Precision@50 of 0.660 using a random split at the page level. That result is based on just one split, and in the test set, the rule only gave non-zero scores to four pages—meaning most of its top 50 were ties at zero. I still need to evaluate performance using a client-level split for a more robust comparison.

## ML-04 — Search Intelligence Data Contract (`work/notebooks/w03_data_contract.ipynb`, committed 15 Nov 2026)

**Data source, unit, window, exclusion, and validation**

- Each row in the daily table `fact_content_daily_performance` (March 2026 partition) represents one page for one client on one day. Query 1 confirmed there are no duplicate (client, page, date) keys.
- For modeling, my unit is a single page, aggregating its daily rows (summing impressions and clicks, averaging position).
- Analysis window: March 1–31, 2026. The "decision moment" is March 15: features use data from March 1–15, and outcomes are measured from March 16–31. June 2026 is kept aside as a sealed test month.
- The proxy for ranking is whether a page's daily impressions drop in the outcome window (see label below).
- Exclusion: I exclude any page with fewer than 100 impressions in March 1–15, since there isn't enough data for a trend.
- Since pages from the same client aren't independent, model validation holds out whole clients, not just random pages.

**Label, features, context, and exclusions**

**Label:** `declined` = 1 if a page's average daily impressions in March 16–31 are below 80% of its average in March 1–15. This is only a proxy for real decline, and using two short windows adds noise. No column from the outcome window (March 16–31) is allowed as a feature.

**Features (all from March 1–15, knowable at the decision moment):**

- `impr_early`: sums impressions only up to March 15
- `clicks_early`: sums clicks only up to March 15
- `ctr_early`: divides those two early totals
- `pos_early`: averages daily position on days with a recorded position up to March 15
- `active_days_early`: counts days with impressions up to March 15

**Context columns:**

- `client_hash_id`
- `content_hash_id`
- `report_date`

**Excluded columns and rationale:**

- Late-window impressions (these form the outcome and would leak the label)
- `sessions_ai` and `ai_*` columns (sparse across the warehouse per the data guide; I did not check March separately, and they belong to a different modeling lane)

**Data completeness, analytics, and NULL handling**

- Of the 9.84 million daily rows (55 clients, 331,437 pages), about 37% have search data and about 4% have analytics data. AI-referral columns are session counts that probably depend on analytics, but this still needs confirmation.
- The analytics flag can be TRUE, FALSE, or NULL. Filtering with `IS TRUE` counts only explicit TRUE values, so NULL rows are intentionally treated as "not available."

**Features: knowable at the decision moment because:**

- `impr_early`: sums impressions only up to March 15
- `clicks_early`: sums clicks only up to March 15
- `ctr_early`: divides those two early totals
- `pos_early`: averages daily position on days with a recorded position up to March 15
- `active_days_early`: counts days with impressions up to March 15

**Results note**

On one slice (March 2026, 55 clients), the honest quick score on held-out clients was AUC = 0.538 and Precision@50 = 0.100 in this saved run, against a base rate of 0.181, so not better than chance on this split. Precision@50 is fragile here: the small tree gives only a few distinct scores, so the top 50 is mostly ties, and an earlier run of the same code gave 0.160. Adding a column from the outcome window raised Precision@50 to 1.000 (AUC 0.831), showing how leakage inflates a score. I removed it and kept the honest number. This is a directional result from one split; I haven't tested other time windows or longer histories yet.

**Named limitation**

Only about 37% of March's daily rows include search data and about 4% have analytics data, so this analysis covers a minority of rows.

- **Receipts (March 2026 slice):** 9,841,378 daily rows, 55 clients, 331,437 pages, 31 days (2026-03-01 to 2026-03-31); no duplicate keys. Search data on 3,611,061 rows, analytics data on 413,966, both on 364,347; 3,018,741 rows with a NULL analytics flag. 176,738 pages with search data, 77,540 kept after the volume floor, 32.8% declined. Honest quick score on held-out clients: Precision@50 0.100, AUC 0.538, base rate 0.181; with the deliberate leak: Precision@50 1.000, AUC 0.831.

## ML-07 — Baseline Action Score and Top-20 Review (`work/notebooks/w04_baseline_score.ipynb`, committed 19 Nov 2026)

A page should be considered for a refresh review if it gets a solid number of views, isn't brand new, and hasn't been updated for quite a while. Pages that are both highly visible and more out-of-date get higher priority for review.

**My rule**

The rule uses only three inputs: impressions in the last 90 days, days since the last update, and total content age. For each page, the rule calculates a score as:

score = impressions_90d × staleness_weight

where staleness_weight = days_since_last_update (capped at 180) / 180.

- **Eligibility:** at least 500 impressions in the last 90 days, at least 90 days since the last update, and content at least 90 days old. I chose these thresholds to focus on established, visible pages that might benefit most from a refresh, and to avoid noise from recent or little-seen content.
- **Reason code:** STALE_VISIBLE
- **Action:** refresh review

The label indicating whether a page is declining is used only in signal checks and evaluation, not as an input to the ranking.

**Signal check 1: staleness** (the signal behind FlyRank's refresh flags)

- **Belief:** Pages left untouched for longer are more likely to see declines.
- **What the table shows:** Declining rate is 51.1% (0–30 days since update, n=20,480), 58.9% (31–90 days, n=175), 61.1% (91–180 days, n=9,171), and 47.1% (181+ days, n=174).
- **Verdict: MIXED.** The declining rate rises with staleness up to 180 days, but then drops in the oldest group, which has few pages and may not be reliable.
- **Extra note:** Days since update takes only a handful of values: 20 and 104 days account for most pages (68%). The 104-day group covers 8,773 pages, making up almost all of the 91–180 group. This pattern is consistent with a site-wide update or sync across many pages.

**Signal check 2: volume**

- **Belief:** More visible pages are at greater risk of decline, so higher volume should correlate with a higher declining rate.
- **What the table shows:** Declining rate is 38.9% (1–100 impressions, n=8,006), 60.4% (101–500, n=5,279), 62.1% (501–3,000, n=8,432), and 57.0% (3,000+, n=8,283). Among the largest pages: 60.6% (3k–10k), 55.0% (10k–30k), 46.2% (30k+).
- **Verdict: MIXED.** The declining rate increases with volume at the low end but falls off among the biggest pages, suggesting that extremely high volume may be linked to greater stability.

The rule ranks 6,575 of 30,000 pages (21.9%) and writes the ranked queue to work/outputs/baseline_action_score.csv.

**Weak picks**

- **Weak pick 1:** The top 20 are all extremely large pages (176,000+ impressions), yet pages in the 30k+ group have the lowest declining rate (46.2%), below the base rate of 54.2%.
- **Weak pick 2:** Rank 10 has over 200,000 impressions but a CTR of 0.00, which may be a tracking error.
- **What it means for the baseline:** Precision@10 is 0.600, @20 is 0.450 and @50 is 0.440, against a base rate of 0.542. Overall that is no better than random picking, and below it at 20 and 50. This is the baseline the Week 5 model has to beat.

**Leakage check**

- **Inputs used:** impressions_90d, days_since_last_update, content_age_days. Not used: trend_direction, trend_pct, any product flags.
- **Caveat:** The starter data is a single snapshot, and impressions_90d covers the same 90 days as the label (last 30 vs previous 30), so this is an association check, not a forecast.

- **Receipts:** base rate 54.2%. Staleness buckets declining rate: 51.1% (0–30 days, n=20,480), 58.9% (31–90, n=175), 61.1% (91–180, n=9,171), 47.1% (181+, n=174). Volume buckets: 38.9% (1–100, n=8,006), 60.4% (101–500, n=5,279), 62.1% (501–3,000, n=8,432), 57.0% (3,000+, n=8,283). Queue: 6,575 of 30,000 pages. Precision@10 / @20 / @50 = 0.600 / 0.450 / 0.440 vs base rate 0.542. The full top-20 review is in the notebook. Metrics file: `work/outputs/baseline_metrics.json`.

## Freestyle plan (draft, to confirm and put in my own words)

**Direction:** Growth / Recovery / Momentum Prediction, staged. Decline is the first target; growth and recovery are later extensions. AI-referred sessions could be a later extension if the data allows.

**Question:** can features from a page's prior window show which pages are likely to lose visibility in the next window, so an editor knows where to look first?

**Design**
- **Data and windows:** the warehouse daily table, with prior-window features and a future-window outcome built per client, because history depth differs. June 2026 stays sealed as the test month. Months come from a setting, so new months can be added without rewriting code.
- **Label:** decline = a page's future daily impressions below 80% of its prior daily average, for pages with enough prior impressions and search data confirmed with `IS TRUE`. Features come only from the prior window, and nothing built from the outcome window is allowed.
- **Peer comparison (from the seasonality idea):** compare a page's change with its client's typical page and, where history exists, with the same period a year earlier. A decline that matches a client-wide dip is treated as possibly seasonal (monitor, or plan a refresh before the season) and not as an urgent review.
- **Maturity window:** an article gets time after a post or refresh before its trend is judged, because of Google algorithm updates. This is a setting by action type (the Week 4 lecture gave 2–4 weeks for a content fix, 4–6 for a snippet fix, 4–8 for a page-one push, and 6–12 for a refresh).
- **Baseline first:** a transparent rule (impressions fell over the last 14 days compared with the 14 before), with a reason code and an action, scored by Precision@K against the base rate. A model has to beat it, validated on whole held-out clients.
- **Output:** a ranked future-risk list with reason codes. An editor reviews the top pages.
- **Careful words:** observed and directional decision support. No claim that an action causes recovery, and seasonality is treated as a risk, not a proven cause.

**First results (scratch notebook, not submitted):** see the exploration section below.

**Known challenges:** weak signal (my ML-04 quick score was near the base rate), only about 17 months of history, seasonality and algorithm updates, and the size of the warehouse (iterate on one month and cache).

**Next steps:** run more month pairs and see whether the client-wide-dip share is stable; add a spike check; build the prior-window baseline and score it on held-out clients; confirm with the track how the lane change affects the Week 5 data.

---

## Exploration, not submitted — freestyle windows

A scratch notebook for the Growth / Recovery / Momentum direction I chose (not committed): features from a prior window, outcome in a future window, declined = future daily impressions below 80% of prior, compared with the page's client. February to March 2026: 80,322 eligible pages across 37 clients, 26.7% declined. Of the 21,482 declined pages, 17,837 fell while their client's typical page held steady, 3,609 fell with a client-wide dip, and 36 had an unknown client. A check on the same transition a year earlier found almost no sign of a repeated dip, with last-year information for only 20,511 pages.

**What this suggests**

On the February to March 2026 slice, 17,837 of the 21,482 declined pages fell while their client's typical page did not fall by 10% or more, and 3,609 declined along with a client-wide dip (36 had no reliable client comparison). This suggests that most page-level declines may be specific to the page, possibly because of content issues or competition, or because they are only temporary fluctuations such as a fading spike, while a smaller share could be due to broader, possibly seasonal or market factors affecting the whole client. If I were the editor, I would review the pages that declined while their client held steady first, since these are more likely to be page-specific and a targeted review seems the better use of limited time. The pages that declined along with their client's dip I would monitor or revisit later, as their trend may reflect factors a page edit cannot change. I can't say whether these results hold for other time periods. The same-period check from a year earlier covered only about a quarter of the pages (20,511), so it does not settle whether the client-wide dips are seasonal.

**What I will test next**

Next, I will compare additional month pairs, such as March to April and April to May, to see whether the client-wide dips stay near 17% of declined pages or change over time. Then I will check for spikes by looking at whether a page's peak day in the earlier window explains some of the sharpest drops, meaning the decline may just be a fading spike and not a true loss. Finally, I will evaluate a simple recent-trend rule as a baseline, scoring it by Precision@K against the base rate on held-out clients.
