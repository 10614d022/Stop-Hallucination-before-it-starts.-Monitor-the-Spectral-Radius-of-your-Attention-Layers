# Spectral Instability Leaderboard

This leaderboard aggregates external replication results for the spectral-instability hypothesis.

## Submission fields
Use `leaderboard_template.csv` as the canonical schema.

## Required metrics
- collapse_rate
- mean_event_time
- mean_lead_time
- mean_quality_score

## Ranking suggestions
Primary:
1. lower collapse_rate
2. higher mean_event_time
3. acceptable quality_score

## Scientific rule
A strong result must improve stability **without** severe quality collapse.
