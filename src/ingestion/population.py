from google.cloud import bigquery
import pandas as pd

client = bigquery.Client()


# Queries the github_timeline public dataset for the previous month
query = """
    SELECT
        actor.login                            AS login,
        actor.id                               AS actor_id,
        COUNT(*)                               AS n_events,
        COUNT(DISTINCT repo.name)              AS n_repos,
        COUNT(DISTINCT DATE(created_at))       AS active_days,
        COUNTIF(type = 'PushEvent')            AS n_pushes,
        COUNTIF(type = 'PullRequestEvent')     AS n_prs,
        MIN(created_at)                        AS first_seen,
        MAX(created_at)                        AS last_seen,
        ARRAY_AGG(DISTINCT repo.name LIMIT 20) AS repos
    FROM `githubarchive.day.2026*`
    WHERE _TABLE_SUFFIX BETWEEN '1020' AND '1027'
        AND type IN ('PushEvent', 'PullRequestEvent', 'PullRequestReviewEvent')
        AND NOT ENDS_WITH(actor.login, '[bot]')
    GROUP BY login, actor_id
"""
df = client.query(query).to_dataframe()