def rollup_minute_summary(conn, since_ts_ms: int | None = None):
    cur = conn.cursor()

    base_query = """
    INSERT INTO events_summary (
        granularity,
        device_id,
        event_type,
        event_source,
        bucket_start_ms,
        event_count
    )
    SELECT
        '1m' AS granularity,
        device_id,
        event_type,
        event_source,
        (timestamp_ms / 60000) * 60000 AS bucket_start_ms,
        COUNT(*) AS event_count
    FROM events
    WHERE timestamp_ms <= (strftime('%s','now') * 1000) - 10000
    """

    params = []

    if since_ts_ms is not None:
        base_query += " AND timestamp_ms > ?"
        params.append(since_ts_ms)

    base_query += """
    GROUP BY
        device_id,
        event_type,
        event_source,
        bucket_start_ms
    ON CONFLICT (
        granularity,
        device_id,
        event_type,
        event_source,
        bucket_start_ms
    )
    DO UPDATE SET
        event_count = event_count + excluded.event_count;
    """

    cur.execute(base_query, params)
    conn.commit()
