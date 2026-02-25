BEAT_SCHEDULE = {
    "daily-close": {
        "task": "xerolens_worker.tasks.run_agent.daily_close",
        "schedule": "0 1 * * *",
    },
    "hourly-revenue-reconciliation": {
        "task": "xerolens_worker.tasks.reconcile_revenue.reconcile_revenue",
        "schedule": "0 * * * *",
    },
}
