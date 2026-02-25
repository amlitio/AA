BEAT_SCHEDULE = {
    "daily-close": {
        "task": "xerolens_worker.tasks.run_agent.daily_close",
        "schedule": "0 1 * * *",
    }
}
