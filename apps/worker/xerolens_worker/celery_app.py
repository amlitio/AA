from __future__ import annotations


class MockCeleryApp:
    def task(self, fn):
        return fn


app = MockCeleryApp()
