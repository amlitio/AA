"""Database tool contract used by workflows."""


class DbTool:
    def upsert_case(self, tenant_id, transaction_id, risk, confidence, findings, evidence):
        raise NotImplementedError

    def save_actions(self, case_id, actions):
        raise NotImplementedError

    def mark_action_executed(self, case_id, action):
        raise NotImplementedError
