"""Abstract-style Xero tool contract used by workflows."""


class XeroTool:
    def get_invoice(self, tenant_id: str, invoice_id: str):
        raise NotImplementedError

    def execute_action(self, tenant_id: str, invoice_id: str, action):
        raise NotImplementedError
