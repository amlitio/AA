"""Document lookup and extraction tool contract."""


class DocsTool:
    def find_candidates(self, **kwargs):
        raise NotImplementedError

    def extract(self, tenant_id: str, doc_id: str):
        raise NotImplementedError
