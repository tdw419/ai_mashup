from typing import List, Dict, Any
from db_operations import get_db_connection

class DatabaseExplorer:
    def __init__(self, db_path="lancedb_data"):
        self.db = get_db_connection(db_path)

    def list_recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        if "documents" not in self.db.table_names():
            return []
        table = self.db.open_table("documents")
        return table.to_pandas().to_dict("records")

    def filter_by_stage(self, stage: str, limit: int = 50):
        if "documents" not in self.db.table_names():
            return []
        table = self.db.open_table("documents")
        return table.search(where=f"stage = '{stage}'").limit(limit).to_pandas().to_dict("records")

    def filter_by_workflow(self, workflow_id: str, limit: int = 100):
        if "documents" not in self.db.table_names():
            return []
        table = self.db.open_table("documents")
        return table.search(where=f"workflow_id = '{workflow_id}'").limit(limit).to_pandas().to_dict("records")

    def search(self, text: str, k: int = 10):
        if "documents" not in self.db.table_names():
            return []
        table = self.db.open_table("documents")
        return table.search(text).limit(k).to_pandas().to_dict("records")

    def get_doc(self, doc_id: str):
        if "documents" not in self.db.table_names():
            return None
        table = self.db.open_table("documents")
        result = table.search(where=f"id = '{doc_id}'").limit(1).to_pandas()
        return result.to_dict("records")[0] if not result.empty else None
