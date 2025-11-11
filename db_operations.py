import lancedb
import uuid
from datetime import datetime
from typing import List
from lancedb.pydantic import LanceModel, Vector

class Document(LanceModel):
    id: str
    content: str
    embedding: Vector(1536)
    stage: str
    category: str
    purpose: str
    model_used: str
    workflow_id: str
    iteration: int
    timestamp: str
    contains_pxos_primitives: bool
    keywords: List[str]

def get_db_connection(db_path="lancedb_data"):
    """Establishes a connection to the LanceDB database."""
    return lancedb.connect(db_path)

def save_to_db(db, data):
    """Saves data to the 'documents' table in LanceDB."""
    if "documents" not in db.table_names():
        db.create_table("documents", schema=Document)

    table = db.open_table("documents")
    # Pydantic model instances are added directly
    table.add([Document(**data)])
