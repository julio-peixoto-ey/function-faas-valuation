import os
import json
import datetime
from typing import Any, Dict

def save_json(data: Any, output_dir: str = "extraction_results") -> str:
    """
    Grava o dicionário/objeto de extração em um arquivo .json e
    devolve o caminho completo salvo.
    Aceita dataclasses com método to_dict(), Pydantic models ou dict.
    """
    if hasattr(data, "to_dict"):
        data = data.to_dict()  # type: ignore[attr-defined]

    if not isinstance(data, (dict, list)):
        raise TypeError("Dados de entrada devem ser dict, list ou possuir .to_dict()")

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
    file_path = os.path.join(output_dir, f"extraction_{timestamp}.json")

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return file_path
