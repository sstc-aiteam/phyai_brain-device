"""
Flask API Route 共用工具。
"""

from flask import jsonify, request


def json_response(response):
    """
    將 Service response 轉換成 Flask JSON response。
    """
    if not isinstance(response, dict):
        response = {
            "status": "error",
            "module": "system",
            "action": "json_response",
            "result": False,
            "data": None,
            "error": (
                "Service response 必須是 dict，"
                f"實際為 {type(response).__name__}"
            ),
            "error_type": "TypeError",
        }

        return jsonify(response), 500

    if response.get("status") != "error":
        return jsonify(response), 200

    error_type = response.get("error_type")

    if error_type in {
        "ValueError",
        "TypeError",
        "KeyError",
    }:
        status_code = 400

    elif error_type in {
        "TimeoutError",
        "ConnectionError",
        "ConnectionRefusedError",
    }:
        status_code = 503

    else:
        status_code = 500

    return jsonify(response), status_code


def get_json_body():
    """
    取得 JSON object。

    沒有 request body 時回傳空 dict。
    """
    data = request.get_json(silent=True)

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise ValueError(
            "request body 必須是 JSON object"
        )

    return data


def request_error(
    module,
    action,
    message,
    error_type="ValueError",
):
    """
    建立 Route 層輸入格式錯誤 response。
    """
    response = {
        "status": "error",
        "module": module,
        "action": action,
        "result": False,
        "data": None,
        "error": str(message),
        "error_type": error_type,
    }

    return json_response(response)