def success(module: str, action: str, result=True, data=None, **kwargs):
    """
    統一 success / failed response。

    result=False 時，status 會自動變成 failed。
    """

    response = {
        "status": "success" if result is not False else "failed",
        "module": module,
        "action": action,
        "result": result,
    }

    if data is not None:
        response["data"] = data

    response.update(kwargs)

    return response


def error(module: str, action: str, error=None, message=None, **kwargs):
    """
    統一 error response。
    error 可傳 Exception。
    message 可傳自訂錯誤訊息。
    """

    if message is None:
        message = str(error) if error is not None else "unknown error"

    response = {
        "status": "error",
        "module": module,
        "action": action,
        "message": message,
    }

    response.update(kwargs)

    return response


def not_found(module: str, action: str, message="not found", data=None, **kwargs):
    response = {
        "status": "not_found",
        "module": module,
        "action": action,
        "message": message,
    }

    if data is not None:
        response["data"] = data

    response.update(kwargs)

    return response