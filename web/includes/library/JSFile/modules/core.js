/**
 * core.js
 * ES Module: common API helper and output/status rendering.
 */

export function resolveApiUrl(path)
{
    if (typeof path !== "string" || !path.trim())
    {
        throw new TypeError("path 必須是非空字串");
    }

    const normalizedPath = path.trim();

    if (/^(https?:)?\/\//.test(normalizedPath))
    {
        return normalizedPath;
    }

    const configuredBase = (
        typeof window !== "undefined"
        && typeof window.ROBOT_API_BASE === "string"
    )
        ? window.ROBOT_API_BASE.trim()
        : "";

    if (!configuredBase)
    {
        return normalizedPath;
    }

    if (configuredBase.includes("{path}"))
    {
        return configuredBase.replace(
            "{path}",
            encodeURIComponent(normalizedPath),
        );
    }

    return `${configuredBase}${normalizedPath}`;
}

export async function postJson(url, data = {}, options = {})
{
    const resolvedUrl = resolveApiUrl(url);

    const response = await fetch(resolvedUrl, { method: "POST", headers: { "Content-Type": "application/json", ...(options.headers ?? {}) }, body: JSON.stringify(data ?? {}), cache: options.cache ?? "no-store", signal: options.signal});

    let result;

    try
    {
        result = await response.json();
    }
    catch
    {
        throw new Error(`伺服器回傳的內容不是 JSON，HTTP ${response.status}`);
    }

    if (!response.ok)
    {
        throw new Error( result?.error || result?.message || `HTTP ${response.status}`);
    }

    return result;
}

export async function getJson(url, options = {})
{
    const resolvedUrl = resolveApiUrl(url);

    const response = await fetch(resolvedUrl, { method: "GET", headers: { ...(options.headers ?? {}) }, cache: options.cache ?? "no-store", signal: options.signal });

    let result;

    try
    {
        result = await response.json();
    }
    catch
    {
        throw new Error(`伺服器回傳的內容不是 JSON，HTTP ${response.status}`);
    }

    if (!response.ok)
    {
        throw new Error( result?.error || result?.message || `HTTP ${response.status}`);
    }

    return result;
}

export function sleep(ms)
{
    return new Promise((resolve) => setTimeout(resolve, ms));
}

export function getElement(target)
{
    if (target instanceof HTMLElement)
    {
        return target;
    }

    if (typeof target === "string" && target.trim())
    {
        const element = document.getElementById(target.trim());

        if (element)
        {
            return element;
        }
    }

    throw new Error("找不到指定的 DOM 元素");
}

export function scrollToBottom(div)
{
    const container = getElement(div);

    container.scrollTop = container.scrollHeight;
}

export function updateMessage(div, role, content, options = {})
{
    const container = getElement(div);
    const row = document.createElement("div");
    row.className = `message-row ${role}`;

    const message = document.createElement("div");
    message.className = "message";

    if (options.error === true)
    {
        message.classList.add("error");
    }

    message.textContent = String(content ?? "");

    row.appendChild(message);
    container.appendChild(row);

    scrollToBottom(container);

    return message;
}


export async function updateStreamMessage(div, role, content, options = {})
{
    const container = getElement(div);
    const row = document.createElement("div");
    row.className = `message-row ${role}`;

    const message = document.createElement("div");
    message.className = "message";

    if (options.error === true)
    {
        message.classList.add("error");
    }

    row.appendChild(message);
    container.appendChild(row);

    const typingSpeed = options.speed ?? 25;
    const normalizedContent = String(content ?? "");

    for (const character of normalizedContent)
    {
        message.textContent += character;

        scrollToBottom(container);

        await sleep(typingSpeed);
    }

    return message;
}

export function updateStatus(div, content, options = {})
{
    const container = getElement(div);

    const statusElement = container.querySelector("#status");
    const errorElement = container.querySelector("#error");

    if (!statusElement)
    {
        throw new Error(`指定容器內找不到 #status 元素`);
    }

    if (!errorElement)
    {
        throw new Error(`指定容器內找不到 #error 元素`);
    }

    const normalizedContent = String(content ?? "");

    if (options.error === true)
    {
        errorElement.textContent = normalizedContent;
        errorElement.classList.add("error");
    }
    else
    {
        statusElement.textContent = normalizedContent;
        errorElement.textContent = "";
        errorElement.classList.remove("error");
    }
}


export function showReturn(div, action, response) 
{
    const container = getElement(div);
    container.textContent = JSON.stringify({ action, response }, null, 2);
}


export function toggleHidden(divid)
{
    var div = document.getElementById(divid);
    if (div.classList.contains('hidden'))
    {
        div.classList.remove('hidden');
    }
    else
    {
        div.classList.add('hidden');
    }
}


window.resolveRobotApiUrl = resolveApiUrl;
