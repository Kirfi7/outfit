import httpx


def make_httpx_client(timeout: httpx.Timeout, proxy_url: str = "") -> httpx.AsyncClient:
    if not proxy_url:
        return httpx.AsyncClient(timeout=timeout, trust_env=False)
    try:
        return httpx.AsyncClient(timeout=timeout, trust_env=False, proxy=proxy_url)
    except TypeError:
        return httpx.AsyncClient(timeout=timeout, trust_env=False, proxies={"http": proxy_url, "https": proxy_url})
