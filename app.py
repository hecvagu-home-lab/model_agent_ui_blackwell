"""
webui.py — standalone Streamlit front-end for model-agent.

This has no dependency on model-agent's source tree — it's a pure HTTP
client against model-agent-api (see /opt/ai/model-agent/app/api.py).
That's the whole point of splitting it into its own project: this
container can be rebuilt, redeployed, or torn down independently of
model-agent itself.
"""

import os
import requests
import httpx
import streamlit as st

#API_URL = os.environ.get("MODEL_AGENT_API_URL", "http://localhost:8080").rstrip("/")
API_URL = os.getenv("MODEL_AGENT_API_URL", "http://model-agent-api:8500").rstrip("/")


st.set_page_config(page_title="model-agent", layout="wide")


def _human_size(num_bytes) -> str:
    size = float(num_bytes or 0)
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} PiB"


def api_get(path: str, **kwargs):
    resp = httpx.get(f"{API_URL}{path}", timeout=30.0, **kwargs)
    resp.raise_for_status()
    return resp.json()


def api_post(path: str, json: dict = None, timeout: float = 600.0):
    """
    Sends POST request to model-agent-api. 
    Default timeout set to 600 seconds (10 minutes) for large model downloads.
    """
    # Passing None to httpx disables timeout completely if preferred
    return httpx.post(f"{API_URL}{path}", json=json, timeout=timeout)


def api_delete(path: str, params: dict, timeout: float = 180.0):
    return httpx.delete(f"{API_URL}{path}", params=params, timeout=timeout)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("model-agent")
    st.caption("Local runtime provisioner")
    st.divider()
    st.text(f"API\n{API_URL}")

    try:
        health = api_get("/health")
        st.success(f"Connected ({health.get('service')})")
    except httpx.HTTPError as exc:
        st.error(f"Cannot reach API: {exc}")
        st.stop()

    st.divider()
    if st.button("Refresh", use_container_width=True):
        st.rerun()
    st.caption("No authentication on this UI or the API — internal LAN only.")


tab_repos, tab_sync, tab_adopt = st.tabs(["📦 Managed Repos", "⬇️ Sync", "📥 Adopt"])


# ---------------------------------------------------------------------------
# Tab 1 — Repos (list + remove)
# ---------------------------------------------------------------------------

with tab_repos:
    try:
        summaries = api_get("/repos")
    except httpx.HTTPError as exc:
        st.error(f"Failed to list repos: {exc}")
        summaries = []

    if not summaries:
        st.info("No models currently managed by model-agent.")
    else:
        total_bytes = sum(s["total_size_bytes"] for s in summaries)
        st.metric("Total managed", f"{_human_size(total_bytes)} across {len(summaries)} repo(s)")

        rows = [
            {
                "Repo": s["repo_id"],
                "Size": _human_size(s["total_size_bytes"]),
                "Artifacts": s["artifact_count"],
                "Pinned": "yes" if s["any_pinned"] else "-",
            }
            for s in summaries
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Remove a repo")
        st.caption(
            "With cascade on, this deletes local blobs + placement "
            "directories AND the copy in model-manager + MinIO — full teardown."
        )
        repo_ids = [s["repo_id"] for s in summaries]
        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
        with col1:
            selected_repo = st.selectbox("Repo to remove", repo_ids)
        with col2:
            force = st.checkbox("Force")
        with col3:
            cascade = st.checkbox("Cascade", value=True)
        with col4:
            st.write("")
            st.write("")
            if st.button("Remove", type="primary"):
                resp = api_delete(
                    f"/repos/{selected_repo}",
                    params={"force": str(force).lower(), "cascade": str(cascade).lower()},
                )
                if resp.status_code < 400:
                    st.success(resp.json())
                    st.rerun()
                else:
                    st.error(f"{resp.status_code}: {resp.text}")


# ---------------------------------------------------------------------------
# Tab 2 — Sync
# ---------------------------------------------------------------------------

with tab_sync:
    st.subheader("Materialize a repo for a backend")
    st.caption(
        "This call blocks until the sync finishes (or fails) — large "
        "first-time downloads can take a while."
    )

    with st.form("sync_form"):
        sync_repo_id = st.text_input("Repo ID", placeholder="Qwen/Qwen2.5-Coder-7B-Instruct")
        sync_backend = st.selectbox("Backend", ["vllm"])
        sync_placement_root = st.text_input("Placement root", value="/opt/models/vllm")
        sync_unit_id = st.text_input("Unit ID (optional)", value="")
        submitted = st.form_submit_button("Run Sync", type="primary")

    if submitted:
        if not sync_repo_id or not sync_placement_root:
            st.error("Repo ID and placement root are required.")
        else:
            payload = {
                "repo_id": sync_repo_id,
                "backend": sync_backend,
                "placement_root": sync_placement_root,
            }
            if sync_unit_id.strip():
                payload["unit_id"] = sync_unit_id.strip()

            with st.spinner(f"Syncing '{sync_repo_id}'... this may take a while."):
                resp = api_post("/sync", json=payload)

            if resp.status_code < 400:
                st.success(resp.json())
            else:
                st.error(f"{resp.status_code}: {resp.text}")


# ---------------------------------------------------------------------------
# Tab 3 — Adopt
# ---------------------------------------------------------------------------

with tab_adopt:
    st.subheader("Fold an existing local file into the managed store")

    with st.form("adopt_form"):
        adopt_local_path = st.text_input(
            "Local file path (as seen inside model-agent-api's container)",
            placeholder="/opt/models/comfyui/checkpoints/sd_xl_base_1.0.safetensors",
        )
        adopt_repo_id = st.text_input("Repo ID", placeholder="stabilityai/stable-diffusion-xl-base-1.0")
        adopt_submitted = st.form_submit_button("Run Adopt", type="primary")

    if adopt_submitted:
        if not adopt_local_path or not adopt_repo_id:
            st.error("Both fields are required.")
        else:
            payload = {"local_path": adopt_local_path, "repo_id": adopt_repo_id}
            with st.spinner(f"Adopting '{adopt_local_path}'..."):
                resp = api_post("/adopt", json=payload)

            if resp.status_code < 400:
                st.success(resp.json())
            else:
                st.error(f"{resp.status_code}: {resp.text}")
