"""
webui.py — standalone Streamlit front-end for model-agent.

This has no dependency on model-agent's source tree — it's a pure HTTP
client against model-agent-api (see /opt/ai/model-agent/app/api.py).
That's the whole point of splitting it into its own project: this
container can be rebuilt, redeployed, or torn down independently of
model-agent itself.
"""

import os
import time
import requests
import httpx
import streamlit as st
from typing import List, Dict, Optional
import json
import pandas as pd

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
    return httpx.post(f"{API_URL}{path}", json=json, timeout=timeout)


def api_delete(path: str, params: dict, timeout: float = 180.0):
    return httpx.delete(f"{API_URL}{path}", params=params, timeout=timeout)


def get_subfolders(backend: str, placement_root: str, fallback: List[str]) -> List[str]:
    """
    Fetches the real subfolder list from the API, cached in session_state
    per (backend, placement_root) pair so it's not re-fetched on every rerun.
    Falls back to a hardcoded list if the API call fails for any reason.
    """
    cache_key = f"subfolders_{backend}_{placement_root}"
    if cache_key not in st.session_state:
        try:
            data = api_get("/backends/{}/subfolders".format(backend), params={"placement_root": placement_root})
            st.session_state[cache_key] = data.get("subfolders") or fallback
        except Exception:
            st.session_state[cache_key] = fallback
    return st.session_state[cache_key]


def _build_artifact_table(discovered_artifacts, backend, comfy_subfolder, dest_options_by_backend):
    """
    Turns the raw discovered_artifacts list into a DataFrame shaped for
    st.data_editor: one row per artifact, id kept as a hidden column for
    lookups, name/size/type shown read-only, select + destination editable.
    """
    rows = []
    for artifact in discovered_artifacts:
        artifact_id = artifact.get("id")
        artifact_name = artifact.get("name", "unknown")
        name_lower = artifact_name.lower()

        suggested_folder = "checkpoints"
        if "taeh3" in name_lower or "taesd" in name_lower or "vae_approx" in name_lower:
            suggested_folder = "vae_approx"
        elif "vae" in name_lower:
            suggested_folder = "vae"
        elif "encoder" in name_lower or "clip" in name_lower or "text" in name_lower:
            suggested_folder = "text_encoders"
        elif "lora" in name_lower or "turbo" in name_lower:
            suggested_folder = "loras"
        elif "diffusion" in name_lower or "unet" in name_lower:
            suggested_folder = "diffusion_models"

        if backend == "comfyui":
            default_dest = suggested_folder if suggested_folder != "checkpoints" else comfy_subfolder
        else:
            default_dest = suggested_folder

        rows.append({
            "id": artifact_id,
            "Select": False,
            "Artifact Name": artifact_name,
            "Type": artifact.get("artifact_type", artifact.get("type", "unknown")),
            "Size": _human_size(artifact.get("size_bytes", 0)),
            "Destination Folder": default_dest,
        })

    return pd.DataFrame(rows)


def _status_label(repo_id: str, artifact_id: str) -> str:
    key = f"{repo_id}_{artifact_id}"
    status = st.session_state.download_status.get(key)
    return {
        "downloading": "⬇️ Downloading",
        "completed": "✅ Done",
        "error": "❌ Error",
    }.get(status, "⏳ Pending")


def _check_existing_locations(repo_id, placement_root, backend, table):
    """
    Asks model-agent-api where each artifact's file already lives on
    disk, if anywhere. Returns {artifact_id: subfolder_or_None}.
    subfolder is "" if the file sits directly in placement_root.
    """
    items = [
        {
            "artifact_id": row["id"],
            "filename": row["Artifact Name"],
            "subfolder": row["Destination Folder"],
        }
        for _, row in table.iterrows()
        if row["id"]
    ]
    if not items:
        return {}
    try:
        resp = httpx.post(
            f"{API_URL}/files/check-existing",
            json={"placement_root": placement_root, "items": items},
            timeout=15.0,
        )
        resp.raise_for_status()
        return resp.json().get("results", {})
    except Exception:
        # Best-effort hint only — if the check fails, leave destinations
        # and On Disk status as originally guessed rather than blocking
        # the table from rendering.
        return {}


def _resolve_destination(row, found_locations):
    """
    If the file was found on disk somewhere, use that real location as
    the Destination Folder instead of the naming-heuristic guess. "" means
    found directly in placement_root; None means not found at all.
    """
    found = found_locations.get(row["id"])
    if found is not None:
        return found if found != "" else "(root)"
    return row["Destination Folder"]



def _on_backend_change():
    new_backend = st.session_state["sync_backend"]
    st.session_state["sync_root"] = "/opt/models/comfyui" if new_backend == "comfyui" else "/opt/models/vllm"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _sync_artifacts(repo_id, artifacts, backend, placement_root, subfolders=None, status_placeholder=None, base_table=None):
    """
    artifacts: dict mapping artifact_id -> artifact_name. Names may repeat
    within a repo (e.g. multiple units each having a config.json); id is
    what's guaranteed unique, so all lookups here go through id.
    """
    subfolders = subfolders or {}

    with st.spinner(f"Syncing {len(artifacts)} artifact(s) from {repo_id}..."):
        for artifact_id, name in artifacts.items():
            key = f"{repo_id}_{artifact_id}"
            st.session_state.download_status[key] = "downloading"
            if status_placeholder is not None and base_table is not None:
                live_table = base_table.copy()
                live_table["Status"] = live_table["id"].apply(lambda aid: _status_label(repo_id, aid))
                status_placeholder.dataframe(live_table.drop(columns=["id"]), hide_index=True, width="stretch")

            target_subfolder = subfolders.get(artifact_id, "")
            final_placement = os.path.join(placement_root, target_subfolder).rstrip("/")
            payload = {
                "repo_id": repo_id,
                "artifacts": [artifact_id],
                "backend": backend,
                "placement_root": final_placement,
            }

            try:
                resp = api_post(f"/discover/{repo_id}/sync-batch", json=payload)
                if resp.status_code < 400:
                    st.session_state.download_status[key] = "completed"
                else:
                    st.session_state.download_status[key] = "error"
                    st.error(f"Error syncing {name}: {resp.text}")
            except Exception as e:
                st.session_state.download_status[key] = "error"
                st.error(f"Exception syncing {name}: {e}")

            if status_placeholder is not None and base_table is not None:
                live_table = base_table.copy()
                live_table["Status"] = live_table["id"].apply(lambda aid: _status_label(repo_id, aid))
                status_placeholder.dataframe(live_table.drop(columns=["id"]), hide_index=True, width="stretch")

        st.success("✅ Sync batch complete!")
        st.rerun()


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
        st.success(f"Connected ({health.get('service', 'ok')})")
    except Exception as exc:
        st.error(f"Cannot reach API: {exc}")
        st.stop()

    st.divider()
    if st.button("Refresh", width="stretch"):
        st.rerun()
    st.caption("No authentication on this UI or the API — internal LAN only.")


tab_repos, tab_sync, tab_adopt = st.tabs(["📦 Managed Repos", "⬇️ Sync", "📥 Adopt"])


# ---------------------------------------------------------------------------
# Tab 1 — Repos (list + remove)
# ---------------------------------------------------------------------------

with tab_repos:
    try:
        summaries = api_get("/repos")
    except Exception as exc:
        st.error(f"Failed to list repos: {exc}")
        summaries = []

    if not summaries:
        st.info("No models currently managed by model-agent.")
    else:
        total_bytes = sum(s.get("total_size_bytes", 0) for s in summaries)
        st.metric("Total managed", f"{_human_size(total_bytes)} across {len(summaries)} repo(s)")

        rows = [
            {
                "Repo": s.get("repo_id"),
                "Size": _human_size(s.get("total_size_bytes", 0)),
                "Artifacts": s.get("artifact_count", 0),
                "Pinned": "yes" if s.get("any_pinned") else "-",
            }
            for s in summaries
        ]
        st.dataframe(rows, width="stretch", hide_index=True)

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

    selected_subfolders = {}

    st.subheader("📦 Sync Artifacts from MinIO to Backend")
    st.caption("Discover available artifacts from MinIO, select which ones to download, and sync to your backend.")

    if 'download_status' not in st.session_state:
        st.session_state.download_status = {}
    if 'download_progress' not in st.session_state:
        st.session_state.download_progress = {}
    if 'selected_artifacts' not in st.session_state:
        st.session_state.selected_artifacts = []
    if 'discovered_artifacts' not in st.session_state:
        st.session_state.discovered_artifacts = []

    col_backend, col_repo, col_discover = st.columns([2, 3, 2])

    with col_backend:
        backend = st.selectbox(
            "🎯 Backend",
            ["comfyui", "vllm"],
            key="sync_backend",
            on_change=_on_backend_change,
        )
        default_root = "/opt/models/comfyui" if backend == "comfyui" else "/opt/models/vllm"
        placement_root = st.text_input("📁 Placement root", value=default_root, key="sync_root")

    with col_repo:
        repo_id = st.text_input(
            "📦 Repo ID (HuggingFace format)",
            placeholder="Comfy-Org/MiniMax-H3",
            key="sync_repo_id",
            help="Format: organization/repo_name"
        )

    with col_discover:
        st.write("")
        st.write("")
        if st.button("🔍 Discover from MinIO", type="primary", width="stretch"):
            if not repo_id:
                st.error("Please enter a Repo ID first")
            else:
                with st.spinner(f"Discovering artifacts from {repo_id}..."):
                    try:
                        resp = httpx.get(f"{API_URL}/discover/{repo_id}/artifacts", timeout=60.0)
                        if resp.status_code == 200:
                            data = resp.json()
                            st.session_state.discovered_artifacts = data.get("artifacts", [])
                            st.session_state.selected_artifacts = []
                            st.success(f"Found {len(st.session_state.discovered_artifacts)} artifacts")
                            st.rerun()
                        else:
                            st.error(f"Error {resp.status_code}: {resp.text}")
                    except Exception as e:
                        st.error(f"Discovery failed: {e}")

    if st.session_state.discovered_artifacts:
        st.divider()
        st.subheader(f"📋 Artifacts in {repo_id}")

        comfy_subfolder = None
        if backend == "comfyui":
            comfy_fallback_options = ["checkpoints", "diffusion_models", "loras", "vae", "vae_approx", "clip", "text_encoders", "unet", "controlnet", "embeddings"]
            comfy_subfolder_options = get_subfolders(backend, placement_root, comfy_fallback_options)
            default_index = comfy_subfolder_options.index("checkpoints") if "checkpoints" in comfy_subfolder_options else 0
            comfy_subfolder = st.selectbox(
                "📂 Default ComfyUI Subfolder",
                comfy_subfolder_options,
                index=default_index,
                key="comfy_subfolder_default"
            )

        st.write("### Select artifacts to sync:")

        # Destination folder choices — same source as before (real subfolders from
        # the API, falling back to a hardcoded list), just computed once up front
        # instead of per-row.
        fallback_options = (
            ["checkpoints", "diffusion_models", "loras", "vae", "vae_approx", "clip", "text_encoders", "unet", "controlnet", "embeddings"]
            if backend == "comfyui" else ["models", "weights", "checkpoints"]
        )
        dest_options = get_subfolders(backend, placement_root, fallback_options)

        artifact_table = _build_artifact_table(
            st.session_state.discovered_artifacts,
            backend=backend,
            comfy_subfolder=comfy_subfolder,
            dest_options_by_backend=dest_options,
        )

        # Must run before dest_options / Destination Folder are finalized below —
        # both depend on knowing where files were actually found on disk.
        found_locations = _check_existing_locations(repo_id, placement_root, backend, artifact_table)

        # Make sure any auto-detected location is selectable even if it wasn't
        # in the discovered/fallback subfolder list.
        found_subfolders = {v if v else "(root)" for v in found_locations.values() if v is not None}
        dest_options = sorted(set(dest_options) | found_subfolders)

        artifact_table["Destination Folder"] = artifact_table.apply(_resolve_destination, axis=1, args=(found_locations,))
        artifact_table["On Disk"] = artifact_table["id"].apply(
            lambda aid: "✅ Yes" if found_locations.get(aid) is not None else ("❓ Unknown" if aid not in found_locations else "— No")
        )

        edited_table = st.data_editor(
            artifact_table,
            hide_index=True,
            width="stretch",
            disabled=["Artifact Name", "Type", "Size", "On Disk", "Status"],
            column_order=["Select", "Artifact Name", "Type", "Size", "Destination Folder", "On Disk", "Status"],
            column_config={
                "id": None,  # hides the id column from display while keeping it in the data
                "Select": st.column_config.CheckboxColumn("Select", width="small"),
                "Artifact Name": st.column_config.TextColumn("Artifact Name", width="large"),
                "Type": st.column_config.TextColumn("Type", width="small"),
                "Size": st.column_config.TextColumn("Size", width="small"),
                "Destination Folder": st.column_config.SelectboxColumn(
                    "Destination Folder",
                    options=dest_options,
                    width="medium",
                ),
                "On Disk": st.column_config.TextColumn("On Disk", width="small"),
            },
            key=f"artifact_editor_{repo_id}",
        )

        edited_table["Status"] = edited_table["id"].apply(lambda aid: _status_label(repo_id, aid))

        # Rebuild selection + destination state from the edited table, keyed by
        # artifact id (never by name — two artifacts in the same repo can share
        # a filename, e.g. config.json across multiple units).
        selected_rows = edited_table[edited_table["Select"] == True]

        st.session_state.selected_artifacts = [
            f"{repo_id}_{aid}" for aid in selected_rows["id"].tolist()
        ]

        selected_subfolders = dict(zip(edited_table["id"], edited_table["Destination Folder"]))

        st.divider()

        # Must exist before either sync button below can reference it.
        status_placeholder = st.empty()

        col_sync_all, col_sync_selected, col_status_info = st.columns([1, 1, 2])

        with col_sync_all:
            if st.button("📥 Sync All Artifacts", type="primary", width="stretch"):
                all_artifacts = {
                    a.get("id"): a.get("name")
                    for a in st.session_state.discovered_artifacts
                    if a.get("id") and a.get("name")
                }
                if all_artifacts:
                    _sync_artifacts(
                        repo_id=repo_id,
                        artifacts=all_artifacts,
                        backend=backend,
                        placement_root=placement_root,
                        subfolders=selected_subfolders,
                        status_placeholder=status_placeholder,
                        base_table=edited_table,
                    )
                else:
                    st.warning("No artifacts to sync")

        with col_sync_selected:
            if st.button("📥 Sync Selected", type="primary", width="stretch"):
                if st.session_state.selected_artifacts:
                    id_to_name = {a.get("id"): a.get("name") for a in st.session_state.discovered_artifacts}
                    selected_ids = [a.replace(f"{repo_id}_", "") for a in st.session_state.selected_artifacts]
                    selected_artifacts_map = {
                        aid: id_to_name[aid] for aid in selected_ids if aid in id_to_name
                    }
                    _sync_artifacts(
                        repo_id=repo_id,
                        artifacts=selected_artifacts_map,
                        backend=backend,
                        placement_root=placement_root,
                        subfolders=selected_subfolders,
                        status_placeholder=status_placeholder,
                        base_table=edited_table,
                    )
                else:
                    st.warning("No artifacts selected")

        with col_status_info:
            st.caption("💡 Select artifacts using the checkboxes, then click 'Sync Selected'")

    else:
        st.info("👆 Enter a Repo ID and click 'Discover from MinIO' to see available artifacts")


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
