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


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _sync_artifacts(repo_id: str, artifacts: Dict[str, str], backend: str, placement_root: str, subfolders: Dict[str, str] = None):
    """
    Sincroniza artifacts asociando cada uno a su subcarpeta correspondiente.
    `artifacts` maps artifact_id -> artifact_name. Names may repeat within
    a repo (e.g. multiple units each having a config.json); id is what's
    guaranteed unique.
    """
    subfolders = subfolders or {}

    with st.spinner(f"Syncing {len(artifacts)} artifact(s) from {repo_id}..."):
        for artifact_id, name in artifacts.items():
            key = f"{repo_id}_{artifact_id}"
            st.session_state.download_status[key] = "downloading"

            target_subfolder = subfolders.get(artifact_id, "")
            final_placement = os.path.join(placement_root, target_subfolder).rstrip("/")

            payload = {
                "repo_id": repo_id,
                "artifacts": [artifact_id],   # was [name] — model-manager now filters by id
                "backend": backend,
                "placement_root": final_placement,
            }

            try:
                resp = api_post(f"/discover/{repo_id}/sync-batch", json=payload)
                if resp.status_code < 400:
                    st.session_state.download_status[key] = "completed"
                    st.session_state.download_progress[key] = 100
                else:
                    st.session_state.download_status[key] = "error"
                    st.error(f"Error syncing {name}: {resp.text}")
            except Exception as e:
                st.session_state.download_status[key] = "error"
                st.error(f"Exception syncing {name}: {e}")

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
            key="sync_backend"
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
        
        col_select, col_artifact, col_size, col_dest, col_status = st.columns([0.5, 3, 1, 2, 1.5])
        with col_select:
            st.write("**Select**")
        with col_artifact:
            st.write("**Artifact Name**")
        with col_size:
            st.write("**Size**")
        with col_dest:
            st.write("**Destination Folder**")
        with col_status:
            st.write("**Status**")
        
        st.divider()
        
        for idx, artifact in enumerate(st.session_state.discovered_artifacts):
            artifact_name = artifact.get("name", "unknown")
            artifact_id = artifact.get("id")
            if not artifact_id:
                # Fallback for any artifact payload missing an id — keeps old
                # behavior (and the same crash risk) only in that edge case.
                artifact_id = f"{artifact_name}_{idx}"
            artifact_key = f"{repo_id}_{artifact_id}"
            artifact_size = artifact.get("size_bytes", 0)
            artifact_type = artifact.get("artifact_type", artifact.get("type", "unknown"))
            
            # Mapeo inteligente con prioridad a vae_approx para modelos TAESD
            suggested_folder = "checkpoints"
            name_lower = artifact_name.lower()
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
            
            c_select, c_artifact, c_size, c_dest, c_status = st.columns([0.5, 3, 1, 2, 1.5])
            
            with c_select:
                is_selected = st.checkbox(
                    f"Select {artifact_name}",
                    key=f"sel_{artifact_key}",
                    value=artifact_key in st.session_state.selected_artifacts,
                    label_visibility="collapsed"
                )
                if is_selected and artifact_key not in st.session_state.selected_artifacts:
                    st.session_state.selected_artifacts.append(artifact_key)
                elif not is_selected and artifact_key in st.session_state.selected_artifacts:
                    st.session_state.selected_artifacts.remove(artifact_key)
            
            with c_artifact:
                st.write(f"**{artifact_name}**")
                st.caption(f"Type: {artifact_type}")
            
            with c_size:
                st.write(_human_size(artifact_size))
                        
            with c_dest:
                fallback_options = (
                    ["checkpoints", "diffusion_models", "loras", "vae", "vae_approx", "clip", "text_encoders", "unet", "controlnet", "embeddings"]
                    if backend == "comfyui" else ["models", "weights", "checkpoints"]
                )
                dest_options = get_subfolders(backend, placement_root, fallback_options)

            dest_key = f"dest_{artifact_key}"
            if dest_key not in st.session_state:
                st.session_state[dest_key] = default_dest

            chosen_folder = st.selectbox(
                f"Destination for {artifact_name}",
                dest_options,
                key=dest_key,
                label_visibility="collapsed"
            )
            selected_subfolders[artifact_id] = chosen_folder

            with c_status:
                if artifact_key in st.session_state.download_status:
                    status = st.session_state.download_status[artifact_key]
                    if status == "completed":
                        st.success("✅ Done")
                    elif status == "downloading":
                        progress = st.session_state.download_progress.get(artifact_key, 0)
                        st.progress(progress, text=f"{progress}%")
                    elif status == "error":
                        st.error("❌ Error")
                else:
                    st.write("⏳ Pending")
        
        st.divider()
        
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
                        subfolders=selected_subfolders
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
                        subfolders=selected_subfolders
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
