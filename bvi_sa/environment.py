"""Build and update navigation environments for the BVI simulation.

This module is part of the BVI ACT-R navigation simulation workflow.
"""

import hashlib
import time
from pathlib import Path

import networkx as nx
import osmnx as ox


def _get_graph_cache_path(center_point, dist):
    """Handle get graph cache path behavior."""
    cache_root = Path(__file__).resolve().parents[1] / "cache" / "graphs"
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_key = f"{center_point[0]:.6f}_{center_point[1]:.6f}_{dist}_walk"
    cache_name = f"{hashlib.sha1(cache_key.encode('utf-8')).hexdigest()}.graphml"
    return cache_root / cache_name


def _configure_osmnx_cache():
    """Handle configure osmnx cache behavior."""
    cache_dir = Path(__file__).resolve().parents[1] / "cache" / "osmnx"
    cache_dir.mkdir(parents=True, exist_ok=True)
    ox.settings.use_cache = True
    ox.settings.cache_folder = str(cache_dir)


def _download_graph_with_retry(center_point, dist, max_retries=3):
    """Handle download graph with retry behavior."""
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            return ox.graph_from_point(center_point, dist=dist, network_type="walk")
        except Exception as error:
            last_error = error
            if attempt < max_retries:
                wait_seconds = 1.5 * attempt
                print(
                    f"环境下载失败（第{attempt}次）: {error}; {wait_seconds:.1f}s后重试..."
                )
                time.sleep(wait_seconds)
    raise RuntimeError("环境下载连续失败") from last_error


def build_route_phases(graph, start_node, goal_node):
    """Handle build route phases behavior."""
    try:
        path_nodes = nx.shortest_path(graph, start_node, goal_node)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return [{"type": "path", "edges": [], "nodes": [start_node]}]

    intersection_nodes = {n for n in graph.nodes if graph.degree[n] >= 3}

    phases = []
    current_path_edges: list = []
    current_path_nodes: list = [path_nodes[0]]

    for i in range(len(path_nodes) - 1):
        u, v = path_nodes[i], path_nodes[i + 1]
        current_path_edges.append((u, v))
        current_path_nodes.append(v)

        if v in intersection_nodes and v != goal_node:
            phases.append(
                {
                    "type": "path",
                    "edges": list(current_path_edges),
                    "nodes": list(current_path_nodes),
                }
            )
            phases.append(
                {
                    "type": "crossing",
                    "node": v,
                }
            )
            current_path_edges = []
            current_path_nodes = [v]

    if current_path_edges:
        phases.append(
            {
                "type": "path",
                "edges": list(current_path_edges),
                "nodes": list(current_path_nodes),
            }
        )

    if not phases:
        phases = [{"type": "path", "edges": [], "nodes": [start_node]}]

    return phases


def load_environment(center_point=(-33.8688, 151.2093), dist=500):
    """Handle load environment behavior."""
    _configure_osmnx_cache()
    graph_cache_path = _get_graph_cache_path(center_point, dist)

    graph = None
    if graph_cache_path.exists():
        try:
            graph = ox.load_graphml(graph_cache_path)
            print(f"Loaded cached street network: {graph_cache_path}")
        except Exception as error:
            print(f"读取本地图缓存失败，改为重新下载: {error}")

    if graph is None:
        print("Downloading Sydney street network (small area for simulation)...")
        graph = _download_graph_with_retry(center_point, dist)
        try:
            ox.save_graphml(graph, graph_cache_path)
            print(f"Saved street network cache: {graph_cache_path}")
        except Exception as error:
            print(f"保存本地图缓存失败: {error}")

    start_node = 8588632
    goal_node = 13644962503
    route_phases = build_route_phases(graph, start_node, goal_node)

    n_crossings = sum(1 for p in route_phases if p["type"] == "crossing")
    print(f"Network: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    print(f"Start: {start_node}, Goal: {goal_node}")
    print(f"Route phases: {len(route_phases)} 段（含 {n_crossings} 个路口阶段）")
    return graph, start_node, goal_node, route_phases
